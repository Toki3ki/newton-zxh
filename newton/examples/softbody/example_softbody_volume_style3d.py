# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare volume change during symmetric two-ended stretching.

Blue: VBD; orange: SIPD-Semi; green: SNH-matched scalar PD.
Both X end faces are clamped and pulled apart continuously while running. Interior
particles contract freely. Gravity, contact and material damping are disabled.
The scalar method uses a different energy from the Stable Neo-Hookean methods.

Success criteria: finite states, prescribed grip positions, positive tet volumes.

Runs:
- `python -m newton.examples softbody_volume_style3d`
- `python -m newton.examples softbody_volume_style3d --viewer null --num-frames 180 --test`
"""

from collections import deque
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
import newton.examples


@wp.kernel
def move_grips(
    rest: wp.array[wp.vec3],
    mass: wp.array[float],
    displacement: float,
    velocity: float,
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
):
    i = wp.tid()
    if mass[i] == 0.0:
        sign = float(1.0)
        if rest[i][0] < 0.0:
            sign = -1.0
        positions[i] = rest[i] + wp.vec3(sign * displacement, 0.0, 0.0)
        velocities[i] = wp.vec3(sign * velocity, 0.0, 0.0)


class _Layer:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder = newton.ModelBuilder()
        newton.solvers.SolverStyle3D.register_custom_attributes(builder)
        builder.gravity = (0.0, 0.0, 0.0)
        youngs_modulus = args.youngs_modulus
        poissons_ratio = args.poissons_ratio
        mu = youngs_modulus / (2.0 * (1.0 + poissons_ratio))
        lam = youngs_modulus * poissons_ratio / (
            (1.0 + poissons_ratio) * (1.0 - 2.0 * poissons_ratio)
        )
        builder.add_soft_grid(
            # Newton's default viewer uses Z as the up axis.
            pos=wp.vec3(-0.9, -0.27, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=10,
            dim_y=3,
            dim_z=3,
            cell_x=0.18,
            cell_y=0.18,
            cell_z=0.18,
            density=800.0,
            k_mu=mu,
            k_lambda=lam,
            k_damp=0.0,
            fix_left=True,
            fix_right=True,
            particle_radius=0.02,
            color=wp.vec3(*args.color),
        )
        builder.color()

        self.model = builder.finalize()
        self.model.soft_contact_ke = 2.0e3
        self.model.soft_contact_kd = 2.0e1
        self.model.soft_contact_mu = 0.3

        if args.tet_solver == "vbd":
            self.solver = newton.solvers.SolverVBD(
                self.model,
                iterations=args.iterations,
                particle_enable_self_contact=False,
                particle_enable_tile_solve=False,
            )
        else:
            self.solver = newton.solvers.SolverStyle3D(
                self.model,
                config=newton.solvers.SolverStyle3D.Config(
                    iterations=args.iterations,
                    linear_iterations=8,
                    enable_surface_constraints=False,
                    enable_tet_sipd=True,
                    tet_solver=args.tet_solver,
                    tet_sipd_relaxation=1.0,
                    tet_pd_volume_ratio=args.pd_volume_ratio,
                ),
            )
            self.solver.collision.stiff_vf = 0.0
            self.solver.collision.stiff_ee = 0.0
            self.solver.collision.stiff_ef = 0.0

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = newton.CollisionPipeline(self.model).contacts()

        self.fixed = self.model.particle_mass.numpy() == 0.0
        self.rest = self.model.particle_q.numpy().copy()

        self.strain_rate = args.strain_rate
        self.youngs_modulus = youngs_modulus
        self.poissons_ratio = poissons_ratio
        self.mu = mu
        self.lam = lam
        self.length = float(np.ptp(self.rest[:, 0]))
        self.tets = self.model.tet_indices.numpy()
        self.inv_dm = self.model.tet_poses.numpy().astype(np.float64)
        self.rest_volumes = 1.0 / (6.0 * np.abs(np.linalg.det(self.inv_dm)))
        self.volume_ratio = self.min_j = self.max_j = 1.0

    def grip_motion(self, time):
        # Motion uses simulation time, so the viewer pause also stops the grips.
        speed = 0.5 * self.length * self.strain_rate
        return speed * time, speed

    def apply_grips(self, state, time):
        displacement, velocity = self.grip_motion(time)
        wp.launch(
            move_grips,
            dim=self.model.particle_count,
            inputs=[self.model.particle_q, self.model.particle_mass, displacement, velocity],
            outputs=[state.particle_q, state.particle_qd],
            device=self.model.device,
        )

    def step(self):
        for substep in range(self.sim_substeps):
            time = self.sim_time + (substep + 1) * self.sim_dt
            self.apply_grips(self.state_0, time)
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.apply_grips(self.state_1, time)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt
        q = self.state_0.particle_q.numpy().astype(np.float64)
        vertices = q[self.tets]
        ds = np.stack([vertices[:, j] - vertices[:, 0] for j in (1, 2, 3)], axis=-1)
        jacobians = np.linalg.det(ds @ self.inv_dm)
        self.volume_ratio = float(np.dot(jacobians, self.rest_volumes) / self.rest_volumes.sum())
        self.min_j = float(jacobians.min())
        self.max_j = float(jacobians.max())

    def test_final(self):
        if self.min_j <= 0.0:
            raise AssertionError(f"{self.name}: inverted tetrahedron, min J={self.min_j}")
        print(f"{self.name}: V/V0={self.volume_ratio:.5f}, J=[{self.min_j:.5f}, {self.max_j:.5f}]")


class Example:
    def __init__(self, viewer, args):
        if args.iterations < 1:
            raise ValueError("iterations must be positive")
        self.viewer = viewer
        self.sim_time = 0.0
        if not np.isfinite(args.strain_rate) or args.strain_rate <= 0.0:
            raise ValueError("strain-rate must be finite and positive")
        if not np.isfinite(args.youngs_modulus) or args.youngs_modulus <= 0.0:
            raise ValueError("youngs-modulus must be finite and positive")
        if not np.isfinite(args.poissons_ratio) or not -1.0 < args.poissons_ratio < 0.5:
            raise ValueError("poissons-ratio must be finite and satisfy -1 < nu < 0.5")
        self.layers = []
        viewer.picking_enabled = False
        specs = [
            ("vbd", "VBD", (0.1, 0.35, 0.9)),
            ("semi_implicit", "SIPD-Semi", (0.95, 0.30, 0.04)),
            ("pd", "Scalar PD - SNH match", (0.12, 0.65, 0.25)),
        ]
        if args.tet_solver != "compare":
            specs = [spec for spec in specs if spec[0] == args.tet_solver]
            if not specs:
                specs = [("pcg", "SIPD-PCG", (0.95, 0.30, 0.04))]
        for index, (kind, name, color) in enumerate(specs):
            layer = _Layer(
                viewer,
                SimpleNamespace(
                    tet_solver=kind,
                    iterations=args.iterations,
                    color=color,
                    strain_rate=args.strain_rate,
                    pd_volume_ratio=args.pd_volume_ratio,
                    youngs_modulus=args.youngs_modulus,
                    poissons_ratio=args.poissons_ratio,
                ),
            )
            layer.name = name
            layer.color = color
            layer.volume_history = deque([1.0], maxlen=1800)
            viewer.activate(name)
            viewer.set_model(layer.model)
            shift = (index - 0.5 * (len(specs) - 1)) * 1.5
            viewer.set_layer_transform(name, wp.transform(wp.vec3(0.0, shift, 0.0), wp.quat_identity()))
            self.layers.append(layer)
        self.iterations = args.iterations
        self.pd_volume_ratio = args.pd_volume_ratio
        self.youngs_modulus = args.youngs_modulus
        self.poissons_ratio = args.poissons_ratio
        self.time_history = deque([0.0], maxlen=1800)
        self._plot_window_positioned = False
        self._owns_implot_context = False
        if hasattr(viewer, "register_ui_callback"):
            viewer.register_ui_callback(self._plot_panel, position="free")

    def step(self):
        for layer in self.layers:
            self.viewer.activate(layer.name)
            layer.step()
            layer.volume_history.append(layer.volume_ratio)
        self.sim_time += self.layers[0].frame_dt
        self.time_history.append(self.sim_time)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        for layer in self.layers:
            self.viewer.activate(layer.name)
            self.viewer.log_state(layer.state_0)
        self.viewer.end_frame()

    def _plot_panel(self, ui):
        from imgui_bundle import implot  # noqa: PLC0415

        if not self._plot_window_positioned:
            size = ui.get_io().display_size
            width = min(520.0, max(360.0, size.x * 0.38))
            height = min(360.0, size.y - 30.0)
            ui.set_next_window_pos(ui.ImVec2(size.x - width - 10.0, size.y - height - 10.0))
            ui.set_next_window_size(ui.ImVec2(width, height))
            self._plot_window_positioned = True
        expanded, _ = ui.begin("Plots - Volume Comparison")
        if expanded:
            if implot.get_current_context() is None:
                implot.create_context()
                self._owns_implot_context = True
            ui.text("V/V0 = 1: conserved volume. Click legend to toggle curves.")
            if implot.begin_plot("Volume ratio", ui.ImVec2(-1.0, -1.0)):
                implot.setup_axes("Time (s)", "V/V0", implot.AxisFlags_.auto_fit, implot.AxisFlags_.auto_fit)
                times = np.asarray(self.time_history, dtype=np.float64)
                for layer in self.layers:
                    implot.plot_line(
                        layer.name,
                        times,
                        np.asarray(layer.volume_history, dtype=np.float64),
                        spec=implot.Spec(line_color=ui.ImVec4(*layer.color, 1.0), line_weight=2.0),
                    )
                implot.end_plot()
        ui.end()

    def cleanup(self):
        if self._owns_implot_context:
            from imgui_bundle import implot  # noqa: PLC0415

            implot.destroy_context()
            self._owns_implot_context = False

    def gui(self, ui):
        ui.text("Blue: VBD | Orange: SIPD-Semi | Green: Scalar PD SNH match")
        ui.text(f"Outer iterations: {self.iterations} | substeps: 4 | material damping: OFF")
        ui.text(f"E={self.youngs_modulus:g} Pa | nu={self.poissons_ratio:g}")
        ui.text(f"PD volume multiplier={self.pd_volume_ratio:g}; PCG: 8")
        ui.text("Both X ends pulled apart; gravity/contact OFF; end cross-sections clamped.")
        layer = self.layers[0]
        displacement, _ = layer.grip_motion(self.sim_time)
        ui.text(f"Length ratio: {1.0 + 2.0 * displacement / layer.length:.3f}; strain rate: {layer.strain_rate:.2f}/s")
        ui.text("Continuous stretching. Pause the viewer to stop the grips.")
        for layer in self.layers:
            ui.text(
                f"{layer.name}: V/V0={layer.volume_ratio:.4f} "
                f"({100.0 * (layer.volume_ratio - 1.0):+.2f}%), "
                f"J=[{layer.min_j:.3f}, {layer.max_j:.3f}]"
            )

    def test_post_step(self):
        for layer in self.layers:
            q = layer.state_0.particle_q.numpy()
            v = layer.state_0.particle_qd.numpy()
            if not np.isfinite(q).all() or not np.isfinite(v).all():
                raise AssertionError(f"{layer.name}: non-finite state")
            expected = layer.rest[layer.fixed].copy()
            displacement, _ = layer.grip_motion(layer.sim_time)
            expected[:, 0] += np.sign(expected[:, 0]) * displacement
            np.testing.assert_allclose(q[layer.fixed], expected, atol=1.0e-6, rtol=0.0)

    def test_final(self):
        self.test_post_step()
        for layer in self.layers:
            layer.test_final()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--tet-solver",
            choices=["compare", "vbd", "semi_implicit", "pcg", "pd"],
            default="compare",
            help="Choose one method or compare all three solver paths",
        )
        parser.add_argument("--iterations", type=int, default=16)
        parser.add_argument("--strain-rate", type=float, default=0.5, help="Engineering strain per simulation second")
        parser.add_argument("--youngs-modulus", type=float, default=2.5e5, help="Young's modulus [Pa]")
        parser.add_argument("--poissons-ratio", type=float, default=0.49, help="Poisson's ratio")
        parser.add_argument(
            "--pd-volume-ratio",
            type=float,
            default=1.0,
            help="Multiplier of the matched 3*lambda volume weight; 0 disables volume preservation",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
