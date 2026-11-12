# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compare blue VBD, orange Style3D local and green Style3D global cables.

Identical zigzag geometry, endpoint spin, gravity and substeps; independent
simulation models prevent cross-contact. Stiffness and outer iterations match
by default; explicit PD overrides support material and convergence experiments.
White material-axis ticks expose axial spin on otherwise symmetric capsules.
Success: finite bounded states and matching prescribed endpoint rotations.
This is a visual diagnostic, not an assertion of equal material response.

Run: python -m newton.examples cable_twist_compare
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.cable.example_cable_twist import Example as TwistExample
from newton.examples.cable.example_cable_twist import spin_first_capsules_kernel


@wp.kernel
def material_ticks(poses: wp.array[wp.transform], starts: wp.array[wp.vec3], ends: wp.array[wp.vec3]):
    i = wp.tid()
    starts[i] = wp.transform_get_translation(poses[i])
    ends[i] = wp.transform_point(poses[i], wp.vec3(0.09, 0.0, 0.0))


class Example:
    def __init__(self, viewer, args):
        self.viewer, self.args = viewer, args
        if min(args.substeps, args.iterations, args.linear_iterations) < 1:
            raise ValueError("Iteration counts must be positive.")
        self.pd_stiffness = getattr(args, "pd_angular_stiffness", 2.5e4)
        self.pd_iterations = getattr(args, "pd_iterations", None)
        if self.pd_iterations is None:
            self.pd_iterations = args.iterations
        if self.pd_iterations < 1:
            raise ValueError("PD iteration count must be positive.")
        if (
            not np.isfinite([args.angular_stiffness, self.pd_stiffness]).all()
            or min(args.angular_stiffness, self.pd_stiffness) <= 0
        ):
            raise ValueError("Angular stiffness must be finite and positive.")
        self.sim_time = 0.0
        self.dt = 1.0 / (60 * args.substeps)
        self.models, self.solvers, self.states, self.controls, self.pipelines, self.layer_contacts = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        display = newton.ModelBuilder()
        self.pins, self.rates = [], []
        rotation_mode = getattr(args, "rotation_solver", "both")
        if rotation_mode in ("global", "both") and args.dynamic_scale:
            raise ValueError("Global director comparison requires fixed scale.")
        self.layer_modes = ["vbd", "local", "global"] if rotation_mode == "both" else ["vbd", rotation_mode]
        colors = [(0.15, 0.55, 1.0), (1.0, 0.4, 0.08), (0.2, 0.9, 0.4)]
        for index, mode in enumerate(self.layer_modes):
            color = colors[index]
            builder = newton.ModelBuilder()
            builder.default_shape_cfg.ke = 1e6
            builder.default_shape_cfg.kd = 0
            builder.default_shape_cfg.mu = 0  # VIPER currently has frictionless contacts.
            rod = TwistExample.create_cable_geometry_with_turns(
                None, pos=wp.vec3(-1.6, -2.6 + index * 3.6, 0.02), num_elements=64, length=6.4, radius=0.02
            )
            bodies, _ = builder.add_rod(
                rod=rod,
                stretch_stiffness=1e6,
                bend_stiffness=args.angular_stiffness if mode == "vbd" else self.pd_stiffness,
                twist_stiffness=args.angular_stiffness if mode == "vbd" else self.pd_stiffness,
                bend_damping=0,
                twist_damping=0,
                color=color,
                label=mode,
            )
            pin = bodies[0]
            builder.body_mass[pin] = 0
            builder.body_inv_mass[pin] = 0
            builder.body_inertia[pin] = wp.mat33(0.0)
            builder.body_inv_inertia[pin] = wp.mat33(0.0)
            builder.add_ground_plane()
            builder.color()
            model = builder.finalize(device=args.device)
            pipeline = newton.CollisionPipeline(model)
            contacts = pipeline.contacts()
            if index == 0:
                solver = newton.solvers.SolverVBD(model, iterations=args.iterations)
            else:
                solver = newton.solvers.SolverStyle3D(
                    model,
                    config=newton.solvers.SolverStyle3D.Config(
                        enable_cable_viper=True,
                        cable_viper_fixed_scale=not args.dynamic_scale,
                        cable_viper_rotation_solver=mode,
                        cable_viper_rotation_proximal=getattr(args, "rotation_proximal", 1.0),
                        iterations=self.pd_iterations,
                        linear_iterations=args.linear_iterations,
                    ),
                )
            self.models.append(model)
            self.solvers.append(solver)
            self.states.append([model.state(), model.state()])
            self.controls.append(model.control())
            self.pipelines.append(pipeline)
            self.layer_contacts.append(contacts)
            self.pins.append(wp.array([pin], dtype=int, device=model.device))
            self.rates.append(wp.array([args.spin_rate], dtype=float, device=model.device))
            # Display model combines geometry only; each solver owns its original model.
            display.add_builder(builder)
        self.model = display.finalize(device=args.device)
        self.display_state = self.model.state()
        self.tick_start = wp.zeros(self.model.body_count, dtype=wp.vec3, device=self.model.device)
        self.tick_end = wp.zeros_like(self.tick_start)
        viewer.set_model(self.model)
        viewer.set_camera(pos=wp.vec3(7, -9, 9), pitch=-40, yaw=130)
        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture(device=self.model.device) as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        """Advance both solvers with identical prescribed motion and time steps."""
        for _ in range(self.args.substeps):
            for i in range(len(self.models)):
                a, b = self.states[i]
                a.clear_forces()
                wp.launch(
                    spin_first_capsules_kernel,
                    dim=1,
                    inputs=[self.pins[i], self.rates[i], self.dt],
                    outputs=[a.body_q, b.body_q],
                    device=self.models[i].device,
                )
                self.pipelines[i].collide(a, self.layer_contacts[i])
                self.solvers[i].step(a, b, self.controls[i], self.layer_contacts[i], self.dt)
                # Stable buffer identity across CUDA graph replays, including odd substeps.
                a.assign(b)

    def step(self):
        """Advance one frame without rendering overhead."""
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += 1.0 / 60

    def render(self):
        """Show original capsule geometry and material-axis markers."""
        offset = 0
        for model, states in zip(self.models, self.states, strict=True):
            wp.copy(self.display_state.body_q, states[0].body_q, dest_offset=offset, count=model.body_count)
            offset += model.body_count
        wp.launch(
            material_ticks,
            dim=self.model.body_count,
            inputs=[self.display_state.body_q, self.tick_start, self.tick_end],
            device=self.model.device,
        )
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.display_state)
        self.viewer.log_lines("material_x_axes", self.tick_start, self.tick_end, (1.0, 1.0, 1.0))
        self.viewer.end_frame()

    def gui(self, ui):
        """Identify the solvers and explain the comparison settings."""
        ui.text(
            " | ".join(
                f"{color}: {mode}" for color, mode in zip(["Blue", "Orange", "Green"], self.layer_modes, strict=False)
            )
        )
        ui.text("White ticks: material X axes (show twist)")
        ui.text(
            f"Substeps {self.args.substeps} | VBD outer {self.args.iterations} | PD outer {self.pd_iterations}"
            f" | PD PCG {self.args.linear_iterations}"
        )
        ui.text(f"Bend/twist stiffness: VBD {self.args.angular_stiffness:g} | PD {self.pd_stiffness:g}")
        ui.text(f"Spin {self.args.spin_rate} rad/s | fixed scale: {not self.args.dynamic_scale}")
        ui.text("Both: no friction or material damping. Different discretization/contact response.")

    def test_final(self):
        """Verify finite states, bounded motion and consistent endpoint drives."""
        rotations = []
        for states in self.states:
            poses = states[0].body_q.numpy()
            if not np.isfinite(poses).all() or np.max(np.abs(poses[:, :3])) > 100:
                raise ValueError("Cable comparison became unbounded.")
            rotations.append(poses[0, 3:])
        for rotation in rotations[1:]:
            np.testing.assert_allclose(rotations[0], rotation, atol=1e-5)


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--linear-iterations", type=int, default=10)
    parser.add_argument("--angular-stiffness", type=float, default=1e3)
    parser.add_argument("--spin-rate", type=float, default=0.5)
    parser.add_argument("--rotation-solver", choices=["local", "global", "both"], default="both")
    parser.add_argument("--rotation-proximal", type=float, default=1.0)
    parser.add_argument(
        "--dynamic-scale", action="store_true", help="Enable VIPER scale dynamics instead of Cosserat mode."
    )
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
