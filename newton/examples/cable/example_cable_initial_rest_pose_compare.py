# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compare VBD and Style3D local/global rotation with different initial/rest poses.

Each solver has two cables: a helix with straight rest shape (negative X),
and a straight cable with helical rest shape (positive X). Geometry, authored
stiffness, density, time steps and nonlinear sweep counts match across solvers.
Scene construction is shared with cable_initial_rest_pose, including friction
and damping inputs. VBD uses them; Viper-PD currently omits those terms. Its
fixed-section angular energies also differ; this is a capability comparison.

Success criteria:
- All three methods preserve finite, bounded states and connected capsules.
- Both initial/rest cases move in response to their nonzero initial strain.
- Each method starts from the same poses, velocities and structural rest data,
  up to the display translation. Independent models prevent cross-method contact.

Runs:
- ``python -m newton.examples cable_initial_rest_pose_compare``
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.cable.example_cable_initial_rest_pose import create_scene
from newton.examples.cable.example_cable_twist_compare import material_ticks


class Example:
    def __init__(self, viewer, args):
        self.viewer, self.args = viewer, args
        if min(args.substeps, args.iterations, args.linear_iterations) < 1:
            raise ValueError("Substeps and iteration counts must be positive.")
        self.sim_time = 0.0
        self.dt = 1.0 / (60 * args.substeps)
        self.layer_modes = ["vbd", "local", "global"]
        self.models, self.solvers, self.states, self.controls, self.pipelines, self.layer_contacts = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        self.initial_poses, self.cable_bodies, self.joint_geometry = [], [], []
        self.colors = [(0.15, 0.55, 1.0), (1.0, 0.4, 0.08), (0.2, 0.9, 0.4)]
        display = newton.ModelBuilder()
        for index, (mode, color) in enumerate(zip(self.layer_modes, self.colors, strict=True)):
            builder, cables = create_scene(
                centers=(wp.vec3(-3.0, (index - 1) * 1.8, 0.245), wp.vec3(3.0, (index - 1) * 1.8, 0.245)),
                color=color,
            )
            model = builder.finalize(device=args.device)
            pipeline = newton.CollisionPipeline(model, contact_matching="latest")
            if mode == "vbd":
                solver = newton.solvers.SolverVBD(model, iterations=args.iterations)
            else:
                solver = newton.solvers.SolverStyle3D(
                    model,
                    config=newton.solvers.SolverStyle3D.Config(
                        enable_cable_viper=True,
                        cable_viper_fixed_scale=True,
                        cable_viper_rotation_solver=mode,
                        iterations=args.iterations,
                        linear_iterations=args.linear_iterations,
                    ),
                )
            self.models.append(model)
            self.solvers.append(solver)
            self.states.append([model.state(), model.state()])
            self.controls.append(model.control())
            self.pipelines.append(pipeline)
            self.layer_contacts.append(pipeline.contacts())
            self.cable_bodies.append(cables)
            self.initial_poses.append(model.body_q.numpy())
            self.joint_geometry.append(
                (
                    model.joint_parent.numpy(),
                    model.joint_child.numpy(),
                    model.joint_X_p.numpy(),
                    model.joint_X_c.numpy(),
                )
            )
            display.add_builder(builder)
        self.model = display.finalize(device=args.device)
        self.display_state = self.model.state()
        self.tick_start = wp.zeros(self.model.body_count, dtype=wp.vec3, device=self.model.device)
        self.tick_end = wp.zeros_like(self.tick_start)
        viewer.set_model(self.model)
        viewer.set_camera(pos=wp.vec3(0.0, -10.5, 10.5), pitch=-45.0, yaw=90.0)
        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture(device=self.model.device) as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        """Advance all methods using identical time steps and contact settings."""
        for _ in range(self.args.substeps):
            for i, solver in enumerate(self.solvers):
                a, b = self.states[i]
                a.clear_forces()
                self.pipelines[i].collide(a, self.layer_contacts[i])
                solver.step(a, b, self.controls[i], self.layer_contacts[i], self.dt)
                # Keep capture inputs stable, including with odd substep counts.
                a.assign(b)

    def step(self):
        """Advance one simulation frame."""
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += 1.0 / 60

    def render(self):
        """Render the six cables with material-axis markers."""
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
        """Explain the layout and shared settings."""
        ui.text("Blue: VBD | Orange: Viper-PD local | Green: Viper-PD global")
        ui.text("Left (-X): helix initial / straight rest")
        ui.text("Right (+X): straight initial / helix rest")
        ui.text("White ticks: material X axes. All cables are free, with fixed section.")
        ui.text("Original scene: bend/twist 500, material damping 1, cable friction 0.1")
        ui.text(f"Substeps {self.args.substeps} | outer {self.args.iterations} | PD PCG {self.args.linear_iterations}")
        ui.text("Viper-PD currently omits friction and material damping.")
        ui.text("Same authored parameters; angular energies and convergence differ.")

    def test_post_step(self):
        """Reject non-finite states, excessive motion and disconnected capsule anchors."""
        for mode, states, initial, geometry in zip(
            self.layer_modes, self.states, self.initial_poses, self.joint_geometry, strict=True
        ):
            poses, velocities = states[0].body_q.numpy(), states[0].body_qd.numpy()
            if not np.isfinite(poses).all() or not np.isfinite(velocities).all():
                raise ValueError(f"{mode}: non-finite cable state.")
            if np.max(np.linalg.norm(poses[:, :3] - initial[:, :3], axis=1)) > 10:
                raise ValueError(f"{mode}: cable exceeded the scene bounds.")
            np.testing.assert_allclose(np.linalg.norm(poses[:, 3:], axis=1), 1.0, atol=1e-4)
            parents, children, xp, xc = geometry
            for j in np.flatnonzero(parents >= 0):
                p = wp.transform_point(wp.transform(*poses[parents[j]]), wp.vec3(*xp[j, :3]))
                c = wp.transform_point(wp.transform(*poses[children[j]]), wp.vec3(*xc[j, :3]))
                if wp.length(p - c) > 0.025:
                    raise ValueError(f"{mode}: joint {j} anchor gap exceeds the capsule radius.")

    def test_final(self):
        """Verify both initially strained cases respond in each solver."""
        self.test_post_step()
        if self.sim_time < 0.1:
            return
        for mode, states, initial, cables in zip(
            self.layer_modes, self.states, self.initial_poses, self.cable_bodies, strict=True
        ):
            poses = states[0].body_q.numpy()
            for bodies in cables:
                # Quaternion sign is immaterial to orientation.
                dots = np.abs(np.sum(poses[bodies, 3:] * initial[bodies, 3:], axis=1))
                if np.max(1.0 - dots) < 1e-6:
                    raise ValueError(f"{mode}: an initially strained cable did not rotate.")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--substeps", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--linear-iterations", type=int, default=10)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
