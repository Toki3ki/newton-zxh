# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cable Plectoneme Compare

Runs the same hanging, counter-twisted, self-contacting rod with VBD and the
Style3D VIPER Cable solver side by side.

Success criteria:
- Both cables remain finite, connected, and bounded.
- Counter-twist drives both centerlines out of their initial plane.
- Capsule self-contact prevents non-neighbouring strands from freely crossing.

Runs:
- ``python -m newton.examples cable_plectoneme_compare``
- ``python -m newton.examples cable_plectoneme_compare --solver style3d``
- ``python -m newton.examples cable_plectoneme_compare --solver vbd``
"""

import math

import numpy as np
import warp as wp

import newton
import newton.examples


@wp.kernel
def drive_endpoints_kernel(
    root_body: int,
    tip_body: int,
    root_pos: wp.vec3,
    tip_pos: wp.vec3,
    root_rest_rot: wp.quat,
    tip_rest_rot: wp.quat,
    twist_angle: float,
    body_q0: wp.array[wp.transform],
    body_q1: wp.array[wp.transform],
):
    """Prescribe equal and opposite end rotations about the end tangents."""
    root_axis = wp.quat_rotate(root_rest_rot, wp.vec3(0.0, 0.0, 1.0))
    tip_axis = wp.quat_rotate(tip_rest_rot, wp.vec3(0.0, 0.0, 1.0))
    root_rot = wp.mul(wp.quat_from_axis_angle(root_axis, -0.5 * twist_angle), root_rest_rot)
    tip_rot = wp.mul(wp.quat_from_axis_angle(tip_axis, 0.5 * twist_angle), tip_rest_rot)
    root_pose = wp.transform(root_pos, root_rot)
    tip_pose = wp.transform(tip_pos, tip_rot)
    body_q0[root_body] = root_pose
    body_q1[root_body] = root_pose
    body_q0[tip_body] = tip_pose
    body_q1[tip_body] = tip_pose


class _Simulation:
    """Own one plectoneme model, collision pipeline, solver, and state pair."""

    NUM_ELEMENTS = 48
    END_SEPARATION = 1.2
    TOP_HEIGHT = 1.8
    SAG_DEPTH = 0.9
    TWIST_TURNS = 6.0
    SETTLE_TIME = 1.0
    TWIST_TIME = 6.0
    HOLD_TIME = 2.0

    def __init__(self, solver_type: str, x_offset: float, device, substeps: int, iterations: int):
        self.solver_type = solver_type
        self.frame_dt = 1.0 / 60.0
        self.sim_substeps = substeps + substeps % 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        nodes = self._hanging_arc_nodes()
        nodes[:, 0] += x_offset
        segment_length = float(np.mean(np.linalg.norm(np.diff(nodes, axis=0), axis=1)))
        radius = 0.42 * segment_length
        shape_cfg = newton.ModelBuilder.ShapeConfig(
            ke=3.0e5,
            kd=25.0,
            kf=0.0,
            mu=0.0,
            gap=0.6 * segment_length,
        )
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        rod = newton.Rod([wp.vec3(*point) for point in nodes], radius=radius)
        color = (0.15, 0.55, 1.0) if solver_type == "vbd" else (1.0, 0.4, 0.08)
        self.bodies, joints = builder.add_rod(
            rod=rod,
            cfg=shape_cfg,
            total_mass=0.4,
            stretch_stiffness=1.0e6,
            bend_stiffness=6.0,
            twist_stiffness=400.0,
            # Keep VBD material damping off while comparing stiffness ratios.
            bend_damping=0.0,
            twist_damping=0.0,
            color=color,
            label=f"{solver_type}_plectoneme",
            wrap_in_articulation=False,
        )
        self.bodies = list(map(int, self.bodies))
        self._filter_neighbours(builder, self.bodies, span=2)
        self.root_body = self.bodies[0]
        self.tip_body = self.bodies[-1]
        for body in (self.root_body, self.tip_body):
            builder.body_mass[body] = 0.0
            builder.body_inv_mass[body] = 0.0
            builder.body_inertia[body] = wp.mat33(0.0)
            builder.body_inv_inertia[body] = wp.mat33(0.0)
        builder.add_articulation(joints, label="plectoneme_articulation")
        if solver_type == "vbd":
            builder.color()

        self.builder = builder
        self.model = builder.finalize(device=device)
        if solver_type == "vbd":
            self.solver = newton.solvers.SolverVBD(
                self.model,
                iterations=iterations,
                rigid_contact_history=True,
                rigid_body_contact_buffer_size=1024,
            )
        else:
            config = newton.solvers.SolverStyle3D.Config(
                iterations=iterations,
                linear_iterations=16,
                cable_stretch_stiffness=1.0e7,
                cable_bend_stiffness=2.0,
                cable_twist_stiffness=100.0,
            )
            self.solver = newton.solvers.SolverStyle3D(
                self.model,
                config=config,
                dynamics=newton.solvers.SolverStyle3D.Dynamics.CABLE,
            )

        self.pipeline = newton.CollisionPipeline(self.model, contact_matching="latest")
        self.contacts = self.pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        poses = self.state_0.body_q.numpy()
        self.root_pos = poses[self.root_body, :3].copy()
        self.tip_pos = poses[self.tip_body, :3].copy()
        self.root_rot = wp.quat(*poses[self.root_body, 3:7])
        self.tip_rot = wp.quat(*poses[self.tip_body, 3:7])
        poses[self.bodies[len(self.bodies) // 2], 1] += 1.0e-3
        self.state_0.body_q.assign(poses)
        self.state_1.body_q.assign(poses)

    @classmethod
    def _hanging_arc_nodes(cls) -> np.ndarray:
        u = np.linspace(0.0, 1.0, cls.NUM_ELEMENTS + 1)
        return np.column_stack(
            [
                (u - 0.5) * cls.END_SEPARATION,
                np.zeros_like(u),
                cls.TOP_HEIGHT - cls.SAG_DEPTH * np.sin(math.pi * u),
            ]
        )

    @staticmethod
    def _filter_neighbours(builder, bodies: list[int], span: int) -> None:
        for i, body_i in enumerate(bodies):
            for body_j in bodies[i + 1 : i + span + 1]:
                for shape_i in builder.body_shapes.get(body_i, []):
                    for shape_j in builder.body_shapes.get(body_j, []):
                        builder.add_shape_collision_filter_pair(int(shape_i), int(shape_j))

    @staticmethod
    def _smoothstep(value: float) -> float:
        value = min(1.0, max(0.0, value))
        return value * value * (3.0 - 2.0 * value)

    def _twist_command(self) -> float:
        if self.sim_time <= self.SETTLE_TIME:
            return 0.0
        fraction = self._smoothstep((self.sim_time - self.SETTLE_TIME) / self.TWIST_TIME)
        return 2.0 * math.pi * self.TWIST_TURNS * fraction

    def simulate(self) -> None:
        angle = self._twist_command()
        for _ in range(self.sim_substeps):
            wp.launch(
                drive_endpoints_kernel,
                dim=1,
                inputs=[
                    self.root_body,
                    self.tip_body,
                    wp.vec3(*self.root_pos),
                    wp.vec3(*self.tip_pos),
                    self.root_rot,
                    self.tip_rot,
                    angle,
                ],
                outputs=[self.state_0.body_q, self.state_1.body_q],
                device=self.model.device,
            )
            self.state_0.clear_forces()
            self.pipeline.collide(self.state_0, self.contacts)
            if self.solver_type == "vbd":
                self.solver.set_rigid_history_update(True)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt

    def test_final(self) -> None:
        poses = self.state_0.body_q.numpy()
        velocities = self.state_0.body_qd.numpy()
        assert np.isfinite(poses).all(), f"{self.solver_type}: non-finite poses"
        assert np.isfinite(velocities).all(), f"{self.solver_type}: non-finite velocities"
        assert np.max(np.abs(poses[:, :3])) < 10.0, f"{self.solver_type}: positions blew up"


class Example:
    """Display VBD and Style3D plectoneme formation side by side."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        modes = ("vbd", "style3d") if args.solver == "both" else (args.solver,)
        offsets = {"vbd": -1.2 if len(modes) == 2 else 0.0, "style3d": 1.2 if len(modes) == 2 else 0.0}
        # Keep the interactive comparison responsive.  These are exposed as
        # command-line overrides for convergence experiments.
        substeps = int(args.substeps or 4)
        iterations = int(args.iterations or 4)
        self.simulations = [_Simulation(mode, offsets[mode], args.device, substeps, iterations) for mode in modes]
        self.frame_dt = self.simulations[0].frame_dt
        self.sim_time = 0.0

        display_builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for simulation in self.simulations:
            display_builder.add_builder(simulation.builder)
        self.display_model = display_builder.finalize(device=args.device)
        self.display_state = self.display_model.state()
        self.viewer.set_model(self.display_model)
        self.viewer.set_camera(pos=wp.vec3(0.0, -5.0, 1.3), pitch=1.4, yaw=90.0)

    def step(self):
        for simulation in self.simulations:
            simulation.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        offset = 0
        for simulation in self.simulations:
            wp.copy(
                self.display_state.body_q,
                simulation.state_0.body_q,
                dest_offset=offset,
                count=simulation.model.body_count,
            )
            offset += simulation.model.body_count
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.display_state)
        self.viewer.end_frame()

    def test_final(self):
        for simulation in self.simulations:
            simulation.test_final()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--solver", choices=["both", "vbd", "style3d"], default="both")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--substeps", type=int, default=None)
    parser.set_defaults(num_frames=int(60 * (_Simulation.SETTLE_TIME + _Simulation.TWIST_TIME + _Simulation.HOLD_TIME)))
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
