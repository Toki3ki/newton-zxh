# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Style3D Cable Bundle Elastoplastic Loading

Runs the cable-bundle obstacle scene with the Style3D VIPER Cable solver.
Each Darboux site stores a local plastic reference strain. The global solve
holds it fixed; a bend return mapping updates it after each converged step.

Success criteria:
- Seven cables are bent into an S pattern by four moving capsules.
- After release, the elastic cables remain finite and recover toward straight.

Run with ``python -m newton.examples cable_bundle_hysteresis_style3d``.
"""

import numpy as np
import warp as wp

import newton
import newton.examples


@wp.kernel
def move_obstacles_triwave(
    bodies: wp.array[int],
    init_y: wp.array[float],
    init_z: wp.array[float],
    period: float,
    time: float,
    stop_time: float,
    release_time: float,
    body_q0: wp.array[wp.transform],
    body_q1: wp.array[wp.transform],
):
    """Apply the load-hold-release obstacle motion."""
    i = wp.tid()
    body = bodies[i]
    pose = body_q0[body]
    position = wp.transform_get_translation(pose)
    rotation = wp.transform_get_rotation(pose)
    if time >= release_time:
        new_position = wp.vec3(position[0], position[1], init_z[i] + 10.0)
    else:
        sample_time = wp.min(time, stop_time)
        fraction = sample_time / period
        fraction = fraction - wp.floor(fraction)
        fraction = fraction + 0.5
        fraction = fraction - wp.floor(fraction)
        triangle = 2.0 * (1.0 - wp.abs(2.0 * fraction - 1.0)) - 1.0
        new_position = wp.vec3(position[0], triangle * init_y[i], init_z[i])
    new_pose = wp.transform(new_position, rotation)
    body_q0[body] = new_pose
    body_q1[body] = new_pose


def _bundle_offsets(count: int, radius: float, gap_multiplier: float) -> list[tuple[float, float]]:
    """Return one center and a compact first ring of cable offsets."""
    if count == 1:
        return [(0.0, 0.0)]
    distance = 2.0 * radius * gap_multiplier
    offsets = [(0.0, 0.0)]
    for i in range(count - 1):
        angle = 2.0 * np.pi * i / (count - 1)
        offsets.append((float(distance * np.cos(angle)), float(distance * np.sin(angle))))
    return offsets


class Example:
    """Bend and release a seven-cable Style3D bundle."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / 60.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.num_cables = 7
        self.num_elements = args.segments
        self.cable_length = 4.0
        self.cable_radius = 0.02
        with_plasticity = not args.no_plasticity and (
            args.bend_yield_strain > 0.0 or args.twist_yield_strain > 0.0
        )
        self.with_plasticity = with_plasticity
        self.with_bend_plasticity = with_plasticity and args.bend_yield_strain > 0.0

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.rigid_gap = 0.05
        builder.default_shape_cfg.ke = 1.0e5
        builder.default_shape_cfg.kd = 1.0e1
        builder.default_shape_cfg.mu = 1.0

        start_x = -0.5
        start_z = 0.3
        offsets = _bundle_offsets(self.num_cables, self.cable_radius, 1.1)
        self.cable_body_ids = []
        for cable, (offset_y, offset_z) in enumerate(offsets):
            rod = newton.Rod.create_straight(
                start=wp.vec3(start_x, offset_y, start_z + offset_z),
                direction=wp.vec3(1.0, 0.0, 0.0),
                length=self.cable_length,
                segment_count=self.num_elements,
                radius=self.cable_radius,
            )
            bodies, _ = builder.add_rod(
                rod=rod,
                total_mass=0.4,
                label=f"bundle_cable_{cable}",
                color=(0.95, 0.35, 0.12),
            )
            self.cable_body_ids.append(bodies)

        obstacle_cfg = newton.ModelBuilder.ShapeConfig(mu=0.0)
        self.obstacle_bodies = []
        obstacle_y = []
        obstacle_z = []
        for obstacle in range(4):
            x = 0.5 + 2.0 * obstacle / 3.0
            y = 0.3 if obstacle % 2 == 0 else -0.3
            z = 0.3
            body = builder.add_body(xform=wp.transform(wp.vec3(x, y, z), wp.quat_identity()))
            builder.add_shape_capsule(body, radius=0.05, half_height=0.35, cfg=obstacle_cfg)
            builder.body_mass[body] = 0.0
            builder.body_inv_mass[body] = 0.0
            builder.body_inertia[body] = wp.mat33(0.0)
            builder.body_inv_inertia[body] = wp.mat33(0.0)
            self.obstacle_bodies.append(body)
            obstacle_y.append(y)
            obstacle_z.append(z)
        builder.add_ground_plane()

        self.model = builder.finalize(device=args.device)
        config = newton.solvers.SolverStyle3D.Config(
            iterations=6,
            linear_iterations=16,
            cable_stretch_stiffness=1.0e5,
            cable_bend_stiffness=1.0e2,
            cable_twist_stiffness=1.0e2,
            cable_bend_yield_strain=args.bend_yield_strain if with_plasticity else 0.0,
            cable_twist_yield_strain=args.twist_yield_strain if with_plasticity else 0.0,
        )
        self.solver = newton.solvers.SolverStyle3D(
            self.model,
            config=config,
            dynamics=newton.solvers.SolverStyle3D.Dynamics.CABLE,
        )
        # The measured peak is about 5,000 contacts; leave headroom without
        # making each matrix-free Cable contact multiply scan 50,000 empty slots.
        self.pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=10000)
        self.contacts = self.pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.obstacle_bodies_wp = wp.array(self.obstacle_bodies, dtype=int, device=args.device)
        self.obstacle_init_y = wp.array(obstacle_y, dtype=float, device=args.device)
        self.obstacle_init_z = wp.array(obstacle_z, dtype=float, device=args.device)
        self.obstacle_period = 2.0
        self.obstacle_stop_time = 1.0
        self.obstacle_release_time = 2.0

        self.viewer.set_model(self.model)

    def simulate(self):
        """Advance all substeps for one display frame."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            wp.launch(
                move_obstacles_triwave,
                dim=len(self.obstacle_bodies),
                inputs=[
                    self.obstacle_bodies_wp,
                    self.obstacle_init_y,
                    self.obstacle_init_z,
                    self.obstacle_period,
                    self.sim_time,
                    self.obstacle_stop_time,
                    self.obstacle_release_time,
                ],
                outputs=[self.state_0.body_q, self.state_1.body_q],
                device=self.model.device,
            )
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def step(self):
        """Advance the example by one frame."""
        self.simulate()

    def render(self):
        """Render the current simulation state."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify stability and the expected released elastic/plastic shape."""
        poses = self.state_0.body_q.numpy()
        velocities = self.state_0.body_qd.numpy()
        bodies = [body for cable in self.cable_body_ids for body in cable]
        if not np.isfinite(poses[bodies]).all() or not np.isfinite(velocities[bodies]).all():
            raise ValueError("Style3D cable bundle contains non-finite values.")
        if np.max(np.abs(poses[bodies, :3])) >= 10.0:
            raise ValueError("Style3D cable bundle exceeded the example bounds.")
        if self.sim_time < self.obstacle_release_time + 0.4:
            return
        nodes = self.solver._cable._node_position.numpy()
        straightness = []
        for node_range in self.solver._cable.component_offsets:
            points = nodes[node_range.start : node_range.stop]
            arc_length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            straightness.append(float(np.linalg.norm(points[-1] - points[0])) / max(arc_length, 1.0e-8))
        straightness = np.asarray(straightness)
        if self.with_bend_plasticity and np.min(straightness) >= 0.99:
            raise ValueError(f"Plastic bundle retained no measurable bend: {straightness.round(3)}")
        if not self.with_plasticity and np.min(straightness) < 0.9:
            raise ValueError(f"Elastic bundle did not recover: {straightness.round(3)}")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--segments", type=int, default=40)
        parser.add_argument("--no-plasticity", action="store_true", help="Use a purely elastic cable")
        parser.add_argument("--bend-yield-strain", type=float, default=0.12, help="Bend yield strain [rad]")
        parser.add_argument("--twist-yield-strain", type=float, default=0.0, help="Twist yield strain [rad]")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
