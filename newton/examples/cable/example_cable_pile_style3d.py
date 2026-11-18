# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Style3D Cable Pile

Drops alternating layers of wavy cables onto a frictional ground plane. This
Style3D variant exercises global VIPER stretch/bend/twist constraints together
with cable-to-cable and cable-to-ground contact.

Success criteria:
- The cable states remain finite while the pile falls and contacts the ground.
- Cables do not pass substantially through the ground or acquire explosive velocities.

Run with ``python -m newton.examples cable_pile_style3d``.
"""

import math

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    """Drop a layered cable pile using the Style3D VIPER Cable solver."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / 60.0
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.layers = args.layers
        self.lanes_per_layer = args.lanes
        self.cable_radius = 0.012

        segment_length = 0.05
        cable_length = args.segments * segment_length
        lane_spacing = max(8.0 * self.cable_radius, 0.15)
        layer_gap = self.cable_radius * 3.0

        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.0
        builder.default_shape_cfg.mu = 1.0
        builder.default_shape_cfg.ke = 1.0e5
        builder.default_shape_cfg.kd = 0.0

        cable_shape_cfg = newton.ModelBuilder.ShapeConfig(
            density=builder.default_shape_cfg.density,
            ke=builder.default_shape_cfg.ke,
            kd=builder.default_shape_cfg.kd,
            kf=builder.default_shape_cfg.kf,
            ka=builder.default_shape_cfg.ka,
            mu=builder.default_shape_cfg.mu,
            restitution=builder.default_shape_cfg.restitution,
        )
        ground_cfg = newton.ModelBuilder.ShapeConfig(mu=1.0e9, ke=1.0e5, kd=0.0)
        builder.add_ground_plane(cfg=ground_cfg)

        for layer in range(self.layers):
            along_x = layer % 2 == 0
            direction = wp.vec3(1.0, 0.0, 0.0) if along_x else wp.vec3(0.0, 1.0, 0.0)
            normal = wp.vec3(0.0, 1.0, 0.0) if along_x else wp.vec3(1.0, 0.0, 0.0)
            z = 0.3 + layer * layer_gap
            for lane in range(self.lanes_per_layer):
                offset = (lane - 0.5 * (self.lanes_per_layer - 1)) * lane_spacing
                center = wp.vec3(0.0, offset, z) if along_x else wp.vec3(offset, 0.0, z)
                rod = newton.Rod.create_straight(
                    start=center - 0.5 * cable_length * direction,
                    direction=direction,
                    length=cable_length,
                    segment_count=args.segments,
                )
                points = [wp.vec3(*(float(value) for value in point)) for point in rod.points]
                for i in range(len(points)):
                    phase = 4.0 * math.pi * i / args.segments
                    points[i] += normal * (0.025 * cable_length * math.sin(phase))

                rod = newton.Rod(points, radius=self.cable_radius)
                rod.compute_frames(twist_total=0.0)
                builder.add_rod(
                    rod=rod,
                    cfg=cable_shape_cfg,
                    stretch_stiffness=5.0e5,
                    bend_stiffness=1.0e2,
                    bend_damping=2.0e1,
                    label=f"cable_l{layer}_{lane}",
                    color=(0.95, 0.35, 0.12),
                )

        builder.color()
        self.model = builder.finalize(device=args.device)
        self.pipeline = newton.CollisionPipeline(
            self.model,
            rigid_contact_max=args.rigid_contact_max,
            contact_matching="sticky",
        )
        self.contacts = self.pipeline.contacts()
        config = newton.solvers.SolverStyle3D.Config(
            iterations=args.iterations,
            linear_iterations=args.linear_iterations,
            cable_stretch_stiffness=5.0e5,
            cable_bend_stiffness=1.0e2,
            cable_twist_stiffness=1.0e2,
        )
        self.solver = newton.solvers.SolverStyle3D(
            self.model,
            config=config,
            dynamics=newton.solvers.SolverStyle3D.Dynamics.CABLE,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "camera"):
            self.viewer.camera.fov = 40.0

    def simulate(self):
        """Execute all simulation substeps for one display frame."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def step(self):
        """Advance the example by one display frame."""
        self.simulate()

    def render(self):
        """Render the current simulation state."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify that the cable pile remains finite and bounded."""
        poses = self.state_0.body_q.numpy()
        velocities = self.state_0.body_qd.numpy()
        assert np.isfinite(poses).all(), "Non-finite cable poses"
        assert np.isfinite(velocities).all(), "Non-finite cable velocities"
        assert np.min(poses[:, 2]) > -0.5, "Cables penetrated the ground substantially"
        assert np.max(np.abs(velocities)) < 5.0e2, "Cable velocities became unstable"

    @staticmethod
    def create_parser():
        """Create command-line options for pile size and solver cost."""
        parser = newton.examples.create_parser()
        parser.add_argument("--layers", type=int, default=10)
        parser.add_argument("--lanes", type=int, default=10)
        parser.add_argument("--segments", type=int, default=40)
        parser.add_argument("--substeps", type=int, default=4)
        parser.add_argument("--iterations", type=int, default=4)
        parser.add_argument("--linear-iterations", type=int, default=12)
        parser.add_argument("--rigid-contact-max", type=int, default=50000)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
