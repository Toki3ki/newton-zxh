# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Style3D Cable Initial and Rest Poses

Demonstrates that the Style3D VIPER Cable solver distinguishes the initial
configuration from the structural-rest configuration:

- an initially helical cable has a straight structural rest pose;
- an initially straight cable has a helical structural rest pose.

The initial and rest segment lengths match, so the motion primarily tests the
Darboux bending/twist rest strain instead of axial stretch.

Success criteria:
- The orange helix relaxes toward a straighter shape.
- The blue straight cable bends toward its helical structural-rest shape.
- Both cables remain finite, connected, and bounded.

Run with ``python -m newton.examples cable_initial_rest_pose_style3d``.
"""

import math

import numpy as np
import warp as wp

import newton
import newton.examples


def _create_helix_points(
    center: wp.vec3,
    *,
    num_segments: int,
    axis_length: float,
    radius: float,
    turns: float,
) -> list[wp.vec3]:
    """Create uniformly sampled helix points around the X-axis."""
    return [
        wp.vec3(
            float(center[0]) + axis_length * (i / num_segments - 0.5),
            float(center[1]) + radius * math.cos(2.0 * math.pi * turns * i / num_segments),
            float(center[2]) + radius * math.sin(2.0 * math.pi * turns * i / num_segments),
        )
        for i in range(num_segments + 1)
    ]


class Example:
    """Simulate different initial and rest poses with Style3D Cable dynamics."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / 60.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        num_segments = 60
        cable_radius = 0.025
        helix_radius = 0.22
        helix_axis_length = 2.6
        helix_turns = 2.5

        builder = newton.ModelBuilder()
        cable_cfg = builder.default_shape_cfg.copy()
        cable_cfg.density = 800.0
        cable_cfg.ke = 5.0e4
        cable_cfg.kd = 0.0
        cable_cfg.mu = 0.1

        cable_height = cable_radius + helix_radius
        left_center = wp.vec3(0.0, -0.75, cable_height)
        right_center = wp.vec3(0.0, 0.75, cable_height)

        left_points = _create_helix_points(
            left_center,
            num_segments=num_segments,
            axis_length=helix_axis_length,
            radius=helix_radius,
            turns=helix_turns,
        )
        left_rod = newton.Rod(left_points, radius=cable_radius, rest_straight=True)
        left_bodies, _ = builder.add_rod(
            rod=left_rod,
            cfg=cable_cfg,
            total_mass=0.8,
            label="straight_rest_helix_initial",
            color=(0.95, 0.35, 0.12),
        )

        right_rest_points = _create_helix_points(
            right_center,
            num_segments=num_segments,
            axis_length=helix_axis_length,
            radius=helix_radius,
            turns=helix_turns,
        )
        right_rest = newton.Rod(right_rest_points)
        segment_length = float(wp.length(right_rest_points[1] - right_rest_points[0]))
        right_rod = newton.Rod.create_straight(
            start=wp.vec3(-0.5 * num_segments * segment_length, right_center[1], right_center[2]),
            direction=wp.vec3(1.0, 0.0, 0.0),
            length=num_segments * segment_length,
            segment_count=num_segments,
            radius=cable_radius,
            rest_points=right_rest.points,
            rest_quaternions=right_rest.quaternions,
        )
        right_bodies, _ = builder.add_rod(
            rod=right_rod,
            cfg=cable_cfg,
            total_mass=0.8,
            label="helix_rest_straight_initial",
            color=(0.95, 0.35, 0.12),
        )
        self.cable_bodies = [left_bodies, right_bodies]

        builder.add_ground_plane()
        self.model = builder.finalize(device=args.device)
        config = newton.solvers.SolverStyle3D.Config(
            iterations=4,
            linear_iterations=12,
            cable_stretch_stiffness=1.0e5,
            cable_bend_stiffness=5.0e2,
            cable_twist_stiffness=5.0e2,
        )
        self.solver = newton.solvers.SolverStyle3D(
            self.model,
            config=config,
            dynamics=newton.solvers.SolverStyle3D.Dynamics.CABLE,
        )
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.0, -4.5, 4.2), pitch=-35.0, yaw=90.0)
        if hasattr(self.viewer, "camera"):
            if hasattr(self.viewer.camera, "look_at"):
                self.viewer.camera.look_at(wp.vec3(0.0, 0.0, 0.15))
            if hasattr(self.viewer.camera, "fov"):
                self.viewer.camera.fov = 48.0

    def simulate(self):
        """Advance one display frame."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Advance the example by one frame."""
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        """Render the current simulation state."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify both cables remain finite and bounded."""
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        bodies = [body for cable in self.cable_bodies for body in cable]
        if not np.isfinite(body_q[bodies]).all() or not np.isfinite(body_qd[bodies]).all():
            raise ValueError("Style3D Cable state contains non-finite values.")
        if np.max(np.abs(body_q[bodies, :3])) >= 10.0:
            raise ValueError("Style3D Cable body positions exceeded the example bounds.")


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
