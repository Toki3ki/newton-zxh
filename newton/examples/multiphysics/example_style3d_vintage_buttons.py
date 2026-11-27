# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Suspend two independent elastic patches through two dynamic brass buttons.

The patches are thin tetrahedral solids, initially flat and stress-free. Only
particle-body attachments connect them through the buttons. A weight attached
at the lower right corner stretches the patches under gravity. Right-drag the
weight with the viewer's rigid-body picking control, then release it to see recoil.
All deformable and affine-body degrees of freedom share one Style3D solve.

Success criteria:
- The top edge stays attached to the wooden support.
- Both buttons and the weight remain attached, with visible gravity-driven sag.
- The soft bodies remain above the base and all states remain finite.

Runs:
- `python -m newton.examples style3d_vintage_buttons`
  - Pulling and releasing the weight moves both patches and produces recoil.
- `python -m newton.examples style3d_vintage_buttons --test --viewer null --num-frames 250`
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverStyle3D


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / 50.0
        self.sim_substeps = 3
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        builder = newton.ModelBuilder()
        SolverStyle3D.register_custom_attributes(builder)
        self.patch_ranges = []
        colors = []
        young, poisson = 3.0e4, 0.3
        for width, height, bottom, depth, color in (
            (1.4, 0.90, 2.30, 0.032, (0.40, 0.10, 0.14)),
            (1.7, 1.23, 1.15, -0.032, (0.16, 0.43, 0.45)),
        ):
            start, tri_start = builder.particle_count, builder.tri_count
            builder.add_soft_grid(
                pos=wp.vec3(-width / 2.0, depth - 0.0175, bottom),
                rot=wp.quat_identity(),
                vel=wp.vec3(),
                dim_x=20,
                dim_y=2,
                dim_z=16,
                cell_x=width / 20,
                cell_y=0.0175,
                cell_z=height / 16,
                density=80.0,
                k_mu=young / (2.0 * (1.0 + poisson)),
                k_lambda=young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson)),
                k_damp=0.0,
                particle_radius=0.003,
            )
            self.patch_ranges.append((start, builder.particle_count))
            colors.extend([color] * (builder.tri_count - tri_start))

        brass = (0.76, 0.53, 0.22)
        wood = (0.25, 0.13, 0.075)
        # Embedded anchors must not be repelled by their attachment proxies.
        support_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_particle_collision=False)
        support = builder.add_body(xform=wp.transform((0.0, 0.032, 3.20), wp.quat_identity()), label="wooden_support")
        builder.add_shape_box(support, hx=0.96, hy=0.075, hz=0.06, cfg=support_cfg, color=wood)
        for x in (-0.99, 0.99):
            builder.add_shape_box(
                support,
                xform=wp.transform((x, 0.12, -1.60), wp.quat_identity()),
                hx=0.055,
                hy=0.065,
                hz=1.60,
                cfg=support_cfg,
                color=wood,
            )
        builder.add_shape_box(
            -1,
            xform=wp.transform((-0.05, 0.08, 0.10), wp.quat_identity()),
            hx=1.15,
            hy=0.60,
            hz=0.10,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
            color=wood,
        )
        positions = np.asarray(builder.particle_q)
        self.top_indices = np.flatnonzero(positions[:, 2] > 3.14)
        self.top_rest = positions[self.top_indices].copy()
        self._attach(builder, support, self.top_indices)

        self.button_bodies = []
        button_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_particle_collision=False)
        disk_rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi / 2)
        for x in (-0.52, 0.52):
            center = np.array((x, 0.0, 2.345))
            mass = 400.0 * 4.0 / 3.0 * np.pi * 0.13**3
            body = builder.add_body(
                xform=wp.transform(center, wp.quat_identity()),
                mass=mass,
                inertia=wp.mat33(np.eye(3) * (0.4 * mass * 0.13**2)),
                label=f"button_{len(self.button_bodies)}",
            )
            for y in (-0.065, 0.065):
                builder.add_shape_cylinder(
                    body,
                    xform=wp.transform((0.0, y, 0.0), disk_rotation),
                    radius=0.105,
                    half_height=0.012,
                    cfg=button_cfg,
                    color=brass,
                )
            builder.add_shape_cylinder(
                body,
                xform=wp.transform(wp.vec3(), disk_rotation),
                radius=0.025,
                half_height=0.065,
                cfg=button_cfg,
                color=brass,
            )
            indices = np.flatnonzero(np.linalg.norm(positions - center, axis=1) < 0.13)
            self._attach(builder, body, indices)
            self.button_bodies.append(body)

        self.weight_body = builder.add_body(
            xform=wp.transform((0.74, -0.032, 1.0), wp.quat_identity()),
            label="brass_weight",
        )
        weight_cfg = newton.ModelBuilder.ShapeConfig(density=500.0, has_particle_collision=False)
        builder.add_shape_cylinder(self.weight_body, radius=0.105, half_height=0.18, cfg=weight_cfg, color=brass)
        builder.add_shape_sphere(
            self.weight_body,
            xform=wp.transform((0.0, 0.0, 0.20), wp.quat_identity()),
            radius=0.075,
            cfg=newton.ModelBuilder.ShapeConfig(density=100.0, has_particle_collision=False),
            color=brass,
        )
        lower = self.patch_ranges[1][0]
        indices = np.flatnonzero(np.linalg.norm(positions - (0.74, -0.032, 1.20), axis=1) < 0.075)
        self._attach(builder, self.weight_body, indices[indices >= lower])

        self.model = builder.finalize()
        self.model.tri_color = wp.array(colors, dtype=wp.vec3, device=self.model.device)
        self.model.soft_contact_ke = 2.0e4
        self.model.soft_contact_kd = 10.0
        self.solver = SolverStyle3D(
            self.model,
            dynamics=SolverStyle3D.Dynamics.ALL,
            config=SolverStyle3D.Config(iterations=10, linear_iterations=10),
        )
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(3.6, -6.5, 3.2), -10.0, 119.0)
        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    @staticmethod
    def _attach(builder, body, indices):
        if len(indices) == 0:
            raise ValueError("The attachment region contains no soft-body vertices.")
        inverse = wp.transform_inverse(builder.body_q[body])
        for index in indices:
            builder.add_attachment_particle_body(
                int(index),
                body,
                body_point=wp.transform_point(inverse, wp.vec3(builder.particle_q[index])),
                stiffness=2.0e4,
                damping=2.0,
            )

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph is None:
            self.simulate()
        else:
            wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt

    def test_post_step(self):
        """Keep the supported assembly finite and its upper edge attached."""
        particles = self.state_0.particle_q.numpy()
        for values in (
            particles,
            self.state_0.particle_qd.numpy(),
            self.state_0.body_q.numpy(),
            self.state_0.body_qd.numpy(),
        ):
            assert np.isfinite(values).all(), "The coupled assembly became non-finite"
        assert np.max(np.linalg.norm(particles[self.top_indices] - self.top_rest, axis=1)) < 0.01

    def test_final(self):
        """Transmit gravity through both buttons without detaching the weight."""
        particles = self.state_0.particle_q.numpy()
        poses = self.state_0.body_q.numpy()
        for particle, body, point in zip(
            self.model.attachment_particle_body_particle.numpy(),
            self.model.attachment_particle_body_body.numpy(),
            self.model.attachment_particle_body_body_point.numpy(),
            strict=True,
        ):
            anchor = np.array(wp.transform_point(wp.transform(*poses[body]), wp.vec3(point)))
            assert np.linalg.norm(particles[particle] - anchor) < 0.015, "An anchor detached"
        assert particles[:, 2].min() > 0.25, "A patch reached the base"
        assert np.all(poses[self.button_bodies, 2] < 2.34), "The buttons did not load the upper patch"
        assert poses[self.weight_body, 2] < 0.99, "The weight did not stretch the lower patch"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
