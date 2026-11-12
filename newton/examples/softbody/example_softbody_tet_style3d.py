# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SIPD tetrahedral elastic body with the Style3D solver.

This example exercises the tetrahedral pairwise-force path added to Style3D.
The body is fixed along its bottom layer, falls under gravity, and collides
with the ground plane.

Runs:
- `python -m newton.examples softbody_tet_style3d --viewer gl`
- `python -m newton.examples softbody_tet_style3d --viewer null --num-frames 2`
"""

import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder = newton.ModelBuilder()
        newton.solvers.SolverStyle3D.register_custom_attributes(builder)
        builder.add_ground_plane()
        builder.add_soft_grid(
            # Newton's default viewer uses Z as the up axis.
            pos=wp.vec3(-0.45, -0.45, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=5,
            dim_y=5,
            dim_z=3,
            cell_x=0.18,
            cell_y=0.18,
            cell_z=0.18,
            density=800.0,
            k_mu=2.0e4,
            k_lambda=4.0e4,
            k_damp=2.0e1,
            fix_left=True,
            particle_radius=0.02,
            color=wp.vec3(0.2, 0.55, 0.9),
        )
        builder.color()

        self.model = builder.finalize()
        self.model.soft_contact_ke = 2.0e3
        self.model.soft_contact_kd = 2.0e1
        self.model.soft_contact_mu = 0.3

        self.solver = newton.solvers.SolverStyle3D(
            self.model,
            config=newton.solvers.SolverStyle3D.Config(
                iterations=8,
                linear_iterations=8,
                enable_surface_constraints=False,
                enable_tet_sipd=True,
                tet_solver="semi_implicit",
                tet_sipd_relaxation=1.0,
            ),
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        # A short smoke test: the body must remain finite and near the scene.
        p_lower = wp.vec3(-3.0, -1.0, -3.0)
        p_upper = wp.vec3(3.0, 3.0, 3.0)
        newton.examples.test_particle_state(
            self.state_0,
            "SIPD tetrahedral body remains finite",
            lambda q, _qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
