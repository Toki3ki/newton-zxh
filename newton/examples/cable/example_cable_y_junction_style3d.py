# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Style3D Cable N-Way Junction

Simulates a graph-topology rod whose N branches share one Style3D Cable
node. Stretch, mass, contact, and write-back share that node; ordinary Darboux
bend/twist sites remain active inside each branch. At the degree-N junction,
the N-1 Builder ROD joints form an angular tree that preserves the reference
branch angles without introducing all N(N-1)/2 pairwise frame energies.

Success criteria:
- All N branches remain connected at one shared node while hanging.
- The pinned branch tip remains fixed and all Cable states remain finite.

Run with ``python -m newton.examples cable_y_junction_style3d``.
"""

import argparse
import math

import numpy as np
import warp as wp

import newton
import newton.examples


def _branch_directions(count: int) -> list[wp.vec3]:
    """Return ``count`` uniformly spaced branch directions in the XY plane."""
    return [
        wp.vec3(math.cos(2.0 * math.pi * i / count), math.sin(2.0 * math.pi * i / count), 0.0)
        for i in range(count)
    ]


class Example:
    """Suspend a graph-topology N-way cable using Style3D Cable dynamics."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / 60.0
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.branch_count = args.branches

        radius = 0.01
        segment_length = 0.03
        junction = wp.vec3(0.0, 0.0, 1.25)
        points = [junction]
        edges = []
        for direction in _branch_directions(self.branch_count):
            previous = 0
            for i in range(1, args.segments + 1):
                points.append(junction + direction * (i * segment_length))
                current = len(points) - 1
                edges.append((previous, current))
                previous = current

        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.002
        builder.default_shape_cfg.ke = 1.0e4
        builder.default_shape_cfg.kd = 0.0
        builder.default_shape_cfg.mu = 1.0
        rod = newton.Rod(points, edges=edges, radius=radius)
        self.cable_bodies, _ = builder.add_rod(
            rod=rod,
            cfg=builder.default_shape_cfg.copy(),
            stretch_stiffness=1.0e7,
            bend_stiffness=5.0e3,
            bend_damping=1.0e3,
            label="style3d_y_graph",
            color=(0.95, 0.35, 0.12),
        )

        self.pinned_body = int(self.cable_bodies[args.segments - 1])
        builder.body_mass[self.pinned_body] = 0.0
        builder.body_inv_mass[self.pinned_body] = 0.0
        builder.body_inertia[self.pinned_body] = wp.mat33(0.0)
        builder.body_inv_inertia[self.pinned_body] = wp.mat33(0.0)
        if args.ground:
            builder.add_ground_plane()

        self.model = builder.finalize(device=args.device)
        self.model.set_gravity((0.0, 0.0, args.gravity_z))
        config = newton.solvers.SolverStyle3D.Config(
            iterations=args.iterations,
            linear_iterations=args.linear_iterations,
            cable_stretch_stiffness=1.0e7,
            cable_bend_stiffness=2.0e2,
            cable_twist_stiffness=2.0e2,
        )
        self.solver = newton.solvers.SolverStyle3D(
            self.model,
            config=config,
            dynamics=newton.solvers.SolverStyle3D.Dynamics.CABLE,
        )
        self.pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=args.rigid_contact_max)
        self.contacts = self.pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.pinned_pose = self.state_0.body_q.numpy()[self.pinned_body].copy()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(2.1, 0.0, 1.1), pitch=0.0, yaw=-180.0)

    def simulate(self):
        """Execute all substeps for one display frame."""
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
        """Render the current Cable proxy state."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify finite state, shared junction topology, and the pinned tip."""
        poses = self.state_0.body_q.numpy()[self.cable_bodies]
        velocities = self.state_0.body_qd.numpy()[self.cable_bodies]
        if not np.isfinite(poses).all() or not np.isfinite(velocities).all():
            raise ValueError("Style3D N-way junction contains non-finite state.")
        degree = self.solver._cable._node_degree_np
        if int(np.max(degree)) != self.branch_count:
            raise ValueError(
                f"Expected one degree-{self.branch_count} junction, got maximum degree {int(np.max(degree))}."
            )
        pinned = self.state_0.body_q.numpy()[self.pinned_body]
        if np.max(np.abs(pinned[:3] - self.pinned_pose[:3])) > 1.0e-4:
            raise ValueError("Pinned N-way junction tip moved unexpectedly.")

    @staticmethod
    def create_parser():
        """Create options for topology resolution and solver cost."""
        parser = newton.examples.create_parser()
        parser.add_argument("--branches", type=int, default=3, choices=range(2, 33))
        parser.add_argument("--segments", type=int, default=20)
        parser.add_argument("--substeps", type=int, default=4)
        parser.add_argument("--iterations", type=int, default=4)
        parser.add_argument("--linear-iterations", type=int, default=16)
        parser.add_argument("--rigid-contact-max", type=int, default=4096)
        parser.add_argument("--gravity-z", type=float, default=-9.81)
        parser.add_argument("--ground", action=argparse.BooleanOptionalAction, default=True)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
