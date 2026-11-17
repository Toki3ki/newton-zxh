# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example Cable Twist Compare

Runs the zigzag twist route from ``example_cable_twist`` with the VBD rod solver
and the Style3D VIPER cable module side by side.

A root capsule spins continuously about its own axis while the rest of the cable
is free. Both solvers see identical geometry, boundary motion, damping, and time
step, so their twist profiles can be compared directly. The Style3D cable path has
no contact response yet, so the scene contains no colliders and no gravity.

Success criteria:
- The cable stays finite, bounded, and connected for the whole run.
- The driven twist reaches at least the first few joints and decays with distance.
- Every material frame stays orthonormal.
- The final state is identical in kind for both solvers: connected and stable.

Runs:
- `python -m newton.examples cable_twist_compare`
- `python -m newton.examples cable_twist_compare --solver style3d`
- `python -m newton.examples cable_twist_compare --solver vbd`
"""

import numpy as np
import warp as wp

import newton
import newton.examples

TWIST_RATE = 1.5  # Root twist rate [rad/s].


@wp.kernel
def spin_root_capsule_kernel(
    body_index: wp.array[wp.int32],
    twist_rate: float,
    dt: float,
    body_q0: wp.array[wp.transform],
    body_q1: wp.array[wp.transform],
):
    """Rotate the driven root capsule about its own capsule axis."""
    body = body_index[0]
    transform = body_q0[body]
    rotation = wp.transform_get_rotation(transform)
    # The capsule axis is the body-local +Z axis, which is also the rod frame axis.
    axis = wp.quat_rotate(rotation, wp.vec3(0.0, 0.0, 1.0))
    spun = wp.mul(wp.quat_from_axis_angle(axis, twist_rate * dt), rotation)
    pose = wp.transform(wp.transform_get_translation(transform), spun)
    body_q0[body] = pose
    body_q1[body] = pose


def _rotation_angle(frame0: np.ndarray, frame1: np.ndarray) -> float:
    """Return the angle of ``frame0^T frame1`` in radians."""
    relative = frame0.T @ frame1
    return float(np.arccos(np.clip(0.5 * (np.trace(relative) - 1.0), -1.0, 1.0)))


def _twist_angle(frame0: np.ndarray, frame1: np.ndarray, chord: np.ndarray) -> float:
    """Return the Bishop twist between two frames about their shared chord."""
    tangent = chord / max(float(np.linalg.norm(chord)), 1.0e-12)
    normal0 = frame0[:, 0] - tangent * np.dot(frame0[:, 0], tangent)
    normal1 = frame1[:, 0] - tangent * np.dot(frame1[:, 0], tangent)
    if np.linalg.norm(normal0) < 1.0e-6 or np.linalg.norm(normal1) < 1.0e-6:
        return 0.0
    normal0 = normal0 / np.linalg.norm(normal0)
    normal1 = normal1 / np.linalg.norm(normal1)
    return float(np.arctan2(np.dot(tangent, np.cross(normal0, normal1)), np.dot(normal0, normal1)))


class _CableSimulation:
    """Own one solver, model, and state for one side of the comparison."""

    def __init__(self, solver_type: str, pos: wp.vec3, device):
        self.solver_type = solver_type

        # Simulation cadence, shared by both solvers.
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Cable layout: the zigzag route of example_cable_twist at a smaller scale.
        self.num_elements = 32
        self.segment_length = 0.1
        self.cable_length = self.num_elements * self.segment_length
        self.cable_radius = 0.02
        self.y_separation = 3.0

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        rod = self.create_zigzag_rod(pos=pos)
        color = (0.15, 0.55, 1.0) if self.solver_type == "vbd" else (1.0, 0.4, 0.08)

        if self.solver_type == "vbd":
            # Match the rod joint stiffness of example_cable_twist.
            self.rod_bodies, _rod_joints = builder.add_rod(
                rod=rod,
                stretch_stiffness=1.0e6,
                bend_stiffness=1.0e3,
                twist_stiffness=1.0e3,
                bend_damping=1.0e1,
                twist_damping=1.0e1,
                color=color,
                label="cable",
            )
        else:
            self.rod_bodies, _rod_joints = builder.add_rod(rod=rod, total_mass=0.4, color=color, label="cable")

        # Match example_cable_twist: prescribe only the spinning root and leave
        # every remaining segment, including the tip, dynamic.
        self.root_body = self.rod_bodies[0]
        builder.body_mass[self.root_body] = 0.0
        builder.body_inv_mass[self.root_body] = 0.0
        builder.body_inertia[self.root_body] = wp.mat33(0.0)
        builder.body_inv_inertia[self.root_body] = wp.mat33(0.0)
        self.root_bodies = wp.array([self.root_body], dtype=wp.int32)

        if self.solver_type == "vbd":
            builder.color()

        self.builder = builder
        self.model = builder.finalize(device=device)

        if self.solver_type == "vbd":
            self.solver = newton.solvers.SolverVBD(self.model, iterations=5)
        else:
            # ``EA`` in N, ``EI`` and ``GJ`` in N*m^2, close to the VBD section above.
            config = newton.solvers.SolverStyle3D.Config(
                # This 65-block system is launch-bound on the GPU. Two nonlinear
                # updates with a short PCG solve are sufficient for this smooth,
                # contact-free twist drive.
                iterations=2,
                linear_iterations=16,
                cable_stretch_stiffness=1.0e3,
                cable_bend_stiffness=5.0,
                cable_twist_stiffness=1.0e1,
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
        self.rest_frames = self._material_frames()

    def create_zigzag_rod(self, pos: wp.vec3) -> newton.Rod:
        """Create the +Y -> +X -> -Y -> +X zigzag route with three 90-degree turns."""
        num_points = self.num_elements + 1
        segments_per_leg = self.num_elements // 4
        points = []
        for i in range(num_points):
            if i <= segments_per_leg:
                x = 0.0
                y = i * self.segment_length
            elif i <= 2 * segments_per_leg:
                x = (i - segments_per_leg) * self.segment_length
                y = segments_per_leg * self.segment_length
            elif i <= 3 * segments_per_leg:
                x = segments_per_leg * self.segment_length
                y = segments_per_leg * self.segment_length - (i - 2 * segments_per_leg) * self.segment_length
            else:
                x = segments_per_leg * self.segment_length + (i - 3 * segments_per_leg) * self.segment_length
                y = 0.0
            points.append(pos + wp.vec3(x, y, 0.0))
        rod = newton.Rod(points, radius=self.cable_radius)
        rod.compute_frames()
        return rod

    def _material_frames(self) -> np.ndarray:
        """Return each segment's world material frame from the output state."""
        poses = self.state_0.body_q.numpy()
        frames = np.zeros((len(self.rod_bodies), 3, 3))
        for segment, body in enumerate(self.rod_bodies):
            rotation = np.array(wp.quat_to_matrix(wp.quat(*poses[body][3:])))
            frames[segment] = rotation.reshape(3, 3)
        return frames

    def twist_profile(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the per-joint twist angles [rad] and the per-segment rotation angles.

        The joint twist is measured between neighboring material frames about their
        shared chord, so it isolates transmitted twist from rigid rotation of the
        whole cable. The segment angle is measured against the initial frame.
        """
        frames = self._material_frames()
        centers = self.state_0.body_q.numpy()[:, :3]
        joint_twist = np.zeros(len(self.rod_bodies) - 1)
        for segment in range(len(self.rod_bodies) - 1):
            chord = centers[self.rod_bodies[segment + 1]] - centers[self.rod_bodies[segment]]
            joint_twist[segment] = abs(_twist_angle(frames[segment], frames[segment + 1], chord))
        rotation = np.array([_rotation_angle(self.rest_frames[i], frames[i]) for i in range(len(self.rod_bodies))])
        return joint_twist, rotation

    def simulate(self, viewer=None):
        """Execute all simulation substeps for one frame."""
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.launch(
                kernel=spin_root_capsule_kernel,
                dim=1,
                inputs=[self.root_bodies, TWIST_RATE, self.sim_dt],
                outputs=[self.state_0.body_q, self.state_1.body_q],
            )
            if viewer is not None:
                viewer.apply_forces(self.state_0)
            if (substep % 10) == 0:
                self.collision_pipeline.collide(self.state_0, self.contacts)
            if self.solver_type == "vbd":
                self.solver.set_rigid_history_update((substep % 10) == 0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def test_final(self):
        """Check stability, connectivity, frame orthonormality, and twist transfer."""
        poses = self.state_0.body_q.numpy()
        velocities = self.state_0.body_qd.numpy()
        assert np.isfinite(poses).all(), "Non-finite values in body poses"
        assert np.isfinite(velocities).all(), "Non-finite values in body velocities"
        assert (np.abs(poses) < 1.0e3).all(), "Body positions too large (>1000)"

        for segment in range(len(self.rod_bodies) - 1):
            body0 = self.rod_bodies[segment]
            body1 = self.rod_bodies[segment + 1]
            distance = np.linalg.norm(poses[body1][:3] - poses[body0][:3])
            assert distance < self.segment_length * 1.1, f"Segments {segment}-{segment + 1} separated: {distance:.3f} m"

        frames = self._material_frames()
        for segment, frame in enumerate(frames):
            np.testing.assert_allclose(frame @ frame.T, np.eye(3), atol=1.0e-4, err_msg=f"segment {segment}")

        joint_twist, rotation = self.twist_profile()
        # The prescribed root must drive twist into the otherwise free cable,
        # so both the first joint and a mid-route joint must respond.
        assert joint_twist[0] > 1.0e-3, f"Twist did not enter the cable: {joint_twist[0]:.5f} rad"
        assert joint_twist[len(joint_twist) // 2] > 1.0e-3, (
            f"Twist did not reach mid-route: {joint_twist[len(joint_twist) // 2]:.5f} rad"
        )
        assert np.max(rotation[1:]) > 1.0e-3, "The free cable did not respond to the driven root"


class Example:
    """Display VBD and Style3D simultaneously, or one solver when requested."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        modes = ("vbd", "style3d") if args.solver == "both" else (args.solver,)
        positions = {
            "vbd": wp.vec3(-2.0 if len(modes) == 2 else 0.0, 0.0, 0.0),
            "style3d": wp.vec3(2.0 if len(modes) == 2 else 0.0, 0.0, 0.0),
        }
        self.simulations = [_CableSimulation(mode, positions[mode], args.device) for mode in modes]
        self.sim_time = 0.0
        self.frame_dt = self.simulations[0].frame_dt

        display_builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for simulation in self.simulations:
            display_builder.add_builder(simulation.builder)
        self.display_model = display_builder.finalize(device=args.device)
        self.display_state = self.display_model.state()
        self.viewer.set_model(self.display_model)

    def simulate(self):
        """Advance every selected solver by one frame."""
        interactive_viewer = self.viewer if len(self.simulations) == 1 else None
        for simulation in self.simulations:
            simulation.simulate(interactive_viewer)

    def step(self):
        """Advance simulation by one frame."""
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        """Copy both independent states into the combined display model."""
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
        """Run the same stability and twist-transfer checks for every solver."""
        for simulation in self.simulations:
            simulation.test_final()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--solver",
        type=str,
        default="both",
        choices=["both", "vbd", "style3d"],
        help="Rod solver display: both (default), style3d, or vbd",
    )
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
