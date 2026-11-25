# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Press an isochorically plastic pad with a cross or apple die.

The die cycles down into an elastic tetrahedral pad, holds briefly, then lifts.
Style3D's per-volume yield strain lets the pad recover elastically below yield
and retain an indentation after the press is released.

Success criteria:
- The kinematic die completes repeated press-and-release cycles.
- The soft pad and press remain finite throughout the run.
- Plastic rest stretch accumulates in the pad after pressing and remains after release.

Runs:
- `python -m newton.examples style3d_plasticity_press`
- `python -m newton.examples style3d_plasticity_press --die apple`

In apple mode an opposing hammer stamps "NEWTON" after the apple lifts.
"""

import math
from itertools import pairwise

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import BodyFlags
from newton.solvers import SolverStyle3D

FPS = 60
SUBSTEPS = 2
SUBSTEP_DT = 1.0 / (FPS * SUBSTEPS)
PRESS_PERIOD = 1.5
OPEN_ANGLE = 1.28
PRESS_DOWN_TIME = 0.7
PRESS_HOLD_TIME = 0.3
PRESS_UP_TIME = 0.7
# Pad dimensions [m]; subdivision counts change resolution without changing size.
PAD_LENGTH = 0.72
PAD_WIDTH = 0.72
PAD_HEIGHT = 0.18
PAD_DIM_X = 21 * 6
PAD_DIM_Y = 21 * 6
PAD_DIM_Z = 9
PAD_YOUNGS_MODULUS = 5.0e3
PAD_POISSON_RATIO = 0.49
PAD_YIELD_STRAIN = 0.18
APPLE_OFFSET_X = 0.05


def _extrude_profile(outline, half_height):
    """Cap a counterclockwise, star-shaped outline around its local origin."""
    n = len(outline)
    ring = np.arange(n)
    next_ring = np.roll(ring, -1)
    bottom = np.column_stack((outline, np.full(n, -half_height)))
    top = np.column_stack((outline, np.full(n, half_height)))
    vertices = np.vstack((bottom, top, (0, 0, -half_height), (0, 0, half_height)))
    faces = np.vstack(
        (
            np.column_stack((np.full(n, 2 * n), next_ring, ring)),
            np.column_stack((np.full(n, 2 * n + 1), ring + n, next_ring + n)),
            np.column_stack((ring, next_ring, next_ring + n)),
            np.column_stack((ring, next_ring + n, ring + n)),
        )
    )
    return newton.Mesh(vertices.astype(np.float32), faces.astype(np.int32).ravel())


@wp.func
def _press_motion(time: float):
    q = float(0.0)
    qd = float(0.0)
    if time < PRESS_DOWN_TIME:
        phase = time / PRESS_DOWN_TIME
        q = -OPEN_ANGLE * 0.5 * (1.0 - wp.cos(math.pi * phase))
        qd = -OPEN_ANGLE * 0.5 * (math.pi / PRESS_DOWN_TIME) * wp.sin(math.pi * phase)
    elif time < PRESS_DOWN_TIME + PRESS_HOLD_TIME:
        q = -OPEN_ANGLE
    elif time < PRESS_DOWN_TIME + PRESS_HOLD_TIME + PRESS_UP_TIME:
        phase = (time - PRESS_DOWN_TIME - PRESS_HOLD_TIME) / PRESS_UP_TIME
        q = -OPEN_ANGLE * 0.5 * (1.0 + wp.cos(math.pi * phase))
        qd = OPEN_ANGLE * 0.5 * (math.pi / PRESS_UP_TIME) * wp.sin(math.pi * phase)
    return OPEN_ANGLE + q, qd


@wp.kernel
def _drive_press_kernel(
    dt: float,
    hammer_period: float,
    hammer_speed: float,
    after_hammers: float,
    clock: wp.array[float],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
):
    period = hammer_period * float(wp.min(2, joint_q.shape[0])) + after_hammers
    time = clock[0] % period
    angle, rate = _press_motion(time * hammer_speed)
    joint_q[0] = angle
    joint_qd[0] = rate * hammer_speed
    if joint_q.shape[0] >= 2:
        angle, rate = _press_motion(((time + period - hammer_period) % period) * hammer_speed)
        joint_q[1] = angle
        joint_qd[1] = rate * hammer_speed
    clock[0] += dt


def _add_letter_press(builder, steel, gold):
    """Build the opposing die from continuous strokes in the pad's top-view lettering frame."""
    builder.add_shape_box(
        -1, xform=wp.transform((-0.52, 0, 0.14), wp.quat_identity()), hx=0.055, hy=0.065, hz=0.14, color=steel
    )
    body = builder.add_link(is_kinematic=True, label="letter_arm")
    builder.add_shape_cylinder(
        body,
        xform=wp.transform((0.17, 0, 0.064), wp.quat_from_axis_angle(wp.vec3(0, 1, 0), math.pi / 2)),
        radius=0.022,
        half_height=0.17,
        color=steel,
    )
    text_center = 0.265  # Leave a strip of untouched pad between the apple and the text.
    builder.add_shape_box(
        body, xform=wp.transform((text_center, 0, 0.052), wp.quat_identity()), hx=0.085, hy=0.335, hz=0.014, color=steel
    )
    # From the lettering hinge looking toward the apple, screen right is -Y and up is +X.
    letters = (
        (((0, 0), (0, 4), (2, 0), (2, 4)),),  # N
        (((2, 4), (0, 4), (0, 0), (2, 0)), ((0, 2), (1.7, 2))),  # E
        (((0, 4), (0, 0), (1, 1.7), (2, 0), (2, 4)),),  # W
        (((0, 4), (2, 4)), ((1, 4), (1, 0))),  # T
        (((0, 0), (0, 4), (2, 4), (2, 0), (0, 0)),),  # O
        (((0, 0), (0, 4), (2, 0), (2, 4)),),  # N
    )
    for i, strokes in enumerate(letters):
        for stroke in strokes:
            for (u, v), (w, h) in pairwise(stroke):
                dx, dy = (h - v) * 0.030, (u - w) * 0.038
                x = text_center + ((v + h) / 2 - 1.75) * 0.030
                y = ((len(letters) - 1) / 2 - i) * 0.108 - ((u + w) / 2 - 1) * 0.038
                rotation = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), math.atan2(dy, dx))
                builder.add_shape_box(
                    body,
                    xform=wp.transform((x, y, 0), rotation),
                    hx=math.hypot(dx, dy) / 2 + 0.002,
                    hy=0.009,
                    hz=0.035,
                    color=gold,
                )
    hinge = builder.add_joint_revolute(
        -1, body, parent_xform=wp.transform((-0.52, 0, 0.302), wp.quat_identity()), axis=wp.vec3(0, -1, 0)
    )
    builder.add_articulation([hinge], label="letter_press")
    builder.joint_q[builder.joint_q_start[hinge]] = OPEN_ANGLE


class Example:
    hammer_period = PRESS_PERIOD
    iterations = 5
    linear_iterations = 10
    hammer_speed = 1.0
    after_hammers = 0.0

    def __init__(self, viewer, args):
        self.die = getattr(args, "die", "cross")
        self.viewer = viewer
        self.sim_time = 0.0
        self.frame_dt = 1.0 / FPS

        builder = newton.ModelBuilder()
        SolverStyle3D.register_custom_attributes(builder)
        builder.add_ground_plane()

        self._build_workpieces(builder)

        # A shallower stroke limits lateral flow around the narrow leaf and stem cleft.
        pivot = wp.transform(wp.vec3(0.52, 0.0, 0.302), wp.quat_identity())
        steel = wp.vec3(0.12, 0.14, 0.19)
        gold = wp.vec3(0.80, 0.52, 0.10)
        builder.add_shape_box(
            -1,
            xform=wp.transform((0.52, 0.0, 0.14), wp.quat_identity()),
            hx=0.055,
            hy=0.065,
            hz=0.14,
            color=steel,
        )
        press_body = builder.add_link(is_kinematic=True, label="swing_arm")
        head_x = -0.52 + (APPLE_OFFSET_X if self.die == "apple" else 0.0)
        across = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), math.pi / 2)
        builder.add_shape_cylinder(
            press_body,
            xform=wp.transform((head_x / 2, 0, 0.055), across),
            radius=0.022,
            half_height=-head_x / 2,
            color=steel,
        )
        head = wp.transform((head_x, 0, 0), wp.quat_identity())
        if self.die == "cross":
            for hx, hy in ((0.22, 0.055), (0.055, 0.22)):
                builder.add_shape_box(press_body, xform=head, hx=hx, hy=hy, hz=0.042, color=gold)
        else:
            angle = np.linspace(-np.pi, np.pi, 96, endpoint=False)
            # Shorten the radius at the stem and blossom ends to form the apple's clefts.
            radius = 1 - 0.38 * np.exp(-((angle / 0.29) ** 2)) - 0.16 * np.exp(-(((np.pi - np.abs(angle)) / 0.32) ** 2))
            outline = radius[:, None] * np.column_stack(
                (
                    0.17 * np.cos(angle),
                    0.16 * np.sin(angle) * (1 + 0.20 * np.cos(angle)),
                )
            )
            builder.add_shape_mesh(
                press_body, xform=head, mesh=_extrude_profile(outline, 0.042), color=(0.57, 0.80, 0.04)
            )
            leaf_tilt = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), -0.52)
            leaf = np.column_stack((0.063 * np.cos(angle), 0.033 * np.sin(angle) * np.abs(np.sin(angle))))
            builder.add_shape_mesh(
                press_body,
                xform=wp.transform((head_x + 0.189, -0.032, 0), leaf_tilt),
                mesh=_extrude_profile(leaf, 0.042),
                color=(0.12, 0.48, 0.04),
            )
        builder.add_shape_cylinder(
            press_body,
            xform=wp.transform((head_x, 0, 0.065), wp.quat_identity()),
            radius=0.056,
            half_height=0.024,
            color=steel,
        )
        hinge = builder.add_joint_revolute(-1, press_body, parent_xform=pivot, axis=newton.Axis.Y)
        builder.add_articulation([hinge], label="press")
        builder.joint_q[0] = OPEN_ANGLE
        if self.die == "apple":
            _add_letter_press(builder, steel, gold)
        self._add_feed_press(builder)
        self.model = builder.finalize()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        self.model.tri_color = wp.full(self.model.tri_count, wp.vec3(0.78, 0.25, 0.07), device=self.model.device)
        self.model.soft_contact_ke = 2.0e4
        self.model.soft_contact_kd = 2.0
        self.model.soft_contact_kf = 10.0
        self.model.soft_contact_mu = 0.5

        self.solver = SolverStyle3D(
            self.model,
            dynamics=SolverStyle3D.Dynamics.ALL,
            config=SolverStyle3D.Config(iterations=self.iterations, linear_iterations=self.linear_iterations),
        )
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model, soft_contact_gap=0.01)
        self.contacts = self.collision_pipeline.contacts()
        self.clock = wp.zeros(1, dtype=float, device=self.model.device)
        self.graph = None

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(-1.05, -1.45, 1.40), -30.0, 53.0)
        if self.model.device.is_cuda and not wp.config.verify_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def _build_workpieces(self, builder):
        # Steel platen under the pad.
        builder.add_shape_box(
            -1,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.075), q=wp.quat_identity()),
            hx=0.43,
            hy=0.43,
            hz=0.045,
            color=wp.vec3(0.08, 0.10, 0.14),
        )

        mu = PAD_YOUNGS_MODULUS / (2.0 * (1.0 + PAD_POISSON_RATIO))
        lam = PAD_YOUNGS_MODULUS * PAD_POISSON_RATIO / ((1.0 + PAD_POISSON_RATIO) * (1.0 - 2.0 * PAD_POISSON_RATIO))

        builder.add_soft_grid(
            pos=wp.vec3(-PAD_LENGTH / 2, -PAD_WIDTH / 2, 0.12),
            rot=wp.quat_identity(),
            vel=wp.vec3(),
            # Resolve the letter strokes across the stamping surface.
            dim_x=PAD_DIM_X,
            dim_y=PAD_DIM_Y,
            dim_z=PAD_DIM_Z,
            cell_x=PAD_LENGTH / PAD_DIM_X,
            cell_y=PAD_WIDTH / PAD_DIM_Y,
            cell_z=PAD_HEIGHT / PAD_DIM_Z,
            density=250.0,
            k_mu=mu,
            k_lambda=lam,
            k_damp=0.0,
            particle_radius=0.008,
            label="plastic_pad",
            custom_attributes={"style3d:volume_yield_strain": PAD_YIELD_STRAIN},
        )
        # Bond the underside to the platen so the swinging die cannot drag the sample away.
        for i, point in enumerate(builder.particle_q):
            if np.isclose(point[2], 0.12):
                builder.particle_flags[i] &= ~newton.ParticleFlags.ACTIVE

    def _add_feed_press(self, builder):
        pass

    def _advance_belt(self):
        pass

    def simulate(self):
        for _ in range(SUBSTEPS):
            self._advance_belt()
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            wp.launch(
                _drive_press_kernel,
                dim=1,
                inputs=[SUBSTEP_DT, self.hammer_period, self.hammer_speed, self.after_hammers, self.clock],
                outputs=[self.state_0.joint_q, self.state_0.joint_qd],
                device=self.model.device,
            )
            newton.eval_fk(
                self.model,
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.state_0,
                body_flag_filter=BodyFlags.KINEMATIC,
            )
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, SUBSTEP_DT)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph is None:
            self.simulate()
        else:
            wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt

    def test_post_step(self):
        """Keep the press and plastic pad finite as the die moves."""
        assert np.isfinite(self.state_0.particle_q.numpy()).all(), "The pad became non-finite."
        assert np.isfinite(self.state_0.body_q.numpy()).all(), "The press became non-finite."

    def test_final(self):
        """Retain a measurable permanent shape change after the first press."""
        stretch = self.state_0.style3d.tet_plastic_rest_stretch.numpy()
        assert np.isfinite(stretch).all(), "Plastic rest stretch became non-finite."
        plastic_change = np.linalg.norm(stretch - np.eye(3), axis=(1, 2))
        assert np.max(plastic_change) > 1.0e-3, "The press did not produce a permanent indentation."
        rest = self.model.particle_q.numpy()
        xy = np.abs(rest[:, :2])
        top = np.isclose(rest[:, 2], rest[:, 2].max())
        if self.die == "cross":
            footprint = top & (xy.max(axis=1) < 0.205) & (xy.min(axis=1) < 0.025)
        else:
            footprint = top & (np.abs(rest[:, 0] - APPLE_OFFSET_X) < 0.09) & (xy[:, 1] < 0.12)
        corners = top & (xy.min(axis=1) > 0.18)
        heights = self.state_0.particle_q.numpy()[:, 2]
        assert np.median(heights[corners]) - np.median(heights[footprint]) > 0.01, "The imprint disappeared."

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--die", choices=("cross", "apple"), default="cross", help="Stamp fitted to the swing arm.")
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
