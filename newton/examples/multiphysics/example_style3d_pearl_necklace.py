# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Drop a three-strand pearl necklace onto a neck-and-shoulder display.

Six circular-section VIPER cables form the left and right halves of three strands.
Only their twelve ends attach to the dynamic rear clasp and front pendant.
Each fitting has three distinct attachment points on each side. Pearls
have closed, drilled sphere meshes for cable contact; hidden sphere proxies
handle rigid contacts without filling the holes seen by the cords. Pearls can
slide and rotate freely, with no attachment or joint to the cords. The scene
uses enlarged jewelry dimensions in meters to make the holes easy to inspect.
Pearls taper toward both fittings and are largest halfway along each cord.
With --scarf, a simulated cloth scarf sits inside the necklace on the bust. Its front slit
is closed by two vertically arranged dynamic brass buttons using particle-body
attachments. The scarf has no fixed vertices; contact supports it on the bust.
Play Animation in Example Options drives a procedural walking gait with torso
sway, twist, vertical bob and alternating turns. The necklace and scarf follow by contact.
The Running toggle smoothly blends into a faster gait with stronger bob and forward lean.
Masses are illustrative: up to 0.3 g per pearl and 35 g for the pendant, rather than
material densities inferred from the enlarged geometry or measured product data.

Success criteria:
- All pearls stay threaded and the twelve cord ends remain attached.
- Three separate strands connect matching eyelets on the front and rear fittings.
- Cable nodes and rigid bodies remain finite and above the display base.
- The bore has clearance around the cord; no solid sphere repels the cord.
- The scarf remains finite and both buttons stay attached to its front lips.

Runs:
- `python -m newton.examples style3d_pearl_necklace`
  - Release the whole necklace from rest above the display; gravity and contact determine its landing.
- `python -m newton.examples style3d_pearl_necklace --drop-height 0`
  - Start directly around the neck for comparison with the drop.
- `python -m newton.examples style3d_pearl_necklace --display table`
  - Right-drag the pendant or a pearl on the table; release to observe recoil.
- `python -m newton.examples style3d_pearl_necklace --scarf`
  - Add the loose buttoned scarf inside the necklace; the default display has no cloth.
- `python -m newton.examples style3d_pearl_necklace --play-animation`
  - Loop a walking gait; uncheck Play Animation to hold the current pose.
- `python -m newton.examples style3d_pearl_necklace --pull-test --test --viewer null --num-frames 300`
  - A bounded spring pulls the pendant after settling, then releases it.
"""

import argparse

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverStyle3D, style3d

# Solver controls: edit here and restart the example.
SIM_SUBSTEPS = 6
SOLVER_ITERATIONS = 15  # Nonlinear iterations per substep.
LINEAR_ITERATIONS = 15  # PCG iterations per nonlinear solve.

SCARF_HEM_HEIGHT = 0.012  # Initial lower edge above the table [m].

STRAND_COUNT = 3
PEARL_RADIUS = 0.018
MIN_PEARL_RADIUS = 0.009
BORE_RADIUS = 0.004
CORD_RADIUS = 0.0015
CORD_SPACING = 0.006
# Illustrative physical weights despite the enlarged display geometry.
PEARL_MASS = 0.0003  # kg for the largest bead; smaller beads use the same density
PENDANT_MASS = 0.035  # kg
CORD_LINEAR_DENSITY = 0.00028  # kg/m
# Elliptical horizontal sections: height, half-width, half-depth [m].
BUST_SECTIONS = np.array(
    [
        (0.025, 0.29, 0.14),
        (0.10, 0.34, 0.17),
        (0.22, 0.37, 0.18),
        (0.30, 0.31, 0.17),
        (0.40, 0.19, 0.135),
        (0.50, 0.12, 0.105),
        (0.60, 0.115, 0.10),
        (0.82, 0.115, 0.10),
    ]
)
BUST_PIVOT = wp.vec3(0.0, 0.0, float(BUST_SECTIONS[0, 0]))
BUST_STEPS_PER_MINUTE = 110.0
BUST_STRIDE_SECONDS = 120.0 / BUST_STEPS_PER_MINUTE
BUST_BOB_HEIGHT = 0.024  # Peak-to-peak vertical travel [m].
BUST_RUN_DEFAULT = True
BUST_RUN_STEPS_PER_MINUTE = 170.0
BUST_RUN_BOB_HEIGHT = 0.060
BUST_RUN_LEAN_DEGREES = 10.0
BUST_GAIT_TRANSITION_SECONDS = 0.35  # Exponential blending time constant.
BUST_TURN_DEGREES = 90.0
BUST_TURN_START_SECONDS = 4.0  # Walk forward before the first turn.
BUST_TURN_SECONDS = 2.0
BUST_TURN_HOLD_SECONDS = 2.0  # Keep walking in each new direction.


@wp.func
def _bust_heading(time: float):
    # Front -> left -> front -> right -> front, with no reset jump.
    elapsed = wp.max(time - BUST_TURN_START_SECONDS, 0.0)
    segment_seconds = BUST_TURN_SECONDS + BUST_TURN_HOLD_SECONDS
    segment = int(wp.floor(elapsed / segment_seconds)) % 4
    u = wp.clamp((elapsed % segment_seconds) / BUST_TURN_SECONDS, 0.0, 1.0)
    blend = u * u * u * (10.0 + u * (-15.0 + 6.0 * u))
    heading = blend
    if segment == 1:
        heading = 1.0 - blend
    elif segment == 2:
        heading = -blend
    elif segment == 3:
        heading = blend - 1.0
    return heading * BUST_TURN_DEGREES * wp.pi / 180.0


@wp.func
def _bust_gait_blend(time: float):
    u = wp.clamp(time / 1.5, 0.0, 1.0)
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


@wp.func
def _bust_offset(time: float, phase: float, run_blend: float):
    # One lateral cycle per stride, two vertical bobs (one per footstep).
    local_offset = _bust_gait_blend(time) * wp.vec3(
        0.008 * wp.sin(phase),
        0.0,
        0.5 * wp.lerp(BUST_BOB_HEIGHT, BUST_RUN_BOB_HEIGHT, run_blend) * (1.0 - wp.cos(2.0 * phase)),
    )
    heading = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), _bust_heading(time))
    return wp.quat_rotate(heading, local_offset)


@wp.func
def _bust_rotation(time: float, phase: float, run_blend: float):
    scale = _bust_gait_blend(time) * wp.pi / 180.0
    pitch = wp.quat_from_axis_angle(
        wp.vec3(1.0, 0.0, 0.0),
        scale * (wp.lerp(3.0, BUST_RUN_LEAN_DEGREES, run_blend) + wp.lerp(1.5, 2.5, run_blend) * wp.sin(2.0 * phase)),
    )
    roll = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), scale * 4.0 * wp.sin(phase))
    yaw = wp.quat_from_axis_angle(
        wp.vec3(0.0, 0.0, 1.0), _bust_heading(time) + scale * wp.lerp(3.0, 5.0, run_blend) * wp.cos(phase)
    )
    return yaw * roll * pitch


@wp.func
def _advance_bust_gait(gait: wp.vec2, running: int, dt: float):
    if dt == 0.0:
        return gait
    blend = wp.lerp(float(running), gait[1], wp.exp(-dt / BUST_GAIT_TRANSITION_SECONDS))
    cadence = wp.lerp(BUST_STEPS_PER_MINUTE, BUST_RUN_STEPS_PER_MINUTE, 0.5 * (gait[1] + blend))
    return wp.vec2(gait[0] + wp.pi * cadence / 60.0 * dt, blend)


@wp.kernel
def _animate_bust(
    body: int,
    pivot: wp.vec3,
    time: wp.array[float],
    playing: wp.array[int],
    gait: wp.array[wp.vec2],
    running: wp.array[int],
    dt: float,
    advance: bool,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Prescribe only the bust, with a twist matching the solver's pose prediction."""
    t = time[0]
    g = gait[0]
    h = float(0.0)
    if playing[0] != 0:
        h = dt
    if advance:
        t += h
        time[0] = t
        g = _advance_bust_gait(g, running[0], h)
        gait[0] = g
    next_g = _advance_bust_gait(g, running[0], h)
    rotation = _bust_rotation(t, g[0], g[1])
    position = pivot + _bust_offset(t, g[0], g[1])
    linear = wp.vec3(0.0)
    angular = wp.vec3(0.0)
    if playing[0] != 0:
        linear = (_bust_offset(t + h, next_g[0], next_g[1]) - _bust_offset(t, g[0], g[1])) / dt
        # Invert the normalized Euler quaternion step used for kinematic contacts.
        delta = _bust_rotation(t + h, next_g[0], next_g[1]) * wp.quat_inverse(rotation)
        angular = wp.vec3(delta[0], delta[1], delta[2]) * (2.0 / (dt * delta[3]))
    body_q[body] = wp.transform(position, rotation)
    body_qd[body] = wp.spatial_vector(linear, angular)


def _pendant_pose(display, *, scarf=False):
    """Place the orb upright on the bust, or face-up on the table."""
    if display == "table":
        return wp.transform((0.0, -0.365, 0.032), wp.quat_identity())
    return wp.transform(
        (0.0, -0.195 if scarf else -0.175, 0.390), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi / 2)
    )


def _orbit_point(angle):
    """A tilted orbit: its upper half passes behind the central orb."""
    return np.array((0.095 * np.cos(angle), -0.010 + 0.028 * np.sin(angle), -0.016 * np.sin(angle)))


def _front_eyelet(side, layer):
    """Return the ring root and eyelet center, ordered inner strand to outer."""
    root = _orbit_point(np.arccos(side * (0.30, 0.63, 0.92)[layer]))
    return root, root + np.array((side * 0.0015, 0.005, 0.0))


def _drilled_pearl(radius):
    """Make an outward-oriented closed sphere with a cylindrical Z-axis bore."""
    segments, rings = 40, 25
    theta = np.linspace(np.arcsin(BORE_RADIUS / radius), np.pi - np.arcsin(BORE_RADIUS / radius), rings)
    phi = np.arange(segments) * (2.0 * np.pi / segments)
    vertices = np.stack(
        (
            radius * np.sin(theta[:, None]) * np.cos(phi),
            radius * np.sin(theta[:, None]) * np.sin(phi),
            np.broadcast_to(radius * np.cos(theta[:, None]), (rings, segments)),
        ),
        axis=-1,
    ).reshape(-1, 3)
    indices = []
    # Closing the last (bottom) ring to the first gives the inward-facing bore.
    for i in range(rings):
        for j in range(segments):
            a, b = i * segments + j, ((i + 1) % rings) * segments + j
            c, d = ((i + 1) % rings) * segments + (j + 1) % segments, i * segments + (j + 1) % segments
            indices.extend((a, b, c, a, c, d))
    return newton.Mesh(vertices, indices, roughness=0.20, metallic=0.12)


def _bust_mesh():
    """Loft a closed neck-and-shoulder display without external assets."""
    sections = BUST_SECTIONS
    segments = 80
    phi = np.arange(segments) * 2.0 * np.pi / segments
    vertices = [(rx * np.cos(p), ry * np.sin(p), z) for z, rx, ry in sections for p in phi]
    indices = []
    for i in range(len(sections) - 1):
        for j in range(segments):
            a, b = i * segments + j, (i + 1) * segments + j
            c, d = (i + 1) * segments + (j + 1) % segments, i * segments + (j + 1) % segments
            indices.extend((a, d, c, a, c, b))
    for ring, sign in ((0, -1), (len(sections) - 1, 1)):
        center = len(vertices)
        vertices.append((0.0, 0.0, sections[ring, 0]))
        for j in range(segments):
            a, b = ring * segments + j, ring * segments + (j + 1) % segments
            indices.extend((center, b, a) if sign < 0 else (center, a, b))
    return newton.Mesh(vertices, indices, compute_inertia=False, roughness=0.85)


def _cord_path(side, layer, display, *, scarf=False):
    """Resample each branch uniformly so cable segments resolve every bore."""
    t = np.linspace(0.0, 1.0, 2001)
    if display == "table":
        dense = np.column_stack(
            (
                side * (0.035 + 0.015 * layer + 0.025 * t + (0.19 + 0.04 * layer) * np.sin(np.pi * t)),
                (0.32 + 0.035 * layer) * np.cos(np.pi * t),
                np.full_like(t, PEARL_RADIUS + 0.004),
            )
        )
    else:
        angle = 0.28 + (np.pi - 0.56) * t
        blend = (1.0 - np.cos(np.pi * t)) / 2.0
        z = (0.575 - 0.043 * layer) * (1.0 - blend) + (0.440 - 0.043 * layer) * blend
        clearance = 0.025 if scarf else 0.005
        rx = np.interp(z, BUST_SECTIONS[:, 0], BUST_SECTIONS[:, 1]) + PEARL_RADIUS + clearance
        ry = np.interp(z, BUST_SECTIONS[:, 0], BUST_SECTIONS[:, 2]) + PEARL_RADIUS + clearance
        dense = np.column_stack((side * rx * np.sin(angle), ry * np.cos(angle), z))
    # Author the initial cord to meet the actual upper-rim eyelet. This blend
    # only constructs rest geometry; no runtime curve-following force is used.
    _, eyelet = _front_eyelet(side, layer)
    endpoint = np.array(wp.transform_point(_pendant_pose(display, scarf=scarf), wp.vec3(eyelet)))
    blend = np.clip((t - 0.65) / 0.35, 0.0, 1.0)
    blend = blend * blend * (3.0 - 2.0 * blend)
    dense += blend[:, None] * (endpoint - dense[-1])
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    arc = np.linspace(0.0, distance[-1], int(np.ceil(distance[-1] / CORD_SPACING)) + 1)
    points = np.column_stack([np.interp(arc, distance, dense[:, axis]) for axis in range(3)])
    return points, arc


def _pearl_layout(length):
    """Pack graded beads by local diameter, with small beads at both fittings."""
    start, end = 0.020, length - 0.020
    samples = np.linspace(start, end, 4001)
    radius = MIN_PEARL_RADIUS + (PEARL_RADIUS - MIN_PEARL_RADIUS) * np.sin(np.linspace(0.0, np.pi, len(samples)))
    packing = np.r_[0.0, np.cumsum(np.diff(samples) / (radius[:-1] + radius[1:] + 0.002))]
    centers = np.interp(np.linspace(0.0, packing[-1], int(packing[-1]) + 1), packing, samples)
    radii = MIN_PEARL_RADIUS + (PEARL_RADIUS - MIN_PEARL_RADIUS) * np.sin(np.pi * (centers - start) / (end - start))
    return centers, radii


@wp.kernel
def _pull_pendant(
    body: int,
    target: wp.array[wp.vec3],
    active: wp.array[float],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_mass: wp.array[float],
    body_com: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
):
    """Apply a bounded spring through forces, leaving all poses solver-owned."""
    if active[0] > 0.0:
        center = wp.transform_point(body_q[body], body_com[body])
        error = target[0] - center
        error *= wp.min(1.0, 0.20 / wp.max(wp.length(error), 1.0e-8))
        force = body_mass[body] * (100.0 * error - 20.0 * wp.spatial_top(body_qd[body]))
        wp.atomic_add(body_f, body, wp.spatial_vector(force, wp.vec3(0.0)))


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.display = args.display
        self.scarf_enabled = self.display == "bust" and getattr(args, "scarf", False)
        self.play_animation = self.display == "bust" and getattr(args, "play_animation", False)
        self.running = BUST_RUN_DEFAULT
        self.pull_test = args.pull_test
        self.drop_height = getattr(args, "drop_height", 0.6)
        if not np.isfinite(self.drop_height) or self.drop_height < 0.0:
            raise ValueError("drop-height must be finite and nonnegative")
        self.frame_dt = 1.0 / 60.0
        self.sim_substeps = SIM_SUBSTEPS
        if self.sim_substeps < 1:
            raise ValueError("SIM_SUBSTEPS must be positive")
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        builder = newton.ModelBuilder()
        # Millimeter cables need millimeter contact search distances. The
        # generic 0.1 m gap visits most triangles of many neighboring pearls.
        builder.rigid_gap = 0.0015
        # Millimeter cable segments have valid moments below the generic 1e-10 floor.
        builder.inertia_absolute_tolerance = 1.0e-16
        builder.inertia_correction = 0.0
        SolverStyle3D.register_custom_attributes(builder)
        # Milligram-scale endpoint beads need compliant contact to avoid chatter
        # as the discrete bore contacts change. Keep attachment stiffness separate.
        surface_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ke=1.0e3, kd=1.0, kf=0.01, mu=0.05)
        table_shape = builder.add_shape_box(
            -1,
            xform=wp.transform((0.0, 0.0, -0.025), wp.quat_identity()),
            hx=0.65,
            hy=0.65,
            hz=0.025,
            cfg=surface_cfg,
            color=(0.15, 0.20, 0.23),
        )
        self.bust = -1
        if self.display == "bust":
            self.bust = builder.add_link(
                xform=wp.transform(BUST_PIVOT, wp.quat_identity()), is_kinematic=True, label="neck_display"
            )
            bust_shape = builder.add_shape_mesh(
                self.bust,
                xform=wp.transform(-BUST_PIVOT, wp.quat_identity()),
                mesh=_bust_mesh(),
                cfg=surface_cfg,
                color=(0.72, 0.68, 0.58),
            )
            # A prescribed display can tilt through its support table. Jewelry
            # and cloth still collide with both; no other pair is excluded here.
            builder.add_shape_collision_filter_pair(bust_shape, table_shape)

        paths = [
            _cord_path(side, layer, self.display, scarf=self.scarf_enabled)
            for layer in range(STRAND_COUNT)
            for side in (-1, 1)
        ]
        rear_points = np.array([path[0][0] for path in paths])
        fitting_rotation = (
            wp.quat_identity()
            if self.display == "table"
            else wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi / 2)
        )
        rear_center = rear_points.mean(axis=0)
        gold = (0.65, 0.45, 0.17)
        fitting_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ke=1.0e3, kd=1.0, kf=2.0, mu=0.25)
        self.clasp = builder.add_link(
            xform=wp.transform(rear_center, fitting_rotation),
            mass=0.010,
            inertia=wp.mat33(np.diag((1.0e-5, 5.0e-6, 1.4e-5))),
            label="rear_clasp",
        )
        self.pendant = builder.add_link(
            xform=_pendant_pose(self.display, scarf=self.scarf_enabled),
            mass=PENDANT_MASS,
            # Keep the same mass distribution when adjusting the pendant weight.
            inertia=wp.mat33(PENDANT_MASS * np.diag((8.0e-4, 1.6e-3, 2.0e-3))),
            label="front_pendant",
        )
        self._add_ring(builder, self.clasp, np.zeros(3), 0.028, 0.047, 0.003, fitting_cfg, gold)
        inverse = wp.transform_inverse(builder.body_q[self.clasp])
        for endpoint in rear_points:
            local = np.array(wp.transform_point(inverse, wp.vec3(endpoint)))
            self._add_ring(builder, self.clasp, local, 0.006, 0.005, 0.0018, fitting_cfg, gold, segments=8)
            theta = np.arctan2(local[1] / 0.047, local[0] / 0.028)
            root = np.array((0.028 * np.cos(theta), 0.047 * np.sin(theta), 0.0))
            self._add_bar(builder, self.clasp, root, local, 0.0025, fitting_cfg, gold)
        builder.add_shape_ellipsoid(self.clasp, rx=0.025, ry=0.018, rz=0.010, cfg=fitting_cfg, color=gold)
        self._add_bar(
            builder, self.clasp, np.array((-0.023, 0.0, 0.0)), np.array((0.023, 0.0, 0.0)), 0.003, fitting_cfg, gold
        )
        self._add_pendant(builder, self.pendant, fitting_cfg, gold)

        pearl_density = PEARL_MASS / _drilled_pearl(PEARL_RADIUS).mass
        pearl_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0, has_particle_collision=False, ke=100.0, kd=0.2, kf=0.002, mu=0.05
        )
        proxy_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, is_visible=False, ke=1.0e3, kd=0.5, kf=0.002, mu=0.05)
        self.cord_bodies, self.pearl_bodies, self.pearl_sides = [], [], []
        self.cord_attachments = []
        cable_shapes, pearl_mesh_shapes, pearl_proxy_shapes = [], [], []
        self.pearl_radii = []
        for side_index, (points, arc) in enumerate(paths):
            rod = newton.Rod(points, radius=CORD_RADIUS)
            rod.compute_frames(twist_total=0.0)
            first_shape = builder.shape_count
            bodies, _joints = builder.add_rod(
                rod=rod,
                total_mass=CORD_LINEAR_DENSITY * arc[-1],
                # Low tangential drag lets milligram beads slide through the bores.
                cfg=newton.ModelBuilder.ShapeConfig(ke=300.0, kd=0.2, kf=0.002, mu=0.05),
                label=f"elastic_cable_{side_index}",
                color=(0.60, 0.33, 0.12),
            )
            self.cord_bodies.append(bodies)
            cable_shapes.extend(range(first_shape, builder.shape_count))
            for fitting, segment, endpoint in (
                (self.clasp, bodies[0], points[0]),
                (self.pendant, bodies[-1], points[-1]),
            ):
                local_fitting = wp.transform_point(wp.transform_inverse(builder.body_q[fitting]), wp.vec3(endpoint))
                local_segment = wp.transform_point(wp.transform_inverse(builder.body_q[segment]), wp.vec3(endpoint))
                joint = builder.add_joint_ball(
                    fitting,
                    segment,
                    parent_xform=wp.transform(local_fitting, wp.quat_identity()),
                    child_xform=wp.transform(local_segment, wp.quat_identity()),
                    label=f"cable_endpoint_{side_index}_{fitting}",
                )
                self.cord_attachments.append(joint)

            bead_positions, radii = _pearl_layout(arc[-1])
            for bead, (s, radius) in enumerate(zip(bead_positions, radii, strict=True)):
                center = np.array([np.interp(s, arc, points[:, axis]) for axis in range(3)])
                j = np.clip(np.searchsorted(arc, s), 1, len(points) - 1)
                tangent = points[j] - points[j - 1]
                tangent /= np.linalg.norm(tangent)
                pose = wp.transform(center, wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), wp.vec3(tangent)))
                # Use the drilled volume's mass properties, rather than the solid proxy's.
                pearl_mesh = _drilled_pearl(radius)
                body = builder.add_link(
                    xform=pose,
                    mass=pearl_density * pearl_mesh.mass,
                    inertia=wp.mat33(np.asarray(pearl_mesh.inertia) * pearl_density),
                    label=f"pearl_{side_index}_{bead}",
                )
                pearl_mesh_shapes.append(
                    builder.add_shape_mesh(body, mesh=pearl_mesh, cfg=pearl_cfg, color=(0.93, 0.88, 0.73))
                )
                pearl_proxy_shapes.append(builder.add_shape_sphere(body, radius=radius, cfg=proxy_cfg))
                self.pearl_bodies.append(body)
                self.pearl_sides.append(side_index)
                self.pearl_radii.append(radius)

        # Lift only the necklace. The scarf starts resting around the display.
        lift = wp.vec3(0.0, 0.0, self.drop_height)
        for i, pose in enumerate(builder.body_q):
            if i == self.bust:
                continue
            builder.body_q[i] = wp.transform(wp.transform_get_translation(pose) + lift, wp.transform_get_rotation(pose))
        self.scarf_buttons = []
        self.scarf_attachment_count = 0
        if self.scarf_enabled:
            self._add_scarf(builder)

        # Drilled meshes collide only with cables; sphere proxies serve bead/bead
        # and display contact. A sphere against a cable would seal the bore.
        cable_shape_set = set(cable_shapes)
        for mesh_shape in pearl_mesh_shapes:
            for other_shape in range(builder.shape_count):
                if other_shape not in cable_shape_set and other_shape != mesh_shape:
                    builder.add_shape_collision_filter_pair(mesh_shape, other_shape)
        fitting_shapes = [shape for shape, body in enumerate(builder.shape_body) if body in (self.clasp, self.pendant)]
        for cable_shape in cable_shapes:
            for other_shape in pearl_proxy_shapes + fitting_shapes:
                builder.add_shape_collision_filter_pair(cable_shape, other_shape)

        self.model = builder.finalize()
        self.model.soft_contact_ke = 3.0e2
        self.model.soft_contact_kd = 0.2
        # Cloth/shape tangential response: allow jewelry to slip over the scarf.
        # These shared soft-contact values also apply to the supporting bust.
        self.model.soft_contact_kf = 0.1
        self.model.soft_contact_mu = 0.1
        self.solver = SolverStyle3D(
            self.model,
            dynamics=(
                SolverStyle3D.Dynamics.ALL
                if self.scarf_enabled
                else SolverStyle3D.Dynamics.AFFINE_BODY | SolverStyle3D.Dynamics.CABLE
            ),
            config=SolverStyle3D.Config(
                iterations=SOLVER_ITERATIONS,
                linear_iterations=LINEAR_ITERATIONS,
                cable_stretch_stiffness=60.0,
                cable_bend_stiffness=2.0e-6,
                cable_twist_stiffness=1.0e-6,
            ),
        )
        if self.solver.collision is not None:
            self.solver.collision.radius = 0.001
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.pipeline = newton.CollisionPipeline(
            self.model, rigid_contact_max=20000, soft_contact_max=16 * self.model.particle_count
        )
        self.contacts = self.pipeline.contacts()
        self.viewer.set_model(self.model)
        if self.display == "table":
            self.viewer.set_camera(wp.vec3(0.0, -0.95, 1.35), -54.0, 90.0)
        else:
            self.viewer.set_camera(wp.vec3(0.5, -1.9, 1.1 + 0.5 * self.drop_height), -17.0, 105.0)
        self.pull_origin = wp.transform_point(
            wp.transform(*self.state_0.body_q.numpy()[self.pendant]), wp.vec3(self.model.body_com.numpy()[self.pendant])
        )
        self.pull_target = wp.array([self.pull_origin], dtype=wp.vec3, device=self.model.device)
        self.pull_active = wp.zeros(1, dtype=float, device=self.model.device)
        self.animation_gait = wp.array([wp.vec2(0.0, float(self.running))], dtype=wp.vec2, device=self.model.device)
        self.animation_running = wp.array([int(self.running)], dtype=int, device=self.model.device)
        self.animation_time = wp.zeros(1, dtype=float, device=self.model.device)
        self.animation_playing = wp.array([int(self.play_animation)], dtype=int, device=self.model.device)
        self.solver.reset(self.state_0)
        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def _add_scarf(self, builder):
        """An unpinned wrap with a front slit bridged by two dynamic buttons."""
        columns, rows = 160, 32
        upper_heights = np.linspace(0.21, 0.66, rows + 1)
        if not 0.003 < SCARF_HEM_HEIGHT <= upper_heights[0]:
            raise ValueError("SCARF_HEM_HEIGHT must be above the cloth contact radius and at most 0.21 m")
        extra_rows = int(np.ceil((upper_heights[0] - SCARF_HEM_HEIGHT) / (upper_heights[1] - upper_heights[0])))
        heights = np.concatenate((np.linspace(SCARF_HEM_HEIGHT, upper_heights[0], extra_rows + 1)[:-1], upper_heights))
        rows = len(heights) - 1
        vertices = []
        for z in heights:
            # Extend the existing hem downward without tapering around the base.
            # Preserve the upper mesh and button rows while adding free fabric.
            profile_z = max(z, upper_heights[0])
            rx = np.interp(profile_z, BUST_SECTIONS[:, 0], BUST_SECTIONS[:, 1]) + 0.014
            ry = np.interp(profile_z, BUST_SECTIONS[:, 0], BUST_SECTIONS[:, 2]) + 0.014
            opening = np.arcsin(0.006 / rx)
            for phi in np.linspace(-np.pi / 2 + opening, 3 * np.pi / 2 - opening, columns + 1):
                # Circumferential ease and shallow rest folds supply excess fabric.
                # These author the initial mesh only; no runtime shape constraint.
                fold = 0.009 * (1.0 - np.cos(10.0 * (phi + np.pi / 2))) / 2.0
                vertices.append(((rx + fold) * np.cos(phi), (ry + fold) * np.sin(phi), z))
        vertices = np.array(vertices)
        indices = []
        for row in range(rows):
            for column in range(columns):
                a = row * (columns + 1) + column
                b = a + columns + 1
                indices.extend((a, a + 1, b + 1, a, b + 1, b))
        # Intrinsic triangle metrics avoid initial membrane strain on the loft.
        # Shared physical vertices retain cloth connectivity across UV seams.
        triangles = vertices[np.array(indices).reshape(-1, 3)]
        u, v = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        lengths = np.linalg.norm(u, axis=1)
        projection = np.sum(u * v, axis=1) / lengths
        panel = np.zeros((len(triangles), 3, 2))
        panel[:, 1, 0] = lengths
        panel[:, 2, 0] = projection
        panel[:, 2, 1] = np.sqrt(np.sum(v * v, axis=1) - projection**2)
        start, tri_start = builder.particle_count, builder.tri_count
        style3d.add_cloth_mesh(
            builder,
            pos=wp.vec3(),
            rot=wp.quat_identity(),
            vel=wp.vec3(),
            vertices=vertices,
            indices=indices,
            panel_verts=panel.reshape(-1, 2),
            panel_indices=np.arange(len(indices)),
            density=0.20,
            tri_aniso_ke=wp.vec3(400.0, 400.0, 200.0),
            edge_aniso_ke=wp.vec3(1.0e-5),
            # Leave room for triangle interiors across the bust's section creases.
            particle_radius=0.003,
            label="buttoned_scarf",
        )
        builder.tri_color[tri_start:] = [(0.06, 0.30, 0.33)] * (builder.tri_count - tri_start)
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_particle_collision=False, ke=1000.0, kd=1.0, mu=0.25)
        for height in (0.54, 0.25):
            row = int(np.argmin(abs(heights - height)))
            center = vertices[row * (columns + 1)].copy()
            center[0] = 0.0
            mass, radius = 0.003, 0.023
            disk_offset, disk_half_height, shank_radius = 0.005, 0.0015, 0.004
            body = builder.add_link(
                xform=wp.transform(center, wp.quat_identity()),
                mass=mass,
                inertia=wp.mat33(
                    mass
                    * np.diag((0.25 * radius**2 + disk_offset**2, 0.5 * radius**2, 0.25 * radius**2 + disk_offset**2))
                ),
                label=f"scarf_button_{len(self.scarf_buttons)}",
            )
            # Same double-sided construction as style3d_vintage_buttons:
            # front and rear discs joined by a central shank, all on one body.
            disk_rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi / 2)
            for y in (-disk_offset, disk_offset):
                builder.add_shape_cylinder(
                    body,
                    xform=wp.transform((0.0, y, 0.0), disk_rotation),
                    radius=radius,
                    half_height=disk_half_height,
                    cfg=cfg,
                    color=(0.76, 0.53, 0.22),
                )
            builder.add_shape_cylinder(
                body,
                xform=wp.transform(wp.vec3(), disk_rotation),
                radius=shank_radius,
                half_height=disk_offset,
                cfg=cfg,
                color=(0.76, 0.53, 0.22),
            )
            # Two small patches, one on each lip, sew onto the same rigid button.
            inverse = wp.transform_inverse(builder.body_q[body])
            for r in (row - 1, row, row + 1):
                for column in (0, columns):
                    particle = start + r * (columns + 1) + column
                    builder.add_attachment_particle_body(
                        particle,
                        body,
                        body_point=wp.transform_point(inverse, wp.vec3(builder.particle_q[particle])),
                        stiffness=500.0,
                        damping=0.05,
                    )
                    self.scarf_attachment_count += 1
            self.scarf_buttons.append(body)

    @staticmethod
    def _add_pendant(builder, body, cfg, color):
        """Build a solid orb, tilted Saturn ring, short upper eyelets and cross."""
        builder.add_shape_sphere(
            body, radius=0.033, xform=wp.transform((0.0, -0.008, 0.0), wp.quat_identity()), cfg=cfg, color=color
        )
        # Two rails form the orbit band. The near rail passes in front of the
        # orb's lower half, instead of surrounding a flat central disc.
        for scale in (1.0, 0.93):
            for j in range(48):
                start = _orbit_point(j * 2.0 * np.pi / 48)
                end = _orbit_point((j + 1) * 2.0 * np.pi / 48)
                start[0] *= scale
                end[0] *= scale
                start[1:] *= scale
                end[1:] *= scale
                Example._add_bar(builder, body, start, end, 0.0025, cfg, color)
        for layer in range(STRAND_COUNT):
            for side in (-1, 1):
                root, center = _front_eyelet(side, layer)
                Example._add_ring(builder, body, center, 0.0035, 0.0045, 0.0013, cfg, color, segments=12)
                Example._add_bar(builder, body, root, center + np.array((0.0, -0.0045, 0.0)), 0.0015, cfg, color)
        # A short pedestal places the cross above the orb, rather than running
        # a long shaft through an empty oval frame.
        Example._add_bar(builder, body, np.array((0.0, 0.023, 0.0)), np.array((0.0, 0.073, 0.0)), 0.003, cfg, color)
        Example._add_bar(
            builder, body, np.array((-0.014, 0.059, 0.0)), np.array((0.014, 0.059, 0.0)), 0.003, cfg, color
        )
        # Equatorial band across the orb, in the local XZ plane.
        for j in range(32):
            a, b = j * 2.0 * np.pi / 32, (j + 1) * 2.0 * np.pi / 32
            start = np.array((0.033 * np.cos(a), -0.008, 0.033 * np.sin(a)))
            end = np.array((0.033 * np.cos(b), -0.008, 0.033 * np.sin(b)))
            Example._add_bar(builder, body, start, end, 0.0015, cfg, color)

    @staticmethod
    def _add_bar(builder, body, start, end, radius, cfg, color):
        delta = end - start
        length = np.linalg.norm(delta)
        if length < 1.0e-6:
            return
        builder.add_shape_capsule(
            body,
            radius=radius,
            half_height=length / 2.0,
            xform=wp.transform(
                (start + end) / 2.0, wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), wp.vec3(delta / length))
            ),
            cfg=cfg,
            color=color,
        )

    @staticmethod
    def _add_ring(builder, body, center, rx, ry, radius, cfg, color, *, segments=20):
        for j in range(segments):
            a, b = j * 2.0 * np.pi / segments, (j + 1) * 2.0 * np.pi / segments
            start = center + np.array((rx * np.cos(a), ry * np.sin(a), 0.0))
            end = center + np.array((rx * np.cos(b), ry * np.sin(b), 0.0))
            Example._add_bar(builder, body, start, end, radius, cfg, color)

    def gui(self, ui):
        """Control the prescribed display without constraining the jewelry."""
        if self.bust < 0:
            ui.text("Animation is available with the bust display.")
            return
        _, self.play_animation = ui.checkbox("Play Animation", self.play_animation)
        _, self.running = ui.checkbox("Running", self.running)
        cadence = BUST_RUN_STEPS_PER_MINUTE if self.running else BUST_STEPS_PER_MINUTE
        bob = BUST_RUN_BOB_HEIGHT if self.running else BUST_BOB_HEIGHT
        ui.text(f"Cadence: {cadence:g} steps/min; bob: {1000 * bob:g} mm")
        ui.text(
            f"Turns: +/-{BUST_TURN_DEGREES:g} deg; {BUST_TURN_SECONDS:g} s turn / {BUST_TURN_HOLD_SECONDS:g} s hold"
        )

    def simulate(self):
        for substep in range(self.sim_substeps):
            if self.bust >= 0:
                wp.launch(
                    _animate_bust,
                    dim=1,
                    inputs=[
                        self.bust,
                        BUST_PIVOT,
                        self.animation_time,
                        self.animation_playing,
                        self.animation_gait,
                        self.animation_running,
                        self.sim_dt,
                        False,
                    ],
                    outputs=[self.state_0.body_q, self.state_0.body_qd],
                    device=self.model.device,
                )
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            if self.pull_test:
                wp.launch(
                    _pull_pendant,
                    dim=1,
                    inputs=[
                        self.pendant,
                        self.pull_target,
                        self.pull_active,
                        self.state_0.body_q,
                        self.state_0.body_qd,
                        self.model.body_mass,
                        self.model.body_com,
                    ],
                    outputs=[self.state_0.body_f],
                    device=self.model.device,
                )
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            if self.bust >= 0:
                wp.launch(
                    _animate_bust,
                    dim=1,
                    inputs=[
                        self.bust,
                        BUST_PIVOT,
                        self.animation_time,
                        self.animation_playing,
                        self.animation_gait,
                        self.animation_running,
                        self.sim_dt,
                        True,
                    ],
                    outputs=[self.state_1.body_q, self.state_1.body_qd],
                    device=self.model.device,
                )
            # Captured frames must finish in the same buffer they started from.
            # Otherwise an odd substep count replays from the penultimate state.
            if self.sim_substeps % 2 == 1 and substep == self.sim_substeps - 1:
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.animation_playing.fill_(int(self.play_animation))
        self.animation_running.fill_(int(self.running))
        if self.pull_test:
            active = 1.0 <= self.sim_time < 3.0
            ramp = np.clip(self.sim_time - 1.0, 0.0, 1.0)
            offset = wp.vec3(0.12 * ramp, -0.08 * ramp, 0.0 if self.display == "table" else 0.03 * ramp)
            self.pull_target.fill_(self.pull_origin + offset)
            self.pull_active.fill_(float(active))
        if self.graph is None:
            self.simulate()
        else:
            wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt

    def threading_errors(self):
        """Measure the nearest cord crossing of each pearl's bore midplane [m]."""
        poses = self.state_0.body_q.numpy()
        centerlines = self.cable_centerlines()
        errors = []
        for body, side in zip(self.pearl_bodies, self.pearl_sides, strict=True):
            rotation = np.asarray(wp.quat_to_matrix(wp.quat(*poses[body, 3:]))).reshape(3, 3)
            local = (centerlines[side] - poses[body, :3]) @ rotation
            valid = (local[:-1, 2] * local[1:, 2] <= 0.0) & (np.abs(local[1:, 2] - local[:-1, 2]) > 1.0e-9)
            a, b = local[:-1][valid], local[1:][valid]
            if len(a):
                crossing = a - (a[:, 2] / (b[:, 2] - a[:, 2]))[:, None] * (b - a)
                errors.append(float(np.linalg.norm(crossing[:, :2], axis=1).min()))
            else:
                errors.append(float("inf"))
        return np.array(errors)

    def cable_centerlines(self):
        """Read the solved VIPER nodes for each connected cable branch."""
        cable = self.solver._cable
        nodes = cable._node_position.numpy()
        return [nodes[list(component)] for component in cable.component_offsets]

    def test_post_step(self):
        """Keep all states finite and every pearl threaded throughout motion."""
        for array in (self.state_0.body_q.numpy(), self.state_0.body_qd.numpy()):
            assert np.isfinite(array).all(), "The necklace became non-finite"
        if self.bust >= 0:
            pivot = self.state_0.body_q.numpy()[self.bust, :3]
            assert np.linalg.norm(pivot[:2]) <= 0.00801
            assert BUST_PIVOT[2] - 1e-7 <= pivot[2] <= BUST_PIVOT[2] + max(BUST_BOB_HEIGHT, BUST_RUN_BOB_HEIGHT) + 1e-7
        assert min(line[:, 2].min() for line in self.cable_centerlines()) > -0.01, "A cable passed through the base"
        assert np.all(self.threading_errors() < BORE_RADIUS + 0.001), "A pearl left its cable"
        assert self.contacts.rigid_contact_count.numpy()[0] <= self.contacts.rigid_contact_max, (
            "Contact buffer overflow"
        )
        if self.scarf_enabled:
            particles = self.state_0.particle_q.numpy()
            assert np.isfinite(particles).all(), "The scarf became non-finite"
            assert particles[:, 2].min() > 0.0, "The scarf passed through the base"
            assert self.contacts.soft_contact_count.numpy()[0] <= self.contacts.soft_contact_max

    def test_final(self):
        """Keep cable ends on their fittings and both scarf lips on the buttons."""
        self.test_post_step()
        assert len(self.cord_attachments) == 4 * STRAND_COUNT
        assert self.model.attachment_particle_body_count == self.scarf_attachment_count
        if self.scarf_enabled:
            particles = self.state_0.particle_q.numpy()
            poses = self.state_0.body_q.numpy()
            for particle, body, point in zip(
                self.model.attachment_particle_body_particle.numpy(),
                self.model.attachment_particle_body_body.numpy(),
                self.model.attachment_particle_body_body_point.numpy(),
                strict=True,
            ):
                assert body in self.scarf_buttons
                target = np.array(wp.transform_point(wp.transform(*poses[body]), wp.vec3(point)))
                assert np.linalg.norm(particles[particle] - target) < 0.005, "A scarf button detached"
        poses = self.state_0.body_q.numpy()
        lines = self.cable_centerlines()
        anchors = self.model.joint_X_p.numpy()
        for side, line in enumerate(lines):
            for endpoint, fitting, joint in (
                (line[0], self.clasp, self.cord_attachments[2 * side]),
                (line[-1], self.pendant, self.cord_attachments[2 * side + 1]),
            ):
                target = np.array(wp.transform_point(wp.transform(*poses[fitting]), wp.vec3(anchors[joint, :3])))
                assert np.linalg.norm(endpoint - target) < 0.005, "A cable end detached"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--scarf",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add the buttoned cloth scarf to the bust (disabled by default).",
    )
    parser.add_argument("--play-animation", action="store_true", help="Animate the bust about its bottom center.")
    parser.add_argument(
        "--display", choices=("table", "bust"), default="bust", help="Display support for the necklace."
    )
    parser.add_argument(
        "--pull-test", action="store_true", help="Pull the pendant with a bounded spring from 1 to 3 seconds."
    )
    parser.add_argument(
        "--drop-height",
        type=float,
        default=0.25,
        help="Lift the whole necklace by this height [m] before releasing it from rest.",
    )
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
