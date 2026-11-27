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
Masses are illustrative: up to 0.3 g per pearl and 35 g for the pendant, rather than
material densities inferred from the enlarged geometry or measured product data.

Success criteria:
- All pearls stay threaded and the twelve cord ends remain attached.
- Three separate strands connect matching eyelets on the front and rear fittings.
- Cable nodes and rigid bodies remain finite and above the display base.
- The bore has clearance around the cord; no solid sphere repels the cord.

Runs:
- `python -m newton.examples style3d_pearl_necklace`
  - Release the whole necklace from rest above the display; gravity and contact determine its landing.
- `python -m newton.examples style3d_pearl_necklace --drop-height 0`
  - Start directly around the neck for comparison with the drop.
- `python -m newton.examples style3d_pearl_necklace --display table`
  - Right-drag the pendant or a pearl on the table; release to observe recoil.
- `python -m newton.examples style3d_pearl_necklace --pull-test --test --viewer null --num-frames 300`
  - A bounded spring pulls the pendant after settling, then releases it.
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverStyle3D

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


def _pendant_pose(display):
    """Place the orb upright on the bust, or face-up on the table."""
    if display == "table":
        return wp.transform((0.0, -0.365, 0.032), wp.quat_identity())
    return wp.transform((0.0, -0.175, 0.390), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi / 2))


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


def _cord_path(side, layer, display):
    """Resample each branch uniformly so particle contacts resolve every bore."""
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
        rx = np.interp(z, BUST_SECTIONS[:, 0], BUST_SECTIONS[:, 1]) + PEARL_RADIUS + 0.005
        ry = np.interp(z, BUST_SECTIONS[:, 0], BUST_SECTIONS[:, 2]) + PEARL_RADIUS + 0.005
        dense = np.column_stack((side * rx * np.sin(angle), ry * np.cos(angle), z))
    # Author the initial cord to meet the actual upper-rim eyelet. This blend
    # only constructs rest geometry; no runtime curve-following force is used.
    _, eyelet = _front_eyelet(side, layer)
    endpoint = np.array(wp.transform_point(_pendant_pose(display), wp.vec3(eyelet)))
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
        self.pull_test = args.pull_test
        self.drop_height = getattr(args, "drop_height", 0.6)
        if not np.isfinite(self.drop_height) or self.drop_height < 0.0:
            raise ValueError("drop-height must be finite and nonnegative")
        self.frame_dt = 1.0 / 60.0
        self.sim_substeps = getattr(args, "substeps", 2)
        if self.sim_substeps < 1:
            raise ValueError("substeps must be positive")
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
        surface_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ke=1.0e3, kd=1.0, kf=2.0, mu=0.25)
        builder.add_shape_box(
            -1,
            xform=wp.transform((0.0, 0.0, -0.025), wp.quat_identity()),
            hx=0.65,
            hy=0.65,
            hz=0.025,
            cfg=surface_cfg,
            color=(0.15, 0.20, 0.23),
        )
        if self.display == "bust":
            builder.add_shape_mesh(-1, mesh=_bust_mesh(), cfg=surface_cfg, color=(0.72, 0.68, 0.58))

        paths = [_cord_path(side, layer, self.display) for layer in range(STRAND_COUNT) for side in (-1, 1)]
        rear_points = np.array([path[0][0] for path in paths])
        fitting_rotation = (
            wp.quat_identity()
            if self.display == "table"
            else wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi / 2)
        )
        rear_center = rear_points.mean(axis=0)
        gold = (0.65, 0.45, 0.17)
        fitting_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0, has_particle_collision=False, ke=1.0e3, kd=1.0, kf=2.0, mu=0.25
        )
        self.clasp = builder.add_link(
            xform=wp.transform(rear_center, fitting_rotation),
            mass=0.010,
            inertia=wp.mat33(np.diag((1.0e-5, 5.0e-6, 1.4e-5))),
            label="rear_clasp",
        )
        self.pendant = builder.add_link(
            xform=_pendant_pose(self.display),
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
            density=0.0, has_particle_collision=False, ke=300.0, kd=0.2, kf=0.2, mu=0.15
        )
        proxy_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0, has_particle_collision=False, is_visible=False, ke=1.0e3, kd=0.5, kf=1.0, mu=0.12
        )
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
                cfg=newton.ModelBuilder.ShapeConfig(ke=300.0, kd=0.2, kf=0.2, mu=0.15),
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

        # Translate the entire necklace once at construction. Body-local anchors
        # and material rest geometry are unchanged; there is no runtime support
        # or prescribed release trajectory. All initial velocities remain zero.
        lift = wp.vec3(0.0, 0.0, self.drop_height)
        for i, pose in enumerate(builder.body_q):
            builder.body_q[i] = wp.transform(wp.transform_get_translation(pose) + lift, wp.transform_get_rotation(pose))

        self.model = builder.finalize()
        self.model.soft_contact_ke = 3.0e2
        self.model.soft_contact_kd = 0.2
        self.model.soft_contact_kf = 0.2
        self.model.soft_contact_mu = 0.15
        self.solver = SolverStyle3D(
            self.model,
            dynamics=SolverStyle3D.Dynamics.AFFINE_BODY | SolverStyle3D.Dynamics.CABLE,
            config=SolverStyle3D.Config(
                iterations=15,
                linear_iterations=30,
                cable_stretch_stiffness=60.0,
                cable_bend_stiffness=2.0e-6,
                cable_twist_stiffness=1.0e-6,
            ),
        )
        # Thin tube walls must not repel their opposite side at the cloth
        # handler's default 6 mm thickness, which exceeds the whole diameter.
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=20000, soft_contact_max=0)
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
        self.solver.reset(self.state_0)
        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

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

    def simulate(self):
        for substep in range(self.sim_substeps):
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
            # Captured frames must finish in the same buffer they started from.
            # Otherwise an odd substep count replays from the penultimate state.
            if self.sim_substeps % 2 == 1 and substep == self.sim_substeps - 1:
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
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
        assert min(line[:, 2].min() for line in self.cable_centerlines()) > -0.01, "A cable passed through the base"
        assert np.all(self.threading_errors() < BORE_RADIUS + 0.001), "A pearl left its cable"
        assert self.contacts.rigid_contact_count.numpy()[0] <= self.contacts.rigid_contact_max, (
            "Contact buffer overflow"
        )

    def test_final(self):
        """Bind only twelve cable endpoints to the freely moving front/rear fittings."""
        self.test_post_step()
        assert len(self.cord_attachments) == 4 * STRAND_COUNT
        assert self.model.attachment_particle_body_count == 0
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
    parser.add_argument("--substeps", type=int, default=6, help="Collision and solver substeps per 60 Hz frame.")
    parser.add_argument(
        "--display", choices=("table", "bust"), default="bust", help="Display support for the necklace."
    )
    parser.add_argument(
        "--pull-test", action="store_true", help="Pull the pendant with a bounded spring from 1 to 3 seconds."
    )
    parser.add_argument(
        "--drop-height",
        type=float,
        default=0.006,
        help="Lift the whole necklace by this height [m] before releasing it from rest.",
    )
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
