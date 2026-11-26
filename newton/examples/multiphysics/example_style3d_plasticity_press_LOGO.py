# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Press an OBJ logo die into a rounded plastic pad.

Success criteria:
- The pad remains finite and retains an indentation after release.

Runs:
- python -m newton.examples style3d_plasticity_press_LOGO
- python -m newton.examples style3d_plasticity_press_LOGO --logo-path path/to/logo.obj
"""

import math
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import BodyFlags
from newton.solvers import SolverStyle3D

FPS = 60
SCENE_SCALE = 10.0  # Uniform geometric scale; set to 1.0 for the original size.
# Absolute contact distances [m], independent of SCENE_SCALE.
PAD_PARTICLE_RADIUS = 0.008  # Particle-to-shape collision radius.
SOFT_CONTACT_GAP = 0.05  # Extra particle-to-shape contact detection distance.
SHAPE_CONTACT_MARGIN = 0.0  # Outward offset of rigid collision surfaces.
SHAPE_CONTACT_GAP = 0.1  # Extra rigid-shape contact detection distance.
SELF_CONTACT_THICKNESS = 0.006  # Full self-contact thickness (twice the internal radius).
SUBSTEPS = 2
SUBSTEP_DT = 1.0 / (FPS * SUBSTEPS)
PRESS_PERIOD = 2.0
OPEN_ANGLE = 1.28
PRESS_DOWN_TIME = 0.4
PRESS_HOLD_TIME = 0.5
PRESS_UP_TIME = 0.9
# Never restart a stroke before the previous lift has completed.
PRESS_PERIOD = max(PRESS_PERIOD, PRESS_DOWN_TIME + PRESS_HOLD_TIME + PRESS_UP_TIME + 0.05)
# Pad dimensions [m]; subdivision counts change resolution without changing size.
PAD_LENGTH = 0.72
PAD_WIDTH = 0.72
PAD_HEIGHT = 0.18
PAD_DIM_X = 23 * 6
PAD_DIM_Y = 23 * 6
PAD_DIM_Z = 18
PAD_YOUNGS_MODULUS = 5.0e3
PAD_POISSON_RATIO = 0.49
PAD_YIELD_STRAIN = 0.18
LOGO_DIE_MAX_SPAN = 0.45
LOGO_PIVOT_HEIGHT = 0.287
LOGO_FRAME_ENABLED = True
LOGO_FRAME_THICKNESS = 0.025
LOGO_FRAME_CLEARANCE = 0.002
PAD_CORNER_POWER = 4.0
PAD_DENSITY = 100.0  # [kg/m^3] 250


def _scale_press_scene(builder, scale):
    """Scale geometry and rest data, preserving density and contact thresholds."""
    for name in ("particle_q", "particle_qd", "body_com", "shape_scale"):
        values = getattr(builder, name)
        values[:] = [wp.vec3(value) * scale for value in values]
    for name in ("shape_transform", "body_q", "joint_X_p", "joint_X_c"):
        values = getattr(builder, name)
        values[:] = [
            wp.transform(wp.transform_get_translation(value) * scale, wp.transform_get_rotation(value))
            for value in values
        ]
    for name, power in (
        # Geometric bounding radius, not a contact-distance threshold.
        ("shape_collision_radius", 1),
        ("edge_rest_length", 1),
        ("particle_mass", 3),
        ("body_mass", 3),
        ("body_inv_mass", -3),
        ("tri_areas", 2),
    ):
        values = getattr(builder, name)
        values[:] = [value * scale**power for value in values]
    for name, power in (("tet_poses", -1), ("tri_poses", -1), ("body_inertia", 5), ("body_inv_inertia", -5)):
        values = getattr(builder, name)
        values[:] = [(np.asarray(value) * scale**power).tolist() for value in values]


def _load_obj_mesh(path, max_span):
    """Load and center a triangulated OBJ surface, scaling its longest XY span."""
    vertices = []
    faces = []
    with Path(path).open(encoding="utf-8") as obj_file:
        for line in obj_file:
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if fields[0] == "v":
                vertices.append(tuple(float(value) for value in fields[1:4]))
            elif fields[0] == "f":
                polygon = [int(value.split("/", 1)[0]) for value in fields[1:]]
                polygon = [index - 1 if index > 0 else len(vertices) + index for index in polygon]
                if len(polygon) == 4:
                    # OBJ quads may have three almost collinear corners.
                    # A fixed fan diagonal creates near-zero-area triangles;
                    # choose the diagonal with the better worst triangle.
                    points = np.asarray([vertices[index] for index in polygon], dtype=np.float64)
                    normal = np.sum(np.cross(points, np.roll(points, -1, axis=0)), axis=0)
                    normal /= max(float(np.linalg.norm(normal)), 1.0e-30)
                    splits = (((0, 1, 2), (0, 2, 3)), ((0, 1, 3), (1, 2, 3)))
                    scores = []
                    for split in splits:
                        quality = []
                        for i, j, k in split:
                            a, b, c = points[[i, j, k]]
                            area = np.dot(np.cross(b - a, c - a), normal)
                            lengths = np.sum((b - a) ** 2 + (c - b) ** 2 + (a - c) ** 2)
                            quality.append(area / max(float(lengths), 1.0e-30))
                        scores.append(min(quality))
                    faces.extend(tuple(polygon[index] for index in face) for face in splits[int(np.argmax(scores))])
                    continue
                for i in range(1, len(polygon) - 1):
                    faces.append((polygon[0], polygon[i], polygon[i + 1]))
    vertices = np.asarray(vertices, dtype=np.float32)
    if vertices.size == 0 or not faces:
        raise ValueError(f"OBJ file contains no triangle mesh: {path}")
    bounds = np.stack((vertices.min(axis=0), vertices.max(axis=0)))
    center = bounds.mean(axis=0)
    centered = vertices - center
    scale = max_span / np.max(bounds[1, :2] - bounds[0, :2])
    # OBJ text uses +X as right and +Y as up. In this scene +X is screen-up
    # and -Y is screen-right, so rotate the artwork into the pad lettering frame.
    vertices = np.column_stack((centered[:, 1], -centered[:, 0], centered[:, 2])) * scale
    return newton.Mesh(vertices, np.asarray(faces, dtype=np.int32).ravel(), compute_inertia=False)


def _round_soft_grid_footprint(builder, particle_start, tet_start, center, half_extents):
    """Round the outer grid and rebuild rest poses and volume-weighted masses."""
    points = np.asarray(builder.particle_q[particle_start:], dtype=np.float32)

    x = (points[:, 0] - center[0]) / half_extents[0]
    y = (points[:, 1] - center[1]) / half_extents[1]
    abs_x = np.abs(x)
    abs_y = np.abs(y)
    square_radius = np.maximum(abs_x, abs_y)
    rounded_radius = np.ones_like(square_radius)
    nonzero = square_radius > 0.0
    rounded_radius[nonzero] = square_radius[nonzero] / (
        abs_x[nonzero] ** PAD_CORNER_POWER + abs_y[nonzero] ** PAD_CORNER_POWER
    ) ** (1.0 / PAD_CORNER_POWER)
    # Spread the corner displacement through the outer half of the grid.
    blend_t = np.clip((square_radius - 0.5) * 2.0, 0.0, 1.0)
    blend = blend_t * blend_t * (3.0 - 2.0 * blend_t)
    scale = 1.0 + blend * (rounded_radius - 1.0)
    points[:, 0] = center[0] + x * scale * half_extents[0]
    points[:, 1] = center[1] + y * scale * half_extents[1]
    builder.particle_q[particle_start:] = [wp.vec3(point) for point in points]

    tet_indices = np.asarray(builder.tet_indices[tet_start:], dtype=np.int32) - particle_start
    masses = np.zeros(len(points), dtype=np.float64)
    minimum_volume = float("inf")
    chunk_size = 65_536
    for chunk_start in range(0, len(tet_indices), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(tet_indices))
        indices = tet_indices[chunk_start:chunk_end]
        p, q, r, s = (points[indices[:, i]] for i in range(4))
        rest_matrices = np.stack((q - p, r - p, s - p), axis=-1)
        volumes = np.linalg.det(rest_matrices) / 6.0
        if np.any(volumes <= 0.0):
            raise ValueError("Rounded soft grid generated an inverted tetrahedron")
        minimum_volume = min(minimum_volume, float(volumes.min()))
        for vertex in range(4):
            np.add.at(masses, indices[:, vertex], PAD_DENSITY * volumes / 4.0)
        inverse_matrices = np.linalg.inv(rest_matrices)
        builder.tet_poses[tet_start + chunk_start : tet_start + chunk_end] = inverse_matrices.tolist()
    builder.particle_mass[particle_start:] = masses.tolist()
    return minimum_volume


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


class Example:
    hammer_period = PRESS_PERIOD
    iterations = 10
    linear_iterations = 10
    hammer_speed = 1.0
    after_hammers = 0.0

    def __init__(self, viewer, args):
        self.logo_path = getattr(args, "logo_path", "D:/Artworks/Mods/LOGO.obj")
        self.viewer = viewer
        self.sim_time = 0.0
        self.frame_dt = 1.0 / FPS

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.margin = SHAPE_CONTACT_MARGIN
        builder.default_shape_cfg.gap = SHAPE_CONTACT_GAP
        SolverStyle3D.register_custom_attributes(builder)
        builder.add_ground_plane()

        self._build_workpieces(builder)

        pivot = wp.transform(wp.vec3(0.52, 0.0, LOGO_PIVOT_HEIGHT), wp.quat_identity())
        steel = wp.vec3(0.12, 0.14, 0.19)
        builder.add_shape_box(
            -1,
            xform=wp.transform((0.52, 0.0, 0.14), wp.quat_identity()),
            hx=0.055,
            hy=0.065,
            hz=0.14,
            color=steel,
        )
        press_body = builder.add_link(is_kinematic=True, label="swing_arm")
        head_x = -0.52
        across = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), math.pi / 2)
        builder.add_shape_cylinder(
            press_body,
            xform=wp.transform((head_x / 2, 0, 0.055), across),
            radius=0.022,
            half_height=-head_x / 2,
            color=steel,
        )
        head = wp.transform((head_x, 0, 0), wp.quat_identity())
        logo_mesh = _load_obj_mesh(self.logo_path, LOGO_DIE_MAX_SPAN)
        logo_mesh.color = wp.vec3(0.82, 0.86, 0.90)
        logo_mesh.metallic = 0.9
        logo_mesh.roughness = 0.24
        builder.add_shape_mesh(press_body, xform=head, mesh=logo_mesh)
        hinge = builder.add_joint_revolute(-1, press_body, parent_xform=pivot, axis=newton.Axis.Y)
        builder.add_articulation([hinge], label="press")
        builder.joint_q[0] = OPEN_ANGLE
        _scale_press_scene(builder, SCENE_SCALE)
        self.model = builder.finalize()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        pad_color = wp.vec3(0.015, 0.015, 0.02)
        self.model.tri_color = wp.full(self.model.tri_count, pad_color, device=self.model.device)
        self.model.soft_contact_ke = 2.0e2  # important for this example.
        self.model.soft_contact_kd = 2.0
        self.model.soft_contact_kf = 10.0
        self.model.soft_contact_mu = 0.5

        self.solver = SolverStyle3D(
            self.model,
            dynamics=SolverStyle3D.Dynamics.ALL,
            config=SolverStyle3D.Config(iterations=self.iterations, linear_iterations=self.linear_iterations),
        )
        self.solver.collision.radius = SELF_CONTACT_THICKNESS / 2.0
        # Disable surface self-repulsion on this solid pad near the rounded corners.
        self.solver.collision.stiff_vf = 0.0
        self.solver.collision.stiff_ee = 0.0
        self.solver.collision.stiff_ef = 0.0
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model, soft_contact_gap=SOFT_CONTACT_GAP)
        self.contacts = self.collision_pipeline.contacts()
        self.clock = wp.zeros(1, dtype=float, device=self.model.device)
        self.graph = None

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(-1.05, -1.45, 1.40) * SCENE_SCALE, -30.0, 53.0)
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

        if LOGO_FRAME_ENABLED:
            # Contact offsets stay in world units while geometry is scaled
            # later. Leave the pad outside the wall's effective surface at rest.
            clearance = LOGO_FRAME_CLEARANCE + (PAD_PARTICLE_RADIUS + SHAPE_CONTACT_MARGIN) / SCENE_SCALE
            inner_x = PAD_LENGTH / 2 + clearance
            inner_y = PAD_WIDTH / 2 + clearance
            half_wall = LOGO_FRAME_THICKNESS / 2
            wall_z = 0.12 + PAD_HEIGHT / 2
            for side in (-1, 1):
                builder.add_shape_box(
                    -1,
                    xform=wp.transform((side * (inner_x + half_wall), 0.0, wall_z), wp.quat_identity()),
                    hx=half_wall,
                    hy=inner_y + LOGO_FRAME_THICKNESS,
                    hz=PAD_HEIGHT / 2,
                    color=wp.vec3(0.18, 0.20, 0.24),
                    label=f"pad_frame_x_{side}",
                )
                builder.add_shape_box(
                    -1,
                    xform=wp.transform((0.0, side * (inner_y + half_wall), wall_z), wp.quat_identity()),
                    hx=inner_x,
                    hy=half_wall,
                    hz=PAD_HEIGHT / 2,
                    color=wp.vec3(0.18, 0.20, 0.24),
                    label=f"pad_frame_y_{side}",
                )

        mu = PAD_YOUNGS_MODULUS / (2.0 * (1.0 + PAD_POISSON_RATIO))
        lam = PAD_YOUNGS_MODULUS * PAD_POISSON_RATIO / ((1.0 + PAD_POISSON_RATIO) * (1.0 - 2.0 * PAD_POISSON_RATIO))

        particle_start = len(builder.particle_q)
        tet_start = builder.tet_count
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
            density=PAD_DENSITY,
            k_mu=mu,
            k_lambda=lam,
            k_damp=0.0,
            particle_radius=PAD_PARTICLE_RADIUS,
            label="plastic_pad",
            custom_attributes={"style3d:volume_yield_strain": PAD_YIELD_STRAIN},
        )
        _round_soft_grid_footprint(builder, particle_start, tet_start, (0.0, 0.0), (PAD_LENGTH / 2, PAD_WIDTH / 2))
        # Bond the underside to the platen so the swinging die cannot drag the sample away.
        for i, point in enumerate(builder.particle_q):
            if np.isclose(point[2], 0.12):
                builder.particle_flags[i] &= ~newton.ParticleFlags.ACTIVE

    def simulate(self):
        for _ in range(SUBSTEPS):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            wp.launch(
                _drive_press_kernel,
                dim=1,
                inputs=[
                    SUBSTEP_DT,
                    self.hammer_period,
                    self.hammer_speed,
                    self.after_hammers,
                    self.clock,
                ],
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
        rest = self.model.particle_q.numpy() / SCENE_SCALE
        xy = np.abs(rest[:, :2])
        top = np.isclose(rest[:, 2], rest[:, 2].max())
        footprint = top & (np.abs(rest[:, 0] - 0.05) < 0.09) & (xy[:, 1] < 0.12)
        corners = top & (xy.min(axis=1) > 0.18)
        heights = self.state_0.particle_q.numpy()[:, 2] / SCENE_SCALE
        assert np.median(heights[corners]) - np.median(heights[footprint]) > 0.01, "The imprint disappeared."

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--logo-path", default="D:/Artworks/Mods/LOGO.obj", help="OBJ mesh for the press die.")
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
