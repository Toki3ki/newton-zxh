# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from ...geometry import ParticleFlags
from ...sim import BodyFlags
from ...sim.contacts import contact_surface_separation
from .affine_body import affine_mass_multiply, inertia_to_second_moment
from .contact import contact_response


@wp.struct
class CableContactTangent:
    """Cache the six possible Cable rows of one rigid contact."""

    active: int
    row0: int
    row1: int
    row2: int
    row3: int
    row4: int
    row5: int
    J0: wp.mat33
    J1: wp.mat33
    J2: wp.mat33
    J3: wp.mat33
    J4: wp.mat33
    J5: wp.mat33
    hessian: wp.mat33


@wp.kernel
def compute_solver_flags_kernel(
    particle_count: int,
    affine_bodies: wp.array[wp.int32],
    particle_masses: wp.array[float],
    particle_flags: wp.array[wp.int32],
    body_masses: wp.array[float],
    body_flags: wp.array[wp.int32],
    # outputs
    solver_flags: wp.array[wp.int32],
):
    """Build ParticleFlags for each particle and all four affine blocks per body."""
    tid = wp.tid()
    if tid < particle_count:
        flags = particle_flags[tid]
        if particle_masses[tid] <= 0.0:
            flags = flags & ~ParticleFlags.ACTIVE
        solver_flags[tid] = flags
    else:
        flags = wp.int32(0)
        affine_row = tid - particle_count
        body = affine_bodies[affine_row]
        source_flags = body_flags[body]
        if (
            body_masses[body] > 0.0
            and (source_flags & BodyFlags.DYNAMIC) != 0
            and (source_flags & BodyFlags.KINEMATIC) == 0
        ):
            flags = ParticleFlags.ACTIVE
        # Translate explicitly because the body and particle enums use different bit positions.
        if (source_flags & BodyFlags.PROXY) != 0:
            flags = flags | ParticleFlags.PROXY

        start = particle_count + 4 * affine_row
        solver_flags[start + 0] = flags
        solver_flags[start + 1] = flags
        solver_flags[start + 2] = flags
        solver_flags[start + 3] = flags


@wp.func
def triangle_deformation_gradient(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, inv_dm: wp.mat22):
    x01, x02 = x1 - x0, x2 - x0
    Fu = x01 * inv_dm[0, 0] + x02 * inv_dm[1, 0]
    Fv = x01 * inv_dm[0, 1] + x02 * inv_dm[1, 1]
    return Fu, Fv


@wp.kernel
def eval_stretch_kernel(
    pos: wp.array[wp.vec3],
    face_areas: wp.array[float],
    inv_dms: wp.array[wp.mat22],
    faces: wp.array2d[wp.int32],
    aniso_ke: wp.array[wp.vec3],
    # outputs
    forces: wp.array[wp.vec3],
):
    """
    Ref. Large Steps in Cloth Simulation, Baraff & Witkin in 1998.
    """
    fid = wp.tid()

    inv_dm = inv_dms[fid]
    face_area = face_areas[fid]
    face = wp.vec3i(faces[fid, 0], faces[fid, 1], faces[fid, 2])

    Fu, Fv = triangle_deformation_gradient(pos[face[0]], pos[face[1]], pos[face[2]], inv_dm)

    len_Fu = wp.length(Fu)
    len_Fv = wp.length(Fv)

    Fu = wp.normalize(Fu) if (len_Fu > 1e-6) else wp.vec3(0.0)
    Fv = wp.normalize(Fv) if (len_Fv > 1e-6) else wp.vec3(0.0)

    dFu_dx = wp.vec3(-inv_dm[0, 0] - inv_dm[1, 0], inv_dm[0, 0], inv_dm[1, 0])
    dFv_dx = wp.vec3(-inv_dm[0, 1] - inv_dm[1, 1], inv_dm[0, 1], inv_dm[1, 1])

    ku = aniso_ke[fid][0]
    kv = aniso_ke[fid][1]
    ks = aniso_ke[fid][2]

    for i in range(3):
        force = -face_area * (
            ku * (len_Fu - 1.0) * dFu_dx[i] * Fu
            + kv * (len_Fv - 1.0) * dFv_dx[i] * Fv
            + ks * wp.dot(Fu, Fv) * (Fu * dFv_dx[i] + Fv * dFu_dx[i])
        )
        wp.atomic_add(forces, face[i], force)


@wp.kernel
def eval_bend_kernel(
    pos: wp.array[wp.vec3],
    edge_rest_area: wp.array[float],
    edge_bending_cot: wp.array[wp.vec4],
    edges: wp.array2d[wp.int32],
    edge_bending_properties: wp.array2d[float],
    # outputs
    forces: wp.array[wp.vec3],
):
    """
    Crouzeix-Raviart isometric bending model from

    "A Quadratic Bending Model for Inextensible Surfaces" (Bergou et al. 2006).

    For one interior edge with local stencil x = (x0, x1, x2, x3)^T,
    the paper defines

        E_b = 1/2 * k * x^T Q x,
        Q = 3 / (A0 + A1) * w^T w,

    where A0 and A1 are the incident triangle rest areas and w is built
    from rest-pose cotangents. The conservative force is

        F_i = -dE_b/dx_i = -k * sum_j Q_ij x_j.
    """
    eid = wp.tid()
    if edges[eid][0] < 0 or edges[eid][1] < 0:
        return
    edge = edges[eid]
    edge_stiff = edge_bending_properties[eid][0] * (3.0 / edge_rest_area[eid])
    bend_weight = wp.vec4(0.0)
    bend_weight[2] = edge_bending_cot[eid][2] + edge_bending_cot[eid][3]
    bend_weight[3] = edge_bending_cot[eid][0] + edge_bending_cot[eid][1]
    bend_weight[0] = -edge_bending_cot[eid][0] - edge_bending_cot[eid][2]
    bend_weight[1] = -edge_bending_cot[eid][1] - edge_bending_cot[eid][3]
    for i in range(4):
        force = wp.vec3(0.0)
        for j in range(4):
            force = force - edge_stiff * bend_weight[i] * bend_weight[j] * pos[edge[j]]
        wp.atomic_add(forces, edge[i], force)


@wp.kernel
def eval_drag_force_kernel(
    spring_stiff: float,
    face_index: wp.array[int],
    drag_pos: wp.array[wp.vec3],
    drag_bary_coord: wp.array[wp.vec3],
    faces: wp.array2d[wp.int32],
    vert_pos: wp.array[wp.vec3],
    # outputs
    forces: wp.array[wp.vec3],
):
    fid = face_index[0]
    if fid != -1:
        coord = drag_bary_coord[0]
        face = wp.vec3i(faces[fid, 0], faces[fid, 1], faces[fid, 2])
        x0 = vert_pos[face[0]]
        x1 = vert_pos[face[1]]
        x2 = vert_pos[face[2]]
        p = x0 * coord[0] + x1 * coord[1] + x2 * coord[2]
        dir = drag_pos[0] - p

        # add force
        force = spring_stiff * dir
        wp.atomic_add(forces, face[0], force * coord[0])
        wp.atomic_add(forces, face[1], force * coord[1])
        wp.atomic_add(forces, face[2], force * coord[2])

        # add hessian
        # dir = wp.normalize(dir)
        # hessian = wp.outer(dir, dir) * spring_stiff
        # hessian_diags[face[0]] += hessian * coord[0]
        # hessian_diags[face[1]] += hessian * coord[1]
        # hessian_diags[face[2]] += hessian * coord[2]


@wp.kernel
def accumulate_dragging_pd_diag_kernel(
    spring_stiff: float,
    face_index: wp.array[int],
    drag_bary_coord: wp.array[wp.vec3],
    faces: wp.array2d[wp.int32],
    particle_flags: wp.array[wp.int32],
    # outputs
    pd_diag: wp.array[float],
):
    fid = face_index[0]
    if fid != -1:
        coord = drag_bary_coord[0]
        face = wp.vec3i(faces[fid, 0], faces[fid, 1], faces[fid, 2])

        if particle_flags[face[0]] & ParticleFlags.ACTIVE:
            pd_diag[face[0]] += spring_stiff * coord[0]

        if particle_flags[face[1]] & ParticleFlags.ACTIVE:
            pd_diag[face[1]] += spring_stiff * coord[1]

        if particle_flags[face[2]] & ParticleFlags.ACTIVE:
            pd_diag[face[2]] += spring_stiff * coord[2]


@wp.kernel
def init_step_kernel(
    particle_count: int,
    affine_bodies: wp.array[wp.int32],
    dt: float,
    gravity: wp.array[wp.vec3],
    particle_world: wp.array[wp.int32],
    f_ext: wp.array[wp.vec3],
    v_curr: wp.array[wp.vec3],
    x_curr: wp.array[wp.vec3],
    x_prev: wp.array[wp.vec3],
    pd_diag: wp.array[float],
    particle_masses: wp.array[float],
    particle_flags: wp.array[wp.int32],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    body_flags: wp.array[wp.int32],
    # outputs
    x_inertia: wp.array[wp.vec3],
    A_diag_base: wp.array[float],
    dx: wp.array[wp.vec3],
    affine_mass_off_diag: wp.array[wp.vec3],
):
    """Initialize particle and affine-body inertia data for one time step."""
    tid = wp.tid()
    if tid < particle_count:
        x_last = x_curr[tid]
        x_prev[tid] = x_last

        if not particle_flags[tid] & ParticleFlags.ACTIVE:
            x_inertia[tid] = x_prev[tid]
            A_diag_base[tid] = 0.0
            dx[tid] = wp.vec3(0.0)
        else:
            v_prev = v_curr[tid]
            mass = particle_masses[tid]
            A_diag_base[tid] = pd_diag[tid] + mass / (dt * dt)
            world_idx = particle_world[tid]
            world_g = gravity[world_idx]
            x_inertia[tid] = x_last + v_prev * dt + (world_g + f_ext[tid] / mass) * (dt * dt)
            dx[tid] = v_prev * dt
    else:
        affine_row = tid - particle_count
        body = affine_bodies[affine_row]
        start = particle_count + 4 * affine_row
        is_kinematic = (body_flags[body] & BodyFlags.KINEMATIC) != 0
        active = body_mass[body] > 0.0 and (is_kinematic or (body_flags[body] & BodyFlags.DYNAMIC) != 0)
        mass = float(0.0)
        moment = wp.mat33(0.0)
        if active:
            mass = body_mass[body] / (dt * dt)
            moment = inertia_to_second_moment(body_inertia[body]) / (dt * dt)

        # Cache the symmetric mass coupling once per step for all PCG products.
        affine_mass_off_diag[affine_row] = wp.vec3(moment[0, 1], moment[0, 2], moment[1, 2])

        for i in range(4):
            row = start + i
            x_prev[row] = x_curr[row]
            dx[row] = wp.vec3(0.0)
            if not active:
                x_inertia[row] = x_curr[row]
            diagonal = mass
            if i > 0:
                diagonal = moment[i - 1, i - 1]
            A_diag_base[row] = diagonal


@wp.kernel
def init_rhs_kernel(
    dt: float,
    x_curr: wp.array[wp.vec3],
    x_inertia: wp.array[wp.vec3],
    particle_count: int,
    affine_bodies: wp.array[int],
    particle_masses: wp.array[float],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    body_flags: wp.array[int],
    # outputs
    rhs: wp.array[wp.vec3],
):
    """Initialize M (x_inertia - x_curr) / dt^2 with one thread per particle or body."""
    tid = wp.tid()

    if tid < particle_count:
        rhs[tid] = (x_inertia[tid] - x_curr[tid]) * particle_masses[tid] / (dt * dt)
        return

    affine_row = tid - particle_count
    body = affine_bodies[affine_row]
    start = particle_count + 4 * affine_row
    if body_mass[body] <= 0.0 or (
        (body_flags[body] & BodyFlags.DYNAMIC) == 0 and (body_flags[body] & BodyFlags.KINEMATIC) == 0
    ):
        rhs[start + 0] = wp.vec3(0.0)
        rhs[start + 1] = wp.vec3(0.0)
        rhs[start + 2] = wp.vec3(0.0)
        rhs[start + 3] = wp.vec3(0.0)
    else:
        inv_dt2 = 1.0 / (dt * dt)
        rhs_com, rhs_a0, rhs_a1, rhs_a2 = affine_mass_multiply(
            body_mass[body],
            body_inertia[body],
            x_inertia[start] - x_curr[start],
            x_inertia[start + 1] - x_curr[start + 1],
            x_inertia[start + 2] - x_curr[start + 2],
            x_inertia[start + 3] - x_curr[start + 3],
        )
        rhs[start + 0] = rhs_com * inv_dt2
        rhs[start + 1] = rhs_a0 * inv_dt2
        rhs[start + 2] = rhs_a1 * inv_dt2
        rhs[start + 3] = rhs_a2 * inv_dt2


@wp.kernel
def prepare_jacobi_preconditioner_kernel(
    particle_count: int,
    A_diag_base: wp.array[float],
    contact_hessian_diags: wp.array[wp.mat33],
    particle_flags: wp.array[wp.int32],
    # outputs
    A_diag: wp.array[wp.mat33],
    inv_A_diag: wp.array[wp.mat33],
):
    """Add the base diagonal and invert the full diagonal for active DOFs.

    A_diag must contain freshly evaluated constraint diagonals for all DOFs.
    Cloth self-contact diagonals remain in the matrix-free product.
    """
    tid = wp.tid()
    diag = wp.identity(3, float) * A_diag_base[tid] + A_diag[tid]
    A_diag[tid] = diag

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        inv_A_diag[tid] = wp.mat33(0.0)
        return
    # Cloth contacts stay in the matrix-free product, but also enter preconditioning.
    if tid < contact_hessian_diags.shape[0]:
        diag += contact_hessian_diags[tid]
    # Constraints can make an affine block invertible even when its mass diagonal is zero.
    # Scale the full block to avoid overflow in the float32 inverse.
    scale = wp.max(wp.max(wp.abs(diag[0, 0]), wp.abs(diag[1, 1])), wp.abs(diag[2, 2]))
    if scale > 0.0:
        inv_scale = 1.0 / scale
        inv_A_diag[tid] = wp.inverse(diag * inv_scale) * inv_scale
    else:
        inv_A_diag[tid] = wp.mat33(0.0)


@wp.kernel
def PD_jacobi_step_kernel(
    rhs: wp.array[wp.vec3],
    x_in: wp.array[wp.vec3],
    inv_A_diag: wp.array[wp.mat33],
    # outputs
    x_out: wp.array[wp.vec3],
):
    tid = wp.tid()
    x_out[tid] = x_in[tid] + inv_A_diag[tid] * rhs[tid]


@wp.kernel
def nonlinear_step_kernel(
    x_in: wp.array[wp.vec3],
    # outputs
    x_out: wp.array[wp.vec3],
    dx: wp.array[wp.vec3],
):
    tid = wp.tid()
    x_out[tid] = x_in[tid] + dx[tid]
    dx[tid] = wp.vec3(0.0)


@wp.kernel
def update_velocity_kernel(
    dt: float,
    particle_count: int,
    velocity_end: int,
    prev_pos: wp.array[wp.vec3],
    pos: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
):
    """Differentiate solved coordinates, optionally damping cloth particles.

    Rows at or beyond ``velocity_end`` are rotation tangents, which carry no
    velocity; their angular velocity is reconstructed by the owning module.
    """
    tid = wp.tid()
    if tid >= velocity_end:
        vel[tid] = wp.vec3(0.0)
    elif tid < particle_count:
        vel[tid] = 0.998 * (pos[tid] - prev_pos[tid]) / dt
    else:
        vel[tid] = (pos[tid] - prev_pos[tid]) / dt


########################################################################################################################
###############################################    VIPER cable kernels    ##############################################
########################################################################################################################

# The cable module binds a contiguous slice of the shared solver buffers that
# holds its center rows first, followed by one rotation-tangent row per segment.
# Every kernel below therefore addresses node ``i`` directly and segment ``e`` at
# ``node_count + e``, without inferring one range from the other.


@wp.func
def cable_rotation_exp(delta: wp.vec3) -> wp.mat33:
    """Map a rotation vector onto SO(3) with the matrix exponential."""
    angle = wp.length(delta)
    if angle < 1.0e-8:
        # First-order expansion, reusing the skew matrix of the rotation vector.
        return wp.mat33(
            1.0,
            -delta[2],
            delta[1],
            delta[2],
            1.0,
            -delta[0],
            -delta[1],
            delta[0],
            1.0,
        )
    return wp.quat_to_matrix(wp.quat_from_axis_angle(delta / angle, angle))


@wp.func
def cable_rotation_log(rotation: wp.mat33) -> wp.vec3:
    """Map a rotation matrix to its principal rotation vector."""
    quat = wp.normalize(wp.quat_from_matrix(rotation))
    if quat[3] < 0.0:
        quat = -quat
    vector = wp.vec3(quat[0], quat[1], quat[2])
    sine = wp.length(vector)
    if sine < 1.0e-8:
        # Half-angle approximation: vector ~ axis * angle / 2.
        return vector * 2.0
    return vector * (2.0 * wp.atan2(sine, quat[3]) / sine)


@wp.kernel
def cable_collect_state_kernel(
    segment_nodes: wp.array2d[int],
    segment_bodies: wp.array[int],
    segment_com: wp.array[wp.vec3],
    segment_half_length: wp.array[float],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    # outputs
    node_position_sum: wp.array[wp.vec3],
    node_velocity_sum: wp.array[wp.vec3],
    node_multiple: wp.array[int],
    frame: wp.array[wp.mat33],
    frame_velocity: wp.array[wp.vec3],
):
    """Convert Newton segment poses into Style3D VIPER node/frame state.

    ``ModelBuilder.add_rod`` represents every rod edge as one capsule body. In
    contrast, the VIPER solve uses one position at every discrete rod node
    (a segment endpoint, shared by adjacent segments) and one material frame per
    segment. These node positions are not capsule-center positions.
    ``Cable._allocate_topology`` has already
    converted the ROD-joint connectivity into ``segment_bodies`` and
    ``segment_nodes`` before this kernel runs. For a two-segment open chain the
    mapping is

    .. code-block:: text

        VIPER nodes:       x0 o----------o x1 ----------o x2
        segment bodies:          body 0          body 1
        material frames:           R0              R1

    For segment ``e`` with body pose ``(p_e, q_e)``, body-local center-of-mass
    offset ``com_e``, and capsule half-length ``h_e``, this kernel samples

    .. code-block:: text

        R_e       = matrix(q_e)
        x_e^-     = transform(pose_e, com_e + (0, 0, -h_e))
        x_e^+     = transform(pose_e, com_e + (0, 0,  h_e))

    Thus, for the common COM-origin convention, ``p_e`` is the segment center
    and the two samples reduce to ``p_e +/- R_e (0, 0, h_e)``. The topology
    table ``segment_nodes[e]`` sends these samples to the two global VIPER node
    rows. At an interior vertex, the adjacent segments both sample the same
    logical node; :func:`cable_average_state_kernel` averages those samples into
    the single node position used by the solve.

    This kernel only evaluates the current values for the previously allocated
    mapping; it neither creates topology nor evaluates elastic constraints. Newton
    keeps the capsule poses for collision, rendering, and state interchange,
    while the Style3D cable advances the reconstructed node positions and
    material frames.
    """
    e = wp.tid()
    body = segment_bodies[e]
    pose = body_q[body]
    half_length = segment_half_length[e]
    com = segment_com[e]
    # Reconstruct the two centerline vertices represented by this capsule body:
    #
    #   start o----------[ body pose / segment center ]----------o end
    #                       material frame R_e
    #
    # ``com`` keeps this expression valid for both COM-origin and legacy
    # start-origin rod bodies.
    start = wp.transform_point(pose, com + wp.vec3(0.0, 0.0, -half_length))
    end = wp.transform_point(pose, com + wp.vec3(0.0, 0.0, half_length))
    node_start = segment_nodes[e, 0]
    node_end = segment_nodes[e, 1]
    wp.atomic_add(node_position_sum, node_start, start)
    wp.atomic_add(node_position_sum, node_end, end)
    wp.atomic_add(node_multiple, node_start, 1)
    wp.atomic_add(node_multiple, node_end, 1)

    velocity = wp.spatial_top(body_qd[body])
    wp.atomic_add(node_velocity_sum, node_start, velocity)
    wp.atomic_add(node_velocity_sum, node_end, velocity)

    # The body quaternion is only the Newton state representation. VIPER keeps
    # the equivalent SO(3) material frame and solves a vec3 tangent increment.
    frame[e] = wp.quat_to_matrix(wp.transform_get_rotation(pose))
    frame_velocity[e] = wp.spatial_bottom(body_qd[body])


@wp.kernel
def cable_average_state_kernel(
    node_multiple: wp.array[int],
    node_position_sum: wp.array[wp.vec3],
    node_velocity_sum: wp.array[wp.vec3],
    # outputs
    node_position: wp.array[wp.vec3],
    node_velocity: wp.array[wp.vec3],
):
    """Average ``end_previous`` and ``start_next`` to remove small gaps at a shared VIPER rod node."""
    tid = wp.tid()
    multiple = node_multiple[tid]
    if multiple == 0:
        node_position[tid] = wp.vec3(0.0)
        node_velocity[tid] = wp.vec3(0.0)
        return
    scale = 1.0 / float(multiple)
    # Interior example: x_i = (end_previous + start_next) / 2.
    node_position[tid] = node_position_sum[tid] * scale
    node_velocity[tid] = node_velocity_sum[tid] * scale


@wp.kernel
def cable_node_begin_kernel(
    dt: float,
    node_mass: wp.array[float],
    node_world: wp.array[int],
    gravity: wp.array[wp.vec3],
    pd_diag: wp.array[float],
    node_position: wp.array[wp.vec3],
    node_velocity: wp.array[wp.vec3],
    # outputs
    node_previous: wp.array[wp.vec3],
    node_inertia: wp.array[wp.vec3],
    dx: wp.array[wp.vec3],
    A_diag_base: wp.array[float],
    solver_flags: wp.array[int],
):
    """Predict VIPER center positions for the whole cable solve."""
    tid = wp.tid()
    node_previous[tid] = node_position[tid]
    dx[tid] = wp.vec3(0.0)
    mass = node_mass[tid]
    if mass <= 0.0:
        node_velocity[tid] = wp.vec3(0.0)
        node_inertia[tid] = node_position[tid]
        A_diag_base[tid] = 0.0
        solver_flags[tid] = 0
        return
    solver_flags[tid] = ParticleFlags.ACTIVE
    A_diag_base[tid] = pd_diag[tid] + mass / (dt * dt)
    node_inertia[tid] = node_position[tid] + node_velocity[tid] * dt + gravity[node_world[tid]] * (dt * dt)


@wp.func
def cable_segment_frame_is_prescribed(mass: float, flags: int) -> bool:
    """Return whether an external driver owns a segment's material frame.

    A zero-mass or kinematic segment is prescribed, mirroring how the affine
    solve treats externally driven bodies.
    """
    return mass <= 0.0 or (flags & BodyFlags.KINEMATIC) != 0


@wp.kernel
def cable_segment_begin_kernel(
    dt: float,
    frame_inertia: wp.array[float],
    frame_velocity: wp.array[wp.vec3],
    pd_diag: wp.array[float],
    segment_bodies: wp.array[int],
    body_q: wp.array[wp.transform],
    body_mass: wp.array[float],
    body_flags: wp.array[int],
    frame: wp.array[wp.mat33],
    # outputs
    frame_begin: wp.array[wp.mat33],
    tangent: wp.array[wp.vec3],
    tangent_previous: wp.array[wp.vec3],
    tangent_velocity: wp.array[wp.vec3],
    tangent_inertia: wp.array[wp.vec3],
    dx: wp.array[wp.vec3],
    A_diag_base: wp.array[float],
    solver_flags: wp.array[int],
):
    """Anchor the material frames and predict their tangent-space targets.

    Rotation rows hold the world-frame tangent increment accumulated since
    ``frame_begin``, so the inertial target of a constant angular velocity is
    simply ``omega * dt``. A prescribed segment takes its frame from the input
    state and keeps no rotation row, so the driver owns both its orientation and
    its angular velocity.
    """
    e = wp.tid()
    body = segment_bodies[e]
    if cable_segment_frame_is_prescribed(body_mass[body], body_flags[body]):
        frame[e] = wp.quat_to_matrix(wp.transform_get_rotation(body_q[body]))
        frame_begin[e] = frame[e]
        tangent[e] = wp.vec3(0.0)
        tangent_previous[e] = wp.vec3(0.0)
        tangent_velocity[e] = wp.vec3(0.0)
        tangent_inertia[e] = wp.vec3(0.0)
        dx[e] = wp.vec3(0.0)
        A_diag_base[e] = 0.0
        solver_flags[e] = 0
        return
    frame_begin[e] = frame[e]
    tangent[e] = wp.vec3(0.0)
    tangent_previous[e] = wp.vec3(0.0)
    tangent_velocity[e] = wp.vec3(0.0)
    dx[e] = wp.vec3(0.0)
    inertia = frame_inertia[e]
    if inertia <= 0.0:
        tangent_inertia[e] = wp.vec3(0.0)
        A_diag_base[e] = 0.0
        solver_flags[e] = 0
        return
    solver_flags[e] = ParticleFlags.ACTIVE
    A_diag_base[e] = pd_diag[e] + inertia / (dt * dt)
    tangent_inertia[e] = frame_velocity[e] * dt


@wp.kernel
def cable_node_inertia_kernel(
    dt: float,
    node_mass: wp.array[float],
    node_position: wp.array[wp.vec3],
    node_inertia: wp.array[wp.vec3],
    # outputs
    rhs: wp.array[wp.vec3],
):
    """Initialize M (x_inertia - x) / dt^2 for VIPER centers."""
    tid = wp.tid()
    rhs[tid] = (node_inertia[tid] - node_position[tid]) * (node_mass[tid] / (dt * dt))


@wp.kernel
def cable_segment_inertia_kernel(
    dt: float,
    frame_inertia: wp.array[float],
    tangent: wp.array[wp.vec3],
    tangent_inertia: wp.array[wp.vec3],
    # outputs
    rhs: wp.array[wp.vec3],
):
    """Initialize J (theta_inertia - theta) / dt^2 for material frames."""
    e = wp.tid()
    rhs[e] = (tangent_inertia[e] - tangent[e]) * (frame_inertia[e] / (dt * dt))


@wp.kernel
def cable_eval_stretch_kernel(
    segment_nodes: wp.array2d[int],
    rest_length: wp.array[float],
    stretch_stiffness: wp.array[float],
    frame: wp.array[wp.mat33],
    node_position: wp.array[wp.vec3],
    # outputs
    rhs_node: wp.array[wp.vec3],
    rhs_rotation: wp.array[wp.vec3],
    diag_rotation: wp.array[wp.mat33],
    frame_edge: wp.array[wp.vec3],
):
    """Accumulate the stretch/alignment force and its frozen rotation tangent.

    With the material frame frozen, the energy is quadratic in the endpoint
    centers, so its position Hessian is the constant ``k [I3, -I3; -I3, I3]``
    stencil assembled into the fixed PD matrix. This kernel only adds the force
    and the frame tangent blocks that the fixed matrix cannot represent.
    """
    e = wp.tid()
    start = segment_nodes[e, 0]
    end = segment_nodes[e, 1]
    rest_edge = frame[e] * wp.vec3(0.0, 0.0, rest_length[e])
    residual = (node_position[end] - node_position[start]) - rest_edge
    stiffness = stretch_stiffness[e]
    force = stiffness * residual
    wp.atomic_add(rhs_node, start, force)
    wp.atomic_add(rhs_node, end, -force)
    wp.atomic_add(rhs_rotation, e, stiffness * wp.cross(rest_edge, residual))
    wp.atomic_add(
        diag_rotation,
        e,
        stiffness * (wp.dot(rest_edge, rest_edge) * wp.identity(3, float) - wp.outer(rest_edge, rest_edge)),
    )
    frame_edge[e] = rest_edge


@wp.kernel
def cable_stretch_matrix_free_kernel(
    segment_nodes: wp.array2d[int],
    stretch_stiffness: wp.array[float],
    frame_edge: wp.array[wp.vec3],
    direction_node: wp.array[wp.vec3],
    direction_rotation: wp.array[wp.vec3],
    # outputs
    product_node: wp.array[wp.vec3],
    product_rotation: wp.array[wp.vec3],
):
    """Add the position-rotation stretch coupling and its transpose to ``A * d``."""
    e = wp.tid()
    start = segment_nodes[e, 0]
    end = segment_nodes[e, 1]
    stiffness = stretch_stiffness[e]
    edge = frame_edge[e]
    coupling = stiffness * wp.cross(edge, direction_rotation[e])
    wp.atomic_add(product_node, start, -coupling)
    wp.atomic_add(product_node, end, coupling)
    relative = direction_node[end] - direction_node[start]
    wp.atomic_add(product_rotation, e, stiffness * wp.cross(relative, edge))


@wp.kernel
def cable_eval_darboux_kernel(
    pairs: wp.array2d[int],
    rest: wp.array[wp.vec3],
    weights: wp.array[wp.vec3],
    frame: wp.array[wp.mat33],
    rhs: wp.array[wp.vec3],
    diagonal: wp.array[wp.mat33],
    off_diagonal: wp.array[wp.mat33],
):
    """Linearize VIPER's dimensionless Darboux strain 2 Im(conj(q0) q1).

    The dual length is absorbed into weights = (EI, EI, GJ) / Ldual.
    Both frames use world-space left increments. Their Jacobians are opposite,
    preserving rigid rotation invariance and a symmetric PSD frozen operator.
    """
    i = wp.tid()
    a = pairs[i, 0]
    b = pairs[i, 1]
    q = wp.normalize(wp.quat_from_matrix(wp.transpose(frame[a]) * frame[b]))
    if q[3] < 0.0:
        q = -q
    v = wp.vec3(q[0], q[1], q[2])
    residual = 2.0 * v - rest[i]
    jac = (q[3] * wp.identity(3, float) - wp.skew(v)) * wp.transpose(frame[a])
    weighted = wp.diag(weights[i])
    force = wp.transpose(jac) * (weighted * residual)
    block = wp.transpose(jac) * weighted * jac
    wp.atomic_add(rhs, a, force)
    wp.atomic_add(rhs, b, -force)
    wp.atomic_add(diagonal, a, block)
    wp.atomic_add(diagonal, b, block)
    off_diagonal[i] = -block


@wp.kernel
def cable_darboux_matrix_free_kernel(
    pairs: wp.array2d[int],
    off_diagonal: wp.array[wp.mat33],
    direction: wp.array[wp.vec3],
    product: wp.array[wp.vec3],
):
    """Apply both off-diagonal blocks of the frozen Darboux operator."""
    i = wp.tid()
    a = pairs[i, 0]
    b = pairs[i, 1]
    block = off_diagonal[i]
    wp.atomic_add(product, a, block * direction[b])
    wp.atomic_add(product, b, wp.transpose(block) * direction[a])


@wp.kernel
def cable_apply_increment_kernel(
    frame_begin: wp.array[wp.mat33],
    tangent: wp.array[wp.vec3],
    # outputs
    frame: wp.array[wp.mat33],
):
    """Retract the accumulated tangent increment onto the material frames."""
    e = wp.tid()
    frame[e] = cable_rotation_exp(tangent[e]) * frame_begin[e]


@wp.kernel
def cable_writeback_kernel(
    dt: float,
    segment_nodes: wp.array2d[int],
    segment_bodies: wp.array[int],
    segment_com: wp.array[wp.vec3],
    body_mass: wp.array[float],
    body_flags: wp.array[int],
    node_position: wp.array[wp.vec3],
    node_previous: wp.array[wp.vec3],
    frame: wp.array[wp.mat33],
    frame_begin: wp.array[wp.mat33],
    # outputs
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    frame_velocity: wp.array[wp.vec3],
):
    """Reconstruct each proxy capsule from its two nodes and its material frame.

    Prescribed segments are left untouched: an external driver owns their pose
    and velocity, exactly as for externally driven affine bodies.
    """
    e = wp.tid()
    body = segment_bodies[e]
    if cable_segment_frame_is_prescribed(body_mass[body], body_flags[body]):
        return
    start = segment_nodes[e, 0]
    end = segment_nodes[e, 1]
    rotation = frame[e]
    center = 0.5 * (node_position[start] + node_position[end])
    previous_center = 0.5 * (node_previous[start] + node_previous[end])
    rotation_quat = wp.normalize(wp.quat_from_matrix(rotation))
    body_q[body] = wp.transform(center - wp.quat_rotate(rotation_quat, segment_com[e]), rotation_quat)
    angular = cable_rotation_log(rotation * wp.transpose(frame_begin[e])) / dt
    frame_velocity[e] = angular
    body_qd[body] = wp.spatial_vector((center - previous_center) / dt, angular)


@wp.kernel
def cable_eval_contact_kernel(
    dt: float,
    node_count: int,
    segment_nodes: wp.array2d[int],
    body_to_segment: wp.array[int],
    segment_com: wp.array[wp.vec3],
    node_position: wp.array[wp.vec3],
    node_previous: wp.array[wp.vec3],
    frame: wp.array[wp.mat33],
    frame_begin: wp.array[wp.mat33],
    body_q: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_ke: wp.array[float],
    shape_kd: wp.array[float],
    contact_count: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    margin0: wp.array[float],
    margin1: wp.array[float],
    contact_ke: wp.array[float],
    contact_kd: wp.array[float],
    # outputs
    rhs: wp.array[wp.vec3],
    diag: wp.array[wp.mat33],
    tangents: wp.array[CableContactTangent],
):
    """Map capsule contacts to two rod-node rows and one frame row per Cable side."""
    tid = wp.tid()
    tangent = CableContactTangent()
    tangent.row0 = -1
    tangent.row1 = -1
    tangent.row2 = -1
    tangent.row3 = -1
    tangent.row4 = -1
    tangent.row5 = -1
    tangents[tid] = tangent
    if tid >= wp.min(contact_count[0], shape0.shape[0]):
        return
    s0 = shape0[tid]
    s1 = shape1[tid]
    if s0 < 0 or s1 < 0:
        return
    b0 = shape_body[s0]
    b1 = shape_body[s1]
    e0 = body_to_segment[b0] if b0 >= 0 else -1
    e1 = body_to_segment[b1] if b1 >= 0 else -1
    if e0 < 0 and e1 < 0:
        return

    p0 = point0[tid]
    p1 = point1[tid]
    previous0 = p0
    previous1 = p1
    r0 = wp.vec3(0.0)
    r1 = wp.vec3(0.0)
    if e0 >= 0:
        a0 = segment_nodes[e0, 0]
        z0 = segment_nodes[e0, 1]
        r0 = frame[e0] * (point0[tid] - segment_com[e0])
        p0 = 0.5 * (node_position[a0] + node_position[z0]) + r0
        previous0 = 0.5 * (node_previous[a0] + node_previous[z0]) + frame_begin[e0] * (point0[tid] - segment_com[e0])
    elif b0 >= 0:
        p0 = wp.transform_point(body_q[b0], point0[tid])
        previous0 = p0
    if e1 >= 0:
        a1 = segment_nodes[e1, 0]
        z1 = segment_nodes[e1, 1]
        r1 = frame[e1] * (point1[tid] - segment_com[e1])
        p1 = 0.5 * (node_position[a1] + node_position[z1]) + r1
        previous1 = 0.5 * (node_previous[a1] + node_previous[z1]) + frame_begin[e1] * (point1[tid] - segment_com[e1])
    elif b1 >= 0:
        p1 = wp.transform_point(body_q[b1], point1[tid])
        previous1 = p1

    n = normal[tid]
    gap = contact_surface_separation(p0, p1, n, margin0[tid], margin1[tid])
    if gap >= 0.0:
        return
    ke = wp.max(0.0, 0.5 * (shape_ke[s0] + shape_ke[s1]))
    kd = wp.max(0.0, 0.5 * (shape_kd[s0] + shape_kd[s1]))
    if contact_ke:
        ke = contact_ke[tid] if contact_ke[tid] > 0.0 else ke
        kd = contact_kd[tid] if contact_kd[tid] > 0.0 else kd
    vn = wp.dot(n, ((p1 - previous1) - (p0 - previous0)) / dt)
    gradient, hessian, _friction_gradient, _friction_hessian = contact_response(
        gap, n, vn, wp.vec3(0.0), ke, kd, 0.0, 0.0, dt
    )
    half_identity = 0.5 * wp.identity(3, float)

    if e0 >= 0:
        a0 = segment_nodes[e0, 0]
        z0 = segment_nodes[e0, 1]
        Jr0 = -wp.skew(r0)
        tangent.row0 = a0
        tangent.row1 = z0
        tangent.row2 = node_count + e0
        tangent.J0 = half_identity
        tangent.J1 = half_identity
        tangent.J2 = Jr0
        wp.atomic_add(rhs, a0, 0.5 * gradient)
        wp.atomic_add(rhs, z0, 0.5 * gradient)
        wp.atomic_add(rhs, node_count + e0, wp.transpose(Jr0) * gradient)
        wp.atomic_add(diag, a0, 0.25 * hessian)
        wp.atomic_add(diag, z0, 0.25 * hessian)
        wp.atomic_add(diag, node_count + e0, wp.transpose(Jr0) * hessian * Jr0)
    if e1 >= 0:
        a1 = segment_nodes[e1, 0]
        z1 = segment_nodes[e1, 1]
        Jr1 = wp.skew(r1)
        tangent.row3 = a1
        tangent.row4 = z1
        tangent.row5 = node_count + e1
        tangent.J3 = -half_identity
        tangent.J4 = -half_identity
        tangent.J5 = Jr1
        wp.atomic_sub(rhs, a1, 0.5 * gradient)
        wp.atomic_sub(rhs, z1, 0.5 * gradient)
        wp.atomic_add(rhs, node_count + e1, wp.transpose(Jr1) * gradient)
        wp.atomic_add(diag, a1, 0.25 * hessian)
        wp.atomic_add(diag, z1, 0.25 * hessian)
        wp.atomic_add(diag, node_count + e1, wp.transpose(Jr1) * hessian * Jr1)
    tangent.active = 1
    tangent.hessian = hessian
    tangents[tid] = tangent


@wp.kernel
def cable_contact_matrix_free_kernel(
    tangents: wp.array[CableContactTangent],
    direction: wp.array[wp.vec3],
    # outputs
    product: wp.array[wp.vec3],
):
    """Apply Cable contact off-diagonal blocks cached by ``cable_eval_contact_kernel``."""
    tangent = tangents[wp.tid()]
    if tangent.active == 0:
        return
    relative = wp.vec3(0.0)
    if tangent.row0 >= 0:
        relative += tangent.J0 * direction[tangent.row0]
    if tangent.row1 >= 0:
        relative += tangent.J1 * direction[tangent.row1]
    if tangent.row2 >= 0:
        relative += tangent.J2 * direction[tangent.row2]
    if tangent.row3 >= 0:
        relative += tangent.J3 * direction[tangent.row3]
    if tangent.row4 >= 0:
        relative += tangent.J4 * direction[tangent.row4]
    if tangent.row5 >= 0:
        relative += tangent.J5 * direction[tangent.row5]
    if tangent.row0 >= 0:
        wp.atomic_add(
            product,
            tangent.row0,
            wp.transpose(tangent.J0) * tangent.hessian * relative
            - wp.transpose(tangent.J0) * tangent.hessian * tangent.J0 * direction[tangent.row0],
        )
    if tangent.row1 >= 0:
        wp.atomic_add(
            product,
            tangent.row1,
            wp.transpose(tangent.J1) * tangent.hessian * relative
            - wp.transpose(tangent.J1) * tangent.hessian * tangent.J1 * direction[tangent.row1],
        )
    if tangent.row2 >= 0:
        wp.atomic_add(
            product,
            tangent.row2,
            wp.transpose(tangent.J2) * tangent.hessian * relative
            - wp.transpose(tangent.J2) * tangent.hessian * tangent.J2 * direction[tangent.row2],
        )
    if tangent.row3 >= 0:
        wp.atomic_add(
            product,
            tangent.row3,
            wp.transpose(tangent.J3) * tangent.hessian * relative
            - wp.transpose(tangent.J3) * tangent.hessian * tangent.J3 * direction[tangent.row3],
        )
    if tangent.row4 >= 0:
        wp.atomic_add(
            product,
            tangent.row4,
            wp.transpose(tangent.J4) * tangent.hessian * relative
            - wp.transpose(tangent.J4) * tangent.hessian * tangent.J4 * direction[tangent.row4],
        )
    if tangent.row5 >= 0:
        wp.atomic_add(
            product,
            tangent.row5,
            wp.transpose(tangent.J5) * tangent.hessian * relative
            - wp.transpose(tangent.J5) * tangent.hessian * tangent.J5 * direction[tangent.row5],
        )
