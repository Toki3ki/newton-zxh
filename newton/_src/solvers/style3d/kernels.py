# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from ...geometry import ParticleFlags
from .linear_solver import NonZeroEntry


@wp.kernel
def deactivate_zero_mass_particles_kernel(
    particle_masses: wp.array[float],
    particle_flags: wp.array[wp.int32],
):
    tid = wp.tid()
    if particle_masses[tid] <= 0.0:
        particle_flags[tid] = particle_flags[tid] & ~ParticleFlags.ACTIVE


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
    pd_diags: wp.array[float],
):
    fid = face_index[0]
    if fid != -1:
        coord = drag_bary_coord[0]
        face = wp.vec3i(faces[fid, 0], faces[fid, 1], faces[fid, 2])

        if particle_flags[face[0]] & ParticleFlags.ACTIVE:
            pd_diags[face[0]] += spring_stiff * coord[0]

        if particle_flags[face[1]] & ParticleFlags.ACTIVE:
            pd_diags[face[1]] += spring_stiff * coord[1]

        if particle_flags[face[2]] & ParticleFlags.ACTIVE:
            pd_diags[face[2]] += spring_stiff * coord[2]


@wp.kernel
def init_step_kernel(
    dt: float,
    gravity: wp.array[wp.vec3],
    particle_world: wp.array[wp.int32],
    f_ext: wp.array[wp.vec3],
    v_curr: wp.array[wp.vec3],
    x_curr: wp.array[wp.vec3],
    x_prev: wp.array[wp.vec3],
    pd_diags: wp.array[float],
    particle_masses: wp.array[float],
    particle_flags: wp.array[wp.int32],
    # outputs
    x_inertia: wp.array[wp.vec3],
    static_A_diags: wp.array[float],
    dx: wp.array[wp.vec3],
):
    tid = wp.tid()
    x_last = x_curr[tid]
    x_prev[tid] = x_last

    if not particle_flags[tid] & ParticleFlags.ACTIVE:
        x_inertia[tid] = x_prev[tid]
        static_A_diags[tid] = 0.0
        dx[tid] = wp.vec3(0.0)
    else:
        v_prev = v_curr[tid]
        mass = particle_masses[tid]
        static_A_diags[tid] = pd_diags[tid] + mass / (dt * dt)
        world_idx = particle_world[tid]
        world_g = gravity[world_idx]
        x_inertia[tid] = x_last + v_prev * dt + (world_g + f_ext[tid] / mass) * (dt * dt)
        dx[tid] = v_prev * dt

        # temp
        # x_curr[tid] = x_last + v_prev * dt


@wp.kernel
def init_rhs_kernel(
    dt: float,
    x_curr: wp.array[wp.vec3],
    x_inertia: wp.array[wp.vec3],
    particle_masses: wp.array[float],
    # outputs
    rhs: wp.array[wp.vec3],
):
    tid = wp.tid()
    rhs[tid] = (x_inertia[tid] - x_curr[tid]) * particle_masses[tid] / (dt * dt)


@wp.kernel
def prepare_jacobi_preconditioner_kernel(
    static_A_diags: wp.array[float],
    contact_hessian_diags: wp.array[wp.mat33],
    particle_flags: wp.array[wp.int32],
    # outputs
    inv_A_diags: wp.array[wp.mat33],
):
    tid = wp.tid()
    diag = wp.identity(3, float) * static_A_diags[tid]
    if particle_flags[tid] & ParticleFlags.ACTIVE:
        diag += contact_hessian_diags[tid]
    inv_A_diags[tid] = wp.inverse(diag) if static_A_diags[tid] > 0.0 else wp.identity(3, float) * 0.0


@wp.kernel
def prepare_jacobi_preconditioner_no_contact_hessian_kernel(
    static_A_diags: wp.array[float],
    # outputs
    inv_A_diags: wp.array[wp.mat33],
):
    tid = wp.tid()
    diag = wp.identity(3, float) * static_A_diags[tid]
    inv_A_diags[tid] = wp.inverse(diag) if static_A_diags[tid] > 0.0 else wp.identity(3, float) * 0.0


@wp.kernel
def PD_jacobi_step_kernel(
    rhs: wp.array[wp.vec3],
    x_in: wp.array[wp.vec3],
    inv_diags: wp.array[wp.mat33],
    # outputs
    x_out: wp.array[wp.vec3],
):
    tid = wp.tid()
    x_out[tid] = x_in[tid] + inv_diags[tid] * rhs[tid]


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
def update_velocity(
    dt: float,
    prev_pos: wp.array[wp.vec3],
    pos: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
):
    particle = wp.tid()
    vel[particle] = 0.998 * (pos[particle] - prev_pos[particle]) / dt


# Cable based on Viper: @https://www.alphaxiv.org/zh-TW/pdf/1906.05260


@wp.func
def rotation_exp(v: wp.vec3):
    """Convert a rotation vector in radians to a quaternion (SO(3) exponential)."""
    a = wp.length(v)
    return wp.quat(
        v[0] * wp.sin(0.5 * a) / wp.max(a, 1.0e-8),
        v[1] * wp.sin(0.5 * a) / wp.max(a, 1.0e-8),
        v[2] * wp.sin(0.5 * a) / wp.max(a, 1.0e-8),
        wp.cos(0.5 * a),
    )


@wp.func
def wrapped_angle(angle: float):
    """Return the principal signed angle in [-pi, pi]."""
    return wp.atan2(wp.sin(angle), wp.cos(angle))


@wp.func
def cable_bishop_twist(qa: wp.quat, qb: wp.quat):
    """Measure rod material spin after shortest-arc tangent transport.

    Local z is the tangent and local x is the material director. Exactly
    antiparallel tangents require the explicit convention below.
    """
    ta = wp.quat_rotate(qa, wp.vec3(0.0, 0.0, 1.0))
    tb = wp.quat_rotate(qb, wp.vec3(0.0, 0.0, 1.0))
    ma = wp.quat_rotate(qa, wp.vec3(1.0, 0.0, 0.0))
    mb = wp.quat_rotate(qb, wp.vec3(1.0, 0.0, 0.0))
    axis = wp.cross(ta, tb)
    cosine = wp.clamp(wp.dot(ta, tb), -1.0, 1.0)
    transport = wp.quat_identity()
    if cosine > -0.999999:
        transport = wp.normalize(wp.quat(axis[0], axis[1], axis[2], 1.0 + cosine))
    elif wp.length(axis) > 1.0e-7:
        transport = wp.quat_from_axis_angle(wp.normalize(axis), wp.atan2(wp.length(axis), cosine))
    else:
        # Antiparallel tangents have no unique shortest transport. Choose the
        # parent material x axis; this singular branch is not differentiable.
        transport = wp.quat_from_axis_angle(ma, wp.pi)
    transported = wp.quat_rotate(transport, ma)
    return wp.atan2(wp.dot(tb, wp.cross(transported, mb)), wp.dot(transported, mb))


@wp.func
def cable_bend_residual(
    j: int,
    ids: wp.array2d[int],
    coefficients: wp.array2d[float],
    x: wp.array[wp.vec3],
    qa: wp.quat,
    target: wp.vec3,
):
    # r = sum_i a_i*c_i - R_a*b0; shared by energy and both derivatives.
    residual = -wp.quat_rotate(qa, target)
    for slot in range(4):
        residual += coefficients[j, slot] * x[ids[j, slot]]
    return residual


@wp.kernel
def eval_cable_bend_twist_energy_kernel(
    ids: wp.array2d[int],
    coefficients: wp.array2d[float],
    pairs: wp.array2d[int],
    targets: wp.array[wp.vec3],
    rest_twist: wp.array[float],
    bend: wp.array[float],
    twist: wp.array[float],
    x: wp.array[wp.vec3],
    frames: wp.array[wp.quat],
    energy: wp.array[float],
):
    j = wp.tid()
    qa, qb = frames[pairs[j, 0]], frames[pairs[j, 1]]
    r = cable_bend_residual(j, ids, coefficients, x, qa, targets[j])
    angle = wrapped_angle(cable_bishop_twist(qa, qb) - rest_twist[j])
    wp.atomic_add(energy, 0, 0.5 * (bend[j] * wp.dot(r, r) + twist[j] * angle * angle))


@wp.kernel
def eval_cable_bend_kernel(
    ids: wp.array2d[int],
    coefficients: wp.array2d[float],
    pairs: wp.array2d[int],
    targets: wp.array[wp.vec3],
    bend: wp.array[float],
    x: wp.array[wp.vec3],
    frames: wp.array[wp.quat],
    active: wp.array[float],
    rhs: wp.array[wp.vec3],
):
    j = wp.tid()
    r = cable_bend_residual(j, ids, coefficients, x, frames[pairs[j, 0]], targets[j])
    for slot in range(4):
        i = ids[j, slot]
        wp.atomic_add(rhs, i, -active[i] * bend[j] * coefficients[j, slot] * r)


@wp.kernel
def eval_cable_rotation_terms_kernel(
    ids: wp.array2d[int],
    coefficients: wp.array2d[float],
    pairs: wp.array2d[int],
    targets: wp.array[wp.vec3],
    rest_twist: wp.array[float],
    bend: wp.array[float],
    twist: wp.array[float],
    x: wp.array[wp.vec3],
    frames: wp.array[wp.quat],
    active: wp.array[float],
    rhs: wp.array[wp.vec3],
    diagonal: wp.array[wp.mat33],
):
    j = wp.tid()
    a, b = pairs[j, 0], pairs[j, 1]
    qa, qb = frames[a], frames[b]
    r = cable_bend_residual(j, ids, coefficients, x, qa, targets[j])
    # Right increment R <- R Exp(delta): d(Ac-Rb0)/d(delta)=R[b0]_x.
    jac = wp.quat_to_matrix(qa) * wp.skew(targets[j])
    wp.atomic_add(rhs, a, -active[a] * bend[j] * wp.transpose(jac) * r)
    wp.atomic_add(diagonal, a, active[a] * bend[j] * wp.transpose(jac) * jac)
    angle = wrapped_angle(cable_bishop_twist(qa, qb) - rest_twist[j])
    for side in range(2):
        derivative = wp.vec3(0.0)
        for axis in range(3):
            delta = wp.vec3(0.0)
            delta[axis] = 0.0005
            plus, minus = float(0.0), float(0.0)
            if side == 0:
                plus = cable_bishop_twist(qa * rotation_exp(delta), qb)
                minus = cable_bishop_twist(qa * rotation_exp(-delta), qb)
            else:
                plus = cable_bishop_twist(qa, qb * rotation_exp(delta))
                minus = cable_bishop_twist(qa, qb * rotation_exp(-delta))
            derivative[axis] = wrapped_angle(plus - minus) / 0.001
        i = pairs[j, side]
        wp.atomic_add(rhs, i, -active[i] * twist[j] * angle * derivative)
        wp.atomic_add(diagonal, i, active[i] * twist[j] * wp.outer(derivative, derivative))


@wp.kernel
def eval_cable_stretch_kernel(
    edges: wp.array2d[int],
    lengths: wp.array[float],
    axial: wp.array[float],
    bulk: wp.array[float],
    n: int,
    rotation_start: int,
    fixed_scale: bool,
    x: wp.array[wp.vec3],
    section_state: wp.array[wp.vec3],
    frames: wp.array[wp.quat],
    active: wp.array[float],
    rhs: wp.array[wp.vec3],
):
    e = wp.tid()
    a, b = edges[e, 0], edges[e, 1]
    scale = float(1.0)
    if not fixed_scale:
        scale = 0.5 * (section_state[n + a][0] + section_state[n + b][0])
    # With R and s fixed: E = wa/2*||d-u||^2 + wv/2*||s^2*d-u||^2,
    # where d=(c_b-c_a)/L and u=R*ez. The position Hessian coefficient
    # is (wa+wv*s^4)/L^2; a dynamic-scale solver must update it as s changes.
    factor = scale * scale
    w = (axial[e] + bulk[e] * factor * factor) / (lengths[e] * lengths[e])
    target = (
        (axial[e] + bulk[e] * factor) / lengths[e] * wp.quat_rotate(frames[rotation_start + e], wp.vec3(0.0, 0.0, 1.0))
    )
    gradient = w * (x[b] - x[a]) - target
    wp.atomic_add(rhs, a, gradient * active[a])
    wp.atomic_add(rhs, b, -gradient * active[b])


@wp.kernel
def solve_cable_rotation_kernel(n: int, diagonal: wp.array[wp.mat33], rhs: wp.array[wp.vec3], dx: wp.array[wp.vec3]):
    # Only rotations use this local solve. Positions are updated
    # separately by the scalar global system, so their increments stay zero.
    i = wp.tid()
    dx[i] = wp.vec3(0.0)
    if i >= n:
        dx[i] = wp.inverse(diagonal[i]) * rhs[i]


@wp.func
def rotation_vector(q: wp.quat):
    r = wp.normalize(q)
    if r[3] < 0.0:
        r = -r
    v = wp.vec3(r[0], r[1], r[2])
    length = wp.length(v)
    return v * (2.0 * wp.atan2(length, r[3]) / wp.max(length, 1.0e-8))


@wp.func
def cable_residual(
    kind: int, a: wp.vec3, b: wp.vec3, s: wp.vec3, t: wp.vec3, r: wp.quat, q: wp.quat, rest: wp.quat, length: float
):
    # Only the axial/alignment residual remains; angular energy is evaluated
    # by the chord-bending and Bishop-twist kernels.
    if kind == 0:
        return (b - a) / length - wp.quat_rotate(r, wp.vec3(0.0, 0.0, 1.0))
    return wp.vec3(0.0)


@wp.kernel
def eval_cable_stretch_rotation_kernel(
    edges: wp.array2d[int],
    lengths: wp.array[float],
    stiffness: wp.array[float],
    rotation_start: int,
    x: wp.array[wp.vec3],
    frames: wp.array[wp.quat],
    active: wp.array[float],
    rhs: wp.array[wp.vec3],
    diagonal: wp.array[wp.mat33],
):
    """Accumulate stretch alignment torque for the local rotation solve.

    E_s = k/2 * ||(c_b-c_a)/L - R*ez||^2.
    For R' = R*Exp(delta), J = R*[ez]_x, rhs = -k*J^T*r,
    and the local Gauss-Newton block is k*J^T*J.
    """
    e = wp.tid()
    i = rotation_start + e
    ez = wp.vec3(0.0, 0.0, 1.0)
    r = (x[edges[e, 1]] - x[edges[e, 0]]) / lengths[e] - wp.quat_rotate(frames[i], ez)
    jac = wp.quat_to_matrix(frames[i]) * wp.skew(ez)
    w = stiffness[e] * active[i]
    wp.atomic_add(rhs, i, -w * wp.transpose(jac) * r)
    wp.atomic_add(diagonal, i, w * wp.transpose(jac) * jac)


@wp.kernel
def cable_scale_terms(
    edges: wp.array2d[int],
    lengths: wp.array[float],
    bulk: wp.array[float],
    radial: wp.array[float],
    n: int,
    rotation_start: int,
    x: wp.array[wp.vec3],
    frames: wp.array[wp.quat],
    active: wp.array[float],
    rhs: wp.array[wp.vec3],
    diagonal: wp.array[wp.mat33],
):
    # E_v=kv/2*||s_bar^2*d-R*ez||^2, E_r=kr/4*((sa-1)^2+(sb-1)^2).
    e = wp.tid()
    a, b = edges[e, 0], edges[e, 1]
    s = 0.5 * (x[n + a][0] + x[n + b][0])
    d = (x[b] - x[a]) / lengths[e]
    r = s * s * d - wp.quat_rotate(frames[rotation_start + e], wp.vec3(0.0, 0.0, 1.0))
    j = s * d
    for node in range(2):
        i = n + edges[e, node]
        g = bulk[e] * wp.dot(j, r) + 0.5 * radial[e] * (x[i][0] - 1.0)
        h = bulk[e] * wp.dot(j, j) + 0.5 * radial[e]
        wp.atomic_add(rhs, i, wp.vec3(-active[i] * g, 0.0, 0.0))
        wp.atomic_add(diagonal, i, wp.diag(wp.vec3(active[i] * h, 0.0, 0.0)))
    i = rotation_start + e
    jac = wp.quat_to_matrix(frames[i]) * wp.skew(wp.vec3(0.0, 0.0, 1.0))
    wp.atomic_add(rhs, i, -active[i] * bulk[e] * wp.transpose(jac) * r)
    wp.atomic_add(diagonal, i, active[i] * bulk[e] * wp.transpose(jac) * jac)


@wp.kernel
def cable_scale_energy(
    edges: wp.array2d[int],
    lengths: wp.array[float],
    bulk: wp.array[float],
    radial: wp.array[float],
    n: int,
    rotation_start: int,
    x: wp.array[wp.vec3],
    frames: wp.array[wp.quat],
    energy: wp.array[float],
):
    e = wp.tid()
    a, b = edges[e, 0], edges[e, 1]
    sa, sb = x[n + a][0], x[n + b][0]
    s = 0.5 * (sa + sb)
    r = s * s * (x[b] - x[a]) / lengths[e] - wp.quat_rotate(frames[rotation_start + e], wp.vec3(0.0, 0.0, 1.0))
    da, db = sa - 1.0, sb - 1.0
    wp.atomic_add(energy, 0, 0.5 * bulk[e] * wp.dot(r, r) + 0.25 * radial[e] * (da * da + db * db))


@wp.kernel
def cable_scale_position_hessian(
    edges: wp.array2d[int],
    lengths: wp.array[float],
    bulk: wp.array[float],
    n: int,
    x: wp.array[wp.vec3],
    active: wp.array[float],
    counts: wp.array[int],
    entries: wp.array2d[NonZeroEntry],
    diagonal: wp.array[float],
):
    # The scale-dependent volume block remains scalar*I3 in the global ELL.
    # Run serially: neighboring edges may update the same ELL entry.
    for e in range(edges.shape[0]):
        a, b = edges[e, 0], edges[e, 1]
        s = 0.5 * (x[n + a][0] + x[n + b][0])
        w = bulk[e] * s * s * s * s / (lengths[e] * lengths[e])
        diagonal[a] += w * active[a]
        diagonal[b] += w * active[b]
        for side in range(2):
            row = int(edges[e, side])
            col = int(edges[e, 1 - side])
            for slot in range(counts[row]):
                entry = entries[slot, row]
                if entry.column_index == col:
                    entry.value -= w * active[row] * active[col]
                    entries[slot, row] = entry


@wp.kernel
def cable_local_inertia_kernel(
    x: wp.array[wp.vec3],
    frames: wp.array[wp.quat],
    predicted: wp.array[wp.vec3],
    predicted_frames: wp.array[wp.quat],
    mass: wp.array[wp.vec3],
    active: wp.array[float],
    rotation_start: int,
    dt: float,
    rhs: wp.array[wp.vec3],
    diag: wp.array[wp.mat33],
):
    i = wp.tid()
    m = mass[i] / (dt * dt)
    error = x[i] - predicted[i]
    if i >= rotation_start:
        error = rotation_vector(wp.quat_inverse(predicted_frames[i]) * frames[i])
    rhs[i] = -wp.cw_mul(m, error) * active[i]
    diag[i] = wp.diag(m) + wp.identity(n=3, dtype=float) * 0.000001


@wp.kernel
def cable_elastic_energy(
    kinds: wp.array[int],
    ids: wp.array2d[int],
    lengths: wp.array[float],
    weights: wp.array[wp.vec3],
    rest: wp.array[wp.quat],
    fixed_scale: bool,
    x: wp.array[wp.vec3],
    frames: wp.array[wp.quat],
    energy: wp.array[float],
):
    c = wp.tid()
    s, t = wp.vec3(x[ids[c, 2]]), wp.vec3(x[ids[c, 3]])
    if fixed_scale:
        s, t = wp.vec3(1.0, 0.0, 0.0), wp.vec3(1.0, 0.0, 0.0)
    f = wp.cw_mul(
        weights[c],
        cable_residual(
            kinds[c], x[ids[c, 0]], x[ids[c, 1]], s, t, frames[ids[c, 4]], frames[ids[c, 5]], rest[c], lengths[c]
        ),
    )
    wp.atomic_add(energy, 0, 0.5 * wp.dot(f, f))


@wp.kernel
def cable_inertial_energy(
    x: wp.array[wp.vec3],
    frames: wp.array[wp.quat],
    predicted: wp.array[wp.vec3],
    predicted_frames: wp.array[wp.quat],
    mass: wp.array[wp.vec3],
    active: wp.array[float],
    n: int,
    rotation_start: int,
    dt: float,
    energy: wp.array[float],
):
    i = wp.tid()
    error = x[i] - predicted[i]
    if i >= rotation_start:
        error = rotation_vector(wp.quat_inverse(predicted_frames[i]) * frames[i])
    value = 0.5 * wp.dot(error, wp.cw_mul(mass[i], error)) / (dt * dt) * active[i]
    if i >= n and i < rotation_start and x[i][0] <= 0.0:
        value = wp.inf
    wp.atomic_add(energy, 0, value)


@wp.kernel
def cable_trial(
    x: wp.array[wp.vec3],
    frames: wp.array[wp.quat],
    dx: wp.array[wp.vec3],
    active: wp.array[float],
    rotation_start: int,
    alpha: float,
    trial_x: wp.array[wp.vec3],
    trial_frames: wp.array[wp.quat],
):
    i = wp.tid()
    d = alpha * active[i] * dx[i]
    trial_x[i] = x[i]
    trial_frames[i] = frames[i]
    if i >= rotation_start:
        trial_frames[i] = wp.normalize(frames[i] * rotation_exp(d))
    else:
        trial_x[i] = x[i] + d


@wp.kernel
def cable_accept(
    trial_x: wp.array[wp.vec3],
    trial_frames: wp.array[wp.quat],
    energy: wp.array[float],
    best_energy: wp.array[float],
    best_x: wp.array[wp.vec3],
    best_frames: wp.array[wp.quat],
):
    i = wp.tid()
    if wp.isfinite(energy[0]) and energy[0] < best_energy[0]:
        best_x[i] = trial_x[i]
        best_frames[i] = trial_frames[i]


@wp.kernel
def cable_accept_energy(energy: wp.array[float], best_energy: wp.array[float]):
    if wp.isfinite(energy[0]) and energy[0] < best_energy[0]:
        best_energy[0] = energy[0]


@wp.kernel
def cable_predict(
    x: wp.array[wp.vec3],
    frames: wp.array[wp.quat],
    velocity: wp.array[wp.vec3],
    active: wp.array[float],
    n: int,
    rotation_start: int,
    gravity: wp.array[wp.vec3],
    dt: float,
    predicted: wp.array[wp.vec3],
    predicted_frames: wp.array[wp.quat],
):
    i = wp.tid()
    target = x[i] + dt * velocity[i] * active[i]
    if i < n:
        target += dt * dt * gravity[0] * active[i]
    elif i < rotation_start:
        target = wp.vec3(wp.max(0.05, target[0]), 0.0, 0.0)
    predicted[i] = target
    predicted_frames[i] = frames[i]
    if i >= rotation_start:
        predicted_frames[i] = wp.normalize(frames[i] * rotation_exp(dt * velocity[i] * active[i]))
    x[i] = target
    frames[i] = predicted_frames[i]


@wp.kernel
def cable_velocity(
    x: wp.array[wp.vec3],
    frames: wp.array[wp.quat],
    old: wp.array[wp.vec3],
    old_frames: wp.array[wp.quat],
    rotation_start: int,
    dt: float,
    velocity: wp.array[wp.vec3],
):
    i = wp.tid()
    velocity[i] = (x[i] - old[i]) / dt
    if i >= rotation_start:
        velocity[i] = rotation_vector(wp.quat_inverse(old_frames[i]) * frames[i]) / dt


@wp.kernel
def cable_update_kinematic_segments(
    bodies: wp.array[int],
    edges: wp.array2d[int],
    endpoints: wp.array2d[wp.vec3],
    body_mass: wp.array[float],
    body_q: wp.array[wp.transform],
    rotation_start: int,
    x: wp.array[wp.vec3],
    frames: wp.array[wp.quat],
):
    e = wp.tid()
    b = bodies[e]
    if body_mass[b] == 0.0:
        x[edges[e, 0]] = wp.transform_point(body_q[b], endpoints[e, 0])
        x[edges[e, 1]] = wp.transform_point(body_q[b], endpoints[e, 1])
        frames[rotation_start + e] = wp.transform_get_rotation(body_q[b])


@wp.kernel
def cable_write_bodies(
    bodies: wp.array[int],
    edges: wp.array2d[int],
    centers: wp.array[wp.vec3],
    x: wp.array[wp.vec3],
    frames: wp.array[wp.quat],
    velocity: wp.array[wp.vec3],
    rotation_start: int,
    body_mass: wp.array[float],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    e = wp.tid()
    b = bodies[e]
    if body_mass[b] > 0.0:
        q = frames[rotation_start + e]
        center = 0.5 * (x[edges[e, 0]] + x[edges[e, 1]])
        body_q[b] = wp.transform(center - wp.quat_rotate(q, centers[e]), q)
        linear = 0.5 * (velocity[edges[e, 0]] + velocity[edges[e, 1]])
        angular = wp.quat_rotate(q, velocity[rotation_start + e])
        body_qd[b] = wp.spatial_vector(linear, angular)


@wp.kernel
def cable_enable_particles(
    nodes: wp.array[int],
    masses: wp.array[float],
    flags: wp.array[int],
):
    """Activate rod centerline particles in the solver-private flags.

    Particle-based rods keep their mapped centerline nodes disabled in the
    shared model flags because rigid-body solvers must not integrate them.
    Style3D owns these position DOFs and enables only the massive nodes here.
    """
    i = nodes[wp.tid()]
    flags[i] = int(ParticleFlags.ACTIVE) if masses[i] > 0.0 else 0
