# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from ...geometry import ParticleFlags


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


@wp.func
def tet_other_local_index(local_i: int, k: int):
    """Return the k-th local vertex other than local_i."""
    return k + 1 if k >= local_i else k


@wp.func
def tet_deformation_gradient(
    pos: wp.array[wp.vec3],
    tet_indices: wp.array2d[wp.int32],
    tet_poses: wp.array[wp.mat33],
    tet_id: int,
):
    i0 = tet_indices[tet_id, 0]
    i1 = tet_indices[tet_id, 1]
    i2 = tet_indices[tet_id, 2]
    i3 = tet_indices[tet_id, 3]

    Ds = wp.matrix_from_cols(
        pos[i1] - pos[i0],
        pos[i2] - pos[i0],
        pos[i3] - pos[i0],
    )

    # tet_poses 保存的是 inverse(Dm)
    return Ds * tet_poses[tet_id]


@wp.kernel
def initialize_tet_inv_qa_kernel(
    rest_pos: wp.array[wp.vec3],
    tet_indices: wp.array2d[wp.int32],
    # outputs: one Q^{-1} [1, 1, 1]^T vector per tet/local vertex
    inv_qa: wp.array[wp.vec3],
):
    """Precompute the SIPD geometric weights for every tet vertex.

    For local vertex i, Q[i] is the Gram matrix of the three rest edges
    emanating from i and inv_qa[i] = Q[i]^{-1} * (1, 1, 1)^T.

    # 把FEM的四面体的节点力拆成边上的成对力
    # SIPD用这个边力形式来组装半隐式求解系统。
    """
    tid = wp.tid()

    for local_i in range(4):
        vertex_i = tet_indices[tid, local_i]
        x_i = rest_pos[vertex_i]

        local_j0 = tet_other_local_index(local_i, 0)
        local_j1 = tet_other_local_index(local_i, 1)
        local_j2 = tet_other_local_index(local_i, 2)

        e0 = rest_pos[tet_indices[tid, local_j0]] - x_i
        e1 = rest_pos[tet_indices[tid, local_j1]] - x_i
        e2 = rest_pos[tet_indices[tid, local_j2]] - x_i

        # matrix_from_cols stores Q column-major. The entries below are
        # symmetric, so this is the Gram matrix e_r dot e_s.
        Q = wp.matrix_from_cols(
            wp.vec3(wp.dot(e0, e0), wp.dot(e1, e0), wp.dot(e2, e0)),
            wp.vec3(wp.dot(e0, e1), wp.dot(e1, e1), wp.dot(e2, e1)),
            wp.vec3(wp.dot(e0, e2), wp.dot(e1, e2), wp.dot(e2, e2)),
        )

        # ModelBuilder rejects degenerate tetrahedra. A tiny regularizer keeps
        # this preprocessing kernel finite for nearly degenerate input.
        Q = Q + wp.float32(1.0e-8) * wp.identity(3, dtype=wp.float32)
        inv_qa[tid * 4 + local_i] = wp.inverse(Q) * wp.vec3(1.0, 1.0, 1.0)


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
    edge_ke = edge_bending_properties[eid][0]
    edge_area = edge_rest_area[eid]
    if edge_ke == 0.0 or edge_area <= 0.0:
        return
    edge_stiff = edge_ke * (3.0 / edge_area)
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


@wp.func
def cofactor_mat33(F: wp.mat33):
    """Return the cofactor matrix of a 3x3 matrix."""
    C = wp.mat33(0.0)
    C[0, 0] = F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1]
    C[0, 1] = F[1, 2] * F[2, 0] - F[1, 0] * F[2, 2]
    C[0, 2] = F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0]
    C[1, 0] = F[0, 2] * F[2, 1] - F[0, 1] * F[2, 2]
    C[1, 1] = F[0, 0] * F[2, 2] - F[0, 2] * F[2, 0]
    C[1, 2] = F[0, 1] * F[2, 0] - F[0, 0] * F[2, 1]
    C[2, 0] = F[0, 1] * F[1, 2] - F[0, 2] * F[1, 1]
    C[2, 1] = F[0, 2] * F[1, 0] - F[0, 0] * F[1, 2]
    C[2, 2] = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    return C


@wp.func
def _tet_pd_volume_stretches(s: wp.vec3d, z_start: wp.vec2d):
    """Minimize distance in feasible log coordinates using an Armijo search.

    q=(exp(u), exp(v), exp(-u-v)) has positive unit product throughout.
    Double precision keeps the small local solve accurate near equilibrium;
    all global simulation arrays remain float32.
    """
    z = z_start
    q = wp.vec3d(wp.exp(z[0]), wp.exp(z[1]), wp.exp(-z[0] - z[1]))
    energy = wp.dot(q - s, q - s) * wp.float64(0.5)
    for _iteration in range(64):
        # Gradient and Hessian of 1/2 ||q(z)-s||^2.
        g = wp.vec2d(
            q[0] * (q[0] - s[0]) - q[2] * (q[2] - s[2]),
            q[1] * (q[1] - s[1]) - q[2] * (q[2] - s[2]),
        )
        if wp.length(g) <= wp.float64(1.0e-9) * (wp.float64(1.0) + wp.dot(q, q)):
            break
        h2 = q[2] * (wp.float64(2.0) * q[2] - s[2])
        h0 = q[0] * (wp.float64(2.0) * q[0] - s[0]) + h2
        h1 = q[1] * (wp.float64(2.0) * q[1] - s[1]) + h2
        determinant = h0 * h1 - h2 * h2
        # An indefinite Newton matrix is replaced by a descent direction;
        # the line search accepts only actual objective decrease.
        direction = -g / (
            wp.abs(h0) + wp.abs(h1) + wp.float64(2.0) * wp.abs(h2) + wp.float64(1.0)
        )
        if h0 > wp.float64(0.0) and determinant > wp.float64(0.0):
            direction = wp.vec2d(-h1 * g[0] + h2 * g[1], h2 * g[0] - h0 * g[1]) / determinant
        slope = wp.dot(g, direction)
        step = wp.float64(1.0)
        accepted = bool(False)
        for _backtrack in range(40):
            trial_z = z + step * direction
            trial_q = wp.vec3d(
                wp.exp(trial_z[0]), wp.exp(trial_z[1]), wp.exp(-trial_z[0] - trial_z[1])
            )
            trial_energy = wp.dot(trial_q - s, trial_q - s) * wp.float64(0.5)
            if wp.isfinite(trial_energy) and trial_energy <= energy + wp.float64(1.0e-4) * step * slope:
                z = trial_z
                q = trial_q
                energy = trial_energy
                accepted = True
                break
            step *= wp.float64(0.5)
        if not accepted:
            break
    return q


@wp.func
def tet_pd_volume_target(F: wp.mat33):
    """Find a feasible local closest-point target with det(Q)=1.

    Warp's U and V are proper rotations; the last singular value carries the
    sign of det(F). No singular-value floor or post-solve rescaling is used.
    Several feasible starts address the nonconvex expanded/inverted cases.
    The line search decreases the distance within each local solve, but does
    not certify a global nearest point. A local/global caller must also retain
    its previous feasible target if it has lower distance at the current F.
    """
    U, sigma, V = wp.svd3(F)
    s = wp.vec3d(wp.float64(sigma[0]), wp.float64(sigma[1]), wp.float64(sigma[2]))
    q = _tet_pd_volume_stretches(s, wp.vec2d(wp.float64(0.0)))
    best = wp.dot(q - s, q - s)
    # Preserve two positive stretches and repair only the third. These starts
    # also escape the symmetric stationary point of large isotropic expansion.
    for omitted in range(3):
        a = (omitted + 1) % 3
        b = (omitted + 2) % 3
        if s[a] > wp.float64(0.0) and s[b] > wp.float64(0.0):
            log_q = wp.vec3d(wp.float64(0.0))
            log_q[a] = wp.log(s[a])
            log_q[b] = wp.log(s[b])
            log_q[omitted] = -log_q[a] - log_q[b]
            candidate = _tet_pd_volume_stretches(s, wp.vec2d(log_q[0], log_q[1]))
            distance = wp.dot(candidate - s, candidate - s)
            if distance < best:
                q = candidate
                best = distance
    diagonal = wp.mat33(0.0)
    diagonal[0, 0] = wp.float32(q[0])
    diagonal[1, 1] = wp.float32(q[1])
    diagonal[2, 2] = wp.float32(q[2])
    return U * diagonal * wp.transpose(V)


@wp.kernel
def eval_tet_pd_constraint_kernel(
    pos: wp.array[wp.vec3],
    tet_indices: wp.array2d[wp.int32],
    tet_poses: wp.array[wp.mat33],
    tet_materials: wp.array2d[float],
    volume_ratio: float,
    forces: wp.array[wp.vec3],
):
    """Evaluate a small-strain-matched PD surrogate for Stable Neo-Hookean.

    The per-tet energy is

        mu*V*||F-R||_F^2 + 3/2*lambda*V*volume_ratio*||F-Q||_F^2,

    where R is the closest proper rotation and det(Q)=1. Its quadratic
    expansion at F=I is ``V*(mu*||eps||^2 + lambda/2*tr(eps)^2)`` when
    ``volume_ratio=1``. This matches Stable Neo-Hookean to second order,
    while retaining a constant scalar PD global operator.
    """
    tid = wp.tid()
    F = tet_deformation_gradient(pos, tet_indices, tet_poses, tid)
    U, _sigma, V = wp.svd3(F)
    correction = wp.identity(3, dtype=float)
    if wp.determinant(U * wp.transpose(V)) < 0.0:
        # Warp's singular values are sorted in descending order.
        correction[2, 2] = -1.0
    R = U * correction * wp.transpose(V)
    inv_dm = tet_poses[tid]
    volume = 1.0 / (6.0 * wp.determinant(inv_dm))
    mu = tet_materials[tid, 0]
    lam = tet_materials[tid, 1]
    P = (2.0 * mu * volume) * (F - R)
    if volume_ratio > 0.0:
        Q = tet_pd_volume_target(F)
        P += (3.0 * lam * volume_ratio * volume) * (F - Q)
    for i in range(4):
        b = wp.vec3(0.0)
        if i == 0:
            b = -(inv_dm[0] + inv_dm[1] + inv_dm[2])
        else:
            b = inv_dm[i - 1]
        wp.atomic_add(forces, tet_indices[tid, i], -(P * b))


@wp.func
def stable_neo_hookean_stress(
    F: wp.mat33,
    mu: float,
    lam: float,
):
    J = wp.determinant(F)
    cof_F = cofactor_mat33(F)

    # SIPD 代码中的参数
    s0 = mu
    s1 = lam + mu

    # P = d Psi / d F
    return s0 * F + (s1 * (J - 1.0) - s0) * cof_F


@wp.func
def stable_neo_hookean_tangent_split(F: wp.mat33, s0: float, s1: float):
    """SIPD stable-Neo-Hookean symmetric tangent split.

    This is the SVD-free form used by the reference implementation. The
    returned matrices are the ``D``, ``D+`` and ``D-`` terms used to build a
    positive local block for the semi-implicit solve.
    """
    J = wp.determinant(F)
    C = F * wp.transpose(F)
    det_C = J * J
    if det_C < 1.0e-4:
        det_C = 1.0e-4

    # C is symmetric, so its cofactor matrix is also its adjugate here.
    C_inv = cofactor_mat33(C) / det_C
    I = wp.identity(3, dtype=wp.float32)

    D_plus = wp.mat33(0.0)
    D_minus = wp.mat33(0.0)
    D = wp.mat33(0.0)

    if J >= 0.0:
        D_plus = s0 * I + C_inv * (J * J * s1)
        D_minus = C_inv * (J * (s0 + s1))
        D = s0 * I - C_inv * (J * s0) + C_inv * (s1 * (J - 1.0) * J)
    else:
        D = s0 * I - C_inv * (J * s0) + C_inv * (s1 * (J - 1.0) * J)
        D_plus = D

    return D, D_plus, D_minus


@wp.kernel
def build_sipd_matrix_diags_kernel(
    scalar_diags: wp.array[float],
    sipd_diag: wp.array[wp.vec3],
    sipd_upper: wp.array[wp.vec3],
    # outputs
    matrix_diags: wp.array[wp.mat33],
):
    """Combine Style3D's scalar diagonal with SIPD's local 3x3 blocks."""
    tid = wp.tid()
    base = scalar_diags[tid]
    d = sipd_diag[tid]
    u = sipd_upper[tid]
    matrix_diags[tid] = wp.matrix_from_rows(
        wp.vec3(base + d[0], u[0], u[1]),
        wp.vec3(u[0], base + d[1], u[2]),
        wp.vec3(u[1], u[2], base + d[2]),
    )


@wp.kernel
def prepare_jacobi_preconditioner_matrix_kernel(
    matrix_diags: wp.array[wp.mat33],
    scalar_diags: wp.array[float],
    contact_hessian_diags: wp.array[wp.mat33],
    particle_flags: wp.array[wp.int32],
    # outputs
    inv_A_diags: wp.array[wp.mat33],
):
    tid = wp.tid()
    diag = matrix_diags[tid]
    if particle_flags[tid] & ParticleFlags.ACTIVE:
        diag += contact_hessian_diags[tid]
    inv_A_diags[tid] = wp.inverse(diag) if scalar_diags[tid] > 0.0 else wp.mat33(0.0)


@wp.kernel
def prepare_jacobi_preconditioner_matrix_no_contact_kernel(
    matrix_diags: wp.array[wp.mat33],
    scalar_diags: wp.array[float],
    # outputs
    inv_A_diags: wp.array[wp.mat33],
):
    tid = wp.tid()
    inv_A_diags[tid] = wp.inverse(matrix_diags[tid]) if scalar_diags[tid] > 0.0 else wp.mat33(0.0)


@wp.func
def tet_metric_damping_force_and_hessian(
    F: wp.mat33,
    F_prev: wp.mat33,
    rest_gradient: wp.vec3,
    damping: float,
    dt: float,
):
    """Return per-volume metric damping force and a positive local tangent.

    Match VBD's objective damping based on C = transpose(F) * F. The history
    is the beginning of the time step, not the previous nonlinear iterate.
    Rigid rotations therefore do not create material damping.
    """
    f0 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    f1 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    f2 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])
    f0_prev = wp.vec3(F_prev[0, 0], F_prev[1, 0], F_prev[2, 0])
    f1_prev = wp.vec3(F_prev[0, 1], F_prev[1, 1], F_prev[2, 1])
    f2_prev = wp.vec3(F_prev[0, 2], F_prev[1, 2], F_prev[2, 2])
    inv_dt = 1.0 / dt
    c00_rate = (wp.dot(f0, f0) - wp.dot(f0_prev, f0_prev)) * inv_dt
    c01_rate = (wp.dot(f0, f1) - wp.dot(f0_prev, f1_prev)) * inv_dt
    c02_rate = (wp.dot(f0, f2) - wp.dot(f0_prev, f2_prev)) * inv_dt
    c11_rate = (wp.dot(f1, f1) - wp.dot(f1_prev, f1_prev)) * inv_dt
    c12_rate = (wp.dot(f1, f2) - wp.dot(f1_prev, f2_prev)) * inv_dt
    c22_rate = (wp.dot(f2, f2) - wp.dot(f2_prev, f2_prev)) * inv_dt
    m = rest_gradient
    dc00_dx = 2.0 * m[0] * f0
    dc01_dx = m[0] * f1 + m[1] * f0
    dc02_dx = m[0] * f2 + m[2] * f0
    dc11_dx = 2.0 * m[1] * f1
    dc12_dx = m[1] * f2 + m[2] * f1
    dc22_dx = 2.0 * m[2] * f2
    force = -damping * (
        c00_rate * dc00_dx
        + 2.0 * c01_rate * dc01_dx
        + 2.0 * c02_rate * dc02_dx
        + c11_rate * dc11_dx
        + 2.0 * c12_rate * dc12_dx
        + c22_rate * dc22_dx
    )
    hessian = (
        damping
        * inv_dt
        * (
            wp.outer(dc00_dx, dc00_dx)
            + 2.0 * wp.outer(dc01_dx, dc01_dx)
            + 2.0 * wp.outer(dc02_dx, dc02_dx)
            + wp.outer(dc11_dx, dc11_dx)
            + 2.0 * wp.outer(dc12_dx, dc12_dx)
            + wp.outer(dc22_dx, dc22_dx)
        )
    )
    return force, hessian


@wp.kernel
def eval_tet_sipd_force_kernel(
    pos: wp.array[wp.vec3],
    rest_pos: wp.array[wp.vec3],
    tet_indices: wp.array2d[wp.int32],
    tet_poses: wp.array[wp.mat33],
    tet_materials: wp.array2d[float],
    inv_qa: wp.array[wp.vec3],
    dt: float,
    pos_prev: wp.array[wp.vec3],
    # Physical forces (-dE/dx), accumulated into the shared Style3D rhs.
    forces: wp.array[wp.vec3],
    sipd_diag: wp.array[wp.vec3],
    sipd_upper: wp.array[wp.vec3],
):
    """Accumulate physical tet forces, metric damping and positive SIPD blocks."""
    tid = wp.tid()

    F = tet_deformation_gradient(
        pos,
        tet_indices,
        tet_poses,
        tid,
    )

    mu = tet_materials[tid, 0]
    lam = tet_materials[tid, 1]
    damping = tet_materials[tid, 2]
    F_prev = F
    if damping > 0.0:
        F_prev = tet_deformation_gradient(pos_prev, tet_indices, tet_poses, tid)

    P = stable_neo_hookean_stress(F, mu, lam)

    s0 = mu
    s1 = lam + mu
    _D, PF1, PF2 = stable_neo_hookean_tangent_split(F, s0, s1)

    # det(inv(Dm)) = 1 / det(Dm)
    # det(Dm) = 6 * rest_volume
    inv_dm_det = wp.determinant(tet_poses[tid])
    rest_volume = 1.0 / (6.0 * inv_dm_det)
    P_e = rest_volume * P

    # 四面体的四个局部顶点
    for local_i in range(4):
        vertex_i = tet_indices[tid, local_i]
        force_i = wp.vec3(0.0)
        A_i = wp.mat33(0.0)

        # inv_qa[tid * 4 + local_i] 对应这个局部顶点的三个几何权重
        a_vec = inv_qa[tid * 4 + local_i]

        for k in range(3):
            local_j = tet_other_local_index(local_i, k)

            vertex_j = tet_indices[tid, local_j]
            a_eij = a_vec[k]

            # C++ direct stress 路径使用参考构型边
            X_ij = rest_pos[vertex_j] - rest_pos[vertex_i]

            # Q^{-1}1 reconstructs -grad(N_i) from the outgoing rest edges.
            # Therefore +a_eij * P_e * X_ij is the physical force -dE/dx_i.
            force_i += a_eij * (P_e * X_ij)

            # Positive/negative SIPD tangent split. This is the local block
            # used by the Style3D diagonal/preconditioner path.
            if a_eij > 0.0:
                A_i += a_eij * PF1
            else:
                A_i += (-a_eij) * PF2

        if damping > 0.0:
            inv_dm = tet_poses[tid]
            m = wp.vec3(0.0)
            if local_i == 0:
                m = wp.vec3(
                    -(inv_dm[0, 0] + inv_dm[1, 0] + inv_dm[2, 0]),
                    -(inv_dm[0, 1] + inv_dm[1, 1] + inv_dm[2, 1]),
                    -(inv_dm[0, 2] + inv_dm[1, 2] + inv_dm[2, 2]),
                )
            else:
                row = local_i - 1
                m = wp.vec3(inv_dm[row, 0], inv_dm[row, 1], inv_dm[row, 2])
            damping_force, damping_hessian = tet_metric_damping_force_and_hessian(F, F_prev, m, damping, dt)
            force_i += rest_volume * damping_force
            A_i += damping_hessian

        # 多个四面体可能共享 vertex_i, 必须原子累加
        wp.atomic_add(forces, vertex_i, force_i)
        wp.atomic_add(sipd_diag, vertex_i, wp.vec3(A_i[0, 0], A_i[1, 1], A_i[2, 2]) * rest_volume)
        wp.atomic_add(sipd_upper, vertex_i, wp.vec3(A_i[0, 1], A_i[0, 2], A_i[1, 2]) * rest_volume)


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
    step_scale: float,
    x_in: wp.array[wp.vec3],
    # outputs
    x_out: wp.array[wp.vec3],
    dx: wp.array[wp.vec3],
):
    tid = wp.tid()
    x_out[tid] = x_in[tid] + step_scale * dx[tid]
    dx[tid] = wp.vec3(0.0)


@wp.kernel
def sipd_semi_implicit_solve_kernel(
    dt: float,
    y_current: wp.array[wp.vec3],
    y_star: wp.array[wp.vec3],
    source: wp.array[wp.vec3],
    A_diag: wp.array[wp.vec3],
    A_upper: wp.array[wp.vec3],
    static_A_diag: wp.array[float],
    contact_hessian: wp.array[wp.mat33],
    particle_masses: wp.array[float],
    particle_flags: wp.array[wp.int32],
    step_scale: float,
    # outputs
    y_out: wp.array[wp.vec3],
):
    """Update each particle from physical forces using a local SIPD block.

    ``source`` shares Style3D's force convention, so the objective gradient
    contains its negative together with the inertial displacement.
    """
    tid = wp.tid()
    current = y_current[tid]

    if not particle_flags[tid] & ParticleFlags.ACTIVE or particle_masses[tid] <= 0.0:
        y_out[tid] = current
        return

    d = A_diag[tid]
    u = A_upper[tid]
    A = wp.matrix_from_rows(
        wp.vec3(d[0], u[0], u[1]),
        wp.vec3(u[0], d[1], u[2]),
        wp.vec3(u[1], u[2], d[2]),
    )
    A += wp.identity(3, dtype=wp.float32) * static_A_diag[tid] + contact_hessian[tid]

    dt2_over_mass = dt * dt / particle_masses[tid]
    hess = wp.identity(3, dtype=wp.float32) + dt2_over_mass * A
    grad = -dt2_over_mass * source[tid] + current - y_star[tid]
    increment = wp.inverse(hess) * (-grad)

    y_out[tid] = current + step_scale * increment


@wp.kernel
def update_velocity(
    dt: float,
    prev_pos: wp.array[wp.vec3],
    pos: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
):
    particle = wp.tid()
    vel[particle] = (pos[particle] - prev_pos[particle]) / dt
