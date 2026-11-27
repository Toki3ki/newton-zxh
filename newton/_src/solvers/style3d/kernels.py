# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from ...core.reset import reset_world_selected
from ...geometry import ParticleFlags
from ...sim import BodyFlags
from .affine_body import affine_mass_multiply, inertia_to_second_moment


@wp.kernel
def eval_particle_body_attachment_kernel(
    particle_count: int,
    particles: wp.array[int],
    bodies: wp.array[int],
    points: wp.array[wp.vec3],
    stiffness: wp.array[float],
    damping: wp.array[float],
    enabled: wp.array[wp.bool],
    body_com: wp.array[wp.vec3],
    x: wp.array[wp.vec3],
    x_prev: wp.array[wp.vec3],
    x_inertia: wp.array[wp.vec3],
    flags: wp.array[int],
    dt: float,
    rhs: wp.array[wp.vec3],
    diag: wp.array[wp.mat33],
    tangents: wp.array[float],
):
    """Accumulate bilateral particle-body spring forces and implicit damping.

    For particle position p and body affine coordinates q = (c, a0, a1, a2),
    c is the world-space COM and A = [a0 a1 a2]. The authored attachment point
    is body-origin-local, so r = points[i] - body_com[body] is COM-local.
    With weights w = (1, r.x, r.y, r.z), the world-space attachment point is
    y = sum_j w[j] q[j] = c + A r and the residual is e = p - y [m].

    Let h = dt [s], k = stiffness[i] [N/m], d = damping[i] [N*s/m], and e_prev
    be the same residual evaluated at the beginning of the substep (x_prev).
    The elastic energy and backward-Euler incremental damping potential are::

        E_elastic = (k / 2) * dot(e, e)
        Phi = E_elastic + (d / (2*h)) * dot(e - e_prev, e - e_prev)
        g = dPhi/de = k*e + (d/h)*(e - e_prev)

    The damping potential is a time-discrete dissipation term, not stored
    elastic energy. Its relative velocity is (e - e_prev) / h. For the block
    coordinates z = (p, c, a0, a1, a2), with I the 3x3 identity::

        J = de/dz = [I, -I, -r.x*I, -r.y*I, -r.z*I]
        rhs += -J.T * g
        H = d2Phi/dz2 = (k + d/h) * J.T * J

    Thus the particle receives -g and the body blocks receive w[j]*g.
    The COM force is +g; at a rigid configuration the torque is (A*r) cross g.
    This kernel adds H_pp = (k+d/h)*I and H_qj,qj = (k+d/h)*w[j]^2*I to diag.
    The companion matrix-product kernel adds all off-diagonal blocks. The
    cached scalar tangent is k+d/h (zero for a disabled attachment).

    Prescribed body coordinates use x_inertia instead of x to match contact
    prediction. Their increments, and those of fixed particles, are held at
    zero by solver flags; the above blocks are restricted to active DOFs by
    the solve. Finite k permits a gap: this is not a hard positional constraint.
    """
    tid = wp.tid()
    tangents[tid] = 0.0
    if not enabled[tid]:
        return
    particle = particles[tid]
    body = bodies[tid]
    start = particle_count + 4 * body
    local = points[tid] - body_com[body]
    weights = wp.vec4(1.0, local[0], local[1], local[2])
    residual = x[particle]
    previous = x_prev[particle]
    for j in range(4):
        point = x[start + j]
        # Match contact prediction for externally driven bodies.
        if (flags[start] & ParticleFlags.ACTIVE) == 0:
            point = x_inertia[start + j]
        residual -= weights[j] * point
        previous -= weights[j] * x_prev[start + j]
    gradient = stiffness[tid] * residual + (damping[tid] / dt) * (residual - previous)
    tangent = stiffness[tid] + damping[tid] / dt
    tangents[tid] = tangent
    identity = wp.identity(3, float)
    wp.atomic_sub(rhs, particle, gradient)
    wp.atomic_add(diag, particle, tangent * identity)
    for j in range(4):
        wp.atomic_add(rhs, start + j, weights[j] * gradient)
        wp.atomic_add(diag, start + j, (tangent * weights[j] * weights[j]) * identity)


@wp.kernel
def particle_body_attachment_off_diag_multiply_kernel(
    particle_count: int,
    particles: wp.array[int],
    bodies: wp.array[int],
    points: wp.array[wp.vec3],
    body_com: wp.array[wp.vec3],
    tangents: wp.array[float],
    x: wp.array[wp.vec3],
    out: wp.array[wp.vec3],
):
    """Add the off-diagonal attachment Hessian product to out.

    Using w = (1, r.x, r.y, r.z) and t = k + d/dt from the force kernel::

        H_p, qj = H_qj, p = -t * w[j] * I
        H_qj, ql = t * w[j] * w[l] * I(j != l)

    For input vector x = (v_p, v_q0, ..., v_q3), set s = sum_j w[j]*v_qj::

        out_p += -t * s
        out_qj += t * w[j] * (s - w[j] * v_qj - v_p)

    The diagonal blocks are already in the assembled matrix and are excluded
    here. Unlike the force kernel, x is a matrix-product vector, not positions.
    """
    tid = wp.tid()
    tangent = tangents[tid]
    if tangent == 0.0:
        return
    particle = particles[tid]
    body = bodies[tid]
    start = particle_count + 4 * body
    local = points[tid] - body_com[body]
    weights = wp.vec4(1.0, local[0], local[1], local[2])
    point = wp.vec3(0.0)
    for j in range(4):
        point += weights[j] * x[start + j]
    wp.atomic_sub(out, particle, tangent * point)
    for j in range(4):
        wp.atomic_add(out, start + j, tangent * weights[j] * (point - weights[j] * x[start + j] - x[particle]))


@wp.kernel
def reset_tet_volume_targets_kernel(
    tet_indices: wp.array2d[wp.int32],
    particle_world: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    reset_all: bool,
    world_count: int,
    volume_targets: wp.array[wp.mat33],
):
    """Reset projection history for tetrahedra in selected worlds."""
    tid = wp.tid()
    selected = reset_all
    if not reset_all:
        for vertex in range(4):
            world = particle_world[tet_indices[tid, vertex]]
            if reset_world_selected(world, world_mask, world_count):
                selected = True
    if selected:
        volume_targets[tid] = wp.identity(3, dtype=float)


@wp.kernel
def compute_solver_flags_kernel(
    particle_count: int,
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
        body = tid - particle_count
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

        start = particle_count + 4 * body
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


@wp.func
def proper_rotation(F: wp.mat33):
    U, _s, V = wp.svd3(F)
    correction = wp.identity(3, dtype=float)
    if wp.determinant(U * wp.transpose(V)) < 0.0:
        correction[2, 2] = -1.0
    return U * correction * wp.transpose(V)


@wp.func
def volume_roots(s: wp.vec3, multiplier: float, small_root: bool):
    result = wp.vec3(0.0)
    for i in range(3):
        root = wp.sqrt(wp.max(0.0, s[i] * s[i] + 4.0 * multiplier))
        if s[i] < 0.0:
            result[i] = 2.0 * multiplier / (root - s[i])
        else:
            result[i] = 0.5 * (s[i] + root)
    if small_root:
        result[2] = -2.0 * multiplier / (s[2] + wp.sqrt(wp.max(0.0, s[2] * s[2] + 4.0 * multiplier)))
    return result


@wp.func
def project_tet_volume(F: wp.mat33, previous: wp.mat33):
    """Choose a feasible SL(3) target without increasing the local PD residual.

    Scalar KKT roots give the compression projection. For expansion, test
    both bracketed branches plus a normalized feasible seed. This is an
    approximate local minimizer, not an exact global closest-point oracle.
    """
    U, s, V = wp.svd3(F)
    # Warp's signed SVD places orientation in the smallest singular value.
    correction = wp.identity(3, dtype=float)
    if wp.determinant(U * wp.transpose(V)) < 0.0:
        correction[2, 2] = -1.0
    Q = U * correction * wp.transpose(V)
    residual = F - Q
    best = wp.ddot(residual, residual)
    residual = F - previous
    if wp.ddot(residual, residual) < best:
        Q = previous
        best = wp.ddot(residual, residual)
    product = s[0] * s[1] * s[2]
    if product > 0.0:
        candidate = F / wp.pow(product, 1.0 / 3.0)
        residual = F - candidate
        if wp.ddot(residual, residual) < best:
            Q = candidate
            best = wp.ddot(residual, residual)
    for branch in range(2):
        lower = float(0.0)
        upper = float(0.0)
        active = bool(False)
        small = branch == 1
        if product <= 1.0 and branch == 0:
            # At this upper bound every positive root is at least one.
            upper = wp.max(1.0, 1.0 - s[2])
            active = True
        elif product > 1.0:
            lower = -0.25 * s[2] * s[2]
            end = volume_roots(s, lower, False)
            end_product = end[0] * end[1] * end[2]
            active = (branch == 0 and end_product <= 1.0) or (branch == 1 and end_product >= 1.0)
        if active:
            for _iteration in range(24):
                middle = 0.5 * (lower + upper)
                roots = volume_roots(s, middle, small)
                below = roots[0] * roots[1] * roots[2] < 1.0
                if below != small:
                    lower = middle
                else:
                    upper = middle
            roots = volume_roots(s, 0.5 * (lower + upper), small)
            roots /= wp.pow(roots[0] * roots[1] * roots[2], 1.0 / 3.0)
            candidate = U * wp.diag(roots) * wp.transpose(V)
            residual = F - candidate
            if wp.ddot(residual, residual) < best:
                Q = candidate
                best = wp.ddot(residual, residual)
    return Q


@wp.func
def tet_deformation_gradient(
    pos: wp.array[wp.vec3], tet_indices: wp.array2d[wp.int32], tet_poses: wp.array[wp.mat33], tet_id: int
):
    """Return ``Ds * inverse(Dm)`` for one tetrahedron."""
    i0 = tet_indices[tet_id, 0]
    ds = wp.matrix_from_cols(
        pos[tet_indices[tet_id, 1]] - pos[i0],
        pos[tet_indices[tet_id, 2]] - pos[i0],
        pos[tet_indices[tet_id, 3]] - pos[i0],
    )
    return ds * tet_poses[tet_id]


@wp.kernel
def eval_tet_constraint_kernel(
    pos: wp.array[wp.vec3],
    tet_indices: wp.array2d[wp.int32],
    tet_poses: wp.array[wp.mat33],
    tet_materials: wp.array2d[float],
    volume_targets: wp.array[wp.mat33],
    plastic_rest_stretch: wp.array[wp.mat33],
    yield_strains: wp.array[float],
    # outputs
    forces: wp.array[wp.vec3],
):
    """Evaluate co-rotational tet elasticity with a volumetric penalty."""
    tid = wp.tid()
    inv_dm = tet_poses[tid]
    F = tet_deformation_gradient(pos, tet_indices, tet_poses, tid)
    if yield_strains.shape[0] > 0 and yield_strains[tid] > 0.0:
        rest_stretch = plastic_rest_stretch[tid]
        shape_target = proper_rotation(F * wp.transpose(rest_stretch)) * rest_stretch
    else:
        shape_target = proper_rotation(F)

    volume = 1.0 / (6.0 * wp.determinant(inv_dm))
    mu = tet_materials[tid, 0]
    lam = tet_materials[tid, 1]
    stress = (2.0 * mu * volume) * (F - shape_target)
    target = project_tet_volume(F, volume_targets[tid])
    volume_targets[tid] = target
    stress += (3.0 * lam * volume) * (F - target)

    for i in range(4):
        gradient = -(inv_dm[0] + inv_dm[1] + inv_dm[2]) if i == 0 else inv_dm[i - 1]
        wp.atomic_add(forces, tet_indices[tid, i], -(stress * gradient))


@wp.kernel
def update_tet_plastic_rest_kernel(
    pos: wp.array[wp.vec3],
    tet_indices: wp.array2d[wp.int32],
    tet_poses: wp.array[wp.mat33],
    yield_strains: wp.array[float],
    previous_stretch: wp.array[wp.mat33],
    plastic_rest_stretch: wp.array[wp.mat33],
):
    """Accumulate yielded isochoric stretch in each tetrahedron's material frame."""
    tid = wp.tid()
    yield_strain = yield_strains[tid]
    previous = previous_stretch[tid]
    plastic_rest_stretch[tid] = previous
    if yield_strain <= 0.0:
        return
    F = tet_deformation_gradient(pos, tet_indices, tet_poses, tid)
    if wp.determinant(F) <= 0.0:
        return
    _u, s, V = wp.svd3(F)
    if wp.min(s) <= 1.0e-6 * wp.max(1.0, wp.max(s)):
        return
    logs = wp.vec3(wp.log(s[0]), wp.log(s[1]), wp.log(s[2]))
    logs -= wp.vec3((logs[0] + logs[1] + logs[2]) / 3.0)
    axes, p = wp.eig3(previous)
    log_previous = axes * wp.diag(wp.vec3(wp.log(p[0]), wp.log(p[1]), wp.log(p[2]))) * wp.transpose(axes)
    delta = V * wp.diag(logs) * wp.transpose(V) - log_previous
    equivalent = wp.sqrt((2.0 / 3.0) * wp.ddot(delta, delta))
    if equivalent <= yield_strain:
        return
    axes_new, logs_new = wp.eig3(log_previous + (1.0 - yield_strain / equivalent) * delta)
    mean = (logs_new[0] + logs_new[1] + logs_new[2]) / 3.0
    for i in range(3):
        logs_new[i] = wp.exp(logs_new[i] - mean)
    plastic_rest_stretch[tid] = axes_new * wp.diag(logs_new) * wp.transpose(axes_new)


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

    if aniso_ke[fid][0] == 0.0 and aniso_ke[fid][1] == 0.0 and aniso_ke[fid][2] == 0.0:
        return

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
    rest_pos: wp.array[wp.vec3],
    use_initial_curvature: bool,
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

    With initial curvature enabled, let c = sum_i w_i x_i and
    r = ||sum_i w_i X_i|| for the initial positions X. Use

        E_b = 1/2 * k * 3/(A0 + A1) * (||c|| - r)^2,
        p = r * c/||c||,
        F_i = -k * 3/(A0 + A1) * w_i * (c - p).

    Holding the rotation-invariant local projection p fixed leaves the existing
    positive semidefinite PD matrix unchanged. At c = 0 use the zero subgradient.
    """
    eid = wp.tid()
    if edges[eid][0] < 0 or edges[eid][1] < 0:
        return
    edge = edges[eid]
    if edge_bending_properties[eid][0] == 0.0:
        return
    edge_stiff = edge_bending_properties[eid][0] * (3.0 / edge_rest_area[eid])
    bend_weight = wp.vec4(0.0)
    bend_weight[2] = edge_bending_cot[eid][2] + edge_bending_cot[eid][3]
    bend_weight[3] = edge_bending_cot[eid][0] + edge_bending_cot[eid][1]
    bend_weight[0] = -edge_bending_cot[eid][0] - edge_bending_cot[eid][2]
    bend_weight[1] = -edge_bending_cot[eid][1] - edge_bending_cot[eid][3]
    curvature = wp.vec3(0.0)
    rest_curvature = wp.vec3(0.0)
    # Relative coordinates avoid cancellation for small shells far from origin.
    for j in range(3):
        curvature += bend_weight[j] * (pos[edge[j]] - pos[edge[3]])
        if use_initial_curvature:
            rest_curvature += bend_weight[j] * (rest_pos[edge[j]] - rest_pos[edge[3]])
    residual = curvature
    if use_initial_curvature and wp.length(curvature) > 1.0e-12:
        residual -= wp.length(rest_curvature) * wp.normalize(curvature)
    for i in range(4):
        wp.atomic_sub(forces, edge[i], edge_stiff * bend_weight[i] * residual)


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
    """Initialize inertia and start active DOFs at their inertial prediction.

    Carry ballistic motion in the nonlinear starting point so a truncated
    solve of stiff elastic constraints does not have to recover it. Prescribed
    DOFs retain their input configuration; contact and attachment kernels use
    their separate inertia targets when evaluating moving boundaries.
    """
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
            x_curr[tid] = x_inertia[tid]
            dx[tid] = wp.vec3(0.0)
    else:
        body = tid - particle_count
        start = particle_count + 4 * body
        is_kinematic = (body_flags[body] & BodyFlags.KINEMATIC) != 0
        active = body_mass[body] > 0.0 and (is_kinematic or (body_flags[body] & BodyFlags.DYNAMIC) != 0)
        mass = float(0.0)
        moment = wp.mat33(0.0)
        if active:
            mass = body_mass[body] / (dt * dt)
            moment = inertia_to_second_moment(body_inertia[body]) / (dt * dt)

        # Cache the symmetric mass coupling once per step for all PCG products.
        affine_mass_off_diag[body] = wp.vec3(moment[0, 1], moment[0, 2], moment[1, 2])

        for i in range(4):
            row = start + i
            x_prev[row] = x_curr[row]
            dx[row] = wp.vec3(0.0)
            if body_mass[body] <= 0.0 and not is_kinematic:
                x_inertia[row] = x_curr[row]
            if active and not is_kinematic:
                x_curr[row] = x_inertia[row]
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

    body = tid - particle_count
    start = particle_count + 4 * body
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
    prev_pos: wp.array[wp.vec3],
    pos: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
):
    tid = wp.tid()
    if tid < particle_count:
        vel[tid] = 0.998 * (pos[tid] - prev_pos[tid]) / dt
    else:
        vel[tid] = (pos[tid] - prev_pos[tid]) / dt
