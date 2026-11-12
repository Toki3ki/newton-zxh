# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Warp VIPER local material updates and scalar position targets.

Notation: c_a, c_b are shared centerline nodes; R_e is a segment's material
frame; s_a, s_b are dimensionless nodal radius scales (reference value 1).
L is the reference segment length, or the dual length for a frame pair.
Each constraint contributes E = 0.5 * ||sqrt(W) * C||^2, componentwise.
``weights`` stores sqrt(W), NOT W; see CableViper for the parameter mapping.
These are the implemented discrete energies, not a claim of exact equivalence
to Newton VBD's geometric bend/transported-twist model.
"""

import warp as wp


@wp.kernel
def cable_contact_terms(
    count: wp.array[int],
    shapes0: wp.array[int],
    shapes1: wp.array[int],
    points0: wp.array[wp.vec3],
    points1: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    margins0: wp.array[float],
    margins1: wp.array[float],
    shape_body: wp.array[int],
    shape_ke: wp.array[float],
    body_edge: wp.array[int],
    body_q: wp.array[wp.transform],
    edges: wp.array2d[int],
    endpoints: wp.array2d[wp.vec3],
    x: wp.array[wp.vec3],
    ids: wp.array2d[int],
    coefficients: wp.array2d[float],
    offsets: wp.array[float],
    stiffness: wp.array[float],
):
    c = wp.tid()
    if c >= count[0]:
        return
    # Freeze the contact support map for this substep:
    # g(c) = offset + n.dot(sum_j a_j*c_j), E_n = k/2 * min(g, 0)^2.
    # The coefficients interpolate capsule support motion from centerline nodes.
    # Radius/frame derivatives, friction and contact damping are not included.
    normal = normals[c]
    offset = -margins0[c] - margins1[c]
    ke = float(0.0)
    for side in range(2):
        shape = int(shapes0[c])
        point = wp.vec3(points0[c])
        sign = float(-1.0)
        if side == 1:
            shape, point, sign = int(shapes1[c]), wp.vec3(points1[c]), 1.0
        body = int(-1)
        edge = int(-1)
        if shape >= 0:
            body = int(shape_body[shape])
            ke += 0.5 * shape_ke[shape]
        world = point
        if body >= 0:
            world = wp.transform_point(body_q[body], point)
            edge = int(body_edge[body])
        ids[c, 2 * side] = 0
        ids[c, 2 * side + 1] = 0
        coefficients[c, 2 * side] = 0.0
        coefficients[c, 2 * side + 1] = 0.0
        if edge >= 0:
            t = wp.clamp((point[2] - endpoints[edge, 0][2]) / (endpoints[edge, 1][2] - endpoints[edge, 0][2]), 0.0, 1.0)
            a, b = int(edges[edge, 0]), int(edges[edge, 1])
            ids[c, 2 * side] = a
            ids[c, 2 * side + 1] = b
            coefficients[c, 2 * side] = sign * (1.0 - t)
            coefficients[c, 2 * side + 1] = sign * t
            # Preserve the support point at linearization, including its offset
            # from the centerline; do not replace it by a different contact.
            offset += sign * wp.dot(normal, world - ((1.0 - t) * x[a] + t * x[b]))
        else:
            offset += sign * wp.dot(normal, world)
    offsets[c] = offset
    stiffness[c] = ke


@wp.kernel
def cable_contact_evaluate(
    count: wp.array[int],
    ids: wp.array2d[int],
    coefficients: wp.array2d[float],
    offsets: wp.array[float],
    stiffness: wp.array[float],
    normals: wp.array[wp.vec3],
    x: wp.array[wp.vec3],
    energy: wp.array[float],
):
    c = wp.tid()
    if c >= count[0]:
        return
    gap = offsets[c]
    for slot in range(4):
        gap += coefficients[c, slot] * wp.dot(normals[c], x[ids[c, slot]])
    penetration = wp.min(gap, 0.0)
    wp.atomic_add(energy, 0, 0.5 * stiffness[c] * penetration * penetration)


@wp.func
def rotation_vector(q: wp.quat):
    r = wp.normalize(q)
    if r[3] < 0.0:
        r = -r
    v = wp.vec3(r[0], r[1], r[2])
    length = wp.length(v)
    return v * (2.0 * wp.atan2(length, r[3]) / wp.max(length, 1.0e-8))


@wp.func
def rotation_exp(v: wp.vec3):
    a = wp.length(v)
    return wp.quat(
        v[0] * wp.sin(0.5 * a) / wp.max(a, 1.0e-8),
        v[1] * wp.sin(0.5 * a) / wp.max(a, 1.0e-8),
        v[2] * wp.sin(0.5 * a) / wp.max(a, 1.0e-8),
        wp.cos(0.5 * a),
    )


@wp.func
def cable_residual(
    kind: int, a: wp.vec3, b: wp.vec3, s: wp.vec3, t: wp.vec3, r: wp.quat, q: wp.quat, rest: wp.quat, length: float
):
    # Let d=(c_b-c_a)/L, u=R_e*e_z, s_bar=(s_a+s_b)/2.
    # For angular rows, theta=Log(R_a^T R_b), theta0=Log(Q0).
    # Log returns a rotation vector in radians, expressed in frame a.
    value = wp.vec3(0.0)
    scale = 0.5 * (s[0] + t[0])
    if kind == 0:
        # Stretch/shear: C_0=d-u. Finite stiffness permits both modes.
        value = (b - a) / length - wp.quat_rotate(r, wp.vec3(0.0, 0.0, 1.0))
    elif kind == 1:
        # Volume-related stretch/shear: C_1=s_bar^2*d-u.
        # For an aligned segment, C_1=0 enforces area*length conservation.
        value = scale * scale * (b - a) / length - wp.quat_rotate(r, wp.vec3(0.0, 0.0, 1.0))
    elif kind == 2:
        # Radial restoration: C_2=(s_bar-1, 0, 0).
        value = wp.vec3(scale - 1.0, 0.0, 0.0)
    elif kind == 3:
        # Scale smoothness: C_3=((s_b-s_a)/L, 0, 0).
        value = wp.vec3((t[0] - s[0]) / length, 0.0, 0.0)
    elif kind == 4:
        # Bend/twist: C_4=(s_bar*theta-theta0)/L; x/y bend, z twists.
        # This is a difference of logs, NOT Log(Q0^T R_a^T R_b).
        # At fixed scale, E_4=0.5*(kb*|theta_xy-theta0_xy|^2
        #                              + kt*(theta_z-theta0_z)^2).
        value = (scale * rotation_vector(wp.quat_inverse(r) * q) - rotation_vector(rest)) / length
    elif kind == 5:
        # Volume-related bending: C_5=P_xy*(s_bar^3*theta-theta0)/L.
        # This extra term is omitted in fixed-section mode, along with C_1..C_3.
        value = (scale * scale * scale * rotation_vector(wp.quat_inverse(r) * q) - rotation_vector(rest)) / length
        value = wp.vec3(value[0], value[1], 0.0)
    return value


@wp.kernel
def eval_cable_viper_constraint_kernel(
    kinds: wp.array[int],
    ids: wp.array2d[int],
    lengths: wp.array[float],
    weights: wp.array[wp.vec3],
    rest: wp.array[wp.quat],
    fixed_scale: bool,
    x: wp.array[wp.vec3],
    frames: wp.array[wp.quat],
    active: wp.array[float],
    jac: wp.array2d[wp.mat33],
    rhs: wp.array[wp.vec3],
):
    c = wp.tid()
    a, b = wp.vec3(x[ids[c, 0]]), wp.vec3(x[ids[c, 1]])
    s, t = wp.vec3(x[ids[c, 2]]), wp.vec3(x[ids[c, 3]])
    if fixed_scale:
        s = wp.vec3(1.0, 0.0, 0.0)
        t = s
    r, q = wp.quat(frames[ids[c, 4]]), wp.quat(frames[ids[c, 5]])
    root = weights[c]
    # f=sqrt(W)*C, J_i=df/d(delta_i), rhs_i += -J_i^T*f.
    # Rotation increments are right-multiplied: R'=R*Exp(delta_theta).
    # Central differences below evaluate the implemented nonlinear residual.
    f = wp.cw_mul(root, cable_residual(kinds[c], a, b, s, t, r, q, rest[c], lengths[c]))
    for slot in range(6):
        if fixed_scale and (slot == 2 or slot == 3):
            jac[c, slot] = wp.mat33(0.0)
            continue
        block = wp.mat33(0.0)
        for axis in range(3):
            d = wp.vec3(0.0)
            d[axis] = 0.0005
            ap, bp, sp, tp, rp, qp = a, b, s, t, r, q
            am, bm, sm, tm, rm, qm = a, b, s, t, r, q
            if slot == 0:
                ap, am = a + d, a - d
            elif slot == 1:
                bp, bm = b + d, b - d
            elif slot == 2:
                sp, sm = s + d, s - d
            elif slot == 3:
                tp, tm = t + d, t - d
            elif slot == 4:
                rp, rm = r * rotation_exp(d), r * rotation_exp(-d)
            else:
                qp, qm = q * rotation_exp(d), q * rotation_exp(-d)
            fp = cable_residual(kinds[c], ap, bp, sp, tp, rp, qp, rest[c], lengths[c])
            fm = cable_residual(kinds[c], am, bm, sm, tm, rm, qm, rest[c], lengths[c])
            column = wp.cw_mul(root, (fp - fm) / 0.001) * active[ids[c, slot]]
            for row in range(3):
                block[row, axis] = column[row]
        jac[c, slot] = block
        wp.atomic_add(rhs, ids[c, slot], -(wp.transpose(block) * f))


@wp.kernel
def cable_local_diagonal(ids: wp.array2d[int], jac: wp.array2d[wp.mat33], diagonal: wp.array[wp.mat33]):
    # Block-Jacobi Gauss--Newton: H_ii += J_i^T*J_i. Repeated slots must
    # first be combined because they represent the same generalized variable.
    # These temporary mat33 blocks are local; they are NOT ELL matrix entries.
    c, slot = wp.tid()
    combined = wp.mat33(0.0)
    for other in range(6):
        if ids[c, other] == ids[c, slot]:
            combined += jac[c, other]
    wp.atomic_add(diagonal, ids[c, slot], wp.transpose(jac[c, slot]) * combined)


@wp.kernel
def cable_local_direction(n: int, diagonal: wp.array[wp.mat33], rhs: wp.array[wp.vec3], dx: wp.array[wp.vec3]):
    # Only scales and rotations use this local solve. Positions are updated
    # separately by the scalar global system, so their increments stay zero.
    i = wp.tid()
    dx[i] = wp.vec3(0.0)
    if i >= n:
        dx[i] = wp.inverse(diagonal[i]) * rhs[i]


@wp.kernel
def cable_position_inertia(
    x: wp.array[wp.vec3],
    predicted: wp.array[wp.vec3],
    mass: wp.array[wp.vec3],
    active: wp.array[float],
    dt: float,
    diagonal: wp.array[float],
    rhs: wp.array[wp.vec3],
):
    i = wp.tid()
    diagonal[i] = mass[i][0] / (dt * dt)
    rhs[i] = diagonal[i] * (predicted[i] - x[i]) * active[i]


@wp.kernel
def cable_position_targets(
    edges: wp.array2d[int],
    lengths: wp.array[float],
    axial: wp.array[float],
    bulk: wp.array[float],
    n: int,
    rotation_start: int,
    fixed_scale: bool,
    x: wp.array[wp.vec3],
    frames: wp.array[wp.quat],
    active: wp.array[float],
    weights: wp.array[float],
    rhs: wp.array[wp.vec3],
):
    e = wp.tid()
    a, b = edges[e, 0], edges[e, 1]
    scale = float(1.0)
    if not fixed_scale:
        scale = 0.5 * (x[n + a][0] + x[n + b][0])
    # Holding R,s fixed makes E_0+E_1 exactly quadratic in Delta=c_b-c_a:
    # E = wa/2*||Delta/L-u||^2 + wv/2*||s_bar^2*Delta/L-u||^2
    #   = k/2*||Delta||^2 - t.dot(Delta) + const,
    # k=(wa+wv*s_bar^4)/L^2, t=(wa+wv*s_bar^2)*u/L.
    # grad_b=k*Delta-t=-grad_a; the Hessian is k*[[1,-1],[-1,1]] x I3.
    factor = scale * scale
    w = (axial[e] + bulk[e] * factor * factor) / (lengths[e] * lengths[e])
    target = (
        (axial[e] + bulk[e] * factor) / lengths[e] * wp.quat_rotate(frames[rotation_start + e], wp.vec3(0.0, 0.0, 1.0))
    )
    gradient = w * (x[b] - x[a]) - target
    weights[e] = w
    wp.atomic_add(rhs, a, gradient * active[a])
    wp.atomic_add(rhs, b, -gradient * active[b])


@wp.kernel
def cable_contact_local(
    count: wp.array[int],
    ids: wp.array2d[int],
    coefficients: wp.array2d[float],
    offsets: wp.array[float],
    stiffness: wp.array[float],
    normals: wp.array[wp.vec3],
    x: wp.array[wp.vec3],
    active: wp.array[float],
    diagonal: wp.array[float],
    rhs: wp.array[wp.vec3],
):
    c = wp.tid()
    if c >= count[0]:
        return
    gap = offsets[c]
    norm = float(0.0)
    for slot in range(4):
        gap += coefficients[c, slot] * wp.dot(normals[c], x[ids[c, slot]])
        norm += wp.abs(coefficients[c, slot]) * active[ids[c, slot]]
    # Project the affine support difference onto its nonpenetrating halfspace.
    # Weighted Cauchy--Schwarz bounds k*A.T*A by this scalar diagonal.
    # This tangent upper bound avoids both anisotropic matrix blocks and a
    # post-solve position correction. It also bounds contacts that activate.
    correction = -wp.min(gap, 0.0) * normals[c]
    for slot in range(4):
        i = ids[c, slot]
        coeff = coefficients[c, slot] * active[i]
        wp.atomic_add(rhs, i, stiffness[c] * coeff * correction)
        wp.atomic_add(diagonal, i, stiffness[c] * wp.abs(coeff) * norm)


@wp.kernel
def cable_scalar_inverse(diagonal: wp.array[float], inverse: wp.array[float]):
    i = wp.tid()
    inverse[i] = 1.0 / diagonal[i]


@wp.kernel
def cable_inertia(
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
    # Local proposal metric M/h^2. For rotations the error is
    # Log(R_pred^T R); using I/h^2 as its Hessian is a local approximation.
    # Acceptance evaluates the nonlinear log-inertia energy below.
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
    # E_inertia = sum_nodes m/(2h^2)*||c-c_pred||^2
    #           + sum_scales m_s/(2h^2)*(s-s_pred)^2
    #           + sum_frames 1/(2h^2)*theta_pred^T*I*theta_pred,
    # theta_pred=Log(R_pred^T R). Positive scale is enforced by infinite energy.
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
def cable_sync_pins(
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
