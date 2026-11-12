# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Objective material damping with full matrix-free Gauss--Newton products.

Every row stores sqrt(kd * measure / dt) times the gradient of one metric
component or dihedral angle. Off-diagonal metric components receive sqrt(2),
matching VBD's Frobenius norm. The full operator is J.T @ J, not just its
vertex diagonal: common translations are therefore in its nullspace.
"""

import warp as wp

from ...geometry import ParticleFlags
from ..vbd.particle_vbd_kernels import compute_angle_derivative, compute_normalized_vector_derivative


@wp.func
def _write_row(
    row: int,
    count: int,
    ids: wp.vec4i,
    gradients: wp.mat44,
    residual: float,
    scale: float,
    local_majorizer: bool,
    row_indices: wp.array2d[int],
    row_jacobians: wp.array2d[wp.vec3],
    forces: wp.array[wp.vec3],
    hessian_diags: wp.array[wp.mat33],
):
    root_scale = wp.sqrt(scale)
    factor = float(1.0)
    if local_majorizer:
        # ||sum_i J_i dx_i||^2 <= n * sum_i ||J_i dx_i||^2.
        factor = float(count)
    for i in range(count):
        g = wp.vec3(gradients[i, 0], gradients[i, 1], gradients[i, 2])
        j = root_scale * g
        row_indices[row, i] = ids[i]
        row_jacobians[row, i] = j
        wp.atomic_add(forces, ids[i], -scale * residual * g)
        wp.atomic_add(hessian_diags, ids[i], factor * wp.outer(j, j))


@wp.kernel
def eval_tet_damping_kernel(
    pos: wp.array[wp.vec3],
    prev: wp.array[wp.vec3],
    indices: wp.array2d[int],
    poses: wp.array[wp.mat33],
    materials: wp.array2d[float],
    dt: float,
    row_offset: int,
    local_majorizer: bool,
    row_indices: wp.array2d[int],
    row_jacobians: wp.array2d[wp.vec3],
    forces: wp.array[wp.vec3],
    hessian_diags: wp.array[wp.mat33],
):
    tet, component = wp.tid()
    damping = materials[tet, 2]
    if damping <= 0.0:
        return
    ids = wp.vec4i(indices[tet, 0], indices[tet, 1], indices[tet, 2], indices[tet, 3])
    inv_dm = poses[tet]
    F = wp.matrix_from_cols(pos[ids[1]] - pos[ids[0]], pos[ids[2]] - pos[ids[0]], pos[ids[3]] - pos[ids[0]]) * inv_dm
    F_prev = wp.matrix_from_cols(
        prev[ids[1]] - prev[ids[0]], prev[ids[2]] - prev[ids[0]], prev[ids[3]] - prev[ids[0]]
    ) * inv_dm
    a = component
    b = component
    weight = float(1.0)
    if component >= 3:
        a = 0
        b = component - 2
        if component == 5:
            a = 1
            b = 2
        weight = 2.0
    fa = wp.vec3(F[0, a], F[1, a], F[2, a])
    fb = wp.vec3(F[0, b], F[1, b], F[2, b])
    fa_prev = wp.vec3(F_prev[0, a], F_prev[1, a], F_prev[2, a])
    fb_prev = wp.vec3(F_prev[0, b], F_prev[1, b], F_prev[2, b])
    residual = wp.dot(fa, fb) - wp.dot(fa_prev, fb_prev)
    gradients = wp.mat44(0.0)
    for i in range(4):
        m = -(inv_dm[0] + inv_dm[1] + inv_dm[2])
        if i > 0:
            m = inv_dm[i - 1]
        g = m[a] * fb + m[b] * fa
        for axis in range(3):
            gradients[i, axis] = g[axis]
    volume = 1.0 / (6.0 * wp.determinant(inv_dm))
    _write_row(
        row_offset + 6 * tet + component, 4, ids, gradients, residual, weight * damping * volume / dt,
        local_majorizer, row_indices, row_jacobians, forces, hessian_diags,
    )


@wp.kernel
def eval_stretch_damping_kernel(
    pos: wp.array[wp.vec3],
    prev: wp.array[wp.vec3],
    indices: wp.array2d[int],
    poses: wp.array[wp.mat22],
    areas: wp.array[float],
    materials: wp.array2d[float],
    dt: float,
    row_offset: int,
    local_majorizer: bool,
    row_indices: wp.array2d[int],
    row_jacobians: wp.array2d[wp.vec3],
    forces: wp.array[wp.vec3],
    hessian_diags: wp.array[wp.mat33],
):
    face, component = wp.tid()
    damping = materials[face, 2]
    if damping <= 0.0:
        return
    ids = wp.vec4i(indices[face, 0], indices[face, 1], indices[face, 2], -1)
    inv_dm = poses[face]
    a = component
    b = component
    weight = float(1.0)
    if component == 2:
        a = 0
        b = 1
        weight = 2.0
    fa = (pos[ids[1]] - pos[ids[0]]) * inv_dm[0, a] + (pos[ids[2]] - pos[ids[0]]) * inv_dm[1, a]
    fb = (pos[ids[1]] - pos[ids[0]]) * inv_dm[0, b] + (pos[ids[2]] - pos[ids[0]]) * inv_dm[1, b]
    fa_prev = (prev[ids[1]] - prev[ids[0]]) * inv_dm[0, a] + (prev[ids[2]] - prev[ids[0]]) * inv_dm[1, a]
    fb_prev = (prev[ids[1]] - prev[ids[0]]) * inv_dm[0, b] + (prev[ids[2]] - prev[ids[0]]) * inv_dm[1, b]
    gradients = wp.mat44(0.0)
    for i in range(3):
        m = -(inv_dm[0] + inv_dm[1])
        if i > 0:
            m = inv_dm[i - 1]
        g = m[a] * fb + m[b] * fa
        for axis in range(3):
            gradients[i, axis] = g[axis]
    _write_row(
        row_offset + 3 * face + component, 3, ids, gradients, wp.dot(fa, fb) - wp.dot(fa_prev, fb_prev),
        weight * damping * areas[face] / dt, local_majorizer, row_indices, row_jacobians, forces, hessian_diags,
    )


@wp.func
def _dihedral_data(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3):
    n1_raw = wp.cross(x2 - x0, x3 - x0)
    n2_raw = wp.cross(x3 - x1, x2 - x1)
    e_raw = x3 - x2
    n1_length, n2_length, e_length = wp.length(n1_raw), wp.length(n2_raw), wp.length(e_raw)
    # Match VBD's degeneracy threshold and angular convention.
    valid = n1_length >= 1.0e-6 and n2_length >= 1.0e-6 and e_length >= 1.0e-6
    n1, n2, e = wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0)
    angle = float(0.0)
    if valid:
        n1, n2, e = n1_raw / n1_length, n2_raw / n2_length, e_raw / e_length
        angle = wp.atan2(wp.dot(wp.cross(n1, n2), e), wp.dot(n1, n2))
    return valid, angle, n1, n2, e, n1_length, n2_length


@wp.kernel
def eval_bend_damping_kernel(
    pos: wp.array[wp.vec3],
    prev: wp.array[wp.vec3],
    indices: wp.array2d[int],
    rest_length: wp.array[float],
    properties: wp.array2d[float],
    dt: float,
    row_offset: int,
    local_majorizer: bool,
    row_indices: wp.array2d[int],
    row_jacobians: wp.array2d[wp.vec3],
    forces: wp.array[wp.vec3],
    hessian_diags: wp.array[wp.mat33],
):
    edge = wp.tid()
    ids = wp.vec4i(indices[edge, 0], indices[edge, 1], indices[edge, 2], indices[edge, 3])
    damping = properties[edge, 1]
    if ids[0] < 0 or ids[1] < 0 or damping <= 0.0:
        return
    x0, x1, x2, x3 = pos[ids[0]], pos[ids[1]], pos[ids[2]], pos[ids[3]]
    valid, angle, n1, n2, e_hat, n1_length, n2_length = _dihedral_data(x0, x1, x2, x3)
    valid_prev, angle_prev, _n1, _n2, _e, _l1, _l2 = _dihedral_data(
        prev[ids[0]], prev[ids[1]], prev[ids[2]], prev[ids[3]]
    )
    if not valid or not valid_prev:
        return
    residual = angle - angle_prev
    if residual > wp.pi:
        residual -= 2.0 * wp.pi
    elif residual < -wp.pi:
        residual += 2.0 * wp.pi
    sin_theta, cos_theta = wp.dot(wp.cross(n1, n2), e_hat), wp.dot(n1, n2)
    gradients = wp.mat44(0.0)
    for i in range(4):
        dn1, dn2 = wp.mat33(0.0), wp.mat33(0.0)
        if i == 0:
            dn1 = compute_normalized_vector_derivative(n1_length, n1, wp.skew(x3 - x2))
        elif i == 1:
            dn2 = compute_normalized_vector_derivative(n2_length, n2, -wp.skew(x3 - x2))
        elif i == 2:
            dn1 = compute_normalized_vector_derivative(n1_length, n1, -wp.skew(x3 - x0))
            dn2 = compute_normalized_vector_derivative(n2_length, n2, wp.skew(x3 - x1))
        else:
            dn1 = compute_normalized_vector_derivative(n1_length, n1, wp.skew(x2 - x0))
            dn2 = compute_normalized_vector_derivative(n2_length, n2, -wp.skew(x2 - x1))
        g = compute_angle_derivative(n1, n2, e_hat, dn1, dn2, sin_theta, cos_theta, wp.skew(n1), wp.skew(n2))
        for axis in range(3):
            gradients[i, axis] = g[axis]
    _write_row(
        row_offset + edge, 4, ids, gradients, residual, damping * rest_length[edge] / dt,
        local_majorizer, row_indices, row_jacobians, forces, hessian_diags,
    )


@wp.kernel
def damping_hessian_multiply_kernel(
    row_indices: wp.array2d[int],
    row_jacobians: wp.array2d[wp.vec3],
    x: wp.array[wp.vec3],
    flags: wp.array[int],
    out: wp.array[wp.vec3],
):
    row = wp.tid()
    value = float(0.0)
    for i in range(4):
        vertex = row_indices[row, i]
        if vertex >= 0 and flags[vertex] & ParticleFlags.ACTIVE:
            value += wp.dot(row_jacobians[row, i], x[vertex])
    for i in range(4):
        vertex = row_indices[row, i]
        if vertex >= 0 and flags[vertex] & ParticleFlags.ACTIVE:
            wp.atomic_add(out, vertex, row_jacobians[row, i] * value)


class MaterialDamping:
    """Cache damping rows at each nonlinear iterate, independently of contacts.

    ``assemble`` adds force to the solver RHS and resets the cached rows and
    vertex diagonals. ``multiply`` returns the complete Gauss--Newton product.
    Reference positions remain those at
    the beginning of the substep across all nonlinear iterations.
    """

    def __init__(self, model):
        self.model = model
        self.tet_rows = 6 * model.tet_count
        self.tri_rows = 3 * model.tri_count
        self.row_count = self.tet_rows + self.tri_rows + model.edge_count
        self.row_indices = wp.full((self.row_count, 4), -1, dtype=int, device=model.device)
        self.row_jacobians = wp.zeros((self.row_count, 4), dtype=wp.vec3, device=model.device)
        self.diagonal = wp.zeros(model.particle_count, dtype=wp.mat33, device=model.device)
        self.Hx = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        self._flags = model.particle_flags

    def assemble(self, pos, pos_prev, dt, forces, flags, *, surface, tets, local):
        self.row_jacobians.zero_()
        self.diagonal.zero_()
        self._flags = flags
        model = self.model
        outputs = [self.row_indices, self.row_jacobians, forces, self.diagonal]
        if tets and self.tet_rows:
            wp.launch(
                eval_tet_damping_kernel, dim=(model.tet_count, 6),
                inputs=[pos, pos_prev, model.tet_indices, model.tet_poses, model.tet_materials,
                        dt, 0, local], outputs=outputs, device=model.device,
            )
        if surface and self.tri_rows:
            wp.launch(
                eval_stretch_damping_kernel, dim=(model.tri_count, 3),
                inputs=[pos, pos_prev, model.tri_indices, model.tri_poses, model.tri_areas, model.tri_materials,
                        dt, self.tet_rows, local], outputs=outputs, device=model.device,
            )
        if surface and model.edge_count:
            wp.launch(
                eval_bend_damping_kernel, dim=model.edge_count,
                inputs=[pos, pos_prev, model.edge_indices, model.edge_rest_length, model.edge_bending_properties,
                        dt, self.tet_rows + self.tri_rows, local], outputs=outputs, device=model.device,
            )

    def multiply(self, x):
        self.Hx.zero_()
        if self.row_count:
            wp.launch(
                damping_hessian_multiply_kernel, dim=self.row_count,
                inputs=[self.row_indices, self.row_jacobians, x, self._flags], outputs=[self.Hx], device=self.model.device,
            )
        return self.Hx


@wp.kernel
def add_diagonal_kernel(a: wp.array[wp.mat33], b: wp.array[wp.mat33], out: wp.array[wp.mat33]):
    i = wp.tid()
    out[i] = a[i] + b[i]


@wp.kernel
def add_vectors_kernel(a: wp.array[wp.vec3], b: wp.array[wp.vec3], out: wp.array[wp.vec3]):
    i = wp.tid()
    out[i] = a[i] + b[i]
