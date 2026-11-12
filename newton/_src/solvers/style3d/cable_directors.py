# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Experimental fixed-section director global solve, using float ELL entries."""

import numpy as np
import warp as wp

from .builder import PDMatrixBuilder
from .cable_viper_kernels import cable_scalar_inverse, rotation_vector
from .linear_solver import PcgSolver, SparseMatrixELL


def director_weights(bend, twist):
    """Match straight-rest small-angle bending and twisting stiffness."""
    # With Q0=I and R_b=R_a*Exp(theta),
    # E_D ~= 0.5*((beta_y+beta_z)*theta_x^2
    #          + (beta_x+beta_z)*theta_y^2 + (beta_x+beta_y)*theta_z^2).
    # Matching diag(kb,kb,kt) gives beta=(kt/2,kt/2,kb-kt/2).
    # Nonnegative weights require kt<=2kb. This matches only the straight-rest
    # small-angle Hessian, not arbitrary-angle or curved-rest log-strain energy.
    if not np.isfinite([bend, twist]).all() or bend < 0 or twist < 0 or twist > 2 * bend:
        raise ValueError("Global directors require 0 <= twist stiffness <= 2 * bend stiffness.")
    return np.array([twist / 2, twist / 2, bend - twist / 2], dtype=np.float32)


@wp.kernel
def pack_directors(frames: wp.array[wp.quat], start: int, directors: wp.array[wp.vec3]):
    i = wp.tid()
    axis = wp.vec3(0.0)
    axis[i % 3] = 1.0
    directors[i] = wp.quat_rotate(frames[start + i // 3], axis)


@wp.kernel
def director_system(
    directors: wp.array[wp.vec3],
    predicted: wp.array[wp.quat],
    x: wp.array[wp.vec3],
    edges: wp.array2d[int],
    lengths: wp.array[float],
    axial: wp.array[float],
    mass: wp.array[wp.vec3],
    active: wp.array[float],
    start: int,
    dt: float,
    base_diagonal: wp.array[float],
    proximal: wp.array[float],
    diagonal: wp.array[float],
    rhs: wp.array[wp.vec3],
):
    i = wp.tid()
    edge, axis = i // 3, i % 3
    enabled = active[start + edge]
    unit = wp.vec3(0.0)
    unit[axis] = 1.0
    inertia = mass[start + edge]
    # Proposal inertia: E_D,inertia=0.5*sum_axis mu_axis*||d_axis-d_pred_axis||^2.
    # Its small-angle angular metric is tr(B)*I3-B, B=diag(mu*h^2).
    # Therefore mu_axis=(tr(I)/2-I_axis)/h^2. Acceptance still uses log inertia.
    mu = wp.max(0.0, 0.5 * (inertia[0] + inertia[1] + inertia[2]) - inertia[axis]) / (dt * dt)
    rhs[i] = enabled * mu * (wp.quat_rotate(predicted[start + edge], unit) - directors[i])
    diagonal[i] = base_diagonal[i] + proximal[i] + enabled * mu
    if axis == 2:
        # E_stretch=wa/2*||(c_b-c_a)/L-d_z||^2 contributes wa to A_zz.
        target = (x[edges[edge, 1]] - x[edges[edge, 0]]) / lengths[edge]
        rhs[i] += enabled * axial[edge] * (target - directors[i])
        diagonal[i] += enabled * axial[edge]


@wp.kernel
def stencil_gradient(
    ids: wp.array2d[int],
    coefficients: wp.array2d[float],
    weights: wp.array[float],
    directors: wp.array[wp.vec3],
    active: wp.array[float],
    rhs: wp.array[wp.vec3],
):
    c = wp.tid()
    residual = wp.vec3(0.0)
    for slot in range(4):
        residual += coefficients[c, slot] * directors[ids[c, slot]]
    for slot in range(4):
        i = ids[c, slot]
        wp.atomic_add(rhs, i, -weights[c] * coefficients[c, slot] * residual * active[i])


@wp.kernel
def stencil_energy(
    ids: wp.array2d[int],
    coefficients: wp.array2d[float],
    weights: wp.array[float],
    frames: wp.array[wp.quat],
    start: int,
    energy: wp.array[float],
):
    c = wp.tid()
    residual = wp.vec3(0.0)
    for slot in range(4):
        i = ids[c, slot]
        axis = wp.vec3(0.0)
        axis[i % 3] = 1.0
        residual += coefficients[c, slot] * wp.quat_rotate(frames[start + i // 3], axis)
    wp.atomic_add(energy, 0, 0.5 * weights[c] * wp.dot(residual, residual))


@wp.kernel
def project_directors(
    directors: wp.array[wp.vec3],
    increment: wp.array[wp.vec3],
    frames: wp.array[wp.quat],
    start: int,
    active: wp.array[float],
    dx: wp.array[wp.vec3],
):
    e = wp.tid()
    matrix = wp.matrix_from_cols(
        directors[3 * e] + increment[3 * e],
        directors[3 * e + 1] + increment[3 * e + 1],
        directors[3 * e + 2] + increment[3 * e + 2],
    )
    # The unconstrained columns need not form a rotation. Closest proper frame:
    # R*=U*diag(1,1,det(U V^T))*V^T = argmin_{R in SO(3)} ||R-D||_F^2.
    # This projection is Euclidean, not an exact constrained energy minimizer.
    u, _sigma, v = wp.svd3(matrix)
    correction = wp.identity(n=3, dtype=float)
    if wp.determinant(u * wp.transpose(v)) < 0.0:
        correction[2, 2] = -1.0
    q = wp.quat_from_matrix(u * correction * wp.transpose(v))
    dx[start + e] = active[start + e] * rotation_vector(wp.quat_inverse(frames[start + e]) * q)


class CableDirectors:
    """Globally coupled director proposals followed by SO(3) projection.

    The owner accepts proposals against chord elasticity and logarithmic
    inertia. This is a projected iteration, not an exact minimizer on SO(3).
    """

    def __init__(self, cable, kinds, ids, weights, lengths, rests):
        self.cable = cable
        n = 3 * len(cable.bodies)
        stencils, coefficients, stiffness = [], [], []
        for kind, row, root_weight, length, rest in zip(kinds, ids, weights, lengths, rests, strict=True):
            if kind != 4:
                continue
            a, b = row[4] - cable.rotation_start, row[5] - cable.rotation_start
            bend, twist = root_weight[0] ** 2 / length**2, root_weight[2] ** 2 / length**2
            beta = director_weights(bend, twist)
            reference = np.array(wp.quat_to_matrix(wp.quat(*rest))).reshape(3, 3)
            # d_e^alpha=R_e*e_alpha. Replace log rotation energy by
            # E_D=0.5*sum_alpha beta_alpha*||d_b^alpha-sum_j Q0[j,alpha]*d_a^j||^2.
            # Each residual is a scalar stencil on four vec3 unknowns:
            # a=[1,-Q0[0,alpha],-Q0[1,alpha],-Q0[2,alpha]].
            # Hessian entries are beta*a_i*a_j (each implicitly times I3).
            for axis in range(3):
                stencils.append([3 * b + axis, 3 * a, 3 * a + 1, 3 * a + 2])
                coefficients.append([1.0, *(-reference[:, axis])])
                stiffness.append(beta[axis])
        device = cable.device
        active = np.repeat(cable.active.numpy()[cable.rotation_start :], 3)
        neighbors = [set() for _ in range(n)]
        for row in stencils:
            for i in row:
                neighbors[i].update(j for j in row if j != i)
        builder = PDMatrixBuilder(n, max_neighbor=max(1, *map(len, neighbors)))
        builder.add_scalar_stencils(stencils, coefficients, stiffness, active=active)
        self.matrix = SparseMatrixELL()
        self.base_diagonal, self.matrix.num_nz, self.matrix.nz_ell = builder.finalize(device)
        self.ids = wp.array(stencils, dtype=int, device=device)
        self.coefficients = wp.array(coefficients, dtype=float, device=device)
        self.weights = wp.array(stiffness, dtype=float, device=device)
        self.active = wp.array(active, dtype=float, device=device)
        # A proximal term controls the unconstrained director proposal. It is
        # a numerical parameter, not an extra physical material stiffness.
        # Add rho/2*||delta_d||^2 to the proposal: A += rho*I, rhs unchanged.
        # rho is excluded from the physical energy checked by the line search.
        per_edge = self.base_diagonal.numpy().reshape(-1, 3).max(axis=1) + cable.axial.numpy()
        rho = np.repeat(np.maximum(per_edge, 1e-6), 3) * cable.config.cable_viper_rotation_proximal
        self.proximal = wp.array(rho, dtype=float, device=device)
        self.directors, self.increment, self.rhs = [wp.zeros(n, dtype=wp.vec3, device=device) for _ in range(3)]
        self.diagonal, self.inverse = [wp.zeros(n, dtype=float, device=device) for _ in range(2)]
        self.pcg = PcgSolver(n, device)

    def step(self, dt):
        c = self.cable
        c.launch(pack_directors, len(self.directors), [c.frames, c.rotation_start, self.directors])
        c.launch(
            director_system,
            len(self.directors),
            [
                self.directors,
                c.predicted_frames,
                c.x,
                c.edges,
                c.segment_lengths,
                c.axial,
                c.mass,
                c.active,
                c.rotation_start,
                dt,
                self.base_diagonal,
                self.proximal,
                self.diagonal,
                self.rhs,
            ],
        )
        c.launch(
            stencil_gradient,
            len(self.ids),
            [self.ids, self.coefficients, self.weights, self.directors, self.active, self.rhs],
        )
        c.launch(cable_scalar_inverse, len(self.directors), [self.diagonal, self.inverse])
        self.pcg.solve(
            self.matrix, self.diagonal, None, self.rhs, self.inverse, self.increment, c.config.linear_iterations
        )
        c.dx.zero_()
        c.launch(
            project_directors,
            len(c.bodies),
            [self.directors, self.increment, c.frames, c.rotation_start, c.active, c.dx],
        )
        c.apply_descent_step(dt)

    def energy(self, frames, energy):
        self.cable.launch(
            stencil_energy,
            len(self.ids),
            [self.ids, self.coefficients, self.weights, frames, self.cable.rotation_start, energy],
        )
