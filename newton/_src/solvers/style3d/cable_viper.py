# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Newton capsule/ROD adapter for the Style3D VIPER constraint path."""

import numpy as np
import warp as wp

from ...geometry import GeoType
from ...sim import JointType
from . import cable_viper_kernels as k
from .builder import PDMatrixBuilder
from .cable_directors import CableDirectors
from .linear_solver import PcgSolver


class CableViper:
    """Device-resident alternating position and material solve using the Style3D PCG infrastructure.

    Initialization reads topology on the host. All stepping runs Warp kernels.
    This experimental path currently accepts pure capsule rod models and static
    scenery. Contacts use frictionless linearized penalties with fixed collision
    radii. External forces, general joints and Dahl are not supported.
    """

    def __init__(self, model, config):
        # Generalized variables: shared nodes c_i, segment frames R_e, and
        # optional nodal scales s_i. Fixed section removes s and its velocity;
        # it does not impose an inextensibility constraint on c.
        self.model, self.config, self.device = model, config, model.device
        self.fixed_scale = config.cable_viper_fixed_scale
        self.rotation_solver = config.cable_viper_rotation_solver
        if self.rotation_solver not in ("local", "global"):
            raise ValueError("cable_viper_rotation_solver must be local or global.")
        if self.rotation_solver == "global" and not self.fixed_scale:
            raise ValueError("Experimental global directors currently require cable_viper_fixed_scale=True.")
        if not np.isfinite(config.cable_viper_rotation_proximal) or config.cable_viper_rotation_proximal <= 0:
            raise ValueError("Director proximal multiplier must be finite and positive.")
        if (
            not np.isfinite(config.cable_viper_volume_ratio)
            or config.cable_viper_volume_ratio < 0
            or not np.isfinite(config.cable_viper_radial_ratio)
            or config.cable_viper_radial_ratio < 0
            or not 0 < config.cable_viper_relaxation <= 1
            or config.linear_iterations < 1
        ):
            raise ValueError("Invalid Cable_Viper material, relaxation, or iteration count.")
        types = model.joint_type.numpy()
        rod_joints = np.flatnonzero(types == int(JointType.ROD))
        if not len(rod_joints):
            raise ValueError("Cable_Viper requires ROD joints.")
        if model.particle_count or np.any((types != int(JointType.ROD)) & (types != int(JointType.FREE))):
            raise ValueError("Cable_Viper currently supports rod-only models with ROD/FREE joints.")
        parents, children = model.joint_parent.numpy(), model.joint_child.numpy()
        bodies = sorted(set(parents[rod_joints]) | set(children[rod_joints]))
        if -1 in bodies:
            raise ValueError("World ROD joints are unsupported; pin a zero-mass capsule instead.")
        mapping = {b: i for i, b in enumerate(bodies)}
        shape_body = model.shape_body.numpy()
        scale, shape_q = model.shape_scale.numpy(), model.shape_transform.numpy()
        body_q, masses = model.body_q.numpy(), model.body_mass.numpy()
        if any(masses[b] > 0 and b not in mapping for b in range(model.body_count)):
            raise ValueError("Cable_Viper does not integrate unrelated dynamic rigid bodies.")
        endpoints, centers, lengths, radii = [], [], [], []
        for b in bodies:
            shapes = np.flatnonzero(shape_body == b)
            if len(shapes) != 1:
                raise ValueError("Each rod body must own exactly one capsule.")
            shape = shapes[0]
            if model.shape_type.numpy()[shape] != int(GeoType.CAPSULE):
                raise ValueError("Cable_Viper requires capsule geometry.")
            center = shape_q[shape, :3]
            length = 2 * scale[shape, 1]
            if length <= 0:
                raise ValueError("Capsule length must be positive.")
            offset = np.array([0, 0, length / 2])
            endpoints.append([center - offset, center + offset])
            centers.append(center)
            lengths.append(length)
            radii.append(scale[shape, 0])
        lengths, radii, endpoints = np.array(lengths), np.array(radii), np.array(endpoints)
        union = list(range(2 * len(bodies)))

        def root(i):
            while union[i] != i:
                i = union[i]
            return i

        xp, xc = model.joint_X_p.numpy(), model.joint_X_c.numpy()
        for j in rod_joints:
            a, b = mapping[parents[j]], mapping[children[j]]
            ai = int(np.argmin(np.linalg.norm(endpoints[a] - xp[j, :3], axis=1)))
            bi = int(np.argmin(np.linalg.norm(endpoints[b] - xc[j, :3], axis=1)))
            union[root(2 * b + bi)] = root(2 * a + ai)
        roots = [root(i) for i in range(len(union))]
        nodes = {r: i for i, r in enumerate(sorted(set(roots)))}
        edges = np.array([[nodes[roots[2 * i]], nodes[roots[2 * i + 1]]] for i in range(len(bodies))])
        n, ne = len(nodes), len(bodies)
        rotation_start = n if self.fixed_scale else 2 * n
        self.n, self.rotation_start, self.size = n, rotation_start, rotation_start + ne
        x = np.zeros((self.size, 3), dtype=np.float32)
        frames = np.tile([0, 0, 0, 1], (self.size, 1)).astype(np.float32)
        counts = np.zeros(n)
        mass = np.ones((self.size, 3), dtype=np.float32) * 1e-8
        active = np.ones(self.size, dtype=np.float32)
        for e, b in enumerate(bodies):
            frames[rotation_start + e] = body_q[b, 3:]
            for endpoint, node in enumerate(edges[e]):
                x[node] += np.array(wp.transform_point(wp.transform(*body_q[b]), wp.vec3(*endpoints[e, endpoint])))
                counts[node] += 1
                mass[node] += masses[b] / 2
                if not self.fixed_scale:
                    mass[n + node, 0] += max(masses[b], 1e-6) * radii[e] ** 2 / 4
                if masses[b] == 0:
                    active[node] = 0
            # Cross-section disc inertia I=m*r^2*diag(1/4,1/4,1/2).
            # Endpoint masses also contribute to transverse motion; this is
            # not the full rigid capsule inertia used by VBD.
            mass[rotation_start + e] = max(masses[b], 1e-6) * radii[e] ** 2 * np.array([0.25, 0.25, 0.5])
            if masses[b] == 0:
                active[rotation_start + e] = 0
        x[:n] /= counts[:, None]
        if not self.fixed_scale:
            x[n : 2 * n, 0] = 1
        kinds, ids, lens, weights, rests = [], [], [], [], []

        def add(kind, row, length, weight, rest=(0, 0, 0, 1)):
            # All energies use 0.5*C^T*W*C; kernels receive sqrt(W).
            if self.fixed_scale and kind in (1, 2, 3, 5):
                return
            kinds.append(kind)
            ids.append(row)
            lens.append(length)
            weights.append(np.sqrt(np.broadcast_to(weight, (3,))))
            rests.append(rest)

        gains = model.joint_target_ke.numpy()
        starts = model.joint_qd_start.numpy()
        qstarts, rest_q = model.joint_q_start.numpy(), model.joint_rest_q.numpy()
        for e, (a, b) in enumerate(edges):
            incident = [j for j in rod_joints if parents[j] == bodies[e] or children[j] == bodies[e]]
            gain = np.mean([gains[starts[j] + 2] for j in incident])
            # Per-segment mapping from incident joint axial stiffness k:
            # W0=k*L^2, W1=W0*volume_ratio, W2=W0*radial_ratio,
            # W3=W2*r^2/4. L^2 cancels the 1/L^2 in C_0 and C_1.
            # This is the current adapter convention, not an EI/EA continuum fit.
            axial = gain * lengths[e] ** 2
            bulk = axial * config.cable_viper_volume_ratio
            row = [
                a,
                b,
                0 if self.fixed_scale else n + a,
                0 if self.fixed_scale else n + b,
                rotation_start + e,
                rotation_start + e,
            ]
            add(0, row, lengths[e], axial)
            add(1, row, lengths[e], bulk)
            add(2, row, lengths[e], [axial * config.cable_viper_radial_ratio, 0, 0])
            add(3, row, lengths[e], [axial * config.cable_viper_radial_ratio * radii[e] ** 2 / 4, 0, 0])
        for j in rod_joints:
            a, b = mapping[parents[j]], mapping[children[j]]
            qa, qb = wp.quat(*xp[j, 3:]), wp.quat(*xc[j, 3:])
            # Q0 converts authored joint rest rotation to the relative material
            # frame: Q0 = Q_parent_anchor * Q_joint_rest * Q_child_anchor^-1.
            rest = qa * wp.quat(*rest_q[qstarts[j] + 3 : qstarts[j] + 7]) * wp.quat_inverse(qb)
            length = (lengths[a] + lengths[b]) / 2
            row = [
                edges[a, 0],
                edges[b, 1],
                0 if self.fixed_scale else n + edges[a, 1],
                0 if self.fixed_scale else n + edges[b, 0],
                rotation_start + a,
                rotation_start + b,
            ]
            bend, twist = gains[starts[j] + 3], gains[starts[j] + 5]
            # W4=L_dual^2*diag(kb,kb,kt); W5=L_dual^2*volume_ratio*diag(kb,kb,0).
            # These are per-joint stiffnesses: the dual length cancels in E4/E5.
            add(4, row, length, np.array([bend, bend, twist]) * length**2, rest)
            add(5, row, length, np.array([bend, bend, 0]) * length**2 * config.cable_viper_volume_ratio, rest)
        self.count = len(kinds)

        def array(data, dtype):
            return wp.array(data, dtype=dtype, device=self.device)

        self.kinds, self.ids = array(kinds, int), wp.array2d(ids, dtype=int, device=self.device)
        self.lengths, self.weights, self.rest = array(lens, float), array(weights, wp.vec3), array(rests, wp.quat)
        self.x, self.frames, self.mass, self.active = (
            array(x, wp.vec3),
            array(frames, wp.quat),
            array(mass, wp.vec3),
            array(active, float),
        )
        self.bodies, self.edges = array(bodies, int), wp.array2d(edges, dtype=int, device=self.device)
        body_edge = np.full(model.body_count, -1, dtype=int)
        body_edge[bodies] = np.arange(ne)
        self.body_edge = array(body_edge, int)
        self.endpoints = wp.array2d(endpoints, dtype=wp.vec3, device=self.device)
        self.centers = array(centers, wp.vec3)
        self.segment_lengths = array(lengths, float)
        self.velocity = wp.zeros(self.size, dtype=wp.vec3, device=self.device)
        self.old, self.predicted = wp.clone(self.x), wp.clone(self.x)
        self.old_frames, self.predicted_frames = wp.clone(self.frames), wp.clone(self.frames)
        self.dx = wp.zeros_like(self.x)
        if self.rotation_solver == "local":
            self.rhs = wp.zeros_like(self.x)
            self.diagonal = wp.zeros(self.size, dtype=wp.mat33, device=self.device)
            self.jacobian = wp.zeros((self.count, 6), dtype=wp.mat33, device=self.device)
        self.matrix_builder = PDMatrixBuilder(n)
        self.matrix, self.edge_slots, self.ell_values = self.matrix_builder.build_rod_pattern(edges, device=self.device)
        self.edge_weights = wp.zeros(ne, dtype=float, device=self.device)
        edge_axial = np.array([weights[i][0] ** 2 for i, kind in enumerate(kinds) if kind == 0])
        self.axial = array(edge_axial, float)
        self.bulk = array(np.zeros(ne) if self.fixed_scale else edge_axial * config.cable_viper_volume_ratio, float)
        self.position_diagonal = wp.zeros(n, dtype=float, device=self.device)
        self.position_inverse = wp.zeros(n, dtype=float, device=self.device)
        self.position_rhs = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.position_dx = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.contacts = None
        self.contact_capacity = -1
        self.pcg = PcgSolver(n, self.device)
        self.trial_x, self.best_x = wp.clone(self.x), wp.clone(self.x)
        self.trial_frames, self.best_frames = wp.clone(self.frames), wp.clone(self.frames)
        self.energy, self.best_energy = [wp.zeros(1, dtype=float, device=self.device) for _ in range(2)]
        self.director_solver = (
            CableDirectors(self, kinds, ids, weights, lens, rests) if self.rotation_solver == "global" else None
        )
        # Keep local log-strain constraints intact; only the experimental
        # energy evaluation substitutes the explicit director quadratic.
        self.energy_weights = self.weights
        if self.director_solver is not None:
            global_energy_weights = np.array(weights).copy()
            global_energy_weights[np.asarray(kinds) == 4] = 0
            self.energy_weights = array(global_energy_weights, wp.vec3)

    def launch(self, kernel, dim, inputs):
        wp.launch(kernel, dim=dim, inputs=inputs, device=self.device)

    def evaluate_energy(self, x, frames, dt, energy):
        # Objective for every line search: E_inertia + E_elastic + E_contact.
        # Local mode uses C_4 (log strain). Global mode zeroes its weight and
        # substitutes director chord energy; it does not count both energies.
        energy.zero_()
        self.launch(
            k.cable_elastic_energy,
            self.count,
            [self.kinds, self.ids, self.lengths, self.energy_weights, self.rest, self.fixed_scale, x, frames, energy],
        )
        if self.director_solver is not None:
            self.director_solver.energy(frames, energy)
        self.launch(
            k.cable_inertial_energy,
            self.size,
            [
                x,
                frames,
                self.predicted,
                self.predicted_frames,
                self.mass,
                self.active,
                self.n,
                self.rotation_start,
                dt,
                energy,
            ],
        )

        self.evaluate_contacts(x, energy)

    def prepare_contacts(self, contacts, state):
        self.contacts = contacts
        if contacts is None:
            return
        capacity = len(contacts.rigid_contact_shape0)
        if capacity != self.contact_capacity:
            self.contact_capacity = capacity
            self.contact_ids = wp.zeros((capacity, 4), dtype=int, device=self.device)
            self.contact_coefficients = wp.zeros((capacity, 4), dtype=float, device=self.device)
            self.contact_offsets = wp.zeros(capacity, dtype=float, device=self.device)
            self.contact_stiffness = wp.zeros(capacity, dtype=float, device=self.device)
        self.launch(
            k.cable_contact_terms,
            capacity,
            [
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                self.model.shape_body,
                self.model.shape_material_ke,
                self.body_edge,
                state.body_q,
                self.edges,
                self.endpoints,
                self.x,
                self.contact_ids,
                self.contact_coefficients,
                self.contact_offsets,
                self.contact_stiffness,
            ],
        )

    def evaluate_contacts(self, x, energy):
        if self.contacts is not None:
            self.launch(
                k.cable_contact_evaluate,
                self.contact_capacity,
                [
                    self.contacts.rigid_contact_count,
                    self.contact_ids,
                    self.contact_coefficients,
                    self.contact_offsets,
                    self.contact_stiffness,
                    self.contacts.rigid_contact_normal,
                    x,
                    energy,
                ],
            )

    def apply_descent_step(self, dt):
        # All candidates share the same linearization. Retain the best feasible
        # energy, including the zero step, without host synchronization.
        wp.copy(self.best_x, self.x)
        wp.copy(self.best_frames, self.frames)
        self.evaluate_energy(self.x, self.frames, dt, self.best_energy)
        for backtrack in range(8):
            alpha = self.config.cable_viper_relaxation * 0.5**backtrack
            self.launch(
                k.cable_trial,
                self.size,
                [
                    self.x,
                    self.frames,
                    self.dx,
                    self.active,
                    self.rotation_start,
                    alpha,
                    self.trial_x,
                    self.trial_frames,
                ],
            )
            self.evaluate_energy(self.trial_x, self.trial_frames, dt, self.energy)
            self.launch(
                k.cable_accept,
                self.size,
                [self.trial_x, self.trial_frames, self.energy, self.best_energy, self.best_x, self.best_frames],
            )
            self.launch(k.cable_accept_energy, 1, [self.energy, self.best_energy])
        wp.copy(self.x, self.best_x)
        wp.copy(self.frames, self.best_frames)

    def step(self, state_in, state_out, dt, contacts=None):
        if self.config.cable_viper_fixed_scale != self.fixed_scale:
            raise ValueError("Recreate SolverStyle3D after changing cable_viper_fixed_scale.")
        if self.config.cable_viper_rotation_solver != self.rotation_solver:
            raise ValueError("Recreate SolverStyle3D after changing cable_viper_rotation_solver.")
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError("dt must be positive and finite.")
        self.launch(
            k.cable_sync_pins,
            len(self.bodies),
            [
                self.bodies,
                self.edges,
                self.endpoints,
                self.model.body_mass,
                state_in.body_q,
                self.rotation_start,
                self.x,
                self.frames,
            ],
        )
        self.prepare_contacts(contacts, state_in)
        wp.copy(self.old, self.x)
        wp.copy(self.old_frames, self.frames)
        self.launch(
            k.cable_predict,
            self.size,
            [
                self.x,
                self.frames,
                self.velocity,
                self.active,
                self.n,
                self.rotation_start,
                self.model.gravity,
                dt,
                self.predicted,
                self.predicted_frames,
            ],
        )
        for _ in range(self.config.iterations):
            # Alternating minimization: propose R/s with c fixed, then solve c
            # with R/s fixed. Both proposals pass the same energy line search.
            if self.director_solver is not None:
                self.director_solver.step(dt)
            else:
                self.launch(
                    k.cable_inertia,
                    self.size,
                    [
                        self.x,
                        self.frames,
                        self.predicted,
                        self.predicted_frames,
                        self.mass,
                        self.active,
                        self.rotation_start,
                        dt,
                        self.rhs,
                        self.diagonal,
                    ],
                )
                self.launch(
                    k.eval_cable_viper_constraint_kernel,
                    self.count,
                    [
                        self.kinds,
                        self.ids,
                        self.lengths,
                        self.weights,
                        self.rest,
                        self.fixed_scale,
                        self.x,
                        self.frames,
                        self.active,
                        self.jacobian,
                        self.rhs,
                    ],
                )
                self.launch(k.cable_local_diagonal, (self.count, 6), [self.ids, self.jacobian, self.diagonal])
                self.launch(k.cable_local_direction, self.size, [self.n, self.diagonal, self.rhs, self.dx])
                self.apply_descent_step(dt)
            self.launch(
                k.cable_position_inertia,
                self.n,
                [self.x, self.predicted, self.mass, self.active, dt, self.position_diagonal, self.position_rhs],
            )
            self.launch(
                k.cable_position_targets,
                len(self.bodies),
                [
                    self.edges,
                    self.segment_lengths,
                    self.axial,
                    self.bulk,
                    self.n,
                    self.rotation_start,
                    self.fixed_scale,
                    self.x,
                    self.frames,
                    self.active,
                    self.edge_weights,
                    self.position_rhs,
                ],
            )
            self.matrix_builder.assemble_rod_matrix(
                self.matrix,
                self.edge_slots,
                self.edges,
                self.edge_weights,
                self.active,
                self.position_diagonal,
                self.ell_values,
            )
            if self.contacts is not None:
                self.launch(
                    k.cable_contact_local,
                    self.contact_capacity,
                    [
                        self.contacts.rigid_contact_count,
                        self.contact_ids,
                        self.contact_coefficients,
                        self.contact_offsets,
                        self.contact_stiffness,
                        self.contacts.rigid_contact_normal,
                        self.x,
                        self.active,
                        self.position_diagonal,
                        self.position_rhs,
                    ],
                )
            self.launch(k.cable_scalar_inverse, self.n, [self.position_diagonal, self.position_inverse])
            # A*delta_c = -grad_c(E), where A = M/h^2 + edge Laplacian
            # + scalar contact majorizer. Every A_ij is a float multiplying I3.
            # There are no angular unknowns in this POSITION matrix; the global
            # director option owns a second scalar ELL system for frame columns.
            self.pcg.solve(
                self.matrix,
                self.position_diagonal,
                None,
                self.position_rhs,
                self.position_inverse,
                self.position_dx,
                self.config.linear_iterations,
            )
            self.dx.zero_()
            wp.copy(self.dx, self.position_dx, count=self.n)
            self.apply_descent_step(dt)
        self.launch(
            k.cable_velocity,
            self.size,
            [self.x, self.frames, self.old, self.old_frames, self.rotation_start, dt, self.velocity],
        )
        wp.copy(state_out.body_q, state_in.body_q)
        wp.copy(state_out.body_qd, state_in.body_qd)
        self.launch(
            k.cable_write_bodies,
            len(self.bodies),
            [
                self.bodies,
                self.edges,
                self.centers,
                self.x,
                self.frames,
                self.velocity,
                self.rotation_start,
                self.model.body_mass,
                state_out.body_q,
                state_out.body_qd,
            ],
        )
