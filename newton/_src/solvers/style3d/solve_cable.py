# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Cable state, topology, and energy data for SolverStyle3D."""

import numpy as np
import warp as wp

from ...geometry import GeoType
from ...sim import JointType
from .builder import _compute_max_neighbors
from .kernels import (
    cable_accept,
    cable_accept_energy,
    cable_enable_particles,
    cable_bishop_twist,
    cable_elastic_energy,
    cable_local_inertia_kernel,
    cable_inertial_energy,
    cable_predict,
    cable_velocity,
    cable_write_bodies,
    cable_scale_energy,
    cable_scale_position_hessian,
    cable_scale_terms,
    cable_trial,
    cable_update_kinematic_segments,
    eval_cable_bend_kernel,
    eval_cable_bend_twist_energy_kernel,
    eval_cable_rotation_terms_kernel,
    eval_cable_stretch_kernel,
    eval_cable_stretch_rotation_kernel,
    solve_cable_rotation_kernel,
)


class _CableWorkspace:
    """Cable topology, state, and energy adapter owned by SolverStyle3D.

    Initialization reads topology on the host and stores cable state.
    SolverStyle3D.step owns the iteration schedule and shared PCG solve.
    Capsule rods may coexist with particle constraints and static scenery.
    Body forces, general joints and Dahl are not supported.
    """

    def __init__(self, model, config):
        # Generalized variables: shared nodes c_i, segment frames R_e, and
        # optional nodal scales s_i. Fixed section removes s and its velocity;
        # it does not impose an inextensibility constraint on c.
        self.model, self.config, self.device = model, config, model.device
        self.fixed_scale = config.cable_fixed_scale
        rod_particle_indices = getattr(model, "rod_particle_indices", None)
        if rod_particle_indices is None or rod_particle_indices.shape[0] == 0:
            self._initialize_empty()
            return
        types = model.joint_type.numpy()
        rod_joints = np.flatnonzero(types == int(JointType.ROD))
        if not len(rod_joints):
            raise ValueError("cable requires ROD joints.")
        # Read standard ModelBuilder.add_rod material and reference data.
        material = np.zeros((model.joint_count, 3))
        reference_rotations = np.tile([0.0, 0.0, 0.0, 1.0], (model.joint_count, 1))
        starts, qstarts = model.joint_qd_start.numpy(), model.joint_q_start.numpy()
        gains, rest_q = model.joint_target_ke.numpy(), model.joint_rest_q.numpy()
        anchors_p, anchors_c = model.joint_X_p.numpy(), model.joint_X_c.numpy()
        for j in rod_joints:
            start, qstart = starts[j], qstarts[j]
            material[j] = gains[start + np.array([2, 3, 5])]
            qa, qb = wp.quat(*anchors_p[j, 3:]), wp.quat(*anchors_c[j, 3:])
            reference_rotations[j] = np.array(qa * wp.quat(*rest_q[qstart + 3 : qstart + 7]) * wp.quat_inverse(qb))
        if np.any((types != int(JointType.ROD)) & (types != int(JointType.FREE))):
            raise ValueError("Style3D has no constraints for these non-ROD/FREE joint types.")
        parents, children = model.joint_parent.numpy(), model.joint_child.numpy()
        bodies = sorted(set(parents[rod_joints]) | set(children[rod_joints]))
        if -1 in bodies:
            raise ValueError("World ROD joints are unsupported; pin a zero-mass capsule instead.")
        mapping = {b: i for i, b in enumerate(bodies)}
        shape_body = model.shape_body.numpy()
        scale, shape_q = model.shape_scale.numpy(), model.shape_transform.numpy()
        body_q, masses = model.body_q.numpy(), model.body_mass.numpy()
        if any(masses[b] > 0 and b not in mapping for b in range(model.body_count)):
            raise ValueError("cable does not integrate unrelated dynamic rigid bodies.")
        endpoints, centers, lengths, radii = [], [], [], []
        for b in bodies:
            shapes = np.flatnonzero(shape_body == b)
            if len(shapes) != 1:
                raise ValueError("Each rod body must own exactly one capsule.")
            shape = shapes[0]
            if model.shape_type.numpy()[shape] != int(GeoType.CAPSULE):
                raise ValueError("cable requires capsule geometry.")
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
        if not hasattr(model, "rod_particle_indices"):
            raise ValueError("Style3D rods require builder.add_rod(..., particle_based=True).")
        edges = model.rod_particle_indices.numpy()[bodies]
        if np.any(edges < 0):
            raise ValueError("Every Style3D rod must have centerline particles.")
        n, ne = model.particle_count, len(bodies)
        self.node_indices = wp.array(np.unique(edges), dtype=int, device=self.device)
        rotation_start = n if self.fixed_scale else 2 * n
        self.n, self.rotation_start, self.size = n, rotation_start, rotation_start + ne
        center = np.zeros((self.size, 3), dtype=np.float32)
        frames = np.tile([0, 0, 0, 1], (self.size, 1)).astype(np.float32)
        center[:n] = model.particle_q.numpy()
        mass = np.ones((self.size, 3), dtype=np.float32) * 1e-8
        active = np.ones(self.size, dtype=np.float32)
        active[:n] = 0
        active[np.unique(edges)] = (model.particle_mass.numpy()[np.unique(edges)] > 0).astype(np.float32)
        mass[:n] = model.particle_mass.numpy()[:, None]
        if not self.fixed_scale:
            center[n:rotation_start, 0] = 1.0
        for e, b in enumerate(bodies):
            frames[rotation_start + e] = body_q[b, 3:]
            for node in edges[e]:
                if not self.fixed_scale:
                    mass[n + node, 0] += masses[b] * radii[e] ** 2 / 4
                if masses[b] == 0:
                    active[node] = 0
            # Cross-section disc inertia I=m*r^2*diag(1/4,1/4,1/2).
            # Endpoint masses also contribute to transverse motion; this is
            # not the full rigid capsule inertia used by VBD.
            mass[rotation_start + e] = max(masses[b], 1e-6) * radii[e] ** 2 * np.array([0.25, 0.25, 0.5])
            if masses[b] == 0:
                active[rotation_start + e] = 0
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

        for e, (a, b) in enumerate(edges):
            incident = [j for j in rod_joints if parents[j] == bodies[e] or children[j] == bodies[e]]
            gain = np.mean([material[j, 0] for j in incident])
            # Per-segment mapping from incident joint axial stiffness k:
            # W0=k*L^2, W1=W0*volume_ratio, W2=W0*radial_ratio,
            # W3=W2*r^2/4. L^2 cancels the 1/L^2 in C_0 and C_1.
            # This is the current adapter convention, not an EI/EA continuum fit.
            axial = gain * lengths[e] ** 2
            row = [
                a,
                b,
                0,
                0,
                rotation_start + e,
                rotation_start + e,
            ]
            add(0, row, lengths[e], axial)
        for j in rod_joints:
            a, b = mapping[parents[j]], mapping[children[j]]
            # Q0 converts authored joint rest rotation to the relative material
            # frame: Q0 = Q_parent_anchor * Q_joint_rest * Q_child_anchor^-1.
            rest = wp.quat(*reference_rotations[j])
            length = (lengths[a] + lengths[b]) / 2
            row = [
                edges[a, 0],
                edges[b, 1],
                0,
                0,
                rotation_start + a,
                rotation_start + b,
            ]
            bend, twist = material[j, 1], material[j, 2]
            # W4=L_dual^2*diag(kb,kb,kt); W5=L_dual^2*volume_ratio*diag(kb,kb,0).
            # These are per-joint stiffnesses: the dual length cancels in E4/E5.
            add(4, row, length, np.array([bend, bend, twist]) * length**2, rest)
        self.count = len(kinds)

        def array(data, dtype):
            return wp.array(data, dtype=dtype, device=self.device)

        self.kinds, self.ids = array(kinds, int), wp.array2d(ids, dtype=int, device=self.device)
        self.lengths, self.weights, self.rest = array(lens, float), array(weights, wp.vec3), array(rests, wp.quat)
        self.center, self.frames, self.mass, self.active = (
            array(center, wp.vec3),
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
        self.old, self.predicted = wp.clone(self.center), wp.clone(self.center)
        self.old_frames, self.predicted_frames = wp.clone(self.frames), wp.clone(self.frames)
        self.dx = wp.zeros_like(self.center)
        self.rhs = wp.zeros_like(self.center)
        self.diagonal = wp.zeros(self.size, dtype=wp.mat33, device=self.device)
        edge_axial = np.array([weights[i][0] ** 2 for i, kind in enumerate(kinds) if kind == 0])
        self.axial = array(edge_axial, float)
        self.bulk = array(np.zeros(ne), float)
        if not self.fixed_scale and config.cable_preserve_volume:
            self.bulk = array(edge_axial * 100.0, float)
        self.radial = array(edge_axial * 0.1, float)
        self.bend_twist = _CableBendTwistData(self, kinds, ids, weights, lens, rests)
        self.trial_center, self.best_center = wp.clone(self.center), wp.clone(self.center)
        self.trial_frames, self.best_frames = wp.clone(self.frames), wp.clone(self.frames)
        self.energy, self.best_energy = [wp.zeros(1, dtype=float, device=self.device) for _ in range(2)]
        # Angular energy is supplied exclusively by chord bending and Bishop twist.
        energy_weights = np.array(weights).copy()
        energy_weights[np.asarray(kinds) == 4] = 0
        self.energy_weights = array(energy_weights, wp.vec3)

    def _initialize_empty(self):
        """Initialize zero-length arrays when no rod constraints were authored."""
        self.n = self.size = self.rotation_start = 0
        self.node_indices = wp.empty(0, dtype=int, device=self.device)
        for name, dtype in (
            ("center", wp.vec3),
            ("predicted", wp.vec3),
            ("mass", wp.vec3),
            ("rhs", wp.vec3),
            ("dx", wp.vec3),
            ("active", float),
            ("frames", wp.quat),
            ("predicted_frames", wp.quat),
            ("diagonal", wp.mat33),
            ("bodies", int),
            ("segment_lengths", float),
            ("axial", float),
            ("bulk", float),
        ):
            setattr(self, name, wp.empty(0, dtype=dtype, device=self.device))
        self.edges = wp.empty((0, 2), dtype=int, device=self.device)
        b = _CableBendTwistData.__new__(_CableBendTwistData)
        b.cable = self
        for name, width, dtype in (("ids", 4, int), ("coefficients", 4, float), ("pairs", 2, int)):
            setattr(b, name, wp.empty((0, width), dtype=dtype, device=self.device))
        for name, dtype in (
            ("targets", wp.vec3),
            ("rest_twist", float),
            ("bend", float),
            ("twist", float),
            ("base_diagonal", float),
        ):
            setattr(b, name, wp.empty(0, dtype=dtype, device=self.device))
        self.bend_twist = b

    def launch(self, kernel, dim, inputs):
        wp.launch(kernel, dim=dim, inputs=inputs, device=self.device)

    def add_position_constraints(self, matrix):
        """Incrementally add cable stretch and bending blocks to a PD matrix builder."""
        if not self.n:
            return
        active = np.ones(self.model.particle_count, dtype=np.float32)
        nodes = self.node_indices.numpy()
        active[nodes] = self.active.numpy()[nodes]
        matrix.add_cable_stretch_constraints(
            self.edges.numpy(), self.segment_lengths.numpy(), self.axial.numpy(), active=active
        )
        matrix.add_cable_bend_constraints(
            self.bend_twist.ids.numpy(),
            self.bend_twist.coefficients.numpy(),
            self.bend_twist.bend.numpy(),
            active=active,
        )

    def bind_position_system(self, matrix, base_diagonal, position_diagonal, rhs, dx):
        """Bind shared Style3D position-system storage used by cable constraints."""
        self.matrix = matrix
        self.bend_twist.matrix = matrix
        self.bend_twist.base_diagonal = base_diagonal
        self.position_diagonal = position_diagonal
        self.position_rhs = rhs
        self.position_dx = dx

    def activate_particles(self, masses, flags):
        """Activate mapped centerline degrees of freedom in solver-private flags."""
        self.launch(cable_enable_particles, len(self.node_indices), [self.node_indices, masses, flags])

    def max_position_neighbors(self):
        """Return the ELL neighbor capacity required by cable position constraints."""
        if not self.n:
            return 0
        edges = self.edges.numpy()
        bends = self.bend_twist.ids.numpy()
        stencils = np.vstack((bends, np.repeat(edges, 2, axis=1)))
        return _compute_max_neighbors(
            self.model.particle_count,
            np.empty((0, 3), dtype=np.int32),
            stencils,
        )

    def frame_begin(self, state, dt):
        """Synchronize cable state, apply kinematic segments, and predict local variables."""
        wp.copy(self.center, state.particle_q, count=self.n)
        wp.copy(self.velocity, state.particle_qd, count=self.n)
        self.position_dx.zero_()
        self.launch(
            cable_update_kinematic_segments,
            len(self.bodies),
            [
                self.bodies,
                self.edges,
                self.endpoints,
                self.model.body_mass,
                state.body_q,
                self.rotation_start,
                self.center,
                self.frames,
            ],
        )
        wp.copy(state.particle_q, self.center, count=self.n)
        wp.copy(self.old, self.center)
        wp.copy(self.old_frames, self.frames)
        self.launch(
            cable_predict,
            self.size,
            [
                self.center,
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

    def local_step(self, dt, positions, step_diagonal, base_entries, contact_energy=None):
        """Update cable frames and assemble cable contributions to the shared position RHS."""
        if self.size == 0:
            return
        self.launch(
            cable_local_inertia_kernel,
            self.size,
            [
                self.center,
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
            eval_cable_stretch_rotation_kernel,
            len(self.edges),
            [
                self.edges,
                self.segment_lengths,
                self.axial,
                self.rotation_start,
                self.center,
                self.frames,
                self.active,
                self.rhs,
                self.diagonal,
            ],
        )
        if not self.fixed_scale:
            self.launch(
                cable_scale_terms,
                len(self.edges),
                [
                    self.edges,
                    self.segment_lengths,
                    self.bulk,
                    self.radial,
                    self.n,
                    self.rotation_start,
                    self.center,
                    self.frames,
                    self.active,
                    self.rhs,
                    self.diagonal,
                ],
            )
        self.launch(
            eval_cable_rotation_terms_kernel,
            len(self.bend_twist.ids),
            [*self.bend_twist.arguments(self.center, self.frames), self.active, self.rhs, self.diagonal],
        )
        self.launch(
            solve_cable_rotation_kernel,
            self.size,
            [self.n, self.diagonal, self.rhs, self.dx],
        )
        # Apply the configured fixed local frame/scale relaxation without line search.
        self.launch(
            cable_trial,
            self.size,
            [
                self.center,
                self.frames,
                self.dx,
                self.active,
                self.rotation_start,
                self.config.cable_relaxation,
                self.trial_center,
                self.trial_frames,
            ],
        )
        wp.copy(self.center, self.trial_center)
        wp.copy(self.frames, self.trial_frames)

        if not self.fixed_scale:
            wp.copy(self.position_diagonal, step_diagonal)
            wp.copy(self.matrix.nz_ell, base_entries)
            self.launch(
                cable_scale_position_hessian,
                1,
                [
                    self.edges,
                    self.segment_lengths,
                    self.bulk,
                    self.n,
                    self.center,
                    self.active,
                    self.matrix.num_nz,
                    self.matrix.nz_ell,
                    self.position_diagonal,
                ],
            )

        self.launch(
            eval_cable_stretch_kernel,
            len(self.edges),
            [
                self.edges,
                self.segment_lengths,
                self.axial,
                self.bulk,
                self.n,
                self.rotation_start,
                self.fixed_scale,
                positions,
                self.center,
                self.frames,
                self.active,
                self.position_rhs,
            ],
        )
        self.launch(
            eval_cable_bend_kernel,
            len(self.bend_twist.ids),
            [
                self.bend_twist.ids,
                self.bend_twist.coefficients,
                self.bend_twist.pairs,
                self.bend_twist.targets,
                self.bend_twist.bend,
                positions,
                self.frames,
                self.active,
                self.position_rhs,
            ],
        )

    def write_body_state(self, state_in, state_out, dt):
        """Compute cable velocities and write capsule poses and velocities."""
        wp.copy(self.center, state_out.particle_q, count=self.n)
        self.launch(
            cable_velocity,
            self.size,
            [
                self.center,
                self.frames,
                self.old,
                self.old_frames,
                self.rotation_start,
                dt,
                self.velocity,
            ],
        )
        wp.copy(self.velocity, state_out.particle_qd, count=self.n)
        wp.copy(state_out.body_q, state_in.body_q)
        wp.copy(state_out.body_qd, state_in.body_qd)
        self.launch(
            cable_write_bodies,
            len(self.bodies),
            [
                self.bodies,
                self.edges,
                self.centers,
                self.center,
                self.frames,
                self.velocity,
                self.rotation_start,
                self.model.body_mass,
                state_out.body_q,
                state_out.body_qd,
            ],
        )

    def evaluate_energy(self, x, frames, dt, energy, contact_energy=None):
        # Objective for every line search: E_inertia + E_elastic + optional E_contact.
        # Chord bending and Bishop twist provide the angular energy.
        energy.zero_()
        self.launch(
            cable_elastic_energy,
            self.count,
            [self.kinds, self.ids, self.lengths, self.energy_weights, self.rest, self.fixed_scale, x, frames, energy],
        )
        self.bend_twist.energy(x, frames, energy)
        if not self.fixed_scale:
            self.launch(
                cable_scale_energy,
                len(self.bodies),
                [
                    self.edges,
                    self.segment_lengths,
                    self.bulk,
                    self.radial,
                    self.n,
                    self.rotation_start,
                    x,
                    frames,
                    energy,
                ],
            )
        self.launch(
            cable_inertial_energy,
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
        if contact_energy is not None:
            contact_energy(x, energy)

    def apply_descent_step(self, dt, initial_step=None, contact_energy=None):
        if self.size == 0:
            return
        # All candidates share the same linearization. Retain the best feasible
        # energy, including the zero step, without host synchronization.
        wp.copy(self.best_center, self.center)
        wp.copy(self.best_frames, self.frames)
        self.evaluate_energy(self.center, self.frames, dt, self.best_energy, contact_energy)
        if initial_step is None:
            initial_step = self.config.cable_relaxation
        for backtrack in range(8):
            alpha = initial_step * 0.5**backtrack
            self.launch(
                cable_trial,
                self.size,
                [
                    self.center,
                    self.frames,
                    self.dx,
                    self.active,
                    self.rotation_start,
                    alpha,
                    self.trial_center,
                    self.trial_frames,
                ],
            )
            self.evaluate_energy(self.trial_center, self.trial_frames, dt, self.energy, contact_energy)
            self.launch(
                cable_accept,
                self.size,
                [
                    self.trial_center,
                    self.trial_frames,
                    self.energy,
                    self.best_energy,
                    self.best_center,
                    self.best_frames,
                ],
            )
            self.launch(cable_accept_energy, 1, [self.energy, self.best_energy])
        wp.copy(self.center, self.best_center)
        wp.copy(self.frames, self.best_frames)


class _CableBendTwistData:
    """Reference stencils and material targets for chord bending and twist.

    E_b=kb/2*||d_b-d_a-R_a*(Q0*ez-ez)||^2, d_e=(c1-c0)/L_e.
    This is an objective curvature approximation, not the VIPER logarithmic
    energy at arbitrary strain. Both local and global use this same objective.
    """

    def __init__(self, cable, kinds, rows, weights, lengths, rests):
        self.cable = cable
        edges = cable.edges.numpy()
        segment_lengths = cable.segment_lengths.numpy()
        ids, coefficients, pairs, targets, rest_twist, bend, twist = [], [], [], [], [], [], []
        for kind, row, weight, length, rest in zip(kinds, rows, weights, lengths, rests, strict=True):
            if kind != 4:
                continue
            a, b = row[4] - cable.rotation_start, row[5] - cable.rotation_start
            la, lb = segment_lengths[a], segment_lengths[b]
            q0 = wp.quat(*rest)
            ids.append([*edges[a], *edges[b]])
            coefficients.append([1 / la, -1 / la, -1 / lb, 1 / lb])
            pairs.append(row[4:6])
            targets.append(np.array(wp.quat_rotate(q0, wp.vec3(0, 0, 1))) - [0, 0, 1])
            rest_twist.append(cable_bishop_twist(wp.quat_identity(), q0))
            bend.append(weight[0] ** 2 / length**2)
            twist.append(weight[2] ** 2 / length**2)
        self.ids = wp.array(ids, dtype=int, device=cable.device)
        self.coefficients = wp.array(coefficients, dtype=float, device=cable.device)
        self.pairs = wp.array(pairs, dtype=int, device=cable.device)
        self.targets = wp.array(targets, dtype=wp.vec3, device=cable.device)
        self.rest_twist = wp.array(rest_twist, dtype=float, device=cable.device)
        self.bend = wp.array(bend, dtype=float, device=cable.device)
        self.twist = wp.array(twist, dtype=float, device=cable.device)

    def arguments(self, x, frames):
        return [
            self.ids,
            self.coefficients,
            self.pairs,
            self.targets,
            self.rest_twist,
            self.bend,
            self.twist,
            x,
            frames,
        ]

    def energy(self, x, frames, energy):
        self.cable.launch(eval_cable_bend_twist_energy_kernel, len(self.ids), [*self.arguments(x, frames), energy])
