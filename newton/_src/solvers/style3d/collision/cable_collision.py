# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Map rigid capsule contacts to Style3D cable centerline particles."""

import warp as wp

from newton import Contacts, Model, State


class CableCollision:
    """Frictionless capsule-contact adapter for a Style3D cable workspace.

    Collision detection remains owned by :class:`newton.CollisionPipeline`.
    This adapter freezes each capsule witness on its centerline segment for one
    solver step and contributes the resulting contact energy, force, and
    Hessian to the shared particle solve.
    """

    def __init__(self, model: Model, cable):
        self.model = model
        self.cable = cable
        self.device = model.device
        self.contacts: Contacts | None = None
        self.contact_hessian_product = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        self._allocate(model.rigid_contact_max or 0)

    def _allocate(self, capacity: int) -> None:
        self.contact_capacity = capacity
        self.contact_enabled = wp.zeros(capacity, dtype=int, device=self.device)
        self.contact_ids = wp.zeros((capacity, 4), dtype=int, device=self.device)
        self.contact_coefficients = wp.zeros((capacity, 4), dtype=float, device=self.device)
        self.contact_offsets = wp.zeros(capacity, dtype=float, device=self.device)
        self.contact_material_stiffness = wp.zeros(capacity, dtype=float, device=self.device)
        self.contact_penalty = wp.zeros(capacity, dtype=float, device=self.device)
        self.contact_multiplier = wp.zeros(capacity, dtype=float, device=self.device)
        self.contact_multiplier_history = wp.zeros(capacity, dtype=float, device=self.device)

    def frame_begin(self, state: State, contacts: Contacts | None) -> None:
        """Freeze capsule witness mappings for the current solver step."""
        self.contacts = contacts
        if contacts is None:
            return
        capacity = len(contacts.rigid_contact_shape0)
        if capacity != self.contact_capacity:
            self._allocate(capacity)
        wp.copy(self.contact_multiplier_history, self.contact_multiplier)
        wp.launch(
            cable_contact_terms,
            dim=capacity,
            inputs=[
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
                self.cable.body_edge,
                state.body_q,
                self.cable.edges,
                self.cable.endpoints,
                self.cable.center,
            ],
            outputs=[
                self.contact_enabled,
                self.contact_ids,
                self.contact_coefficients,
                self.contact_offsets,
                self.contact_material_stiffness,
                self.contact_penalty,
            ],
            device=self.device,
        )
        if contacts.rigid_contact_match_index is None:
            self.contact_multiplier.zero_()
        else:
            wp.launch(
                cable_contact_warmstart,
                dim=capacity,
                inputs=[
                    contacts.rigid_contact_count,
                    self.contact_enabled,
                    contacts.rigid_contact_match_index,
                    self.contact_multiplier_history,
                ],
                outputs=[self.contact_multiplier],
                device=self.device,
            )

    def evaluate_energy(self, positions: wp.array[wp.vec3], energy: wp.array[float]) -> None:
        """Add the frozen-contact ALM energy to a cable line-search objective."""
        if self.contacts is None:
            return
        wp.launch(
            cable_contact_evaluate,
            dim=self.contact_capacity,
            inputs=[
                self.contacts.rigid_contact_count,
                self.contact_enabled,
                self.contact_ids,
                self.contact_coefficients,
                self.contact_offsets,
                self.contact_penalty,
                self.contact_multiplier,
                self.contacts.rigid_contact_normal,
                positions,
            ],
            outputs=[energy],
            device=self.device,
        )

    def accumulate_contact_force(
        self,
        position_diagonal: wp.array[float],
        preconditioner_diagonal: wp.array[float],
        position_rhs: wp.array[wp.vec3],
    ) -> None:
        """Add cable contact force and a scalar preconditioner bound."""
        if self.contacts is None:
            return
        wp.launch(
            cable_contact_condition_penalty,
            dim=self.contact_capacity,
            inputs=[
                self.contacts.rigid_contact_count,
                self.contact_enabled,
                self.contact_ids,
                self.contact_coefficients,
                self.cable.active,
                position_diagonal,
                self.contact_material_stiffness,
            ],
            outputs=[self.contact_penalty],
            device=self.device,
        )
        wp.launch(
            cable_contact_local,
            dim=self.contact_capacity,
            inputs=[
                self.contacts.rigid_contact_count,
                self.contact_enabled,
                self.contact_ids,
                self.contact_coefficients,
                self.contact_offsets,
                self.contact_penalty,
                self.contact_multiplier,
                self.contacts.rigid_contact_normal,
                self.cable.center,
                self.cable.active,
            ],
            outputs=[preconditioner_diagonal, position_rhs],
            device=self.device,
        )

    def hessian_multiply(self, direction: wp.array[wp.vec3]) -> wp.array[wp.vec3]:
        """Return the frozen active-set cable contact Hessian product."""
        self.contact_hessian_product.zero_()
        if self.contacts is not None:
            wp.launch(
                cable_contact_multiply,
                dim=self.contact_capacity,
                inputs=[
                    self.contacts.rigid_contact_count,
                    self.contact_enabled,
                    self.contact_ids,
                    self.contact_coefficients,
                    self.contact_offsets,
                    self.contact_penalty,
                    self.contact_multiplier,
                    self.contacts.rigid_contact_normal,
                    self.cable.center,
                    direction,
                ],
                outputs=[self.contact_hessian_product],
                device=self.device,
            )
        return self.contact_hessian_product

    def linear_iteration_end(self) -> None:
        """Update normal ALM multipliers after an accepted cable position step."""
        if self.contacts is None:
            return
        wp.launch(
            cable_contact_update_multiplier,
            dim=self.contact_capacity,
            inputs=[
                self.contacts.rigid_contact_count,
                self.contact_enabled,
                self.contact_ids,
                self.contact_coefficients,
                self.contact_offsets,
                self.contact_penalty,
                self.contacts.rigid_contact_normal,
                self.cable.center,
            ],
            outputs=[self.contact_multiplier],
            device=self.device,
        )


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
    positions: wp.array[wp.vec3],
    enabled: wp.array[int],
    ids: wp.array2d[int],
    coefficients: wp.array2d[float],
    offsets: wp.array[float],
    material_stiffness: wp.array[float],
    penalty: wp.array[float],
):
    contact = wp.tid()
    if contact >= count[0]:
        enabled[contact] = 0
        return

    normal = normals[contact]
    offset = -margins0[contact] - margins1[contact]
    stiffness = float(0.0)
    cable_side_count = int(0)
    for side in range(2):
        shape = int(shapes0[contact])
        point = wp.vec3(points0[contact])
        sign = float(-1.0)
        if side == 1:
            shape = int(shapes1[contact])
            point = wp.vec3(points1[contact])
            sign = 1.0

        body = int(-1)
        edge = int(-1)
        if shape >= 0:
            body = int(shape_body[shape])
            stiffness += 0.5 * shape_ke[shape]

        world = point
        if body >= 0:
            world = wp.transform_point(body_q[body], point)
            edge = int(body_edge[body])

        ids[contact, 2 * side] = 0
        ids[contact, 2 * side + 1] = 0
        coefficients[contact, 2 * side] = 0.0
        coefficients[contact, 2 * side + 1] = 0.0
        if edge >= 0:
            cable_side_count += 1
            denominator = endpoints[edge, 1][2] - endpoints[edge, 0][2]
            segment_coordinate = wp.clamp((point[2] - endpoints[edge, 0][2]) / denominator, 0.0, 1.0)
            node_a = int(edges[edge, 0])
            node_b = int(edges[edge, 1])
            ids[contact, 2 * side] = node_a
            ids[contact, 2 * side + 1] = node_b
            coefficients[contact, 2 * side] = sign * (1.0 - segment_coordinate)
            coefficients[contact, 2 * side + 1] = sign * segment_coordinate
            centerline = (1.0 - segment_coordinate) * positions[node_a] + segment_coordinate * positions[node_b]
            offset += sign * wp.dot(normal, world - centerline)
        else:
            offset += sign * wp.dot(normal, world)

    enabled[contact] = 1 if cable_side_count > 0 else 0
    offsets[contact] = offset
    material_stiffness[contact] = stiffness
    penalty[contact] = stiffness


@wp.kernel
def cable_contact_warmstart(
    count: wp.array[int],
    enabled: wp.array[int],
    match_index: wp.array[int],
    history: wp.array[float],
    multiplier: wp.array[float],
):
    """Restore the normal multiplier through the pipeline's row matching."""
    contact = wp.tid()
    if contact >= count[0] or enabled[contact] == 0:
        multiplier[contact] = 0.0
        return
    previous = match_index[contact]
    multiplier[contact] = history[previous] if previous >= 0 else 0.0


@wp.kernel
def cable_contact_condition_penalty(
    count: wp.array[int],
    enabled: wp.array[int],
    ids: wp.array2d[int],
    coefficients: wp.array2d[float],
    active: wp.array[float],
    diagonal: wp.array[float],
    material_stiffness: wp.array[float],
    penalty: wp.array[float],
):
    """Scale ALM rho by a diagonal Rayleigh quotient on the free support."""
    contact = wp.tid()
    if contact >= count[0] or enabled[contact] == 0:
        return
    curvature = float(0.0)
    norm_squared = float(0.0)
    for slot in range(4):
        node = ids[contact, slot]
        coefficient = float(0.0)
        for other in range(4):
            if ids[contact, other] == node:
                coefficient += coefficients[contact, other]
        unique = bool(True)
        for previous in range(slot):
            if ids[contact, previous] == node:
                unique = False
        if unique and active[node] > 0.0:
            curvature += coefficient * coefficient * diagonal[node]
            norm_squared += coefficient * coefficient
    support = curvature / norm_squared if norm_squared > 0.0 else 0.0
    penalty[contact] = wp.max(material_stiffness[contact], support)


@wp.kernel
def cable_contact_evaluate(
    count: wp.array[int],
    enabled: wp.array[int],
    ids: wp.array2d[int],
    coefficients: wp.array2d[float],
    offsets: wp.array[float],
    stiffness: wp.array[float],
    multiplier: wp.array[float],
    normals: wp.array[wp.vec3],
    positions: wp.array[wp.vec3],
    energy: wp.array[float],
):
    contact = wp.tid()
    if contact >= count[0] or enabled[contact] == 0:
        return
    gap = offsets[contact]
    for slot in range(4):
        gap += coefficients[contact, slot] * wp.dot(normals[contact], positions[ids[contact, slot]])
    rho = stiffness[contact]
    shifted = wp.max(multiplier[contact] - rho * gap, 0.0)
    if rho > 0.0:
        wp.atomic_add(energy, 0, 0.5 * (shifted * shifted - multiplier[contact] * multiplier[contact]) / rho)


@wp.kernel
def cable_contact_local(
    count: wp.array[int],
    enabled: wp.array[int],
    ids: wp.array2d[int],
    coefficients: wp.array2d[float],
    offsets: wp.array[float],
    stiffness: wp.array[float],
    multiplier: wp.array[float],
    normals: wp.array[wp.vec3],
    positions: wp.array[wp.vec3],
    active: wp.array[float],
    diagonal: wp.array[float],
    rhs: wp.array[wp.vec3],
):
    contact = wp.tid()
    if contact >= count[0] or enabled[contact] == 0:
        return
    gap = offsets[contact]
    norm = float(0.0)
    for slot in range(4):
        gap += coefficients[contact, slot] * wp.dot(normals[contact], positions[ids[contact, slot]])
        norm += wp.abs(coefficients[contact, slot]) * active[ids[contact, slot]]
    force = wp.max(multiplier[contact] - stiffness[contact] * gap, 0.0)
    if force <= 0.0:
        return
    for slot in range(4):
        node = ids[contact, slot]
        coefficient = coefficients[contact, slot] * active[node]
        wp.atomic_add(rhs, node, coefficient * force * normals[contact])
        wp.atomic_add(diagonal, node, stiffness[contact] * wp.abs(coefficient) * norm)


@wp.kernel
def cable_contact_multiply(
    count: wp.array[int],
    enabled: wp.array[int],
    ids: wp.array2d[int],
    coefficients: wp.array2d[float],
    offsets: wp.array[float],
    stiffness: wp.array[float],
    multiplier: wp.array[float],
    normals: wp.array[wp.vec3],
    positions: wp.array[wp.vec3],
    direction: wp.array[wp.vec3],
    product: wp.array[wp.vec3],
):
    """Apply the exact frozen-active-set contact Hessian to a direction."""
    contact = wp.tid()
    if contact >= count[0] or enabled[contact] == 0:
        return
    gap = offsets[contact]
    projected = float(0.0)
    for slot in range(4):
        node = ids[contact, slot]
        coefficient = coefficients[contact, slot]
        gap += coefficient * wp.dot(normals[contact], positions[node])
        projected += coefficient * wp.dot(normals[contact], direction[node])
    if multiplier[contact] - stiffness[contact] * gap <= 0.0:
        return
    for slot in range(4):
        wp.atomic_add(
            product,
            ids[contact, slot],
            stiffness[contact] * coefficients[contact, slot] * projected * normals[contact],
        )


@wp.kernel
def cable_contact_update_multiplier(
    count: wp.array[int],
    enabled: wp.array[int],
    ids: wp.array2d[int],
    coefficients: wp.array2d[float],
    offsets: wp.array[float],
    stiffness: wp.array[float],
    normals: wp.array[wp.vec3],
    positions: wp.array[wp.vec3],
    multiplier: wp.array[float],
):
    """Perform one projected ALM dual update after an accepted position step."""
    contact = wp.tid()
    if contact >= count[0] or enabled[contact] == 0:
        return
    gap = offsets[contact]
    for slot in range(4):
        gap += coefficients[contact, slot] * wp.dot(normals[contact], positions[ids[contact, slot]])
    multiplier[contact] = wp.max(multiplier[contact] - stiffness[contact] * gap, 0.0)
