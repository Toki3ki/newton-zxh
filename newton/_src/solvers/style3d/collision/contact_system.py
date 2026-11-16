# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compose Style3D contact providers behind one solver collision switch."""

import warp as wp

from newton import Contacts, Model, State

from .cable_collision import CableCollision
from .collision import Collision


class ContactSystem:
    """Combine cloth and cable contact contributions for one particle solve."""

    def __init__(
        self,
        model: Model,
        *,
        cloth: Collision | None = None,
        cable: CableCollision | None = None,
    ):
        self.model = model
        self.device = model.device
        self.cloth = cloth
        self.cable = cable
        self.preconditioner_diagonal = wp.zeros(model.particle_count, dtype=float, device=self.device)
        self._zero_contact_hessian_diagonal = wp.zeros(model.particle_count, dtype=wp.mat33, device=self.device)
        self._hessian_product = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)

    @property
    def has_cable(self) -> bool:
        """Whether capsule contacts are part of this contact system."""
        return self.cable is not None

    @property
    def radius(self) -> float:
        """Return the cloth self-contact radius."""
        if self.cloth is None:
            raise AttributeError("Cable-only contact systems have no cloth collision radius.")
        return self.cloth.radius

    @radius.setter
    def radius(self, value: float) -> None:
        if self.cloth is None:
            raise AttributeError("Cable-only contact systems have no cloth collision radius.")
        self.cloth.radius = value

    def frame_begin(self, state: State, contacts: Contacts | None, dt: float) -> None:
        """Prepare every enabled contact provider for one solver step."""
        if self.cloth is not None:
            self.cloth.frame_begin(state.particle_q, state.particle_qd, dt)
        if self.cable is not None:
            self.cable.frame_begin(state, contacts)

    def accumulate_contact_force(
        self,
        dt: float,
        iteration: int,
        state_in: State,
        state_out: State,
        contacts: Contacts | None,
        particle_forces: wp.array[wp.vec3],
        particle_q_prev: wp.array[wp.vec3],
        particle_stiff: wp.array[float],
    ) -> None:
        """Accumulate provider forces and prepare their shared preconditioner."""
        wp.copy(self.preconditioner_diagonal, particle_stiff)
        if self.cloth is not None:
            self.cloth.accumulate_contact_force(
                dt,
                iteration,
                state_in,
                state_out,
                contacts,
                particle_forces,
                particle_q_prev,
                particle_stiff,
            )
        if self.cable is not None:
            self.cable.accumulate_contact_force(
                particle_stiff,
                self.preconditioner_diagonal,
                particle_forces,
            )

    def contact_hessian_diagonal(self) -> wp.array[wp.mat33]:
        """Return the cloth contact block diagonal, or zero for cable-only contact."""
        if self.cloth is None:
            return self._zero_contact_hessian_diagonal
        return self.cloth.contact_hessian_diagonal()

    def hessian_multiply(self, direction: wp.array[wp.vec3]) -> wp.array[wp.vec3]:
        """Return the sum of all matrix-free contact Hessian products."""
        self._hessian_product.zero_()
        if self.cloth is not None:
            product = self.cloth.hessian_multiply(direction)
            wp.launch(
                add_vector_in_place,
                dim=self.model.particle_count,
                inputs=[product],
                outputs=[self._hessian_product],
                device=self.device,
            )
        if self.cable is not None:
            product = self.cable.hessian_multiply(direction)
            wp.launch(
                add_vector_in_place,
                dim=self.model.particle_count,
                inputs=[product],
                outputs=[self._hessian_product],
                device=self.device,
            )
        return self._hessian_product

    def evaluate_cable_energy(self, positions: wp.array[wp.vec3], energy: wp.array[float]) -> None:
        """Add cable contact energy when the cable provider is enabled."""
        if self.cable is not None:
            self.cable.evaluate_energy(positions, energy)

    def linear_iteration_end(self, dx: wp.array[wp.vec3]) -> None:
        """Finalize one accepted nonlinear contact iteration."""
        if self.cloth is not None:
            self.cloth.linear_iteration_end(dx)
        if self.cable is not None:
            self.cable.linear_iteration_end()

    def frame_end(self, state: State, dt: float) -> None:
        """Finalize every enabled contact provider for one solver step."""
        if self.cloth is not None:
            self.cloth.frame_end(state.particle_q, state.particle_qd, dt)

    def rebuild_bvh(self, state: State) -> None:
        """Rebuild the cloth BVH when cloth contact is enabled."""
        if self.cloth is not None:
            self.cloth.rebuild_bvh(state.particle_q)


@wp.kernel
def add_vector_in_place(source: wp.array[wp.vec3], result: wp.array[wp.vec3]):
    """Add one contact Hessian product to the composite result."""
    index = wp.tid()
    result[index] += source[index]
