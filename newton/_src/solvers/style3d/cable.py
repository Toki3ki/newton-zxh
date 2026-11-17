# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""VIPER cable module for :class:`newton.solvers.SolverStyle3D`.

The module owns a staggered set of global rows for every rod-backed deformable
object in the model: one center row per curve node and one material-frame row per
curve segment. It builds the rod's rest data from the proxy capsule bodies, adds
its constant Projective Dynamics stencils to the shared matrix, predicts and
linearizes its own inertia and elastic terms, and reconstructs the proxy bodies
after the solve.

The elastic model is the fixed-scale, global-material-frame form of
*VIPER: Volume Invariant Position-based Elastic Rods*:

- stretch/alignment couples the two endpoint centers of a segment with its
  material frame;
- Darboux bending and twist couple neighboring material frames, preserving
  their relative rest rotation (VIPER equations 14 and 40).

Rotation rows hold the world-frame tangent increment accumulated since
:meth:`Cable.init_step`. Every coupling that is not a scalar multiple of the
``3x3`` identity lives in the matrix-free operator, so the assembled matrix stays
symmetric and positive semidefinite.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ...geometry import GeoType
from ...sim import Contacts, JointType, Model, State
from .builder import PDMatrixBuilder
from .kernels import (
    CableContactTangent,
    cable_apply_increment_kernel,
    cable_average_state_kernel,
    cable_collect_state_kernel,
    cable_contact_matrix_free_kernel,
    cable_darboux_matrix_free_kernel,
    cable_eval_contact_kernel,
    cable_eval_darboux_kernel,
    cable_eval_stretch_kernel,
    cable_node_begin_kernel,
    cable_node_inertia_kernel,
    cable_segment_begin_kernel,
    cable_segment_inertia_kernel,
    cable_stretch_matrix_free_kernel,
    cable_writeback_kernel,
)

__all__ = ["Cable"]


def _quaternion_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Return the rotation matrix of one ``(x, y, z, w)`` quaternion."""
    x, y, z, w = (float(value) for value in quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class Cable:
    """Advance rod-backed deformable objects with fixed-scale VIPER.

    The model's :meth:`~newton.ModelBuilder.add_rod` objects define the cables.
    Each curve contributes its segment bodies in order, and the internal
    :attr:`~newton.JointType.ROD` joints define which segment endpoints share a
    node. Bodies, joints, and shapes stay in the model for collision geometry,
    rendering, and attachments; the module only writes their poses back after a
    solve, so another solver must not also integrate them.

    Cable state is owned by this module after :meth:`bind_dofs`: node centers
    live in the shared solver buffers, and material frames live here because the
    linear solve only stores their tangent increments. The model's initial body
    poses define both the rest configuration and the first cable state.

    A segment whose body has zero mass or is
    :attr:`~newton.BodyFlags.KINEMATIC` keeps its material frame externally
    prescribed: the module reads that frame from the input state each step and
    neither solves nor writes back the segment, exactly as for externally driven
    affine bodies. This is how a spinning root segment drives twist into the rest
    of the cable. Node centers stay solver-owned, so a prescribed segment
    transmits orientation and twist but not a translation attachment.

    Args:
        model: The finalized model that owns the cable proxy bodies.
        stretch_stiffness: Axial rigidity ``EA`` [N]. Zero disables stretch.
        bend_stiffness: Bending rigidity ``EI`` [N·m²]. Zero disables bending.
        twist_stiffness: Torsional rigidity ``GJ`` [N·m²]. Zero disables twist.

    Raises:
        ValueError: If a curve is not a chain of single-capsule segments with at
            most two segments per node, or if a rod joint leaves its curve.

    .. experimental::

        The cable module layout, its configuration names, and the supported
        topology may change without prior notice.
    """

    def __init__(
        self,
        model: Model,
        *,
        stretch_stiffness: float = 1.0e4,
        bend_stiffness: float = 1.0e-2,
        twist_stiffness: float = 1.0e-2,
    ) -> None:
        self.model = model
        self._allocate_topology(stretch_stiffness, bend_stiffness, twist_stiffness)

    @property
    def node_count(self) -> int:
        """Number of VIPER center nodes."""
        return self._node_count

    @property
    def segment_count(self) -> int:
        """Number of VIPER segments, each with one material frame."""
        return self._segment_count

    @property
    def component_offsets(self) -> tuple[range, ...]:
        """Node ranges of the connected rod components, in segment order."""
        return self._component_offsets

    @property
    def body_to_segment(self) -> wp.array[wp.int32]:
        """Map from a model body to its cable segment, or ``-1`` if it is not one."""
        return self._body_to_segment

    def dof_requirements(self) -> tuple[int, int, int]:
        """Return the position, rotation, and scale block counts this module needs."""
        return self._node_count, self._segment_count, 0

    def bind_dofs(
        self,
        *,
        position_start: int,
        position_end: int,
        rotation_start: int,
        rotation_end: int,
        x: wp.array[wp.vec3],
        x_prev: wp.array[wp.vec3],
        x_inertia: wp.array[wp.vec3],
        v: wp.array[wp.vec3],
        dx: wp.array[wp.vec3],
        A_diag_base: wp.array[float],
        pd_diag: wp.array[float],
        solver_flags: wp.array[wp.int32],
    ) -> None:
        """Bind views of the shared solver buffers and seed the initial cable state.

        Args:
            position_start: First cable node row in the shared buffers.
            position_end: Exclusive end of the cable node rows.
            rotation_start: First cable frame row in the shared buffers.
            rotation_end: Exclusive end of the cable frame rows.
            x: Shared solved coordinates, shape [dof_block_count].
            x_prev: Shared coordinates at the start of the step.
            x_inertia: Shared inertial targets.
            v: Shared velocities.
            dx: Shared linear increments.
            A_diag_base: Per-row scalar mass and fixed constraint diagonal.
            pd_diag: Per-row fixed PD diagonal from the assembled matrix.
            solver_flags: Per-row :class:`~newton.ParticleFlags`.
        """
        self._position_start = position_start
        self._position_end = position_end
        self._rotation_start = rotation_start
        self._rotation_end = rotation_end
        if self._node_count == 0:
            # A model without cables still reserves an empty range, which has no
            # addressable view in the shared buffers.
            return
        self._node_position = x[position_start:position_end]
        self._node_previous = x_prev[position_start:position_end]
        self._node_inertia = x_inertia[position_start:position_end]
        self._node_velocity = v[position_start:position_end]
        self._node_dx = dx[position_start:position_end]
        self._node_base_diag = A_diag_base[position_start:position_end]
        self._node_pd_diag = pd_diag[position_start:position_end]
        self._node_flags = solver_flags[position_start:position_end]

        self._tangent = x[rotation_start:rotation_end]
        self._tangent_previous = x_prev[rotation_start:rotation_end]
        self._tangent_inertia = x_inertia[rotation_start:rotation_end]
        self._tangent_velocity = v[rotation_start:rotation_end]
        self._tangent_dx = dx[rotation_start:rotation_end]
        self._tangent_base_diag = A_diag_base[rotation_start:rotation_end]
        self._tangent_pd_diag = pd_diag[rotation_start:rotation_end]
        self._tangent_flags = solver_flags[rotation_start:rotation_end]

        if self._node_count > 0:
            wp.copy(self._node_position, self._initial_node_position)
            wp.copy(self._node_velocity, self._initial_node_velocity)

    def max_matrix_neighbors(self) -> int:
        """Return the largest number of off-diagonal columns any owned row needs."""
        if self._node_count == 0:
            return 0
        columns: list[set[int]] = [set() for _ in range(self._node_count)]
        for start, end in self._segment_nodes_np:
            row, column = int(start), int(end)
            columns[row].add(column)
            columns[column].add(row)
        return max(len(entry) for entry in columns)

    def add_fixed_matrix(self, matrix: PDMatrixBuilder) -> None:
        """Add the constant position stretch stencils to the fixed matrix.

        Darboux bending/twist act on frames and use dynamic block tangents.
        """
        if self._node_count == 0:
            return

        stretch_rows: list[np.ndarray] = []
        stretch_weights: list[np.ndarray] = []
        for segment, (start, end) in enumerate(self._segment_nodes_np):
            stiffness = float(self._stretch_stiffness_np[segment])
            if stiffness <= 0.0:
                continue
            stretch_rows.append(np.array([start, end], dtype=np.int32))
            stretch_weights.append(stiffness * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64))
        if stretch_rows:
            matrix.add_scalar_stencils(np.stack(stretch_rows), np.stack(stretch_weights))

    def init_step(self, state_in: State, dt: float) -> None:
        """Prepare the VIPER node/frame unknowns for one time step.

        The optional :meth:`reset` is only an external-state synchronization:
        it reconstructs the persistent rod-node positions and material frames
        from Newton's proxy-body poses. The two kernels below then perform the
        normal per-step setup, whether or not synchronization was needed:

        - ``cable_node_begin_kernel`` saves the current rod-node positions and
          computes their inertial targets for this step.
        - ``cable_segment_begin_kernel`` saves the current material frames as
          the rotation bases, resets their tangent coordinates to zero, and
          computes the angular inertial targets for this step.

        Thus these kernels do not reconstruct the cable topology or duplicate
        :meth:`reset`; they initialize the incremental solve around the current
        persistent Cable state.

        Args:
            state_in: Input state, used only to synchronize the module the first
                time a step runs. After that the module owns its own state.
            dt: Time step [s].
        """
        if self._node_count == 0:
            return
        if self._needs_sync:
            self.reset(state_in)
            self._needs_sync = False
        # Per-step translational prediction around the current rod-node state.
        wp.launch(
            cable_node_begin_kernel,
            dim=self._node_count,
            inputs=[
                dt,
                self._node_mass,
                self._node_world,
                self.model.gravity,
                self._node_pd_diag,
                self._node_position,
                self._node_velocity,
            ],
            outputs=[
                self._node_previous,
                self._node_inertia,
                self._node_dx,
                self._node_base_diag,
                self._node_flags,
            ],
            device=self.model.device,
        )
        # Per-step rotational tangent coordinates around the current frames.
        wp.launch(
            cable_segment_begin_kernel,
            dim=self._segment_count,
            inputs=[
                dt,
                self._frame_inertia,
                self._frame_velocity,
                self._tangent_pd_diag,
                self._segment_bodies,
                state_in.body_q,
                self.model.body_mass,
                self.model.body_flags,
                self._frame,
            ],
            outputs=[
                self._frame_begin,
                self._tangent,
                self._tangent_previous,
                self._tangent_velocity,
                self._tangent_inertia,
                self._tangent_dx,
                self._tangent_base_diag,
                self._tangent_flags,
            ],
            device=self.model.device,
        )

    def init_rhs(self, dt: float, rhs: wp.array[wp.vec3]) -> None:
        """Initialize the Cable node/frame inertial right-hand side.

        This is the Cable counterpart of the shared ``init_rhs_kernel`` for
        particle and ABD rows.

        Args:
            dt: Time step [s].
            rhs: Right-hand side containing only the Cable rows, with node rows
                followed by material-frame rows.
        """
        if self._node_count == 0:
            return
        node_rhs = rhs[: self._node_count]
        tangent_rhs = rhs[self._node_count :]
        wp.launch(
            cable_node_inertia_kernel,
            dim=self._node_count,
            inputs=[dt, self._node_mass, self._node_position, self._node_inertia],
            outputs=[node_rhs],
            device=self.model.device,
        )
        wp.launch(
            cable_segment_inertia_kernel,
            dim=self._segment_count,
            inputs=[dt, self._frame_inertia, self._tangent, self._tangent_inertia],
            outputs=[tangent_rhs],
            device=self.model.device,
        )

    def eval_cable_constraints(
        self,
        dt: float,
        rhs: wp.array[wp.vec3],
        A_diag: wp.array[wp.mat33],
    ) -> None:
        r"""Accumulate elastic forces and their frozen tangents at the current state.

        The stretch term for segment ``e=(a,b)`` is

        .. math::

            E_s^e = \tfrac12 k_s^e\left\|
                (c_b-c_a)-R_e(0,0,L_e)
            \right\|^2,

        where ``c`` is a VIPER rod-node position, ``R_e`` is the segment
        material frame, and ``L_e`` is the rest length.  At each interior
        frame pair ``(e,e+1)``, the Darboux residual is

        .. math::

            r_D = \operatorname{Log}\!\left(
                (R_e^\mathrm{rest})^T R_{e+1}^\mathrm{rest}
                \; (R_e^T R_{e+1})^{-1}
            \right), \qquad
            E_D = \tfrac12 r_D^T
                \operatorname{diag}(k_b,k_b,k_t) r_D.

        The first two components of ``r_D`` are bending and the component
        about the material tangent is twist.  The kernels accumulate
        ``rhs = -\nabla E`` and the frozen positive-semidefinite tangent
        diagonal into the supplied Cable-only rows; the remaining coupling
        blocks are applied by :meth:`matrix_free_multiply`.

        Args:
            dt: Time step [s]. Reserved for future state-dependent
                volume-invariant terms; the current fixed-scale energy is
                rate independent.
            rhs: Right-hand side containing only the Cable rows, with node rows
                followed by material-frame rows.
            A_diag: Dynamic diagonal containing the same Cable-only row layout.
        """
        if self._node_count == 0:
            return
        node_rhs = rhs[: self._node_count]
        tangent_rhs = rhs[self._node_count :]
        tangent_diag = A_diag[self._node_count :]
        if self._segment_count > 0:
            wp.launch(
                cable_eval_stretch_kernel,
                dim=self._segment_count,
                inputs=[
                    self._segment_nodes,
                    self._rest_length,
                    self._stretch_stiffness,
                    self._frame,
                    self._node_position,
                ],
                outputs=[node_rhs, tangent_rhs, tangent_diag, self._frame_edge],
                device=self.model.device,
            )
        if self._darboux_site_count > 0:
            wp.launch(
                cable_eval_darboux_kernel,
                dim=self._darboux_site_count,
                inputs=[
                    self._darboux_segments,
                    self._darboux_rest,
                    self._darboux_weights,
                    self._frame,
                ],
                outputs=[tangent_rhs, tangent_diag, self._darboux_off_diagonal],
                device=self.model.device,
            )

    def eval_contacts(
        self,
        contacts: Contacts | None,
        body_q: wp.array[wp.transform],
        rhs: wp.array[wp.vec3],
        A_diag: wp.array[wp.mat33],
        dt: float,
    ) -> None:
        r"""Accumulate contacts involving Newton capsule proxies onto Cable rows.

        Newton's collision pipeline supplies the capsule contact pairs.  For a
        contact point with segment-local offset ``r``, this method uses

        .. math::

            p = \tfrac12(x_a+x_b)+Rr, \qquad
            \delta p = \tfrac12\delta x_a+\tfrac12\delta x_b-[Rr]_\times\delta\theta.

        Thus each Cable side writes two rod-node rows and one material-frame
        row. Contacts without a Cable side are ignored here.
        """
        if contacts is None:
            self._contact_tangents.zero_()
            return
        if contacts.rigid_contact_max > len(self._contact_tangents):
            self._contact_tangents = wp.zeros(
                contacts.rigid_contact_max, dtype=CableContactTangent, device=self.model.device
            )
        wp.launch(
            cable_eval_contact_kernel,
            dim=len(self._contact_tangents),
            inputs=[
                dt,
                self._node_count,
                self._segment_nodes,
                self._body_to_segment,
                self._segment_com,
                self._node_position,
                self._node_previous,
                self._frame,
                self._frame_begin,
                body_q,
                self.model.shape_body,
                self.model.shape_material_ke,
                self.model.shape_material_kd,
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                contacts.rigid_contact_stiffness,
                contacts.rigid_contact_damping,
            ],
            outputs=[rhs, A_diag, self._contact_tangents],
            device=self.model.device,
        )

    def matrix_free_multiply(self, direction: wp.array[wp.vec3], product: wp.array[wp.vec3]) -> None:
        """Write the cable contribution to ``A * direction`` that is not assembled.

        This covers the position-rotation stretch coupling, its transpose, and
        the rotation-rotation Darboux coupling. The fixed position stencils and the
        per-row rotation diagonals are already in the assembled operator.

        Args:
            direction: Full-length direction array.
            product: Full-length product array; the cable rows are overwritten.
        """
        if self._node_count == 0:
            return
        direction_node = direction[self._position_start : self._position_end]
        direction_rotation = direction[self._rotation_start : self._rotation_end]
        product_node = product[self._position_start : self._position_end]
        product_rotation = product[self._rotation_start : self._rotation_end]
        product_node.zero_()
        product_rotation.zero_()
        if self._segment_count > 0:
            wp.launch(
                cable_stretch_matrix_free_kernel,
                dim=self._segment_count,
                inputs=[
                    self._segment_nodes,
                    self._stretch_stiffness,
                    self._frame_edge,
                    direction_node,
                    direction_rotation,
                ],
                outputs=[product_node, product_rotation],
                device=self.model.device,
            )
        if self._darboux_site_count > 0:
            wp.launch(
                cable_darboux_matrix_free_kernel,
                dim=self._darboux_site_count,
                inputs=[self._darboux_segments, self._darboux_off_diagonal, direction_rotation],
                outputs=[product_rotation],
                device=self.model.device,
            )
        if len(self._contact_tangents) > 0:
            wp.launch(
                cable_contact_matrix_free_kernel,
                dim=len(self._contact_tangents),
                inputs=[self._contact_tangents, direction[self._position_start : self._rotation_end]],
                outputs=[product[self._position_start : self._rotation_end]],
                device=self.model.device,
            )

    def apply_increment(self) -> None:
        """Retract the solved rotation tangents onto the material frames."""
        if self._segment_count == 0:
            return
        wp.launch(
            cable_apply_increment_kernel,
            dim=self._segment_count,
            inputs=[self._frame_begin, self._tangent],
            outputs=[self._frame],
            device=self.model.device,
        )

    def _from_internal_state(self, state_out: State, dt: float) -> None:
        """Reconstruct the proxy segment bodies from the solved cable state.

        Args:
            state_out: Output state that receives the segment body poses and
                velocities.
            dt: Time step [s].
        """
        if self._segment_count == 0:
            return
        wp.launch(
            cable_writeback_kernel,
            dim=self._segment_count,
            inputs=[
                dt,
                self._segment_nodes,
                self._segment_bodies,
                self._segment_com,
                self.model.body_mass,
                self.model.body_flags,
                self._node_position,
                self._node_previous,
                self._frame,
                self._frame_begin,
            ],
            outputs=[state_out.body_q, state_out.body_qd, self._frame_velocity],
            device=self.model.device,
        )

    def reset(self, state: State) -> None:
        """Resynchronize VIPER node positions and material frames from proxy bodies.

        :meth:`_allocate_topology` has already converted the rod model into the
        fixed lookup tables used here:

        .. code-block:: text

            segment e -> Newton body ``_segment_bodies[e]``
            segment e -> VIPER nodes ``_segment_nodes[e, 0:2]``

        This method does not rediscover that topology. It uses
        :func:`cable_collect_state_kernel` to read each mapped body's current
        pose, reconstruct the segment's two rod-node endpoint samples, and
        extract its material frame. Shared endpoint samples are then reduced by
        :func:`cable_average_state_kernel` into one position per VIPER node.

        Args:
            state: State whose segment body poses define the new cable state.
        """
        if self._node_count == 0:
            return
        self._node_position_sum.zero_()
        self._node_velocity_sum.zero_()
        self._node_multiple.zero_()
        wp.launch(
            cable_collect_state_kernel,
            dim=self._segment_count,
            inputs=[
                self._segment_nodes,
                self._segment_bodies,
                self._segment_com,
                self._segment_half_length,
                state.body_q,
                state.body_qd,
            ],
            outputs=[
                self._node_position_sum,
                self._node_velocity_sum,
                self._node_multiple,
                self._frame,
                self._frame_velocity,
            ],
            device=self.model.device,
        )
        wp.launch(
            cable_average_state_kernel,
            dim=self._node_count,
            inputs=[self._node_multiple, self._node_position_sum, self._node_velocity_sum],
            outputs=[self._node_position, self._node_velocity],
            device=self.model.device,
        )
        self._frame_begin.assign(self._frame)

    ####################################################################################################################
    ################################################    Construction    ################################################
    ####################################################################################################################

    def _allocate_topology(self, stretch_stiffness: float, bend_stiffness: float, twist_stiffness: float) -> None:
        """Build the fixed bridge from Newton rod bodies to VIPER solve rows.

        ``ModelBuilder.add_rod`` leaves one capsule body per rod edge plus ROD
        joints between incident edges. This method consumes that model data once
        when the Style3D cable is constructed and creates the lookup tables

        .. code-block:: text

            _segment_bodies[e]   -> Newton capsule body for segment e
            _segment_nodes[e, 0] -> VIPER node at its first endpoint
            _segment_nodes[e, 1] -> VIPER node at its second endpoint

        It also caches rest-sized capsule geometry, initial material frames,
        lumped node mass/inertia data, and elastic coefficients. Runtime state
        synchronization is deliberately separate: :meth:`reset` passes these
        fixed tables to :func:`cable_collect_state_kernel`, which converts the
        current body poses into node positions and material frames.
        """
        model = self.model
        device = model.device

        self._segment_bodies_np = np.asarray(self._collect_segments(), dtype=np.int32)
        self._segment_count = int(self._segment_bodies_np.size)
        body_to_segment = np.full(model.body_count, -1, dtype=np.int32)
        body_to_segment[self._segment_bodies_np] = np.arange(self._segment_count, dtype=np.int32)
        self._body_to_segment = wp.array(body_to_segment, dtype=int, device=device)
        self._contact_tangents = wp.zeros(0, dtype=CableContactTangent, device=device)
        self._needs_sync = False

        if self._segment_count == 0:
            self._initialize_empty()
            return

        geometry = self._segment_geometry()
        self._segment_nodes_np, self._component_offsets = self._build_nodes(geometry["endpoints"])
        self._node_count = int(self._segment_nodes_np.max()) + 1
        nodes = self._node_state(geometry)
        self._build_elastic_data(
            nodes["position"],
            geometry["frames"],
            geometry["rest_length"],
            bend_stiffness,
            twist_stiffness,
        )

        self._stretch_stiffness_np = (stretch_stiffness / geometry["rest_length"]).astype(np.float32)
        if stretch_stiffness <= 0.0:
            self._stretch_stiffness_np[:] = 0.0

        self._segment_nodes = wp.array2d(self._segment_nodes_np, dtype=int, device=device)
        self._segment_bodies = wp.array(self._segment_bodies_np, dtype=int, device=device)
        self._segment_com = wp.array(geometry["com"].astype(np.float32), dtype=wp.vec3, device=device)
        self._segment_half_length = wp.array(geometry["half_length"].astype(np.float32), dtype=float, device=device)
        self._rest_length = wp.array(geometry["rest_length"].astype(np.float32), dtype=float, device=device)
        self._stretch_stiffness = wp.array(self._stretch_stiffness_np, dtype=float, device=device)
        self._frame_edge = wp.zeros(self._segment_count, dtype=wp.vec3, device=device)
        self._frame = wp.array(geometry["frames"].astype(np.float32), dtype=wp.mat33, device=device)
        self._frame_begin = wp.array(geometry["frames"].astype(np.float32), dtype=wp.mat33, device=device)
        self._frame_velocity = wp.array(geometry["frame_velocity"].astype(np.float32), dtype=wp.vec3, device=device)
        self._frame_inertia = wp.array(
            (np.trace(geometry["inertia"], axis1=1, axis2=2) / 3.0).astype(np.float32),
            dtype=float,
            device=device,
        )

        self._initial_node_position = wp.array(nodes["position"].astype(np.float32), dtype=wp.vec3, device=device)
        self._initial_node_velocity = wp.array(nodes["velocity"].astype(np.float32), dtype=wp.vec3, device=device)
        self._node_mass = wp.array(nodes["mass"].astype(np.float32), dtype=float, device=device)
        self._node_world = wp.array(nodes["world"], dtype=int, device=device)

        self._node_position_sum = wp.zeros(self._node_count, dtype=wp.vec3, device=device)
        self._node_velocity_sum = wp.zeros(self._node_count, dtype=wp.vec3, device=device)
        self._node_multiple = wp.zeros(self._node_count, dtype=int, device=device)
        self._needs_sync = True

    def _initialize_empty(self) -> None:
        """Bind empty topology for a model without curve-backed objects."""
        device = self.model.device
        self._node_count = 0
        self._segment_nodes_np = np.zeros((0, 2), dtype=np.int32)
        self._darboux_segments_np = np.zeros((0, 2), dtype=np.int32)
        self._segment_bodies_np = np.zeros(0, dtype=np.int32)
        self._stretch_stiffness_np = np.zeros(0, dtype=np.float32)
        self._component_offsets = ()
        self._darboux_site_count = 0
        self._segment_nodes = wp.zeros((0, 2), dtype=int, device=device)
        self._segment_bodies = wp.zeros(0, dtype=int, device=device)
        self._segment_com = wp.zeros(0, dtype=wp.vec3, device=device)
        self._segment_half_length = wp.zeros(0, dtype=float, device=device)
        self._rest_length = wp.zeros(0, dtype=float, device=device)
        self._stretch_stiffness = wp.zeros(0, dtype=float, device=device)
        self._frame_edge = wp.zeros(0, dtype=wp.vec3, device=device)
        self._frame = wp.zeros(0, dtype=wp.mat33, device=device)
        self._frame_begin = wp.zeros(0, dtype=wp.mat33, device=device)
        self._frame_velocity = wp.zeros(0, dtype=wp.vec3, device=device)
        self._frame_inertia = wp.zeros(0, dtype=float, device=device)
        self._node_mass = wp.zeros(0, dtype=float, device=device)
        self._node_world = wp.zeros(0, dtype=int, device=device)
        self._initial_node_position = wp.zeros(0, dtype=wp.vec3, device=device)
        self._initial_node_velocity = wp.zeros(0, dtype=wp.vec3, device=device)
        self._node_position_sum = wp.zeros(0, dtype=wp.vec3, device=device)
        self._node_velocity_sum = wp.zeros(0, dtype=wp.vec3, device=device)
        self._node_multiple = wp.zeros(0, dtype=int, device=device)
        self._darboux_segments = wp.zeros((0, 2), dtype=int, device=device)

    def _collect_segments(self) -> list[int]:
        """Return the segment body indices of every deformable curve, in order."""
        model = self.model
        if not model.curve_label or model.curve_body_start is None:
            return []
        body_start = model.curve_body_start.numpy()
        body_end = model.curve_body_end.numpy()
        segments: list[int] = []
        for start, end in zip(body_start, body_end, strict=True):
            segments.extend(range(int(start), int(end)))
        return segments

    def _segment_geometry(self) -> dict[str, np.ndarray]:
        """Return rest geometry, initial frames, and source inertia of each segment.

        Endpoints come from each capsule's rest-sized centerline offset around its
        center of mass, so both ``body_frame_origin`` conventions produce the
        same world positions.

        Raises:
            ValueError: If a segment body does not own exactly one capsule shape.
        """
        model = self.model
        shape_body = model.shape_body.numpy()
        shape_type = model.shape_type.numpy()
        shape_scale = model.shape_scale.numpy()
        shapes_of_body: dict[int, list[int]] = {}
        for shape, body in enumerate(shape_body):
            shapes_of_body.setdefault(int(body), []).append(shape)

        body_com = model.body_com.numpy()
        body_q = model.body_q.numpy()
        body_qd = model.body_qd.numpy()
        body_inertia = model.body_inertia.numpy()

        count = self._segment_count
        half_length = np.zeros(count, dtype=np.float64)
        com = np.zeros((count, 3), dtype=np.float64)
        midpoint = np.zeros((count, 3), dtype=np.float64)
        frames = np.zeros((count, 3, 3), dtype=np.float64)
        endpoints = np.zeros((count, 2, 3), dtype=np.float64)
        frame_velocity = np.zeros((count, 3), dtype=np.float64)
        inertia = np.zeros((count, 3, 3), dtype=np.float64)

        for segment, segment_body in enumerate(self._segment_bodies_np):
            body = int(segment_body)
            shapes = shapes_of_body.get(body, [])
            if len(shapes) != 1:
                raise ValueError(
                    f"Cable: segment body {body} must own exactly one capsule shape, found {len(shapes)} shapes."
                )
            shape = shapes[0]
            if int(shape_type[shape]) != int(GeoType.CAPSULE):
                raise ValueError(
                    f"Cable: segment body {body} uses shape type {int(shape_type[shape])}, "
                    f"but only capsules (GeoType.CAPSULE = {int(GeoType.CAPSULE)}) are supported."
                )
            pose = body_q[body]
            rotation = _quaternion_to_matrix(pose[3:7])
            half = float(shape_scale[shape][1])
            if not np.isfinite(half) or half <= 0.0:
                raise ValueError(f"Cable: segment body {body} needs a positive capsule rest length, got {2.0 * half}.")
            half_length[segment] = half
            com[segment] = body_com[body]
            midpoint[segment] = pose[:3] + rotation @ body_com[body]
            frames[segment] = rotation
            endpoints[segment, 0] = pose[:3] + rotation @ (body_com[body] + np.array([0.0, 0.0, -half]))
            endpoints[segment, 1] = pose[:3] + rotation @ (body_com[body] + np.array([0.0, 0.0, half]))
            frame_velocity[segment] = body_qd[body][3:]
            inertia[segment] = body_inertia[body]

        return {
            "half_length": half_length,
            "rest_length": 2.0 * half_length,
            "com": com,
            "midpoint": midpoint,
            "frames": frames,
            "endpoints": endpoints,
            "frame_velocity": frame_velocity,
            "inertia": inertia,
        }

    def _node_state(self, geometry: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Average segment samples into VIPER node centers, velocities, and masses.

        Averaging alone reproduces an interior node but not a chain endpoint: a
        capsule is rest sized, so an end sample sits at half the *rest* length from
        the initial segment midpoint. Starting from one interior node per
        component, the remaining nodes are therefore reconstructed exactly from
        ``node_a + node_b = 2 * midpoint``, which holds for the initial pose.
        """
        model = self.model
        body_mass = model.body_mass.numpy()
        body_world = model.body_world.numpy()
        body_qd = model.body_qd.numpy()
        position = np.zeros((self._node_count, 3), dtype=np.float64)
        velocity = np.zeros((self._node_count, 3), dtype=np.float64)
        mass = np.zeros(self._node_count, dtype=np.float64)
        world = np.full(self._node_count, -1, dtype=np.int32)
        multiple = np.zeros(self._node_count, dtype=np.float64)
        incident: list[list[int]] = [[] for _ in range(self._node_count)]
        for segment, (start, end) in enumerate(self._segment_nodes_np):
            body = int(self._segment_bodies_np[segment])
            linear = body_qd[body][:3]
            for node, sample in ((int(start), 0), (int(end), 1)):
                position[node] += geometry["endpoints"][segment][sample]
                velocity[node] += linear
                mass[node] += 0.5 * float(body_mass[body])
                multiple[node] += 1.0
                world[node] = int(body_world[body])
                incident[node].append(segment)
        multiple = np.maximum(multiple, 1.0)
        position /= multiple[:, None]
        velocity /= multiple[:, None]

        visited = np.zeros(self._node_count, dtype=bool)
        for component in self._component_offsets:
            seed = next(
                (node for node in component if len(incident[node]) == 2),
                -1,
            )
            if seed < 0:
                # A single-segment component has no interior node to anchor the chain.
                continue
            visited[seed] = True
            pending = [seed]
            while pending:
                node = pending.pop()
                for segment in incident[node]:
                    other = self._far_node(segment, node)
                    if visited[other]:
                        continue
                    position[other] = 2.0 * geometry["midpoint"][segment] - position[node]
                    visited[other] = True
                    pending.append(other)

        return {"position": position, "velocity": velocity, "mass": mass, "world": world}

    def _build_nodes(self, endpoints: np.ndarray) -> tuple[np.ndarray, tuple[range, ...]]:
        """Weld segment endpoints through declared rod joints to form VIPER nodes.

        Welding uses only the declared :attr:`~newton.JointType.ROD` joints of each
        curve: endpoints are never joined by spatial proximity. Within one joint
        the closest endpoint pair is welded, which keeps the mapping valid for
        graphs whose edge order differs from the body order.

        Args:
            endpoints: Initial endpoint positions, shape [segment_count, 2, 3].

        Returns:
            The ``[segment_count, 2]`` node index table and the node range of each
            connected component.

        Raises:
            ValueError: If a rod joint leaves its curve, or if a node is a branch.
        """
        model = self.model
        segment_count = self._segment_count
        handle_parent = list(range(2 * segment_count))
        segment_parent = list(range(segment_count))

        def find(parent: list[int], index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(parent: list[int], index_a: int, index_b: int) -> None:
            root_a = find(parent, index_a)
            root_b = find(parent, index_b)
            if root_a != root_b:
                parent[max(root_a, root_b)] = min(root_a, root_b)

        body_to_segment = {int(body): segment for segment, body in enumerate(self._segment_bodies_np)}
        if model.curve_joint_start is not None:
            joint_start = model.curve_joint_start.numpy()
            joint_end = model.curve_joint_end.numpy()
            joint_type = model.joint_type.numpy()
            joint_parent = model.joint_parent.numpy()
            joint_child = model.joint_child.numpy()
            for start, end in zip(joint_start, joint_end, strict=True):
                for joint in range(int(start), int(end)):
                    if int(joint_type[joint]) != int(JointType.ROD):
                        continue
                    parent_body = int(joint_parent[joint])
                    child_body = int(joint_child[joint])
                    if parent_body not in body_to_segment or child_body not in body_to_segment:
                        raise ValueError(
                            f"Cable: rod joint {joint} connects body {parent_body} to body {child_body}, "
                            "and both bodies must be segments of the same curve."
                        )
                    parent_segment = body_to_segment[parent_body]
                    child_segment = body_to_segment[child_body]
                    distances = [
                        (
                            float(np.linalg.norm(endpoints[parent_segment, side] - endpoints[child_segment, other])),
                            side,
                            other,
                        )
                        for side in (0, 1)
                        for other in (0, 1)
                    ]
                    _, parent_side, child_side = min(distances)
                    union(handle_parent, 2 * parent_segment + parent_side, 2 * child_segment + child_side)
                    union(segment_parent, parent_segment, child_segment)

        # Order components by their smallest segment so node ranges stay contiguous.
        segments_of_root: dict[int, list[int]] = {}
        for segment in range(segment_count):
            segments_of_root.setdefault(find(segment_parent, segment), []).append(segment)
        ordered_components = sorted(segments_of_root.values(), key=min)

        node_of_root: dict[int, int] = {}
        segment_nodes = np.zeros((segment_count, 2), dtype=np.int32)
        component_offsets: list[range] = []
        next_node = 0
        for component in ordered_components:
            first_node = next_node
            for segment in component:
                for side in (0, 1):
                    root = find(handle_parent, 2 * segment + side)
                    if root not in node_of_root:
                        node_of_root[root] = next_node
                        next_node += 1
                    segment_nodes[segment, side] = node_of_root[root]
            component_offsets.append(range(first_node, next_node))

        multiplicity = np.bincount(segment_nodes.ravel(), minlength=next_node)
        if np.any(multiplicity > 2):
            branch = int(np.argmax(multiplicity))
            raise ValueError(
                f"Cable: node {branch} is shared by {int(multiplicity[branch])} segment endpoints. "
                "Branching rod topology is not supported; author an ordered chain instead."
            )
        return segment_nodes, tuple(component_offsets)

    def _build_elastic_data(
        self,
        node_position: np.ndarray,
        frames: np.ndarray,
        rest_length: np.ndarray,
        bend_stiffness: float,
        twist_stiffness: float,
    ) -> None:
        """Build dimensionless rest Darboux vectors and EI/L, GJ/L weights."""
        device = self.model.device
        incident: list[list[int]] = [[] for _ in range(self._node_count)]
        for segment, (start, end) in enumerate(self._segment_nodes_np):
            incident[int(start)].append(segment)
            incident[int(end)].append(segment)

        darboux_sites: list[tuple[int, int, int]] = []
        for node, segments in enumerate(incident):
            if len(segments) != 2 or segments[0] == segments[1]:
                continue
            segment0, segment1 = segments
            darboux_sites.append((segment0, segment1, node))
        darboux_segments = np.zeros((len(darboux_sites), 2), dtype=np.int32)
        for site, (segment0, segment1, _node) in enumerate(darboux_sites):
            darboux_segments[site] = (segment0, segment1)
        self._darboux_site_count = len(darboux_sites)
        self._darboux_segments_np = darboux_segments
        self._darboux_segments = wp.array2d(darboux_segments, dtype=int, device=device)

        darboux_rest = np.zeros((len(darboux_sites), 3), dtype=np.float32)
        darboux_weights = np.zeros_like(darboux_rest)
        for site, (segment0, segment1, _node) in enumerate(darboux_sites):
            relative = frames[segment0].T @ frames[segment1]
            q = np.array(wp.quat_from_matrix(wp.mat33(*relative.ravel())), dtype=np.float64)
            q /= np.linalg.norm(q)
            if q[3] < 0.0:
                q = -q
            darboux_rest[site] = 2.0 * q[:3]
            dual_length = 0.5 * (rest_length[segment0] + rest_length[segment1])
            darboux_weights[site] = np.maximum([bend_stiffness, bend_stiffness, twist_stiffness], 0.0) / dual_length
        self._darboux_rest = wp.array(darboux_rest, dtype=wp.vec3, device=device)
        self._darboux_weights = wp.array(darboux_weights, dtype=wp.vec3, device=device)
        self._darboux_off_diagonal = wp.zeros(len(darboux_sites), dtype=wp.mat33, device=device)

    def _far_node(self, segment: int, node: int) -> int:
        """Return the segment endpoint that is not ``node``."""
        start, end = (int(value) for value in self._segment_nodes_np[segment])
        return end if start == node else start
