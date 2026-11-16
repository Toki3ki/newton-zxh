# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from ...core.types import override
from ...exceptions import NewtonDeprecationWarning
from ...sim import Contacts, Control, Model, ModelBuilder, State
from ..solver import SolverBase
from . import solve_cable
from .builder import PDMatrixBuilder, _compute_max_neighbors
from .collision import CableCollision, Collision, ContactSystem
from .kernels import (
    accumulate_dragging_pd_diag_kernel,
    deactivate_zero_mass_particles_kernel,
    eval_bend_kernel,
    eval_drag_force_kernel,
    eval_stretch_kernel,
    init_rhs_kernel,
    init_step_kernel,
    nonlinear_step_kernel,
    prepare_jacobi_preconditioner_kernel,
    prepare_jacobi_preconditioner_no_contact_hessian_kernel,
    update_velocity,
)
from .linear_solver import PcgSolver, SparseMatrixELL

AttributeAssignment = Model.AttributeAssignment
AttributeFrequency = Model.AttributeFrequency


class _LegacyUnset:
    def __repr__(self) -> str:
        return "_LEGACY_UNSET"


_LEGACY_UNSET: Any = _LegacyUnset()


def _warn_legacy_attribute(name: str, config_name: str) -> None:
    """Point deprecated attribute warnings at the caller's read or assignment."""
    warnings.warn(
        f"SolverStyle3D.{name} is deprecated as of Newton 1.8; use solver.config.{config_name} instead.",
        NewtonDeprecationWarning,
        stacklevel=3,
    )


########################################################################################################################
#################################################    Style3D Solver    #################################################
########################################################################################################################


class SolverStyle3D(SolverBase):
    r"""Projective dynamics based cloth solver.

    References:
        1. Baraff, D. & Witkin, A. "Large Steps in Cloth Simulation."
        2. Liu, T. et al. "Fast Simulation of Mass-Spring Systems."

    Implicit-Euler method solves the following non-linear equation:

    .. math::

        (M / dt^2 + H(x)) \cdot dx &= (M / dt^2) \cdot (x_{prev} + v_{prev} \cdot dt - x) + f_{ext}(x) + f_{int}(x) \\
                                   &= (M / dt^2) \cdot (x_{prev} + v_{prev} \cdot dt + (dt^2 / M) \cdot f_{ext}(x) - x) + f_{int}(x) \\
                                   &= (M / dt^2) \cdot (x_{inertia} - x) + f_{int}(x)

    Notations:
        - :math:`M`: mass matrix
        - :math:`x`: unsolved particle position
        - :math:`H`: Hessian matrix (function of x)
        - :math:`P`: PD-approximated Hessian matrix (constant)
        - :math:`A`: :math:`M / dt^2 + H(x)` or :math:`M / dt^2 + P`
        - :math:`rhs`: Right hand side of the equation: :math:`(M / dt^2) \cdot (x_{inertia} - x) + f_{int}(x)`
        - :math:`res`: Residual: :math:`rhs - A \cdot dx_{init}`, or rhs if :math:`dx_{init} = 0`

    See Also:
        :doc:`newton.solvers.style3d </api/newton_solvers_style3d>` exposes
        helper functions that populate Style3D cloth data on a
        :class:`~newton.ModelBuilder`.

    Example:
        Build a mesh-based cloth with
        :func:`newton.solvers.style3d.add_cloth_mesh`::

            from newton.solvers import style3d

            builder = newton.ModelBuilder()
            SolverStyle3D.register_custom_attributes(builder)
            style3d.add_cloth_mesh(
                builder,
                pos=wp.vec3(0.0, 0.0, 0.0),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                vertices=mesh.vertices.tolist(),
                indices=mesh.indices.tolist(),
                density=0.3,
                tri_aniso_ke=wp.vec3(1.0e2, 1.0e2, 1.0e1),
                edge_aniso_ke=wp.vec3(2.0e-5, 1.0e-5, 5.0e-6),
            )

        Or build a grid with :func:`newton.solvers.style3d.add_cloth_grid`::

            style3d.add_cloth_grid(
                builder,
                pos=wp.vec3(-0.5, 0.0, 2.0),
                rot=wp.quat_identity(),
                dim_x=64,
                dim_y=32,
                cell_x=0.1,
                cell_y=0.1,
                vel=wp.vec3(0.0, 0.0, 0.0),
                mass=0.1,
                tri_aniso_ke=wp.vec3(1.0e2, 1.0e2, 1.0e1),
                edge_aniso_ke=wp.vec3(2.0e-4, 1.0e-4, 5.0e-5),
            )

    """

    @dataclass(kw_only=True)
    class Config:
        """Runtime options for :class:`SolverStyle3D`.

        The solver keeps the supplied object as :attr:`SolverStyle3D.config`.
        Changes made before a subsequent :meth:`SolverStyle3D.step` take
        effect on that step. Captured CUDA graphs must be recaptured to use
        updated configuration values.

        Example:
            Configure a solver for an existing model::

                config = SolverStyle3D.Config(iterations=20, linear_iterations=10)
                solver = SolverStyle3D(model, config=config)
                solver.config.iterations = 30
        """

        iterations: int = 10
        """Number of non-linear iterations per step. Must be at least 1."""

        linear_iterations: int = 10
        """Number of PCG iterations per non-linear iteration, capped by PCG capacity (999 by default)."""

        drag_spring_stiff: float = 1e2
        """Stiffness of the mouse-dragging spring [N/m]."""

        enable_mouse_dragging: bool = False
        """Whether to enable the mouse-dragging kernel."""

        cable_fixed_scale: bool = True
        """Keep the section scale fixed at one."""

        cable_preserve_volume: bool = True
        """Enable the internal dynamic-section volume penalty."""

        cable_relaxation: float = 0.1
        """Fixed relaxation applied to each local cable frame and scale update."""

    config: Config
    """Live solver options, retained by reference when supplied at construction.

    Edit fields or replace this object with a :class:`Config` between calls
    to :meth:`step`. Captured CUDA graphs must be recaptured after changes.
    """

    @property
    def linear_iterations(self) -> int:
        """Legacy alias for :attr:`Config.linear_iterations`.

        .. deprecated:: 1.8
            Use ``solver.config.linear_iterations`` instead.
        """
        _warn_legacy_attribute("linear_iterations", "linear_iterations")
        return self.config.linear_iterations

    @linear_iterations.setter
    def linear_iterations(self, value: int) -> None:
        _warn_legacy_attribute("linear_iterations", "linear_iterations")
        self.config.linear_iterations = value

    @property
    def nonlinear_iterations(self) -> int:
        """Legacy alias for :attr:`Config.iterations`.

        .. deprecated:: 1.8
            Use ``solver.config.iterations`` instead.
        """
        _warn_legacy_attribute("nonlinear_iterations", "iterations")
        return self.config.iterations

    @nonlinear_iterations.setter
    def nonlinear_iterations(self, value: int) -> None:
        _warn_legacy_attribute("nonlinear_iterations", "iterations")
        self.config.iterations = value

    @property
    def drag_spring_stiff(self) -> float:
        """Legacy alias for :attr:`Config.drag_spring_stiff`.

        .. deprecated:: 1.8
            Use ``solver.config.drag_spring_stiff`` instead.
        """
        _warn_legacy_attribute("drag_spring_stiff", "drag_spring_stiff")
        return self.config.drag_spring_stiff

    @drag_spring_stiff.setter
    def drag_spring_stiff(self, value: float) -> None:
        _warn_legacy_attribute("drag_spring_stiff", "drag_spring_stiff")
        self.config.drag_spring_stiff = value

    @property
    def enable_mouse_dragging(self) -> bool:
        """Legacy alias for :attr:`Config.enable_mouse_dragging`.

        .. deprecated:: 1.8
            Use ``solver.config.enable_mouse_dragging`` instead.
        """
        _warn_legacy_attribute("enable_mouse_dragging", "enable_mouse_dragging")
        return self.config.enable_mouse_dragging

    @enable_mouse_dragging.setter
    def enable_mouse_dragging(self, value: bool) -> None:
        _warn_legacy_attribute("enable_mouse_dragging", "enable_mouse_dragging")
        self.config.enable_mouse_dragging = value

    def __init__(
        self,
        model: Model,
        *,
        config: Config | None = None,
        iterations: int = _LEGACY_UNSET,
        linear_iterations: int = _LEGACY_UNSET,
        drag_spring_stiff: float = _LEGACY_UNSET,
        enable_mouse_dragging: bool = _LEGACY_UNSET,
    ):
        """Initialize the Style3D solver.

        .. deprecated:: 1.8
            The ``iterations``, ``linear_iterations``, ``drag_spring_stiff``,
            and ``enable_mouse_dragging`` keyword arguments are deprecated.
            Pass their values through ``config=SolverStyle3D.Config(...)`` instead.

        Args:
            model: The :class:`~newton.Model` containing Style3D attributes to integrate.
            config: Runtime options. If ``None``, use :class:`Config` defaults with
                any supplied legacy keyword options. The solver keeps this object
                publicly as :attr:`config`; changes take effect on the next step.
                An explicit configuration cannot be combined with legacy options.
            iterations: Deprecated keyword option for :attr:`Config.iterations`.
            linear_iterations: Deprecated keyword option for :attr:`Config.linear_iterations`.
            drag_spring_stiff: Deprecated keyword option for :attr:`Config.drag_spring_stiff`.
            enable_mouse_dragging: Deprecated keyword option for :attr:`Config.enable_mouse_dragging`.

        Raises:
            TypeError: If ``config`` is not a :class:`Config` or ``None``, or if
                a configuration is combined with legacy keyword options.
        """

        legacy_options = {
            name: value
            for name, value in (
                ("iterations", iterations),
                ("linear_iterations", linear_iterations),
                ("drag_spring_stiff", drag_spring_stiff),
                ("enable_mouse_dragging", enable_mouse_dragging),
            )
            if value is not _LEGACY_UNSET
        }
        if config is None:
            config = self.Config(**legacy_options)
        elif not isinstance(config, self.Config):
            raise TypeError(f"Expected config to be {self.Config.__qualname__}, got {type(config).__qualname__}.")
        elif legacy_options:
            raise TypeError("Specify either 'config' or legacy solver options, not both.")

        if legacy_options:
            warnings.warn(
                "SolverStyle3D keyword options "
                f"({', '.join(legacy_options)}) are deprecated as of Newton 1.8; "
                "use config=SolverStyle3D.Config(...) instead.",
                NewtonDeprecationWarning,
                stacklevel=2,
            )

        super().__init__(model)
        self.config = config
        if not hasattr(model, "style3d"):
            raise AttributeError(
                "Style3D custom attributes are missing from the model. "
                "Call SolverStyle3D.register_custom_attributes() before building the model."
            )
        self.style3d = model.style3d
        self._cable = solve_cable._CableWorkspace(model, config)
        cable = self._cable
        # Keep solver-specific activity constraints out of the shared model.
        self._particle_flags = (
            wp.clone(model.particle_flags) if model.particle_count else wp.empty(0, dtype=int, device=self.device)
        )
        if model.particle_count > 0:
            wp.launch(
                deactivate_zero_mass_particles_kernel,
                dim=model.particle_count,
                inputs=[model.particle_mass],
                outputs=[self._particle_flags],
                device=self.device,
            )
        cloth_collision = Collision(model) if model.tri_count or model.edge_count else None
        cable_collision = CableCollision(model, cable) if cable.n else None
        self.collision: ContactSystem | None = (
            ContactSystem(model, cloth=cloth_collision, cable=cable_collision)
            if cloth_collision is not None or cable_collision is not None
            else None
        )
        self.linear_solver = PcgSolver(model.particle_count, self.device)

        # Fixed PD matrix
        self.pd_non_diags = SparseMatrixELL()
        self.pd_diags = wp.zeros(model.particle_count, dtype=float, device=self.device)
        self._precompute(model)

        # Non-linear equation variables
        self.dx = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        self.rhs = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        self.x_prev = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        self.x_inertia = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)

        # Static part of A_diag, full A_diag, and inverse of A_diag
        self.static_A_diags = wp.zeros(model.particle_count, dtype=float, device=self.device)
        self.inv_A_diags = wp.zeros(model.particle_count, dtype=wp.mat33, device=self.device)
        self._step_diagonal = wp.zeros(model.particle_count, dtype=float, device=self.device)
        self.A_diags = wp.zeros(model.particle_count, dtype=wp.mat33, device=self.device)
        self._base_entries = wp.clone(self.pd_non_diags.nz_ell)
        cable.bind_position_system(self.pd_non_diags, self.pd_diags, self.static_A_diags, self.rhs, self.dx)

        # Drag info
        self.drag_pos = wp.zeros(1, dtype=wp.vec3, device=self.device)
        self.drag_index = wp.array([-1], dtype=int, device=self.device)
        self.drag_bary_coord = wp.zeros(1, dtype=wp.vec3, device=self.device)

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts,
        dt: float,
        *,
        observables: SolverBase.Observables | None = None,
    ) -> None:
        """Advance the Style3D solver by one time step.

        The solver performs non-linear projective dynamics iterations with
        optional collision handling. During the solve, positions in
        ``state_in`` are updated in-place to the current iterate; the final
        positions and velocities are written to ``state_out``.

        Args:
            state_in: Input :class:`newton.State` (positions updated in-place).
            state_out: Output :class:`newton.State` with the final state.
            control: :class:`newton.Control` input (currently unused).
            contacts: :class:`newton.Contacts` used for collision response.
            dt: Time step in seconds.
            observables: Optional solver observable arrays allocated by :meth:`observables`.
                This solver declares no supported observables, so only an empty
                container is accepted.

        Raises:
            ValueError: If :attr:`Config.iterations` is less than 1. Neither
                input nor output state is modified in this case.
        """
        self.validate_observables(observables, contacts)
        self._require_unreduced_contacts(contacts)

        if self.config.iterations < 1:
            raise ValueError(f"Config.iterations must be at least 1, got {self.config.iterations}.")

        cable = self._cable
        cable.config = self.config

        if cable.n:
            if not cable.fixed_scale and self.collision is not None and self.collision.has_cable and contacts is not None:
                raise ValueError("Dynamic cable radii do not yet support collision contacts.")
            cable.frame_begin(state_in, dt)
        # Model masses and flags may change between solver steps.
        wp.copy(
            self._particle_flags,
            self.model.particle_flags if self.model.particle_flags is not None else self._particle_flags,
        )

        if self.model.particle_count > 0:
            wp.launch(
                deactivate_zero_mass_particles_kernel,
                dim=self.model.particle_count,
                inputs=[self.model.particle_mass],
                outputs=[self._particle_flags],
                device=self.device,
            )

        cable.activate_particles(self.model.particle_mass, self._particle_flags)
        if self.collision is not None:
            self.collision.frame_begin(state_in, contacts, dt)

        wp.launch(
            kernel=init_step_kernel,
            dim=self.model.particle_count,
            inputs=[
                dt,
                self.model.gravity,
                self.model.particle_world,
                state_in.particle_f,
                state_in.particle_qd,
                state_in.particle_q,
                self.x_prev,
                self.pd_diags,
                self.model.particle_mass,
                self._particle_flags,
            ],
            outputs=[
                self.x_inertia,
                self.static_A_diags,
                self.dx,
            ],
            device=self.device,
        )

        if self.config.enable_mouse_dragging:
            wp.launch(
                accumulate_dragging_pd_diag_kernel,
                dim=1,
                inputs=[
                    self.config.drag_spring_stiff,
                    self.drag_index,
                    self.drag_bary_coord,
                    self.model.tri_indices,
                    self._particle_flags,
                ],
                outputs=[self.static_A_diags],
                device=self.device,
            )

        wp.copy(self._step_diagonal, self.static_A_diags)
        if cable.n:
            wp.copy(cable.predicted, self.x_inertia, count=cable.n)

        for _iter in range(self.config.iterations):
            if cable.n:
                wp.copy(cable.center, state_in.particle_q, count=cable.n)
            wp.launch(
                init_rhs_kernel,
                dim=self.model.particle_count,
                inputs=[
                    dt,
                    state_in.particle_q,
                    self.x_inertia,
                    self.model.particle_mass,
                ],
                outputs=[self.rhs],
                device=self.device,
            )

            wp.launch(
                eval_stretch_kernel,
                dim=len(self.model.tri_areas),
                inputs=[
                    state_in.particle_q,
                    self.model.tri_areas,
                    self.model.tri_poses,
                    self.model.tri_indices,
                    self.style3d.tri_aniso_ke,
                ],
                outputs=[self.rhs],
                device=self.device,
            )

            wp.launch(
                eval_bend_kernel,
                dim=len(self.style3d.edge_rest_area),
                inputs=[
                    state_in.particle_q,
                    self.style3d.edge_rest_area,
                    self.style3d.edge_bending_cot,
                    self.model.edge_indices,
                    self.model.edge_bending_properties,
                ],
                outputs=[self.rhs],
                device=self.device,
            )

            cable.local_step(
                dt,
                state_in.particle_q,
                self._step_diagonal,
                self._base_entries,
                contact_energy=None if self.collision is None else self.collision.evaluate_cable_energy,
            )
            if self.config.enable_mouse_dragging:
                wp.launch(
                    eval_drag_force_kernel,
                    dim=1,
                    inputs=[
                        self.config.drag_spring_stiff,
                        self.drag_index,
                        self.drag_pos,
                        self.drag_bary_coord,
                        self.model.tri_indices,
                        state_in.particle_q,
                    ],
                    outputs=[self.rhs],
                    device=self.device,
                )

            if self.collision is not None:
                self.collision.accumulate_contact_force(
                    dt,
                    _iter,
                    state_in,
                    state_out,
                    contacts,
                    self.rhs,
                    self.x_prev,
                    self.static_A_diags,
                )
                wp.launch(
                    prepare_jacobi_preconditioner_kernel,
                    dim=self.model.particle_count,
                    inputs=[
                        self.collision.preconditioner_diagonal,
                        self.collision.contact_hessian_diagonal(),
                        self._particle_flags,
                    ],
                    outputs=[self.inv_A_diags],
                    device=self.device,
                )
            else:
                wp.launch(
                    prepare_jacobi_preconditioner_no_contact_hessian_kernel,
                    dim=self.model.particle_count,
                    inputs=[self.static_A_diags],
                    outputs=[self.inv_A_diags],
                    device=self.device,
                )

            self.linear_solver.solve(
                self.pd_non_diags,
                self.static_A_diags,
                self.dx if _iter == 0 else None,
                self.rhs,
                self.inv_A_diags,
                self.dx,
                self.config.linear_iterations,
                None if self.collision is None else self.collision.hessian_multiply,
            )

            if self.collision is not None:
                self.collision.linear_iteration_end(self.dx)

            wp.launch(
                nonlinear_step_kernel,
                dim=self.model.particle_count,
                inputs=[state_in.particle_q],
                outputs=[state_out.particle_q, self.dx],
                device=self.device,
            )

            state_in.particle_q.assign(state_out.particle_q)

        wp.launch(
            kernel=update_velocity,
            dim=self.model.particle_count,
            inputs=[dt, self.x_prev, state_out.particle_q],
            outputs=[state_out.particle_qd],
            device=self.device,
        )

        if self.collision is not None:
            self.collision.frame_end(state_out, dt)
        if cable.n:
            cable.write_body_state(state_in, state_out, dt)

    def rebuild_bvh(self, state: State):
        if self.collision is not None:
            self.collision.rebuild_bvh(state)

    @override
    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """Declare Style3D custom attributes under the ``style3d`` namespace.

        See Also:
            :ref:`custom_attributes` for the custom attribute system overview.
        """
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tri_aniso_ke",
                frequency=AttributeFrequency.TRIANGLE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.vec3,
                default=wp.vec3(0.0),
                namespace="style3d",
                keep_empty=True,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="edge_rest_area",
                frequency=AttributeFrequency.EDGE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="style3d",
                keep_empty=True,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="edge_bending_cot",
                frequency=AttributeFrequency.EDGE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.vec4,
                default=wp.vec4(0.0, 0.0, 0.0, 0.0),
                namespace="style3d",
                keep_empty=True,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="aniso_ke",
                frequency=AttributeFrequency.EDGE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.vec3,
                default=wp.vec3(0.0),
                namespace="style3d",
            )
        )

    def _precompute(self, model: Model):
        with wp.ScopedTimer("SolverStyle3D::precompute()"):
            if (
                not hasattr(model, "style3d")
                or not hasattr(model.style3d, "tri_aniso_ke")
                or not hasattr(model.style3d, "edge_rest_area")
                or not hasattr(model.style3d, "edge_bending_cot")
            ):
                raise AttributeError(
                    "Style3D custom attributes are missing from the model. "
                    "Call SolverStyle3D.register_custom_attributes() before building the model."
                )

            tri_indices = model.tri_indices.numpy() if model.tri_count else np.empty((0, 3), dtype=np.int32)
            tri_poses = model.tri_poses.numpy().tolist() if model.tri_count else []
            tri_areas = model.tri_areas.numpy().tolist() if model.tri_count else []
            edge_indices = model.edge_indices.numpy() if model.edge_count else np.empty((0, 4), dtype=np.int32)
            edge_bending_properties = model.edge_bending_properties.numpy().tolist() if model.edge_count else []
            tri_aniso_ke = model.style3d.tri_aniso_ke.numpy().tolist() if model.tri_count else []
            edge_rest_area = model.style3d.edge_rest_area.numpy().tolist() if model.edge_count else []
            edge_bending_cot = model.style3d.edge_bending_cot.numpy().tolist() if model.edge_count else []
            def assemble(max_neighbors):
                """Assemble from scratch so failed attempts never contribute partial entries."""
                matrix = PDMatrixBuilder(model.particle_count, max_neighbor=max_neighbors)
                matrix.add_stretch_constraints(tri_indices, tri_poses, tri_aniso_ke, tri_areas)
                matrix.add_bend_constraints(edge_indices, edge_bending_properties, edge_rest_area, edge_bending_cot)
                self._cable.add_position_constraints(matrix)
                return matrix

            try:
                pd_matrix_builder = assemble(32)
            except ValueError:
                # Only overflowing meshes pay for counting stretch and bending connections.
                max_neighbors = max(
                    _compute_max_neighbors(model.particle_count, tri_indices, edge_indices),
                    self._cable.max_position_neighbors(),
                )
                pd_matrix_builder = assemble(max_neighbors)
            self.pd_diags, self.pd_non_diags.num_nz, self.pd_non_diags.nz_ell = pd_matrix_builder.finalize(self.device)

    def _update_drag_info(self, index: int, pos: wp.vec3, bary_coord: wp.vec3):
        """Should be invoked when state changed."""
        # print([index, pos, bary_coord])
        self.drag_bary_coord.fill_(bary_coord)
        self.drag_index.fill_(index)
        self.drag_pos.fill_(pos)
