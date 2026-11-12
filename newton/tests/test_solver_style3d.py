# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from dataclasses import replace
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.style3d.builder import PDMatrixBuilder, _compute_max_neighbors
from newton._src.solvers.style3d.kernels import eval_tet_sipd_force_kernel, initialize_tet_inv_qa_kernel
from newton.exceptions import NewtonDeprecationWarning
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _assert_pd_matrix(test, model, solver):
    """Compare the stored matrix with independently assembled dense element contributions."""
    expected = np.zeros((model.particle_count, model.particle_count), dtype=np.float64)
    for face, pose, area, stiffness in zip(
        model.tri_indices.numpy(),
        model.tri_poses.numpy(),
        model.tri_areas.numpy(),
        model.style3d.tri_aniso_ke.numpy(),
        strict=True,
    ):
        gradients = np.vstack((-pose.sum(axis=0), pose)).astype(np.float64)
        weights = stiffness[:2] + stiffness[2]
        expected[np.ix_(face, face)] += area * (gradients * weights) @ gradients.T
    for edge, prop, area, cot in zip(
        model.edge_indices.numpy(),
        model.edge_bending_properties.numpy(),
        model.style3d.edge_rest_area.numpy(),
        model.style3d.edge_bending_cot.numpy(),
        strict=True,
    ):
        if edge[0] < 0 or edge[1] < 0:
            continue
        weights = np.array([-cot[0] - cot[2], -cot[1] - cot[3], cot[2] + cot[3], cot[0] + cot[1]])
        expected[np.ix_(edge, edge)] += prop[0] * (3.0 / area) * np.outer(weights, weights)
    actual = np.diag(solver.pd_diags.numpy())
    counts = solver.pd_non_diags.num_nz.numpy()
    entries = solver.pd_non_diags.nz_ell.numpy()
    for row, count in enumerate(counts):
        columns = entries["column_index"][:count, row]
        test.assertEqual(len(set(columns.tolist())), count)
        actual[row, columns] += entries["value"][:count, row]
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-4)
    np.testing.assert_allclose(actual, actual.T, rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(actual.sum(axis=1), 0.0, atol=1e-3)


def test_constructor_precomputes_fixed_pd_matrix(test, device):
    """Preserve the ordinary grid's assembled PD stiffness matrix."""
    builder = newton.ModelBuilder()
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    newton.solvers.style3d.add_cloth_grid(
        builder,
        pos=wp.vec3(0.0, 0.0, 1.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=2,
        dim_y=2,
        cell_x=0.1,
        cell_y=0.1,
        mass=0.1,
        tri_aniso_ke=wp.vec3(1.0e2, 1.0e2, 1.0e1),
        edge_aniso_ke=wp.vec3(2.0e-4, 1.0e-4, 5.0e-5),
    )
    model = builder.finalize(device=device)

    solver = newton.solvers.SolverStyle3D(model)
    config = newton.solvers.SolverStyle3D.Config()

    test.assertEqual(solver.config, config)

    test.assertGreater(float(solver.pd_diags.numpy().sum()), 0.0)
    test.assertGreater(int(solver.pd_non_diags.num_nz.numpy().sum()), 0)
    _assert_pd_matrix(test, model, solver)


def test_config_constructor_options(test, device):
    """Match Config and legacy cloth solves, including runtime updates and conflicts."""
    builder = newton.ModelBuilder()
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    newton.solvers.style3d.add_cloth_grid(
        builder,
        pos=(0.0, 0.0, 1.0),
        rot=wp.quat_identity(),
        vel=(0.0, 0.0, 0.0),
        dim_x=1,
        dim_y=1,
        cell_x=0.1,
        cell_y=0.1,
        mass=0.1,
    )
    model = builder.finalize(device=device)
    config = newton.solvers.SolverStyle3D.Config(
        iterations=3,
        linear_iterations=5,
        drag_spring_stiff=42.0,
        enable_mouse_dragging=True,
    )

    solver = newton.solvers.SolverStyle3D(model, config=config)
    test.assertIs(solver.config, config)
    with test.assertWarnsRegex(NewtonDeprecationWarning, "config=SolverStyle3D.Config") as warning:
        legacy_solver = newton.solvers.SolverStyle3D(
            model,
            iterations=config.iterations,
            linear_iterations=config.linear_iterations,
            drag_spring_stiff=config.drag_spring_stiff,
            enable_mouse_dragging=config.enable_mouse_dragging,
        )
    test.assertEqual(warning.filename, __file__)
    test.assertIsNot(legacy_solver.config, config)
    test.assertEqual(legacy_solver.config, config)

    contacts = newton.CollisionPipeline(model).contacts()
    control = model.control()

    def solve(current_solver):
        state_in, state_out = model.state(), model.state()
        current_solver._update_drag_info(0, wp.vec3(0.0, 0.0, 1.1), wp.vec3(1.0, 0.0, 0.0))
        current_solver.step(state_in, state_out, control, contacts, 0.01)
        return state_out.particle_q.numpy(), state_out.particle_qd.numpy()

    outputs = [solve(current_solver) for current_solver in (solver, legacy_solver)]
    np.testing.assert_allclose(outputs[0][0], outputs[1][0], atol=1e-6)
    np.testing.assert_allclose(outputs[0][1], outputs[1][1], atol=1e-6)
    test.assertTrue(np.isfinite(outputs[0][0]).all())
    test.assertGreater(np.linalg.norm(outputs[0][1]), 0.0)

    # Reuse a solver after each individual edit and compare with a fresh solver.
    previous = outputs[0]
    for name, value in (
        ("iterations", 1),
        ("linear_iterations", 1),
        ("drag_spring_stiff", 420.0),
        ("enable_mouse_dragging", False),
    ):
        with test.subTest(runtime_option=name):
            setattr(config, name, value)
            actual = solve(solver)
            expected = solve(newton.solvers.SolverStyle3D(model, config=replace(config)))
            np.testing.assert_allclose(actual[0], expected[0], atol=1e-6)
            np.testing.assert_allclose(actual[1], expected[1], atol=1e-6)
            test.assertGreater(np.max(np.abs(actual[1] - previous[1])), 1e-6)
            legacy_name = "nonlinear_iterations" if name == "iterations" else name
            with test.assertWarnsRegex(NewtonDeprecationWarning, f"solver.config.{name}") as warning:
                test.assertEqual(getattr(solver, legacy_name), value)
            test.assertEqual(warning.filename, __file__)
            previous = actual

    # Replacing the public configuration also updates the next solve.
    solver.config = replace(legacy_solver.config)
    test.assertIsNot(solver.config, config)
    actual = solve(solver)
    np.testing.assert_allclose(actual[0], outputs[0][0], atol=1e-6)
    np.testing.assert_allclose(actual[1], outputs[0][1], atol=1e-6)

    # Legacy setters update the same configuration used by the solver.
    for name in ("iterations", "linear_iterations", "drag_spring_stiff", "enable_mouse_dragging"):
        legacy_name = "nonlinear_iterations" if name == "iterations" else name
        with test.assertWarnsRegex(NewtonDeprecationWarning, f"solver.config.{name}") as warning:
            setattr(solver, legacy_name, getattr(config, name))
        test.assertEqual(warning.filename, __file__)
    test.assertEqual(solver.config, config)
    actual = solve(solver)
    np.testing.assert_allclose(actual[0], previous[0], atol=1e-6)
    np.testing.assert_allclose(actual[1], previous[1], atol=1e-6)

    default_config = newton.solvers.SolverStyle3D.Config()
    with test.assertWarnsRegex(NewtonDeprecationWarning, "config=SolverStyle3D.Config"):
        partial_solver = newton.solvers.SolverStyle3D(model, config=None, iterations=2, drag_spring_stiff=0.0)
    test.assertEqual(partial_solver.config, replace(default_config, iterations=2, drag_spring_stiff=0.0))

    for name, value in (
        ("iterations", default_config.iterations),
        ("linear_iterations", default_config.linear_iterations),
        ("drag_spring_stiff", default_config.drag_spring_stiff),
        ("enable_mouse_dragging", default_config.enable_mouse_dragging),
    ):
        with test.subTest(option=name):
            with test.assertRaisesRegex(TypeError, "either 'config' or legacy solver options"):
                newton.solvers.SolverStyle3D(model, config=config, **{name: value})
    with test.assertRaisesRegex(TypeError, "Expected config to be SolverStyle3D.Config"):
        newton.solvers.SolverStyle3D(model, config={})

    # Invalid runtime edits must fail before either state is modified.
    valid_config = replace(solver.config)
    state_in, state_out = model.state(), model.state()
    state_in.particle_qd.fill_(wp.vec3(1.0, 2.0, 3.0))
    state_out.particle_q.fill_(wp.vec3(4.0, 5.0, 6.0))
    state_out.particle_qd.fill_(wp.vec3(7.0, 8.0, 9.0))
    arrays = (state_in.particle_q, state_in.particle_qd, state_out.particle_q, state_out.particle_qd)
    before = [array.numpy().copy() for array in arrays]
    for iterations in (0, -1):
        with test.subTest(invalid_iterations=iterations):
            if iterations == 0:
                solver.config.iterations = iterations
            else:
                solver.config = replace(valid_config, iterations=iterations)
            with test.assertRaisesRegex(ValueError, "Config.iterations must be at least 1"):
                solver.step(state_in, state_out, control, contacts, 0.01)
            for array, expected in zip(arrays, before, strict=True):
                np.testing.assert_array_equal(array.numpy(), expected)

    solver.config = valid_config
    actual = solve(solver)
    np.testing.assert_allclose(actual[0], previous[0], atol=1e-6)
    np.testing.assert_allclose(actual[1], previous[1], atol=1e-6)


def test_high_valence_pd_matrix(test, device):
    """Preserve stretch and bending entries across the old capacity boundary."""
    for n, outer_ring in ((32, False), (33, False), (20, True)):
        with test.subTest(neighbors=n, outer_ring=outer_ring):
            angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
            vertices = [[0.0, 0.0, 0.0], *np.column_stack((np.cos(angle), np.sin(angle), np.zeros(n))).tolist()]
            triangles = [[0, 1 + i, 1 + (i + 1) % n] for i in range(n)]
            if outer_ring:
                outer_angle = angle + np.pi / n
                vertices += np.column_stack((2 * np.cos(outer_angle), 2 * np.sin(outer_angle), np.zeros(n))).tolist()
                triangles += [[1 + i, n + 1 + i, 1 + (i + 1) % n] for i in range(n)]
            builder = newton.ModelBuilder()
            newton.solvers.SolverStyle3D.register_custom_attributes(builder)
            newton.solvers.style3d.add_cloth_mesh(
                builder,
                pos=(0.0, 0.0, 0.0),
                rot=wp.quat_identity(),
                vel=(0.0, 0.0, 0.0),
                vertices=vertices,
                indices=np.array(triangles).flatten().tolist(),
                density=1.0,
                tri_aniso_ke=wp.vec3(100.0, 100.0, 10.0),
                edge_aniso_ke=wp.vec3(2.0e-4, 1.0e-4, 5.0e-5),
            )
            model = builder.finalize(device=device)
            solver = newton.solvers.SolverStyle3D(
                model, config=newton.solvers.SolverStyle3D.Config(iterations=1, linear_iterations=1)
            )
            test.assertEqual(int(solver.pd_non_diags.num_nz.numpy()[0]), n * (2 if outer_ring else 1))
            _assert_pd_matrix(test, model, solver)


class TestSolverStyle3D(unittest.TestCase):
    def test_capacity_matches_repeated_vertex_assembly(self):
        """Count self-target slots from repeated stencil positions as the kernels do."""
        self.assertEqual(_compute_max_neighbors(0, [], []), 0)
        self.assertEqual(_compute_max_neighbors(4, [], [[-1, 0, 1, 2]]), 0)
        for kind in ("stretch", "bend"):
            with self.subTest(kind=kind):
                faces = [[0, i, i + 1] for i in range(1, 32)] + [[0, 0, 1]]
                edges = [[0, i, i + 1, i + 2] for i in range(1, 31)] + [[0, 0, 1, 2]]
                width = _compute_max_neighbors(33, faces if kind == "stretch" else [], edges if kind == "bend" else [])
                self.assertEqual(width, 33)
                matrix = PDMatrixBuilder(33, max_neighbor=width)
                if kind == "stretch":
                    matrix.add_stretch_constraints(
                        faces, [[[1.0, 0.0], [0.0, 1.0]]] * len(faces), [[1.0] * 3] * len(faces), [0.5] * len(faces)
                    )
                else:
                    matrix.add_bend_constraints(
                        edges, [[1.0, 0.0]] * len(edges), [1.0] * len(edges), [[1.0] * 4] * len(edges)
                    )
                diag, counts, entries = matrix.finalize("cpu")
                self.assertEqual(int(counts.numpy()[0]), 33)
                self.assertEqual(set(entries.numpy()["column_index"][:, 0]), set(range(33)))
                self.assertTrue(np.isfinite(diag.numpy()).all())

    def test_overflow_rejects_incomplete_matrix(self):
        """Raise for undersized stretch and bending assembly, including later finalization."""
        for kind in ("stretch", "bend"):
            with self.subTest(kind=kind):
                builder = PDMatrixBuilder(4, max_neighbor=1)
                with self.assertRaisesRegex(ValueError, "PD matrix capacity 1 exceeded at vertex"):
                    if kind == "stretch":
                        builder.add_stretch_constraints(
                            [[0, 1, 2]], [[[1.0, 0.0], [0.0, 1.0]]], [[1.0, 1.0, 1.0]], [0.5]
                        )
                    else:
                        builder.add_bend_constraints([[0, 1, 2, 3]], [[1.0, 0.0]], [1.0], [[1.0, 1.0, 1.0, 1.0]])
                with self.assertRaisesRegex(ValueError, "PD matrix capacity"):
                    builder.finalize("cpu")


def test_invalid_topology_is_rejected_before_precompute(test, device):
    """Reject malformed cloth topology before Style3D precomputation."""
    builder = newton.ModelBuilder()
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    newton.solvers.style3d.add_cloth_grid(
        builder,
        pos=wp.vec3(0.0, 0.0, 1.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=1,
        dim_y=1,
        cell_x=0.1,
        cell_y=0.1,
        mass=0.1,
    )
    i, j, _ = builder.tri_indices[0]
    builder.tri_indices[0] = (i, j, builder.particle_count)

    with mock.patch.object(newton.solvers.SolverStyle3D, "_precompute") as precompute:
        with test.assertRaisesRegex(ValueError, "tri_indices.*particle count"):
            model = builder.finalize(device=device)
            newton.solvers.SolverStyle3D(
                model, config=newton.solvers.SolverStyle3D.Config(iterations=1, linear_iterations=1)
            )

    precompute.assert_not_called()


def test_zero_mass_isolated_particle_remains_finite(test, device):
    """Keep an isolated zero-mass particle fixed with finite state."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    newton.solvers.style3d.add_cloth_mesh(
        builder,
        pos=(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=(1.0, 2.0, 3.0),
        vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (2.0, 2.0, 0.0)),
        indices=(0, 1, 2),
        density=1.0,
    )
    model = builder.finalize(device=device)
    test.assertEqual(float(model.particle_mass.numpy()[3]), 0.0)
    test.assertTrue(int(model.particle_flags.numpy()[3]) & int(newton.ParticleFlags.ACTIVE))

    solver = newton.solvers.SolverStyle3D(
        model, config=newton.solvers.SolverStyle3D.Config(iterations=1, linear_iterations=1)
    )
    state_0 = model.state()
    state_1 = model.state()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    initial_position = state_0.particle_q.numpy()[3].copy()
    solver.step(state_0, state_1, model.control(), contacts, 0.01)

    positions = state_1.particle_q.numpy()
    velocities = state_1.particle_qd.numpy()
    test.assertTrue(np.isfinite(positions).all())
    test.assertTrue(np.isfinite(velocities).all())
    np.testing.assert_allclose(positions[3], initial_position)
    np.testing.assert_allclose(velocities[3], 0.0)


def test_solver_flags_deactivate_zero_mass_without_mutating_model(test, device):
    """Deactivate zero-mass solver particles without changing shared model flags."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    newton.solvers.style3d.add_cloth_mesh(
        builder,
        pos=(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=(0.0, 0.0, 0.0),
        vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (2.0, 2.0, 0.0)),
        indices=(0, 1, 2),
        density=1.0,
    )
    builder.particle_flags[3] |= newton.ParticleFlags.PROXY
    model = builder.finalize(device=device)
    model_flags = model.particle_flags.numpy().copy()

    solver = newton.solvers.SolverStyle3D(
        model, config=newton.solvers.SolverStyle3D.Config(iterations=1, linear_iterations=1)
    )

    np.testing.assert_array_equal(model.particle_flags.numpy(), model_flags)
    solver_flags = solver._particle_flags.numpy()
    active = int(newton.ParticleFlags.ACTIVE)
    proxy = int(newton.ParticleFlags.PROXY)
    test.assertTrue(np.all((solver_flags[:3] & active) != 0))
    test.assertEqual(int(solver_flags[3]) & active, 0)
    test.assertNotEqual(int(solver_flags[3]) & proxy, 0)


def test_solver_flags_track_runtime_model_changes(test, device):
    """Refresh solver activity after model masses and flags change."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    newton.solvers.style3d.add_cloth_grid(
        builder,
        pos=(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=(0.0, 0.0, 0.0),
        dim_x=1,
        dim_y=1,
        cell_x=0.1,
        cell_y=0.1,
        mass=0.1,
    )
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverStyle3D(
        model, config=newton.solvers.SolverStyle3D.Config(iterations=1, linear_iterations=1)
    )

    masses = model.particle_mass.numpy().copy()
    masses[0] = 0.0
    model.particle_mass.assign(masses)
    model_flags = model.particle_flags.numpy().copy()
    model_flags[1] = 0
    model.particle_flags.assign(model_flags)

    state_0 = model.state()
    state_1 = model.state()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    solver.step(state_0, state_1, model.control(), contacts, 0.01)

    solver_flags = solver._particle_flags.numpy()
    active = int(newton.ParticleFlags.ACTIVE)
    test.assertEqual(int(solver_flags[0]), 0)
    test.assertEqual(int(solver_flags[1]), 0)
    test.assertNotEqual(int(solver_flags[2]) & active, 0)
    np.testing.assert_array_equal(model.particle_flags.numpy(), model_flags)


def test_global_particles_use_global_gravity(test, device):
    """Apply dedicated global gravity to Style3D particles."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -2.0))
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    newton.solvers.style3d.add_cloth_grid(
        builder,
        pos=(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=(0.0, 0.0, 0.0),
        dim_x=1,
        dim_y=1,
        cell_x=0.1,
        cell_y=0.1,
        mass=0.1,
    )
    builder.begin_world(gravity=(0.0, 0.0, 0.0))
    builder.end_world()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverStyle3D(
        model, config=newton.solvers.SolverStyle3D.Config(iterations=1, linear_iterations=1)
    )
    state_0, state_1 = model.state(), model.state()
    contacts = newton.CollisionPipeline(model).contacts()

    solver.step(state_0, state_1, model.control(), contacts, 0.1)

    test.assertTrue(np.all(state_1.particle_qd.numpy()[:, 2] < -1.0e-6))


def test_tet_sipd_preserves_undamped_translation(test, device):
    """Preserve uniform velocity and displacement in an unconstrained elastic tetrahedral grid."""
    velocity = np.array((0.25, -0.125, 0.5), dtype=np.float32)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    builder.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, 1.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(*velocity),
        dim_x=1,
        dim_y=1,
        dim_z=1,
        cell_x=0.5,
        cell_y=0.5,
        cell_z=0.5,
        density=1.0e3,
        k_mu=1.0e5,
        k_lambda=1.0e5,
        k_damp=0.0,
    )
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverStyle3D(
        model,
        config=newton.solvers.SolverStyle3D.Config(
            iterations=1,
            enable_surface_constraints=False,
            enable_tet_sipd=True,
            tet_solver="semi_implicit",
            tet_sipd_relaxation=1.0,
        ),
    )
    state_0, state_1 = model.state(), model.state()
    contacts = newton.CollisionPipeline(model).contacts()
    control = model.control()
    initial_positions = state_0.particle_q.numpy().copy()
    expected_velocity = np.broadcast_to(velocity, initial_positions.shape)
    dt = 1.0 / 64.0

    for step in range(8):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, dt)
        expected_positions = initial_positions + ((step + 1) * dt) * velocity
        np.testing.assert_allclose(state_1.particle_q.numpy(), expected_positions, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(state_1.particle_qd.numpy(), expected_velocity, rtol=0.0, atol=2.0e-5)
        state_0, state_1 = state_1, state_0


def test_tet_sipd_force_matches_energy_gradient(test, device):
    """Match physical SIPD forces to finite differences of the tetrahedral elastic energy."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    for point in ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
        builder.add_particle(point, (0.0, 0.0, 0.0), 1.0)
    builder.add_tetrahedron(0, 1, 2, 3, 100.0, 200.0, 0.0)
    model = builder.finalize(device=device)
    inv_qa = wp.zeros(4, dtype=wp.vec3, device=device)
    wp.launch(
        initialize_tet_inv_qa_kernel,
        dim=1,
        inputs=[model.particle_q, model.tet_indices],
        outputs=[inv_qa],
        device=device,
    )
    q = np.array(((0.1, 0.2, 0.3), (1.2, 0.25, 0.3), (0.2, 1.1, 0.3), (0.1, 0.3, 1.4)))
    pos = wp.array(q.astype(np.float32), dtype=wp.vec3, device=device)
    force = wp.zeros(4, dtype=wp.vec3, device=device)
    diag, upper = wp.zeros_like(force), wp.zeros_like(force)
    wp.launch(
        eval_tet_sipd_force_kernel,
        dim=1,
        inputs=[
            pos,
            model.particle_q,
            model.tet_indices,
            model.tet_poses,
            model.tet_materials,
            inv_qa,
            0.01,
            model.particle_q,
        ],
        outputs=[force, diag, upper],
        device=device,
    )

    def energy(x):
        F = np.column_stack((x[1] - x[0], x[2] - x[0], x[3] - x[0]))
        J = np.linalg.det(F)
        return (50.0 * (np.sum(F * F) - 3.0) - 100.0 * (J - 1.0) + 150.0 * (J - 1.0) ** 2) / 6.0

    expected = np.zeros_like(q)
    epsilon = 1.0e-5
    for vertex in range(4):
        for axis in range(3):
            plus, minus = q.copy(), q.copy()
            plus[vertex, axis] += epsilon
            minus[vertex, axis] -= epsilon
            expected[vertex, axis] = -(energy(plus) - energy(minus)) / (2.0 * epsilon)
    actual = force.numpy()
    np.testing.assert_allclose(actual, expected, rtol=2.0e-5, atol=2.0e-5)
    np.testing.assert_allclose(actual.sum(axis=0), 0.0, atol=2.0e-5)


def test_tet_sipd_ground_contact_is_repulsive(test, device):
    """Move a penetrating tetrahedral grid away from the ground under SIPD contact forces."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    builder.add_ground_plane()
    builder.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, 0.05),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=1,
        dim_y=1,
        dim_z=1,
        cell_x=0.2,
        cell_y=0.2,
        cell_z=0.2,
        density=1000.0,
        k_mu=0.0,
        k_lambda=0.0,
        k_damp=0.0,
        particle_radius=0.1,
    )
    model = builder.finalize(device=device)
    model.soft_contact_ke = 1000.0
    model.soft_contact_kd = 0.0
    model.soft_contact_mu = 0.0
    solver = newton.solvers.SolverStyle3D(
        model,
        config=newton.solvers.SolverStyle3D.Config(
            iterations=1,
            enable_surface_constraints=False,
            tet_solver="semi_implicit",
        ),
    )
    solver.collision.stiff_vf = solver.collision.stiff_ee = solver.collision.stiff_ef = 0.0
    state_in, state_out = model.state(), model.state()
    before = state_in.particle_q.numpy().copy()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state_in, contacts)
    test.assertGreater(int(contacts.soft_contact_count.numpy()[0]), 0)
    solver.step(state_in, state_out, model.control(), contacts, 0.01)
    after = state_out.particle_q.numpy()
    test.assertTrue(np.all(after[before[:, 2] < 0.1, 2] > before[before[:, 2] < 0.1, 2]))


devices = get_test_devices()


add_function_test(
    TestSolverStyle3D,
    "test_high_valence_pd_matrix",
    test_high_valence_pd_matrix,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverStyle3D,
    "test_constructor_precomputes_fixed_pd_matrix",
    test_constructor_precomputes_fixed_pd_matrix,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverStyle3D,
    "test_config_constructor_options",
    test_config_constructor_options,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverStyle3D,
    "test_invalid_topology_is_rejected_before_precompute",
    test_invalid_topology_is_rejected_before_precompute,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverStyle3D,
    "test_zero_mass_isolated_particle_remains_finite",
    test_zero_mass_isolated_particle_remains_finite,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverStyle3D,
    "test_solver_flags_deactivate_zero_mass_without_mutating_model",
    test_solver_flags_deactivate_zero_mass_without_mutating_model,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverStyle3D,
    "test_solver_flags_track_runtime_model_changes",
    test_solver_flags_track_runtime_model_changes,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverStyle3D,
    "test_global_particles_use_global_gravity",
    test_global_particles_use_global_gravity,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverStyle3D,
    "test_tet_sipd_preserves_undamped_translation",
    test_tet_sipd_preserves_undamped_translation,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverStyle3D,
    "test_tet_sipd_force_matches_energy_gradient",
    test_tet_sipd_force_matches_energy_gradient,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverStyle3D,
    "test_tet_sipd_ground_contact_is_repulsive",
    test_tet_sipd_ground_contact_is_repulsive,
    devices=devices,
    check_output=False,
)

if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
