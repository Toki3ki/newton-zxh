# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare VBD, SIPD-Semi and scalar PD on the softbody_hanging beam.

Blue bodies use VBD, orange SIPD-Semi, and green scalar PD. Within each solver, increasing
iteration counts use darker shades. Render-only offsets arrange identical
physics models in three columns. Use --overlay to compare their shapes directly.

Scalar PD uses rotation and unit-volume projections whose quadratic expansion
matches Stable Neo-Hookean at rest. It still differs at finite strain.
PCG uses 10 inner iterations per outer step.

All three paths use identical geometry, Lamé parameters, mass, initial conditions,
gravity, timestep and substeps. Material damping is off by default; --damping
enables the same objective metric damping in all three solvers, using --damping-value.
There are no contacts or picking forces, to isolate the material solve.
The surface triangles are only for display; there are no cloth constraints.
All three solvers use eager execution and reconstruct velocity without attenuation.
SIPD uses full local updates by default; --sipd-relaxation changes that experiment.

The native lower-right Plots panel shows colored live tip sag, maximum speed,
and implicit-Euler residual traces against physical time. Its separate fixed
bent-state plot compares within-timestep convergence against iteration count.
The side panel and optional CSV report tip sag, energy, minimum det(F), and the
implicit-Euler force-balance residual on free vertices of the last substep.
Timing includes CPU launch overhead and completed GPU work, excludes diagnostics
and rendering, and is not a CUDA-graph performance benchmark.
--convergence-csv separately solves the SAME bent starting state and timestep
with several iteration budgets, rather than comparing different trajectories.

Success criteria:
- Three columns of colored hanging beams remain finite and attached at the left.
- Matching iteration budgets have the same physical inputs.
- The panel identifies each solver, iteration budget and latest diagnostics.
- Native live plots use the same colors as the bodies and show data from t=0.

Runs:
- python -m newton.examples softbody_hanging_compare
- python -m newton.examples softbody_hanging_compare --iterations 4 8 16 --overlay
- python -m newton.examples softbody_hanging_compare --damping --damping-value 10
- python -m newton.examples softbody_hanging_compare --viewer null --num-frames 180 --test --metrics-csv outputs/hanging.csv --convergence-csv outputs/hanging_convergence.csv
"""

import argparse
import csv
import time
from collections import deque
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

_SOLVER_SPECS = (
    ("vbd", "VBD", (0.1, 0.35, 0.9)),
    ("sipd", "SIPD-Semi", (0.95, 0.30, 0.04)),
    ("pd", "Scalar PD", (0.12, 0.65, 0.25)),
)
_SOLVER_NAMES = {kind: name for kind, name, _ in _SOLVER_SPECS}


def _pd_volume_targets(F):
    """Evaluate the solver's multistart log-space volume projection in NumPy.

    This float64 diagnostic mirrors the local solve's starts, stopping rule and
    Armijo search. It returns feasible local targets, not certified global
    minimizers. Energy and residual therefore share the local solve's accuracy.
    """
    U, s, Vh = np.linalg.svd(F)
    sign = np.where(np.linalg.det(U @ Vh) < 0.0, -1.0, 1.0)
    U[:, :, 2] *= sign[:, None]
    s[:, 2] *= sign
    starts = np.zeros((4, len(F), 3), dtype=np.float64)
    for omitted in range(3):
        a, b = (omitted + 1) % 3, (omitted + 2) % 3
        valid = (s[:, a] > 0.0) & (s[:, b] > 0.0)
        starts[omitted + 1, valid, a] = np.log(s[valid, a])
        starts[omitted + 1, valid, b] = np.log(s[valid, b])
        starts[omitted + 1, valid, omitted] = -starts[omitted + 1, valid].sum(axis=1)
    z = starts[:, :, :2].reshape(-1, 2).copy()
    s_all = np.tile(s, (4, 1))

    def stretches(coordinates):
        with np.errstate(over="ignore", under="ignore"):
            return np.exp(np.column_stack((coordinates, -coordinates.sum(axis=1))))

    q = stretches(z)
    energy = 0.5 * np.sum((q - s_all) ** 2, axis=1)
    active = np.ones(len(q), dtype=bool)
    for _ in range(64):
        terms = q * (q - s_all)
        g = terms[:, :2] - terms[:, 2, None]
        active &= np.linalg.norm(g, axis=1) > 1.0e-9 * (1.0 + np.sum(q * q, axis=1))
        ids = np.flatnonzero(active)
        if not len(ids):
            break
        h = q[ids] * (2.0 * q[ids] - s_all[ids])
        h0, h1, h2 = h[:, 0] + h[:, 2], h[:, 1] + h[:, 2], h[:, 2]
        determinant = h0 * h1 - h2 * h2
        grad = g[ids]
        direction = -grad / (np.abs(h0) + np.abs(h1) + 2.0 * np.abs(h2) + 1.0)[:, None]
        positive = (h0 > 0.0) & (determinant > 0.0)
        newton_direction = np.column_stack(
            (-h1 * grad[:, 0] + h2 * grad[:, 1], h2 * grad[:, 0] - h0 * grad[:, 1])
        )
        direction[positive] = newton_direction[positive] / determinant[positive, None]
        slope = np.sum(grad * direction, axis=1)
        step = np.ones(len(ids))
        pending = np.ones(len(ids), dtype=bool)
        for _ in range(40):
            rows = np.flatnonzero(pending)
            if not len(rows):
                break
            trial_z = z[ids[rows]] + step[rows, None] * direction[rows]
            trial_q = stretches(trial_z)
            with np.errstate(over="ignore", invalid="ignore"):
                trial_energy = 0.5 * np.sum((trial_q - s_all[ids[rows]]) ** 2, axis=1)
            accepted = np.isfinite(trial_energy) & (
                trial_energy <= energy[ids[rows]] + 1.0e-4 * step[rows] * slope[rows]
            )
            dest = ids[rows[accepted]]
            z[dest], q[dest], energy[dest] = trial_z[accepted], trial_q[accepted], trial_energy[accepted]
            pending[rows[accepted]] = False
            step[rows[~accepted]] *= 0.5
        active[ids[pending]] = False
    best = np.argmin(energy.reshape(4, len(F)), axis=0)
    q = q.reshape(4, len(F), 3)[best, np.arange(len(F))]
    return (U * q[:, None, :]) @ Vh


def _write_rows(path, rows, *, append=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if not append:
            writer.writeheader()
        writer.writerows(rows)


class _SolverLayer:
    def __init__(self, solver_kind, iterations, color, sim_substeps, sim_dt, relaxation, damping=0.0):
        self.solver_kind = solver_kind
        self.iterations = iterations
        self.color = color
        self.layer_id = f"{_SOLVER_NAMES[solver_kind]} / {iterations} iterations"
        self.sim_substeps = sim_substeps
        self.sim_dt = sim_dt
        builder = newton.ModelBuilder()
        newton.solvers.SolverStyle3D.register_custom_attributes(builder)
        builder.add_soft_grid(
            pos=wp.vec3(0.0, 1.0, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=12,
            dim_y=4,
            dim_z=4,
            cell_x=0.1,
            cell_y=0.1,
            cell_z=0.1,
            density=1.0e3,
            k_mu=1.0e5,
            k_lambda=1.0e5,
            k_damp=damping,
            fix_left=True,
            color=color,
        )
        builder.color()
        self.model = builder.finalize()
        if solver_kind == "vbd":
            self.solver = newton.solvers.SolverVBD(
                self.model,
                iterations=iterations,
                particle_enable_self_contact=False,
                particle_enable_tile_solve=False,
            )
        else:
            self.solver = newton.solvers.SolverStyle3D(
                self.model,
                config=newton.solvers.SolverStyle3D.Config(
                    iterations=iterations,
                    enable_surface_constraints=False,
                    enable_tet_sipd=True,
                    tet_solver="pd" if solver_kind == "pd" else "semi_implicit",
                    tet_sipd_relaxation=relaxation,
                ),
            )
            self.solver.collision = None

        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.previous_q = wp.empty_like(self.state_0.particle_q)
        self.previous_qd = wp.empty_like(self.state_0.particle_qd)
        self.rest = self.model.particle_q.numpy().astype(np.float64)
        self.mass = self.model.particle_mass.numpy().astype(np.float64)
        self.free = self.mass > 0.0
        self.tip = np.isclose(self.rest[:, 0], self.rest[:, 0].max())
        self.tets = self.model.tet_indices.numpy()
        self.inv_dm = self.model.tet_poses.numpy().astype(np.float64)
        self.volumes = 1.0 / (6.0 * np.linalg.det(self.inv_dm))
        self.materials = self.model.tet_materials.numpy().astype(np.float64)
        self.gravity = self.model.gravity.numpy()[0].astype(np.float64)
        self.last_metrics = None
        self.last_solve_frame_ms = float("nan")

    def elastic_terms(self, positions):
        """Evaluate energy and its gradient independently in float64 for diagnostics."""
        x = positions[self.tets]
        ds = np.stack((x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]), axis=2)
        F = ds @ self.inv_dm
        J = np.linalg.det(F)
        cof = np.stack(
            (np.cross(F[:, :, 1], F[:, :, 2]), np.cross(F[:, :, 2], F[:, :, 0]), np.cross(F[:, :, 0], F[:, :, 1])),
            axis=2,
        )
        mu = self.materials[:, 0]
        lam_nh = self.materials[:, 1] + mu
        psi = 0.5 * mu * (np.sum(F * F, axis=(1, 2)) - 3.0) - mu * (J - 1.0)
        psi += 0.5 * lam_nh * (J - 1.0) ** 2
        P = mu[:, None, None] * F + (lam_nh * (J - 1.0) - mu)[:, None, None] * cof
        if self.solver_kind == "pd":
            U, _, Vh = np.linalg.svd(F)
            correction = np.ones((len(F), 3))
            correction[:, 2] = np.where(np.linalg.det(U @ Vh) < 0.0, -1.0, 1.0)
            R = (U * correction[:, None, :]) @ Vh
            difference = F - R
            psi = mu * np.sum(difference**2, axis=(1, 2))
            P = (2.0 * mu)[:, None, None] * difference
            volume_ratio = self.solver.config.tet_pd_volume_ratio
            if volume_ratio > 0.0:
                Q = _pd_volume_targets(F)
                volume_difference = F - Q
                volume_weight = 3.0 * self.materials[:, 1] * volume_ratio
                psi += 0.5 * volume_weight * np.sum(volume_difference**2, axis=(1, 2))
                P += volume_weight[:, None, None] * volume_difference
        h = self.volumes[:, None, None] * (P @ self.inv_dm.transpose(0, 2, 1))
        gradient = np.zeros_like(positions)
        np.add.at(gradient, self.tets[:, 0], -np.sum(h, axis=2))
        for vertex in range(1, 4):
            np.add.at(gradient, self.tets[:, vertex], h[:, :, vertex - 1])
        return float(self.volumes @ psi), gradient, float(J.min())

    def diagnostics(self, positions, velocities, previous, previous_velocity):
        elastic, gradient, min_j = self.elastic_terms(positions)
        damping_potential = 0.0
        if np.any(self.materials[:, 2] > 0.0):
            x, x_prev = positions[self.tets], previous[self.tets]
            ds = np.stack((x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]), axis=2)
            ds_prev = np.stack(
                (x_prev[:, 1] - x_prev[:, 0], x_prev[:, 2] - x_prev[:, 0], x_prev[:, 3] - x_prev[:, 0]), axis=2
            )
            F, F_prev = ds @ self.inv_dm, ds_prev @ self.inv_dm
            delta_c = F.transpose(0, 2, 1) @ F - F_prev.transpose(0, 2, 1) @ F_prev
            kd = self.materials[:, 2]
            damping_potential = float(np.sum(self.volumes * kd * np.sum(delta_c**2, axis=(1, 2))) / (2.0 * self.sim_dt))
            P_damp = (2.0 * kd / self.sim_dt)[:, None, None] * (F @ delta_c)
            h_damp = self.volumes[:, None, None] * (P_damp @ self.inv_dm.transpose(0, 2, 1))
            np.add.at(gradient, self.tets[:, 0], -h_damp.sum(axis=2))
            for vertex in range(1, 4):
                np.add.at(gradient, self.tets[:, vertex], h_damp[:, :, vertex - 1])
        predicted = previous + self.sim_dt * previous_velocity + self.sim_dt**2 * self.gravity
        # Stationarity of each solver's own implicit-Euler material objective.
        residual = (self.mass[:, None] / self.sim_dt**2) * (positions - predicted) + gradient
        free_residual = residual[self.free]
        kinetic = 0.5 * np.sum(self.mass[:, None] * velocities**2)
        potential = -np.sum(self.mass * (positions @ self.gravity))
        return {
            "tip_sag_m": float(np.mean(self.rest[self.tip, 2] - positions[self.tip, 2])),
            "com_z_m": float(self.mass @ positions[:, 2] / self.mass.sum()),
            "max_speed_m_s": float(np.linalg.norm(velocities[self.free], axis=1).max()),
            "elastic_J": elastic,
            "kinetic_J": float(kinetic),
            "gravity_J": float(potential),
            "total_J": float(elastic + kinetic + potential),
            "residual_rms_N": float(np.sqrt(np.mean(np.sum(free_residual**2, axis=1)))),
            "residual_max_N": float(np.linalg.norm(free_residual, axis=1).max()),
            "objective_J": float(
                elastic
                + damping_potential
                + 0.5 * np.sum(self.mass[:, None] * (positions - predicted) ** 2) / self.sim_dt**2
            ),
            "min_det_F": min_j,
            "material_damping_Pa_s": float(self.materials[0, 2]),
        }

    def step(self, measure):
        # Same eager execution for all layers; diagnostic copies are outside timing.
        elapsed = 0.0
        for substep in range(self.sim_substeps):
            if substep == self.sim_substeps - 1:
                wp.copy(self.previous_q, self.state_0.particle_q)
                wp.copy(self.previous_qd, self.state_0.particle_qd)
            if measure:
                wp.synchronize_device(self.model.device)
                started = time.perf_counter()
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            if measure:
                wp.synchronize_device(self.model.device)
                elapsed += time.perf_counter() - started
        if measure:
            self.last_solve_frame_ms = 1000.0 * elapsed
        self.last_metrics = self.diagnostics(
            self.state_0.particle_q.numpy().astype(np.float64),
            self.state_0.particle_qd.numpy().astype(np.float64),
            self.previous_q.numpy().astype(np.float64),
            self.previous_qd.numpy().astype(np.float64),
        )
        self.last_metrics["solve_frame_ms"] = self.last_solve_frame_ms

    def convergence(self, budgets):
        """Restart each solve at an identical bent beam, with a fixed inertial target."""
        initial = self.rest.copy()
        initial[:, 2] -= 0.1 * (initial[:, 0] / initial[:, 0].max()) ** 2
        velocity = np.zeros_like(initial)
        initial_row = self.diagnostics(initial, velocity, initial, velocity)
        initial_residual = initial_row["residual_rms_N"]
        rows = []
        for budget in [0, *budgets]:
            if budget == 0:
                metrics = initial_row.copy()
            else:
                state_in, state_out = self.model.state(), self.model.state()
                state_in.particle_q.assign(initial.astype(np.float32))
                if self.solver_kind == "vbd":
                    self.solver.iterations = budget
                else:
                    self.solver.config.iterations = budget
                self.solver.step(state_in, state_out, self.control, None, self.sim_dt)
                metrics = self.diagnostics(
                    state_out.particle_q.numpy().astype(np.float64),
                    state_out.particle_qd.numpy().astype(np.float64),
                    initial,
                    velocity,
                )
            rows.append(
                dict(
                    solver=self.solver_kind,
                    iterations=budget,
                    dt_s=self.sim_dt,
                    residual_relative=metrics["residual_rms_N"] / max(initial_residual, 1.0e-30),
                    **metrics,
                )
            )
        if self.solver_kind == "vbd":
            self.solver.iterations = self.iterations
        else:
            self.solver.config.iterations = self.iterations
        return rows


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.frame = 0
        self.frame_dt = 1.0 / 60.0
        self.sim_substeps = args.substeps
        self.sample_every = args.sample_every
        self.metrics_csv = args.metrics_csv
        self.damping_value = args.damping_value if args.damping else 0.0
        if not np.isfinite(self.damping_value) or self.damping_value < 0.0:
            raise ValueError("Material damping must be finite and nonnegative.")
        self.csv_started = False
        self.plot_times = deque(maxlen=600)
        self.plot_values = {}
        self.plot_visible = {}
        self.plot_mode = 0
        self._plot_window_positioned = False
        self._owns_implot_context = False
        if self.sim_substeps < 1 or self.sample_every < 1 or any(i < 1 for i in args.iterations):
            raise ValueError("Substeps, sample interval and iteration counts must be positive.")
        if not 0.0 < args.sipd_relaxation <= 1.0:
            raise ValueError("SIPD relaxation must lie in (0, 1].")
        self.sim_dt = self.frame_dt / self.sim_substeps
        budgets = sorted(set(args.iterations))
        self.layers = []
        self.viewer.picking_enabled = False
        for column, (kind, _, base) in enumerate(_SOLVER_SPECS):
            for row, budget in enumerate(budgets):
                fraction = row / max(1, len(budgets) - 1)
                # Mix with white: fewer iterations are lighter, more are darker.
                color = tuple((1.0 - fraction) * 0.6 + (0.4 + 0.6 * fraction) * c for c in base)
                layer = _SolverLayer(
                    kind, budget, color, self.sim_substeps, self.sim_dt, args.sipd_relaxation, self.damping_value
                )
                self.layers.append(layer)
                viewer.activate(layer.layer_id)
                viewer.set_model(layer.model)
                offset = wp.vec3(0.0) if args.overlay else wp.vec3((column - 1) * 1.8, row * 0.8, 0.0)
                viewer.set_layer_transform(layer.layer_id, wp.transform(offset, wp.quat_identity()))
                self.plot_values[layer.layer_id] = {
                    key: deque(maxlen=600) for key in ("tip_sag_m", "max_speed_m_s", "residual_rms_N")
                }
                self.plot_visible[layer.layer_id] = True

        self.plot_times.append(0.0)
        for layer in self.layers:
            zero_velocity = np.zeros_like(layer.rest)
            layer.last_metrics = layer.diagnostics(layer.rest, zero_velocity, layer.rest, zero_velocity)
            layer.last_metrics["solve_frame_ms"] = float("nan")
            for key, history in self.plot_values[layer.layer_id].items():
                history.append(layer.last_metrics[key])

        # Dedicated solvers keep the fixed convergence experiment independent
        # of the evolving states and solver work buffers displayed above.
        self.convergence_rows = []
        sweep = sorted({1, 2, 4, 8, 16, 32, 64, *budgets})
        for kind, _, base in _SOLVER_SPECS:
            diagnostic_layer = _SolverLayer(
                kind, budgets[-1], base, self.sim_substeps, self.sim_dt, args.sipd_relaxation, self.damping_value
            )
            self.convergence_rows.extend(diagnostic_layer.convergence(sweep))
        if args.convergence_csv:
            _write_rows(args.convergence_csv, self.convergence_rows)
            print(f"Same-state convergence sweep: {Path(args.convergence_csv).resolve()}")
        if hasattr(viewer, "register_ui_callback"):
            viewer.register_ui_callback(self._plot_panel, position="free")
        print(f"VBD (blue), SIPD-Semi (orange), Scalar PD (green); iterations={budgets}, dt={self.sim_dt:.8f}")
        print("Scalar PD: SNH small-strain-matched shape + volume projections; PCG inner iterations=10.")
        print(f"Material damping={self.damping_value:g} Pa s, no contacts/picking; darker = more iterations.")

    def step(self):
        self.frame += 1
        # Omit the first frame from timing to exclude JIT and lazy setup.
        measure = self.frame > 1 and self.frame % self.sample_every == 0
        for layer in self.layers:
            layer.step(measure)
        self.sim_time = self.frame * self.frame_dt
        self.plot_times.append(self.sim_time)
        for layer in self.layers:
            for key, history in self.plot_values[layer.layer_id].items():
                history.append(layer.last_metrics[key])
        if measure:
            rows = [
                dict(
                    frame=self.frame,
                    time_s=self.sim_time,
                    solver=layer.solver_kind,
                    iterations=layer.iterations,
                    dt_s=self.sim_dt,
                    **layer.last_metrics,
                )
                for layer in self.layers
            ]
            if self.metrics_csv:
                _write_rows(self.metrics_csv, rows, append=self.csv_started)
                self.csv_started = True
            for layer in self.layers:
                values = layer.last_metrics
                print(
                    f"t={self.sim_time:.3f} {layer.layer_id}: sag={values['tip_sag_m']:.5f} m, "
                    f"residual={values['residual_rms_N']:.3e} N, solve={values['solve_frame_ms']:.2f} ms"
                )

    def gui(self, ui):
        ui.text(f"Simulation time: {self.sim_time:.3f} s | dt: {self.sim_dt:.6f} s")
        ui.text("Blue: VBD | Orange: SIPD-Semi | Green: Scalar PD")
        ui.text("PD: SNH small-strain match, shape + volume; inner PCG=10")
        ui.text("Light to dark: increasing iterations")
        ui.text(f"Material damping (all three): {self.damping_value:g} Pa s | no extra velocity attenuation")
        ui.text("No contacts or picking; launch with --damping or --no-damping")
        ui.text("Residual: implicit-Euler force balance, last substep")
        ui.text("Live colored curves: Plots - Hanging Comparison (lower right)")
        ui.text("The same-state convergence plot uses an independent fixed snapshot.")
        for layer in self.layers:
            ui.separator()
            ui.text(layer.layer_id)
            if layer.last_metrics is not None:
                m = layer.last_metrics
                ui.text(f"Tip sag: {m['tip_sag_m']:.4f} m | min J: {m['min_det_F']:.4f}")
                solve_text = f"{m['solve_frame_ms']:.2f} ms" if np.isfinite(m["solve_frame_ms"]) else "not sampled yet"
                ui.text(f"Residual: {m['residual_rms_N']:.3e} N | solve: {solve_text}")
                ui.text(f"Kinetic: {m['kinetic_J']:.4f} J | elastic: {m['elastic_J']:.4f} J")

    @staticmethod
    def _draw_plot(ui, title, traces, *, x_label, y_label, log_y=False, show_legend=False, height=215.0):
        """Draw shared-axis curves with ImPlot bundled in the native viewer UI."""
        from imgui_bundle import implot  # noqa: PLC0415

        flags = 0 if show_legend else implot.Flags_.no_legend
        if implot.begin_plot(title, ui.ImVec2(-1.0, height), flags=flags):
            implot.setup_axes(x_label, y_label, implot.AxisFlags_.auto_fit, implot.AxisFlags_.auto_fit)
            if log_y:
                implot.setup_axis_scale(implot.ImAxis_.y1, implot.Scale_.log10)
            for label, color, xs, ys in traces:
                x, y = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
                if log_y:
                    y = np.maximum(y, 1.0e-12)
                spec = implot.Spec(
                    line_color=ui.ImVec4(*color, 1.0),
                    line_weight=2.0,
                    marker=implot.Marker_.circle if len(x) < 32 else implot.Marker_.none,
                    marker_size=2.5,
                )
                implot.plot_line(label, x, y, spec=spec)
            implot.end_plot()

    def _plot_panel(self, ui):
        """Show the comparison in the native lower-right viewer plot window."""
        from imgui_bundle import implot  # noqa: PLC0415

        io = ui.get_io()
        width = min(520.0, max(360.0, io.display_size.x * 0.38))
        height = min(430.0, io.display_size.y - 30.0)
        if not self._plot_window_positioned:
            ui.set_next_window_pos(
                ui.ImVec2(io.display_size.x - width - 10.0, io.display_size.y - height - 10.0), ui.Cond_.always
            )
            ui.set_next_window_size(ui.ImVec2(width, height), ui.Cond_.always)
            self._plot_window_positioned = True
        expanded, _ = ui.begin("Plots - Hanging Comparison")
        if expanded:
            if implot.get_current_context() is None:
                implot.create_context()
                self._owns_implot_context = True
            ui.text(f"Time: {self.sim_time:.3f}s | Blue VBD / Orange SIPD / Green PD")
            ui.text("PD uses its own material energy and residual.")
            _, self.plot_mode = ui.combo(
                "Plot",
                self.plot_mode,
                [
                    "Convergence: fixed shared state / one dt",
                    "Live tip sag",
                    "Live maximum speed",
                    "Live force balance",
                ],
            )
            if self.plot_mode == 0:
                ui.text("Independent fixed bent state; x-axis is iteration budget.")
                ui.text("All three solvers start from the same state and solve one dt.")
                traces = []
                for kind, label, color in _SOLVER_SPECS:
                    rows = [row for row in self.convergence_rows if row["solver"] == kind]
                    traces.append(
                        (label, color, [row["iterations"] for row in rows], [row["residual_relative"] for row in rows])
                    )
                self._draw_plot(
                    ui,
                    "Residual reduction within one timestep",
                    traces,
                    x_label="iterations",
                    y_label="relative RMS (log)",
                    log_y=True,
                    show_legend=True,
                    height=285.0,
                )
            else:
                ui.text("Darker = more iterations. Toggle overlapping traces.")
                count = len(self.layers) // 3
                for index in range(count):
                    for column in range(3):
                        layer = self.layers[column * count + index]
                        ui.push_style_color(ui.Col_.text, ui.ImVec4(*layer.color, 1.0))
                        _, self.plot_visible[layer.layer_id] = ui.checkbox(
                            layer.layer_id, self.plot_visible[layer.layer_id]
                        )
                        ui.pop_style_color()
                title, key, units, logarithmic = (
                    ("Tip sag", "tip_sag_m", "m", False),
                    ("Maximum particle speed", "max_speed_m_s", "m/s", False),
                    ("Live implicit-Euler force balance", "residual_rms_N", "RMS N (log)", True),
                )[self.plot_mode - 1]
                traces = [
                    (layer.layer_id, layer.color, self.plot_times, self.plot_values[layer.layer_id][key])
                    for layer in self.layers
                    if self.plot_visible[layer.layer_id]
                ]
                self._draw_plot(ui, title, traces, x_label="time [s]", y_label=units, log_y=logarithmic)
        ui.end()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        for layer in self.layers:
            self.viewer.activate(layer.layer_id)
            self.viewer.log_state(layer.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        for layer in self.layers:
            q, v = layer.state_0.particle_q.numpy(), layer.state_0.particle_qd.numpy()
            if not np.isfinite(q).all() or not np.isfinite(v).all():
                raise AssertionError(f"{layer.layer_id}: non-finite state at frame {self.frame}")
            np.testing.assert_allclose(q[~layer.free], layer.rest[~layer.free], atol=1.0e-7, rtol=0.0)
            if np.linalg.norm(q - layer.rest, axis=1).max() > 3.0:
                raise AssertionError(f"{layer.layer_id}: unbounded displacement at frame {self.frame}")

    def test_final(self):
        self.test_post_step()
        for layer in self.layers:
            _, _, min_j = layer.elastic_terms(layer.state_0.particle_q.numpy().astype(np.float64))
            if min_j <= 0.0:
                raise AssertionError(f"{layer.layer_id}: inverted tetrahedron (min J={min_j})")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--iterations", nargs="+", type=int, default=[8])
        parser.add_argument("--substeps", type=int, default=10)
        parser.add_argument(
            "--damping",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Enable the same material damping for all three solvers (default off)",
        )
        parser.add_argument(
            "--damping-value",
            type=float,
            default=10.0,
            help="Material damping in Pa s used with --damping (default 10)",
        )
        parser.add_argument("--sipd-relaxation", type=float, default=1.0)
        parser.add_argument("--overlay", action="store_true", help="Overlay all models without display offsets")
        parser.add_argument("--sample-every", type=int, default=30, help="Diagnostic sampling interval in frames")
        parser.add_argument("--metrics-csv", help="Write time histories, energy, residuals and timing")
        parser.add_argument("--convergence-csv", help="Write a same-state, single-timestep iteration sweep")
        return parser

    def close(self):
        """Release the native plot context owned by a standalone example."""
        # The example browser currently has no teardown callback. Reuse a
        # shared context there; standalone runs release it explicitly below.
        if not self._owns_implot_context:
            return
        from imgui_bundle import implot  # noqa: PLC0415

        if implot.get_current_context() is not None:
            implot.destroy_context()
        self._owns_implot_context = False


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    try:
        newton.examples.run(example, args)
    finally:
        example.close()
