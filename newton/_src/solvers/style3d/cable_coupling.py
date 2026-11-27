# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Two-way point coupling between VIPER segments and affine bodies.

Cable capsule contact points use p = (x_a + x_b)/2 + R r, consistent
with the proxy poses. BALL attachments at segment endpoints instead bind
the actual node, so stretch does not move the attachment off the endpoint.
All responses assemble into the same Newton/PCG system.
"""

import numpy as np
import warp as wp

from ...geometry import ParticleFlags
from ...sim import Contacts, JointType
from ...sim.contacts import contact_surface_separation
from .contact import contact_response

# The implicit solver uses explicit Jacobians and does not expose an adjoint.
wp.set_module_options({"enable_backward": False})


@wp.struct
class Point:
    rows: wp.vec4i
    J0: wp.mat33
    J1: wp.mat33
    J2: wp.mat33
    J3: wp.mat33
    current: wp.vec3
    previous: wp.vec3


@wp.struct
class PointTangent:
    active: int
    side0: Point
    side1: Point
    hessian: wp.mat33


@wp.func
def point_jacobian(p: Point, slot: int):
    result = p.J0
    if slot == 1:
        result = p.J1
    elif slot == 2:
        result = p.J2
    elif slot == 3:
        result = p.J3
    return result


@wp.func
def coupling_point(
    body: int,
    local: wp.vec3,
    endpoint: bool,
    particle_count: int,
    node_start: int,
    rotation_start: int,
    body_to_segment: wp.array[int],
    body_to_affine: wp.array[int],
    segment_nodes: wp.array2d[int],
    half_length: wp.array[float],
    body_com: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    frame: wp.array[wp.mat33],
    frame_begin: wp.array[wp.mat33],
    x: wp.array[wp.vec3],
    previous: wp.array[wp.vec3],
    inertia: wp.array[wp.vec3],
    flags: wp.array[int],
    dt: float,
):
    p = Point()
    p.rows = wp.vec4i(-1)
    p.current = local
    p.previous = local
    if body < 0:
        return p
    segment = body_to_segment[body]
    affine = body_to_affine[body]
    r = local - body_com[body]
    identity = wp.identity(3, float)
    if segment >= 0:
        a = node_start + segment_nodes[segment, 0]
        b = node_start + segment_nodes[segment, 1]
        if endpoint:
            # BALL sites are validated to lie exactly on a centerline endpoint.
            node = a
            if r[2] > 0.0:
                node = b
            p.rows[0] = node
            p.J0 = identity
            p.current = x[node]
            p.previous = previous[node]
        else:
            lever = frame[segment] * r
            p.rows = wp.vec4i(a, b, rotation_start + segment, -1)
            p.J0 = 0.5 * identity
            p.J1 = 0.5 * identity
            p.J2 = -wp.skew(lever)
            p.current = 0.5 * (x[a] + x[b]) + lever
            p.previous = 0.5 * (previous[a] + previous[b]) + frame_begin[segment] * r
    elif affine >= 0:
        start = particle_count + 4 * affine
        p.rows = wp.vec4i(start, start + 1, start + 2, start + 3)
        p.J0 = identity
        p.J1 = r[0] * identity
        p.J2 = r[1] * identity
        p.J3 = r[2] * identity
        p.previous = (
            previous[start] + r[0] * previous[start + 1] + r[1] * previous[start + 2] + r[2] * previous[start + 3]
        )
        if flags[start] & ParticleFlags.ACTIVE:
            p.current = x[start] + r[0] * x[start + 1] + r[1] * x[start + 2] + r[2] * x[start + 3]
        else:
            p.current = (
                inertia[start] + r[0] * inertia[start + 1] + r[1] * inertia[start + 2] + r[2] * inertia[start + 3]
            )
    else:
        p.previous = wp.transform_point(body_q[body], local)
        lever = wp.transform_vector(body_q[body], r)
        p.current = p.previous + dt * (
            wp.spatial_top(body_qd[body]) + wp.cross(wp.spatial_bottom(body_qd[body]), lever)
        )
    return p


@wp.func
def tangent_slot(t: PointTangent, slot: int):
    p = t.side0
    index = slot
    sign = float(-1.0)
    if slot >= 4:
        p = t.side1
        index = slot - 4
        sign = 1.0
    return p.rows[index], sign * point_jacobian(p, index)


@wp.func
def assemble_point(t: PointTangent, gradient: wp.vec3, rhs: wp.array[wp.vec3], diag: wp.array[wp.mat33]):
    for i in range(8):
        row, J = tangent_slot(t, i)
        if row >= 0:
            wp.atomic_sub(rhs, row, wp.transpose(J) * gradient)
            # Include cross terms if two segment sides share a node row.
            for j in range(8):
                column, K = tangent_slot(t, j)
                if column == row:
                    wp.atomic_add(diag, row, wp.transpose(J) * t.hessian * K)


@wp.kernel
def point_off_diagonal_kernel(
    tangents: wp.array[PointTangent], direction: wp.array[wp.vec3], product: wp.array[wp.vec3]
):
    t = tangents[wp.tid()]
    if t.active == 0:
        return
    for i in range(8):
        row, J = tangent_slot(t, i)
        if row >= 0:
            total = wp.vec3(0.0)
            for j in range(8):
                column, K = tangent_slot(t, j)
                if column >= 0 and column != row:
                    total += K * direction[column]
            wp.atomic_add(product, row, wp.transpose(J) * t.hessian * total)


@wp.kernel
def eval_cable_coupling_kernel(
    attachment: bool,
    dt: float,
    stiffness: float,
    damping: float,
    particle_count: int,
    node_start: int,
    rotation_start: int,
    body_to_segment: wp.array[int],
    body_to_affine: wp.array[int],
    segment_nodes: wp.array2d[int],
    half_length: wp.array[float],
    body_com: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    frame: wp.array[wp.mat33],
    frame_begin: wp.array[wp.mat33],
    x: wp.array[wp.vec3],
    previous: wp.array[wp.vec3],
    inertia: wp.array[wp.vec3],
    flags: wp.array[int],
    count: wp.array[int],
    shape_body: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    margin0: wp.array[float],
    margin1: wp.array[float],
    shape_ke: wp.array[float],
    shape_kd: wp.array[float],
    shape_kf: wp.array[float],
    shape_mu: wp.array[float],
    contact_ke: wp.array[float],
    contact_kd: wp.array[float],
    contact_mu: wp.array[float],
    surface_velocity: wp.array[wp.vec3],
    joints: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_enabled: wp.array[bool],
    rhs: wp.array[wp.vec3],
    diag: wp.array[wp.mat33],
    tangents: wp.array[PointTangent],
):
    tid = wp.tid()
    t = PointTangent()
    tangents[tid] = t
    b0 = int(-1)
    b1 = int(-1)
    s0 = int(-1)
    s1 = int(-1)
    local0 = wp.vec3(0.0)
    local1 = wp.vec3(0.0)
    if attachment:
        joint = joints[tid]
        if not joint_enabled[joint]:
            return
        b0 = joint_parent[joint]
        b1 = joint_child[joint]
        local0 = wp.transform_get_translation(joint_X_p[joint])
        local1 = wp.transform_get_translation(joint_X_c[joint])
    else:
        if tid >= wp.min(count[0], shape0.shape[0]):
            return
        s0 = shape0[tid]
        s1 = shape1[tid]
        if s0 < 0 or s1 < 0:
            return
        b0 = shape_body[s0]
        b1 = shape_body[s1]
        e0 = body_to_segment[b0] if b0 >= 0 else -1
        e1 = body_to_segment[b1] if b1 >= 0 else -1
        if e0 < 0 and e1 < 0:
            return
        local0 = point0[tid]
        local1 = point1[tid]
    t.side0 = coupling_point(
        b0,
        local0,
        attachment,
        particle_count,
        node_start,
        rotation_start,
        body_to_segment,
        body_to_affine,
        segment_nodes,
        half_length,
        body_com,
        body_q,
        body_qd,
        frame,
        frame_begin,
        x,
        previous,
        inertia,
        flags,
        dt,
    )
    t.side1 = coupling_point(
        b1,
        local1,
        attachment,
        particle_count,
        node_start,
        rotation_start,
        body_to_segment,
        body_to_affine,
        segment_nodes,
        half_length,
        body_com,
        body_q,
        body_qd,
        frame,
        frame_begin,
        x,
        previous,
        inertia,
        flags,
        dt,
    )
    error = t.side1.current - t.side0.current
    velocity = (error - (t.side1.previous - t.side0.previous)) / dt
    gradient = wp.vec3(0.0)
    if attachment:
        # E = k/2 ||e||^2 + d/(2h) ||e-e_prev||^2;
        # b = -J^T(k e + d (e-e_prev)/h), A = (k+d/h) J^T J.
        gradient = stiffness * error + damping * velocity
        t.hessian = (stiffness + damping / dt) * wp.identity(3, float)
    else:
        n = normals[tid]
        gap = contact_surface_separation(t.side0.current, t.side1.current, n, margin0[tid], margin1[tid])
        if gap >= 0.0:
            return
        ke = wp.max(0.0, 0.5 * (shape_ke[s0] + shape_ke[s1]))
        kd = wp.max(0.0, 0.5 * (shape_kd[s0] + shape_kd[s1]))
        kf = wp.max(0.0, 0.5 * (shape_kf[s0] + shape_kf[s1]))
        mu = wp.sqrt(wp.max(0.0, shape_mu[s0] * shape_mu[s1]))
        if contact_ke:
            ke = contact_ke[tid] if contact_ke[tid] > 0.0 else ke
            kd = contact_kd[tid] if contact_kd[tid] > 0.0 else kd
            mu *= contact_mu[tid] if contact_mu[tid] > 0.0 else 1.0
        if surface_velocity:
            velocity += surface_velocity[tid]
        vn = wp.dot(n, velocity)
        gn, Hn, gt, Ht = contact_response(gap, n, vn, velocity - vn * n, ke, kd, kf, mu, dt)
        gradient = gn + gt
        t.hessian = Hn + Ht
    t.active = 1
    assemble_point(t, gradient, rhs, diag)
    tangents[tid] = t


@wp.kernel
def eval_particle_cable_contact_kernel(
    dt: float,
    particle_count: int,
    node_start: int,
    rotation_start: int,
    body_to_segment: wp.array[int],
    body_to_affine: wp.array[int],
    segment_nodes: wp.array2d[int],
    half_length: wp.array[float],
    body_com: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    frame: wp.array[wp.mat33],
    frame_begin: wp.array[wp.mat33],
    x: wp.array[wp.vec3],
    previous: wp.array[wp.vec3],
    inertia: wp.array[wp.vec3],
    flags: wp.array[int],
    contact_count: wp.array[int],
    contact_particle: wp.array[int],
    contact_shape: wp.array[int],
    contact_point: wp.array[wp.vec3],
    contact_velocity: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    shape_body: wp.array[int],
    shape_margin: wp.array[float],
    shape_mu: wp.array[float],
    particle_radius: wp.array[float],
    ke: float,
    kd: float,
    kf: float,
    mu: float,
    rhs: wp.array[wp.vec3],
    diag: wp.array[wp.mat33],
    tangents: wp.array[PointTangent],
):
    """Map particle/capsule contacts to particle, cable node and frame rows.

    The relative point is p_particle - ((x_a+x_b)/2 + R r). Its Jacobian
    includes the particle and both cable nodes plus the frame rotation.
    Thus the same normal/friction energy produces equal opposite forces;
    the frozen tangent is J^T H J in the shared linear system.
    """
    tid = wp.tid()
    t = PointTangent()
    tangents[tid] = t
    if tid >= wp.min(contact_count[0], contact_particle.shape[0]):
        return
    particle = contact_particle[tid]
    shape = contact_shape[tid]
    if particle < 0 or particle >= particle_count or shape < 0:
        return
    body = shape_body[shape]
    if body < 0:
        return
    segment = body_to_segment[body]
    if segment < 0:
        return
    t.side0 = coupling_point(
        body,
        contact_point[tid],
        False,
        particle_count,
        node_start,
        rotation_start,
        body_to_segment,
        body_to_affine,
        segment_nodes,
        half_length,
        body_com,
        body_q,
        body_qd,
        frame,
        frame_begin,
        x,
        previous,
        inertia,
        flags,
        dt,
    )
    p = Point()
    p.rows = wp.vec4i(particle, -1, -1, -1)
    p.J0 = wp.identity(3, float)
    p.current = x[particle]
    p.previous = previous[particle]
    t.side1 = p
    n = normal[tid]
    gap = wp.dot(n, p.current - t.side0.current) - particle_radius[particle] - shape_margin[shape]
    if gap >= 0.0:
        return
    velocity = ((p.current - p.previous) - (t.side0.current - t.side0.previous)) / dt
    velocity -= frame[segment] * contact_velocity[tid]
    vn = wp.dot(n, velocity)
    friction = wp.sqrt(wp.max(0.0, mu * shape_mu[shape]))
    gn, Hn, gt, Ht = contact_response(gap, n, vn, velocity - vn * n, ke, kd, kf, friction, dt)
    t.active = 1
    t.hessian = Hn + Ht
    assemble_point(t, gn + gt, rhs, diag)
    tangents[tid] = t


class CableCoupling:
    """Shared-system contacts and finite-stiffness BALL endpoint attachments."""

    def __init__(self, model, cable, particle_count, body_to_affine):
        self.model, self.cable = model, cable
        self.particle_count, self.body_to_affine = particle_count, body_to_affine
        mapping = cable.body_to_segment.numpy()
        joints = []
        if model.joint_count:
            parents, children = model.joint_parent.numpy(), model.joint_child.numpy()
            for joint, kind in enumerate(model.joint_type.numpy()):
                if kind != JointType.BALL:
                    continue
                bodies = (parents[joint], children[joint])
                if not any(b >= 0 and mapping[b] >= 0 for b in bodies):
                    continue
                for body, anchor in zip(
                    bodies, (model.joint_X_p.numpy()[joint], model.joint_X_c.numpy()[joint]), strict=True
                ):
                    if body >= 0 and mapping[body] >= 0:
                        local = anchor[:3] - model.body_com.numpy()[body]
                        half = cable._segment_half_length.numpy()[mapping[body]]
                        if np.linalg.norm(local[:2]) > 1e-6 or abs(abs(local[2]) - half) > 1e-6:
                            raise ValueError(
                                "Style3D cable BALL attachments must lie on a segment centerline endpoint."
                            )
                joints.append(joint)
        self.joints = wp.array(joints, dtype=int, device=model.device)
        self.attachments = wp.zeros(len(joints), dtype=PointTangent, device=model.device)
        self.contacts = wp.zeros(0, dtype=PointTangent, device=model.device)
        self.soft_contacts = wp.zeros(0, dtype=PointTangent, device=model.device)
        self.empty_contacts = Contacts(0, 0, device=model.device)

    def evaluate(self, solver, state, contacts, dt):
        cable, model = self.cable, self.model
        if contacts is None:
            self.contacts.zero_()
            self.soft_contacts.zero_()
            contacts = self.empty_contacts
            count = 0
        else:
            count = contacts.rigid_contact_max
        if count == 0:
            self.contacts.zero_()
        if count > len(self.contacts):
            self.contacts = wp.zeros(count, dtype=PointTangent, device=model.device)
        for attachment, tangents in ((False, self.contacts), (True, self.attachments)):
            if not len(tangents) or (not attachment and count == 0):
                continue
            wp.launch(
                eval_cable_coupling_kernel,
                dim=len(tangents),
                inputs=[
                    attachment,
                    dt,
                    solver.config.cable_attachment_stiffness,
                    solver.config.cable_attachment_damping,
                    self.particle_count,
                    cable._position_start,
                    cable._rotation_start,
                    cable.body_to_segment,
                    self.body_to_affine,
                    cable._segment_nodes,
                    cable._segment_half_length,
                    model.body_com,
                    state.body_q,
                    state.body_qd,
                    cable._frame,
                    cable._frame_begin,
                    solver._x,
                    solver._x_prev,
                    solver._x_inertia,
                    solver._particle_flags,
                    contacts.rigid_contact_count,
                    model.shape_body,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    contacts.rigid_contact_point0,
                    contacts.rigid_contact_point1,
                    contacts.rigid_contact_normal,
                    contacts.rigid_contact_margin0,
                    contacts.rigid_contact_margin1,
                    model.shape_material_ke,
                    model.shape_material_kd,
                    model.shape_material_kf,
                    model.shape_material_mu,
                    contacts.rigid_contact_stiffness,
                    contacts.rigid_contact_damping,
                    contacts.rigid_contact_friction,
                    contacts.rigid_contact_surface_velocity,
                    self.joints,
                    model.joint_parent,
                    model.joint_child,
                    model.joint_X_p,
                    model.joint_X_c,
                    model.joint_enabled,
                ],
                outputs=[solver._rhs, solver._A_diag, tangents],
                device=model.device,
            )

        if self.particle_count and contacts.soft_contact_max:
            if contacts.soft_contact_max > len(self.soft_contacts):
                self.soft_contacts = wp.zeros(contacts.soft_contact_max, dtype=PointTangent, device=model.device)
            wp.launch(
                eval_particle_cable_contact_kernel,
                dim=len(self.soft_contacts),
                inputs=[
                    dt,
                    self.particle_count,
                    cable._position_start,
                    cable._rotation_start,
                    cable.body_to_segment,
                    self.body_to_affine,
                    cable._segment_nodes,
                    cable._segment_half_length,
                    model.body_com,
                    state.body_q,
                    state.body_qd,
                    cable._frame,
                    cable._frame_begin,
                    solver._x,
                    solver._x_prev,
                    solver._x_inertia,
                    solver._particle_flags,
                    contacts.soft_contact_count,
                    contacts.soft_contact_particle,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_body_vel,
                    contacts.soft_contact_normal,
                    model.shape_body,
                    model.shape_margin,
                    model.shape_material_mu,
                    model.particle_radius,
                    model.soft_contact_ke,
                    model.soft_contact_kd,
                    model.soft_contact_kf,
                    model.soft_contact_mu,
                ],
                outputs=[solver._rhs, solver._A_diag, self.soft_contacts],
                device=model.device,
            )
        else:
            self.soft_contacts.zero_()

    def multiply(self, direction, product):
        for tangents in (self.contacts, self.attachments, self.soft_contacts):
            if len(tangents):
                wp.launch(
                    point_off_diagonal_kernel,
                    dim=len(tangents),
                    inputs=[tangents, direction],
                    outputs=[product],
                    device=self.model.device,
                )
