#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 Stephane Caron <stephane.caron@normalesup.org>
#
# This file is part of icra-2015.
#
# This code is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# This code is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# this code. If not, see <http://www.gnu.org/licenses/>.


import numpy
import openravepy
import pylab
import time

from matplotlib.patches import Rectangle
from numpy import arange, array, cross, dot, eye, zeros


class RaveRobotModel(object):

    chains = {}  # list of kinematic chains

    def __init__(self, env_file):
        gravity = array([0, 0, -9.81])
        env = openravepy.Environment()
        env.Load(env_file)
        robot = env.GetRobots()[0]
        robot.GetEnv().GetPhysicsEngine().SetGravity(gravity)
        dof_llim, dof_ulim = robot.GetDOFLimits()
        n = robot.GetDOF()

        vel_lim = robot.GetDOFVelocityLimits()
        tau_lim = 100000 * numpy.ones(n)
        for dof in self.dofs:
            # internal limits override those of the robot model
            if dof.vel_limit is not None:
                vel_lim[dof.index] = dof.vel_limit
            if dof.torque_limit is not None:
                tau_lim[dof.index] = dof.torque_limit
        robot.SetDOFVelocityLimits(1000 * vel_lim)  # temporary (OpenRAVE bug)

        self.display_on = False
        self.dof_llim = dof_llim
        self.dof_ulim = dof_ulim
        self.dof_zeros = zeros(n)
        self.env = env
        self.forward_dt = 5e-3
        self.gravity = gravity
        self.nb_dof = n
        self.nb_active_dof = 0
        self.rave = robot
        self.torque_limits = tau_lim
        self.real_torque_limits = tau_lim
        self.total_mass = sum([lnk.GetMass() for lnk in self.rave.GetLinks()])
        self.velocity_limits = vel_lim

        self.collision_handle = self.register_collision_callback()
        for chain, joints in self.chains.iteritems():
            for joint in joints:
                for dof in joint.dofs:
                    dof.llim = self.dof_llim[dof.index]
                    dof.ulim = self.dof_ulim[dof.index]
                joint.ulim = self.dof_ulim[joint.dof_range]
                joint.llim = self.dof_llim[joint.dof_range]

    def get_dof_values(self):
        return self.rave.GetDOFValues()

    def set_dof_values(self, q):
        self.rave.SetDOFValues(q)

    def display(self):
        if not self.display_on:
            self.env.SetViewer('qtcoin')
            self.display_on = True

    @property
    def dofs(self):
        for chain, joints in self.chains.iteritems():
            for joint in joints:
                for dof in joint.dofs:
                    yield dof

    @property
    def links(self):
        return self.rave.GetLinks()

    def get_dof(self, name):
        for dof in self.dofs:
            if dof.name == name:
                return dof
        return None

    def get_dofs(self, *args):
        dofs = [dof
                for chain, joints in self.chains.iteritems()
                for joint in joints
                for dof in joint.dofs
                for identifier in [chain, joint.name, dof.name]
                if identifier in args]
        l = list(set(dofs))
        # convention: DOF ordered by index
        l.sort(key=lambda dof: dof.index)
        return l

    def link_translation_jacobian(self, q, link, link_pt):
        with self.rave:
            self.rave.SetDOFValues(q)
            J = self.rave.ComputeJacobianTranslation(link.GetIndex(), link_pt)
        return J

    def play_trajectory(self, traj, callback=None, dt=3e-2, start=0.,
                        stop=None, nowait=False, slowdown=1.):
        if stop is None:
            stop = traj.duration
        trange = list(arange(start, stop, dt))
        if stop - trange[-1] >= dt:
            trange.append(stop)
        for t in trange:
            if traj.active_dofs is not None:
                q = traj.q_full(t)
                qd = traj.qd_full(t)
                qdd = traj.qdd_full(t)
            else:
                q = traj.q(t)
                qd = traj.qd(t)
                qdd = traj.qdd(t)
            self.rave.SetDOFValues(q)
            if callback:
                callback(t, q, qd, qdd)
            if not nowait:
                time.sleep(slowdown * dt)

    def record(self, callback, fname='output.mpg', codec=13, framerate=30,
               width=800, height=600, timepad=1.):
        vname = self.env.GetViewer().GetName()
        cmd = 'Start %d %d %d codec %d timing simtime filename %s\nviewer %s'
        cmd = cmd % (width, height, framerate, codec, fname, vname)

        recorder = openravepy.RaveCreateModule(self.env, 'viewerrecorder')
        self.env.AddModule(recorder, '')
        recorder.SendCommand(cmd)
        time.sleep(timepad)
        callback()
        time.sleep(timepad)
        recorder.SendCommand('Stop')
        self.env.Remove(recorder)

    def record_trajectory(self, traj):
        def callback():
            self.play_trajectory(traj)
        self.rave.SetDOFValues(traj.q(0))
        self.record(callback)

    def register_collision_callback(self):
        chest = self.rave.GetLink('CHEST_Y_LINK')
        rsho = self.rave.GetLink('R_SHOULDER_R_LINK')
        lsho = self.rave.GetLink('L_SHOULDER_R_LINK')
        rhipy = self.rave.GetLink('R_HIP_Y_LINK')
        rhipp = self.rave.GetLink('R_HIP_P_LINK')
        lhipy = self.rave.GetLink('L_HIP_Y_LINK')
        lhipp = self.rave.GetLink('L_HIP_P_LINK')
        ignored_pairs = [set([chest, rsho]), set([chest, lsho]),
                         set([rhipy, rhipp]), set([lhipy, lhipp])]

        def callback(report, physics, debug=False):
            for link_pair in ignored_pairs:
                if set([report.plink1, report.plink2]) == link_pair:
                    if debug:
                        print "Ignored:\t", report
                    return 1
            if debug:
                print "Not ignored:\t", report
            return 0

        return self.env.RegisterCollisionCallback(callback)

    def self_collides(self, q):
        assert len(q) in [self.nb_dof, self.nb_active_dof]
        with self.rave:  # need to lock environment when calling robot methods
            if len(q) == self.nb_dof:
                self.rave.SetDOFValues(q)
            else:  # len(q) == self.nb_active_dof
                self.set_active_dof_values(q)
            collision = self.rave.CheckSelfCollision()
        return collision

    def compute_com(self, q=None):
        g = zeros(3)
        with self.rave:
            if q is not None:
                self.rave.SetDOFValues(q)
            for link in self.rave.GetLinks():
                g += link.GetMass() * link.GetGlobalCOM()
        return g / self.total_mass

    def compute_link_com_jacobian(self, link):
        index = link.GetIndex()
        com = link.GetGlobalCOM()
        return self.rave.CalculateJacobian(index, com)

    def compute_com_jacobian(self, q=None):
        Jcom = zeros((3, self.nb_dof))
        with self.rave:
            if q is not None:
                self.rave.SetDOFValues(q)
            for link in self.rave.GetLinks():
                Jcom += link.GetMass() * self.compute_link_com_jacobian(link)
        return Jcom / self.total_mass

    def compute_zmp(self, q, qd, qdd):
        global pb_times, total_times, cum_ratio, avg_ratio
        g = self.gravity
        f0 = self.total_mass * g[2]
        tau0 = zeros(3)
        with self.rave:
            self.rave.SetDOFValues(q)
            self.rave.SetDOFVelocities(qd)
            link_velocities = self.rave.GetLinkVelocities()
            link_accelerations = self.rave.GetLinkAccelerations(qdd)
            for link in self.rave.GetLinks():
                mi = link.GetMass()
                ci = link.GetGlobalCOM()
                I_ci = link.GetLocalInertia()
                Ri = link.GetTransform()[0:3, 0:3]
                ri = dot(Ri, link.GetLocalCOM())
                angvel = link_velocities[link.GetIndex()][3:]
                linacc = link_accelerations[link.GetIndex()][:3]
                angacc = link_accelerations[link.GetIndex()][3:]
                ci_ddot = linacc \
                    + cross(angvel, cross(angvel, ri)) \
                    + cross(angacc, ri)
                angmmt = dot(I_ci, angacc) - cross(dot(I_ci, angvel), angvel)
                f0 -= mi * ci_ddot[2]
                tau0 += mi * cross(ci, g - ci_ddot) - dot(Ri, angmmt)
        return cross(array([0, 0, 1]), tau0) * 1. / f0

    def compute_zmp_approx(self, q, qd, qdd):
        global pb_times, total_times, cum_ratio, avg_ratio
        g = self.gravity
        f0 = self.total_mass * g[2]
        tau0 = zeros(3)
        with self.rave:
            self.rave.SetDOFValues(q)
            self.rave.SetDOFVelocities(qd)
            link_velocities = self.rave.GetLinkVelocities()
            link_accelerations = self.rave.GetLinkAccelerations(qdd)
            for link in self.rave.GetLinks():
                mass_frame = link.GetGlobalMassFrame()
                mi = link.GetMass()
                ci = mass_frame[:3, 3]
                Ri = link.GetTransform()[0:3, 0:3]
                ri = dot(Ri, link.GetLocalCOM())
                angvel = link_velocities[link.GetIndex()][3:]
                linacc = link_accelerations[link.GetIndex()][:3]
                angacc = link_accelerations[link.GetIndex()][3:]
                ci_ddot = linacc \
                    + cross(angvel, cross(angvel, ri)) \
                    + cross(angacc, ri)
                f0 -= mi * ci_ddot[2]
                tau0 += mi * cross(ci, g - ci_ddot)
        return cross(array([0, 0, 1]), tau0) * 1. / f0

    def compute_inertia_matrix(self, q, external_torque=None):
        M = zeros((self.nb_dof, self.nb_dof))
        self.rave.SetDOFValues(q)
        for (i, e_i) in enumerate(eye(self.nb_dof)):
            tm, _, _ = self.rave.ComputeInverseDynamics(
                e_i, external_torque, returncomponents=True)
            M[:, i] = tm
        return M

    def compute_forward_dynamics(self, q, qd, tau):
        with self.rave:
            self.rave.SetDOFValues(q)
            self.rave.SetDOFVelocities(qd)
            zero_acc = zeros((self.nb_dof,))
            _, tc, tg = self.rave.ComputeInverseDynamics(
                zero_acc, returncomponents=True)
            M = self.compute_inertia_matrix(q)
            qdd = pylab.linalg.solve(M, tau - tc - tg)
        return qdd


#
# Tool functions
#

def pose_from_pos(p):
    pose = pylab.zeros(7)
    pose[0] = 1.
    pose[4:] = p
    return pose


def put_box(env, pos, size=5e-3):
    box = openravepy.RaveCreateKinBody(env, '')
    box.InitFromBoxes(pylab.array([[0., 0., 0., size, size, size]]), True)
    box.SetName("Box%d" % len(env.GetBodies()))
    env.Add(box)
    box.Enable(False)  # no collision checking
    t = box.GetTransform()
    t[(0, 1, 2), 3] = pos
    box.SetTransform(t)
    return box


def remove_boxes(env):
    for b in env.GetBodies():
        try:
            if b.GetName().index("Box") == 0:
                env.Remove(b)
        except:
            continue


def set_link_transparency(link, alpha):
    for g in link.GetGeometries():
        g.SetTransparency(alpha)


def update_kinbody(env, name, aabb):
    prec = env.GetKinBody(name)
    if prec is not None:
        env.Remove(prec)
    area = openravepy.RaveCreateKinBody(env, '')
    area.SetName(name)
    area.InitFromBoxes(array([array(aabb)]), True)
    g = area.GetLinks()[0].GetGeometries()[0]
    g.SetAmbientColor([.4, .1, .1])
    g.SetDiffuseColor([.5, .1, .1])
    env.Add(area, True)


def display_box(env, pos, box_id='Box', thickness=0.01):
    x, y, z = pos
    aabb = [x, y, z, thickness, thickness, thickness]
    update_kinbody(env, box_id, aabb)


def display_force(env, pos, vec, box_id='Force', f_scale=1e-3, thickness=5e-3):
    x, y, z = pos + .5 * f_scale * vec
    dx, dy, dz = .5 * f_scale * abs(vec)

    def thickify(x):
        return x if abs(x) > thickness else x * thickness / abs(x)

    dx, dy, dz = map(thickify, [dx, dy, dz])
    aabb = [x, y, z, dx, dy, dz]
    update_kinbody(env, box_id, aabb)


def plot_aabb(aabb, color="#AABBDD", ec="none", lw=0, alpha=.5):
    xy = aabb.pos()[:2] - aabb.extents()[:2]
    width, height = 2 * aabb.extents()[:2]
    rect = Rectangle(xy, width, height, color=color, ec=ec, alpha=alpha, lw=lw)
    pylab.gcf().gca().add_artist(rect)
    pylab.plot()
