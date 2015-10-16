#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 Stephane Caron <caron@phare.normalesup.org>
#
# This file is part of surface-contacts-icra-2015.
#
# surface-contacts-icra-2015 is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# surface-contacts-icra-2015 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# surface-contacts-icra-2015. If not, see <http://www.gnu.org/licenses/>.

import cvxopt
import cvxopt.solvers
import numpy

from dmotions.interpolation import full_to_active
from dmotions.vector import norm
from numpy import array, dot, eye, hstack, vstack, zeros
from numpy.linalg import linalg

cvxopt.solvers.options['show_progress'] = False  # disable cvxopt output

DEBUG_IK = False  # printout objective values + animate OpenRAVE model

DOF_SCALE = 0.90  # additional scaling to avoid joint-limit saturation


def solve_ineq(A, b, I, x_max, x_min):
    """Solve A * x = b s.t. x_min <= x <= x_max."""
    qp_P = cvxopt.matrix(dot(A.T, A))
    qp_q = cvxopt.matrix(dot(-b.T, A))
    qp_G = cvxopt.matrix(vstack([+I, -I]))
    qp_h = cvxopt.matrix(hstack([x_max, x_min]))
    x = cvxopt.solvers.qp(qp_P, qp_q, qp_G, qp_h)['x']
    return array(x).reshape((I.shape[0],))


class IKError(Exception):

    def __init__(self, msg=None, last_q=None):
        self.last_q = last_q
        self.msg = msg

    def __str__(self):
        return self.msg


class KinematicTask(object):

    def __init__(self, f, J):
        self.f = f
        self.J = J


class PrioritizedKinematics(object):

    def __init__(self, robot, active_dofs, max_iter=50, conv_thres=1e-3,
                 gain=0.9):
        self.active_dofs = active_dofs
        self.active_indexes = [dof.index for dof in self.active_dofs]
        self.conv_thres = conv_thres
        self.converged = False
        self.gain = gain
        self.max_iter = max_iter
        self.nb_active_dof = len(active_dofs)
        self.robot = robot
        self.tasks = []

    def append(self, Task, *args, **kwargs):
        new_task = Task(self.robot, self.active_dofs, *args, **kwargs)
        return self.tasks.append(new_task)

    def _enforce(self, q_warm):
        self.converged = False
        self.robot.rave.SetDOFValues(q_warm)
        self.robot.rave.SetActiveDOFs(self.active_indexes)
        q_max = array([DOF_SCALE * dof.ulim for dof in self.active_dofs])
        q_min = array([DOF_SCALE * dof.llim for dof in self.active_dofs])
        I = eye(self.nb_active_dof)

        q = full_to_active(q_warm, self.active_dofs)
        self.robot.rave.SetActiveDOFValues(q)

        for itnum in xrange(self.max_iter):
            conv_vect = array([norm(task.f()) for task in self.tasks])
            if numpy.all(conv_vect < self.conv_thres):
                self.converged = True
                break
            if DEBUG_IK:
                conv = ["%10.8f" % x for x in conv_vect]
                print "   %4d: %s" % (itnum, ' '.join(conv))

            ker_proj = eye(self.nb_active_dof)
            dq = zeros(self.nb_active_dof)
            qd_max_reg = self.gain * (q_max - q)
            qd_min_reg = self.gain * (q - q_min)
            for i, task in enumerate(self.tasks):
                J = task.J()
                Jn = dot(J, ker_proj)
                b = -self.gain * task.f() - dot(J, dq)
                In = eye(Jn.shape[0])
                sr_inv = dot(Jn.T, linalg.inv(dot(Jn, Jn.T) + 1e-8 * In))
                dq += dot(sr_inv, b)
                ker_proj = dot(ker_proj, I - dot(linalg.pinv(Jn), Jn))

            qd_max_reg = self.gain * (q_max - q)
            qd_min_reg = self.gain * (q - q_min)
            q += solve_ineq(I, dq, I, qd_max_reg, qd_min_reg)
            self.robot.rave.SetActiveDOFValues(q)

        return self.robot.rave.GetDOFValues()

    def enforce(self, q_warm):
        if DEBUG_IK:
            return self._enforce(q_warm)
        with self.robot.rave:
            return self._enforce(q_warm)


class LinkPoseTask(KinematicTask):

    """Enforce a given pose for a link's referential."""

    def __init__(self, robot, active_dofs, link, target_pose):
        def f():
            return link.GetTransformPose() - target_pose
        J = self.get_jacobian_callback(robot, active_dofs, link)
        super(LinkPoseTask, self).__init__(f, J)

    def get_jacobian_callback(self, robot, active_dofs, link):
        active_indexes = [dof.index for dof in active_dofs]
        index = link.GetIndex()

        def J():
            pose = link.GetTransformPose()
            rot, pos = pose[:4], pose[4:]
            J_trans = robot.rave.CalculateJacobian(index, pos)
            J_rot = robot.rave.CalculateRotationJacobian(index, rot)
            J_full = numpy.vstack([J_rot, J_trans])
            # NB: vstack has same order as GetTransformPose()
            return J_full[:, active_indexes]

        return J


class LinkPointPoseTask(LinkPoseTask):

    """

    Enforce a given pose for a link's referential, with the origin at
    `local_origin` instead of the link's origin.

    """

    def __init__(self, robot, active_dofs, link, local_origin, target_pose):
        def f():
            pose = link.GetTransformPose()
            pose[4:] += dot(link.GetTransform()[0:3, 0:3], local_origin)
            return pose - target_pose

        J = self.get_jacobian_callback(robot, active_dofs, link)
        # NB: call the grandparent constructor
        super(LinkPoseTask, self).__init__(f, J)


class COMTask(KinematicTask):

    """Enforce a given projection of the COM on a plane floor."""

    def __init__(self, robot, active_dofs, target_com):
        assert target_com.shape == (3,)
        active_indexes = [dof.index for dof in active_dofs]

        def f():
            return robot.compute_com() - target_com

        def J():
            return robot.compute_com_jacobian()[:, active_indexes]

        super(COMTask, self).__init__(f, J)


class FloorCOMTask(KinematicTask):

    """Enforce a given projection of the COM on a plane floor."""

    def __init__(self, robot, active_dofs, target_com):
        assert target_com.shape == (2,)
        active_indexes = [dof.index for dof in active_dofs]

        def f():
            return robot.compute_com()[:2] - target_com

        def J():
            return robot.compute_com_jacobian()[(0, 1), :][:, active_indexes]

        super(FloorCOMTask, self).__init__(f, J)


class ElevateBaseLinkTask(KinematicTask):

    """Higher the base link."""

    def __init__(self, robot, active_dofs, target_z):
        z_index = active_dofs.index(robot.get_dof("TRANS_Z"))
        J_mat = zeros((1, len(active_dofs)))
        J_mat[0, z_index] = 1.

        def f():
            return robot.rave.GetDOFValues()[52] - target_z

        def J():
            return J_mat

        super(ElevateBaseLinkTask, self).__init__(f, J)


class RepulsiveLinks(KinematicTask):

    def __init__(self, robot, active_dofs, link1, link2):
        active_indexes = [dof.index for dof in active_dofs]
        dest_shape = (1, len(active_indexes))

        def f():
            diff = link1.GetTransformPose()[4:] - link2.GetTransformPose()[4:]
            return (1. / numpy.dot(diff, diff)).reshape((1,))

        def J():
            diff = link1.GetTransformPose()[4:] - link2.GetTransformPose()[4:]
            J1 = robot.compute_link_com_jacobian(link1)
            J2 = robot.compute_link_com_jacobian(link2)
            J = -2 * numpy.dot(diff, J1 - J2) / numpy.dot(diff, diff) ** 2
            return J[:, active_indexes].reshape(dest_shape)

        super(RepulsiveLinks, self).__init__(f, J)
