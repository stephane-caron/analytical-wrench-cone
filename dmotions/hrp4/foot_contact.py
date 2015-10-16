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

#
# This script (foot_contact.py) is adapted from ClosedChain.py by Quand-Cuong
# Pham.
#

import dmotions
import numpy
import scipy.interpolate

from dmotions.rotation import quat_to_rpy
from dmotions.trajectory import PolynomialChunk, Trajectory
from dmotions.hrp4.ik import IKError
from numpy import array, dot, hstack, linspace, poly1d, zeros, eye
from numpy.linalg import pinv


NB_CHUNKS = 50

GAIN = 0.5

"""

Discussion: the gain should be decided jointly with the number of chunks (the
higher nchunks, the higher the gain, the better the trajectory).  For
a stepping task where norm(dot(Jphi, phid)) seemed to be around 1e-2,
I observed as a rule-of-thumb that gain=float(nchunks) worked nicely, while the
system was unstable below gain=.5*float(nchunks).

"""


class FootContact(object):

    def __init__(self, hrp4, traj):
        assert traj.active_dofs is not None
        assert traj.support_state is not None
        self.dt = traj.duration / NB_CHUNKS
        self.hrp4 = hrp4
        self.init_traj = traj
        self.rave = hrp4.rave
        self.trange = linspace(0, traj.duration, NB_CHUNKS + 1)
        self.use_right_leg = traj.support_state & hrp4.Support.RIGHT
        self.use_left_leg = traj.support_state & hrp4.Support.LEFT

        self.rleg_dofs = hrp4.get_dofs("R_LEG") if self.use_right_leg else []
        self.lleg_dofs = hrp4.get_dofs("L_LEG") if self.use_left_leg else []
        self.theta_dofs = self.rleg_dofs + self.lleg_dofs
        self.theta_max = array([dof.ulim for dof in self.theta_dofs])
        self.theta_min = array([dof.llim for dof in self.theta_dofs])
        self.phi_dofs = [dof for dof in traj.active_dofs
                         if dof not in self.theta_dofs]

    def compensate_trajectory(self):
        theta_values = self.compute_theta_from_phi()
        return self.merge_phi_theta(theta_values)

    def compute_theta_from_phi(self):
        rleg_indices = [dof.index for dof in self.rleg_dofs]
        lleg_indices = [dof.index for dof in self.lleg_dofs]
        theta_indices = rleg_indices + lleg_indices
        phi_indices = [dof.index for dof in self.phi_dofs]
        cur_q = self.init_traj.q_full(0.)
        theta = array(cur_q[theta_indices])
        theta_values = []
        rfoot_index = self.hrp4.right_foot_link.GetIndex()
        lfoot_index = self.hrp4.left_foot_link.GetIndex()

        with self.rave:
            self.rave.SetDOFValues(cur_q)
            dest_pose_left = self.hrp4.left_foot_link.GetTransformPose()
            dest_pose_right = self.hrp4.right_foot_link.GetTransformPose()

            self.q_list = []
            for t in self.trange:
                theta_values.append(array(theta))
                q0 = self.init_traj.q_full(t)
                qd0 = self.init_traj.qd_full(t)
                phi = q0[phi_indices]
                phid = qd0[phi_indices]
                cur_q[phi_indices] = phi
                cur_q[theta_indices] = theta
                self.q_list.append(cur_q.copy())
                self.rave.SetDOFValues(cur_q)
                rleg_thetad, lleg_thetad = [], []
                if self.use_right_leg:
                    rleg_thetad = self.compensate_pose(
                        phi_indices, rleg_indices, rfoot_index, phid,
                        dest_pose_right)
                if self.use_left_leg:
                    lleg_thetad = self.compensate_pose(
                        phi_indices, lleg_indices, lfoot_index, phid,
                        dest_pose_left)
                thetad = hstack([rleg_thetad, lleg_thetad])
                theta += thetad * self.dt
                if (theta < self.theta_min).any():
                    i = list(theta < self.theta_min).index(True)
                    msg = "%s DOF limit exceeded" % self.theta_dofs[i].name
                    raise IKError(cur_q, msg)
                if (theta > self.theta_max).any():
                    i = list(theta > self.theta_max).index(True)
                    msg = "%s DOF limit exceeded" % self.theta_dofs[i].name
                    raise IKError(cur_q, msg)
                t += self.dt

        return array(theta_values)

    def compensate_pose(self, phiindices, thetaindices, constrainedlinkindex,
                        phid, dest_pose):
        link = self.rave.GetLinks()[constrainedlinkindex]
        p = link.GetGlobalCOM()
        pose = link.GetTransformPose()
        Delta_p = numpy.hstack([
            # same ordering as Jacobians below
            pose[4:] - dest_pose[4:],
            quat_to_rpy(pose[:4]) - quat_to_rpy(dest_pose[:4])])

        Jglobalpos = self.rave.CalculateJacobian(
            constrainedlinkindex, p)
        Jglobalrot = self.rave.CalculateAngularVelocityJacobian(
            constrainedlinkindex)
        Jtheta = zeros((6, len(thetaindices)))
        Jphi = zeros((6, len(phiindices)))
        Jtheta[0:3, :] = Jglobalpos[:, thetaindices]
        Jtheta[3:6, :] = Jglobalrot[:, thetaindices]
        Jphi[0:3, :] = Jglobalpos[:, phiindices]
        Jphi[3:6, :] = Jglobalrot[:, phiindices]
        pd = -GAIN * Delta_p / self.dt - dot(Jphi, phid)
        # thetad = linalg.solve(Jtheta, pd)
        sr_inv = dot(Jtheta.T, pinv(dot(Jtheta, Jtheta.T) +
                                    1e-6 * eye(Jtheta.shape[0])))
        thetad = dot(sr_inv, pd)
        return thetad

    def merge_phi_theta(self, theta_values):
        all_dofs = self.theta_dofs + self.phi_dofs
        all_dofs.sort(key=lambda dof: dof.index)
        support_state = self.init_traj.support_state
        q_ref = self.init_traj.q_ref
        trange = self.trange
        tcklist = []
        for i in xrange(theta_values.shape[1]):
            tck = scipy.interpolate.splrep(trange, theta_values[:, i], s=0)
            tcklist.append(tck)
        t = tcklist[0][0]
        chunkslist = []
        for i in xrange(len(t) - 1):
            dof_poly_list = []
            dt = t[i + 1] - t[i]
            if abs(dt) < 1e-5:
                continue
            for dof in self.theta_dofs:
                idof = self.theta_dofs.index(dof)
                tck = tcklist[idof]
                a = 1 / 6. * scipy.interpolate.splev(t[i], tck, der=3)
                b = 0.5 * scipy.interpolate.splev(t[i], tck, der=2)
                c = scipy.interpolate.splev(t[i], tck, der=1)
                d = scipy.interpolate.splev(t[i], tck, der=0)
                dof_poly_list.append((
                    dof.index,
                    poly1d([a, b, c, d])))
            traj_type = type(self.init_traj)
            init_polychunk, t2 = \
                (self.init_traj, t[i]) if traj_type is PolynomialChunk else \
                self.init_traj.chunk_at(t[i])
            assert type(init_polychunk) is PolynomialChunk, \
                "Trajectory has non-polynomial chunks"
            ti_from_chunk_start = t2  # 80 characters limit :p
            for dof in self.phi_dofs:
                idof = self.init_traj.active_dofs.index(dof)
                P = init_polychunk.q_polynoms[idof]
                dof_poly_list.append((
                    dof.index,
                    dmotions.poly1d.translate_start(P, ti_from_chunk_start)))
            dof_poly_list.sort(key=lambda couple: couple[0])
            q_polynoms = zip(*dof_poly_list)[1]
            polychunk = PolynomialChunk(
                dt, q_polynoms, active_dofs=all_dofs, q_ref=q_ref,
                support_state=support_state)
            # assert len(dof_poly_list) == len(all_dofs)
            chunkslist.append(polychunk)
        return Trajectory(
            chunkslist, active_dofs=all_dofs, q_ref=q_ref,
            support_state=support_state)
