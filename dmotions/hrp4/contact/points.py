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


from contact import Contact
from dmotions.rave import display_box, display_force
from dmotions.vector import norm
from numpy import array, dot, vstack, linalg
from scipy.linalg import block_diag


CONTACT_MIN_REACT = 10.  # [N]

CONTACT_FRICTION = 0.8


class DistributedContactPoints(Contact):

    def __init__(self, hrp, support_state, ncontacts=4):
        assert ncontacts == 4  # not extended yet
        assert support_state in [hrp.Support.LEFT, hrp.Support.RIGHT]
        self.hrp = hrp
        self.support_state = support_state
        self.ncontacts = ncontacts

    def compute_jacobian(self, q, returncontacts=False):
        foot_links = self.hrp.get_support_feet(self.support_state)
        contacts, jacobians = [], []
        with self.hrp.rave:
            self.hrp.rave.SetDOFValues(q)
            for foot_link in foot_links:
                i = foot_link.GetIndex()
                T = foot_link.GetTransform()
                if foot_link == self.hrp.left_foot_link:
                    c1 = array([+0.127, +.075, -0.093, 1.])
                    c2 = array([+0.127, -.055, -0.093, 1.])
                    c3 = array([-0.097, -.055, -0.093, 1.])
                    c4 = array([-0.097, +.075, -0.093, 1.])
                else:  # foot_link == self.hrp.right_foot_link
                    c1 = array([+0.127, +.055, -0.093, 1.])
                    c2 = array([+0.127, -.075, -0.093, 1.])
                    c3 = array([-0.097, -.075, -0.093, 1.])
                    c4 = array([-0.097, +.055, -0.093, 1.])
                foot_contacts = [dot(T, c)[0:3] for c in [c1, c2, c3, c4]]
                foot_jacobians = [self.hrp.rave.CalculateJacobian(i, p)
                                  for p in foot_contacts]
                contacts.extend(foot_contacts)
                jacobians.extend(foot_jacobians)
        J = vstack(jacobians)
        assert J.shape == (12 * len(foot_links), 56)
        return (J, contacts) if returncontacts else J

    def compute_inverse_dynamics(self, q, qd, qdd, returncomponents=False):
        with self.hrp.rave:
            self.hrp.rave.SetDOFValues(q)
            self.hrp.rave.SetDOFVelocities(qd)
            tMqdd, tCqd, tg = self.hrp.rave.ComputeInverseDynamics(
                qdd, externalforcetorque=None, returncomponents=True)

        J = self.compute_contact_jacobian(q, qd)

        def solve_unact(v):
            return linalg.lstsq(J.T[50:, :], v[50:])[0]

        def act_proj(v):
            return (v - dot(J.T, solve_unact(v)))[:50]

        if returncomponents:
            return map(act_proj, [tMqdd, tCqd, tg])
        return act_proj(tMqdd + tCqd + tg)

    def get_friction_matrix(self):
        min_fz, mu = CONTACT_MIN_REACT, CONTACT_FRICTION
        T0 = array([
            [+1, +1, -mu],
            [+1, -1, -mu],
            [-1, +1, -mu],
            [-1, -1, -mu],
            [0, 0, -1]])
        T = block_diag(*([T0] * self.ncontacts))
        fc = array([0., 0., 0., 0., -min_fz] * self.ncontacts)
        return T, fc

    def check_contact_forces(self, q, qd, qdd):
        import cvxopt.solvers
        cvxopt.solvers.options['show_progress'] = False  # disable output

        with self.hrp.rave:
            self.hrp.rave.SetDOFValues(q)
            self.hrp.rave.SetDOFVelocities(qd)
            tau = self.rave.ComputeInverseDynamics(qdd)

        J, contacts = self.compute_contact_jacobian(
            q, qd, self.support_state, returncontacts=True)
        ncontacts = len(contacts)
        T, fc = self.get_friction_cones(ncontacts)
        PJT, Ptau = J.T[50:, :], tau[50:]

        # QP: minimize norm(dot(PJT, f) - Ptau)
        #     s.t. dot(T, f) <= fc
        #
        qp_P = cvxopt.matrix(dot(PJT.T, PJT))
        qp_q = cvxopt.matrix(-dot(Ptau, PJT))
        qp_G = cvxopt.matrix(T)
        qp_h = cvxopt.matrix(fc)
        qp_x = cvxopt.solvers.qp(qp_P, qp_q, qp_G, qp_h)['x']
        f = array(qp_x).reshape((3 * ncontacts,))
        error = norm(dot(PJT, f) - Ptau)
        thres = 300 * 1e-2  # norm(Ptau) is around 300 N.m
        if error > thres / 2:  # more than half the allowed error
            print "danger zone: (%f > %f) " % (error, thres / 2)
        assert error < thres, \
            "contact forces cannot realize the motion " \
            "(%f > %f) " % (error, thres)
        return contacts, f

    def display_contact_forces(self, q, qd, qdd):
        contacts, f = self.check_contact_forces(q, qd, qdd)
        for i in xrange(8):
            for s in ["Contact%d" % i, "Force%d" % i]:
                kinbody = self.env.GetKinBody(s)
                if kinbody is not None:
                    self.env.Remove(kinbody)
        for i, c in enumerate(contacts):
            cur_f = f[3 * i:3 * (i + 1)]
            display_box(self.env, c, box_id="Contact%d" % i)
            display_force(self.env, c, cur_f, "Force%d" % i)
