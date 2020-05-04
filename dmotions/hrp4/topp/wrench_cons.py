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


from constraint import TOPPConstraint
from dmotions.hrp4.contact import SurfaceContactWrench
from numpy import allclose, dot, eye, hstack, vstack, zeros, array, ones
from scipy.linalg import inv, qr  # needs to be scipy
from pylab import norm as matnorm


X_real = 224e-3 / 2    # half foot length (same as in the paper)
Y_real = 130e-3 / 2    # half foot width (same as in the paper)
mu_real = 0.4          # friction coefficient

X = X_real * 0.45
Y = Y_real * 0.45
mu = mu_real / 1.4142  # linear friction coefficient

DEBUG_WRENCH = True


def alldot(*args):
    return reduce(dot, args)


class WrenchConstraint(TOPPConstraint):

    def __init__(self, hrp, taumax, support_state, active_dofs):
        self.hrp = hrp
        self.taumax = taumax[:50]
        self.active_dofs = active_dofs
        self.contact = SurfaceContactWrench(hrp, support_state)
        self.last_jacobian = 100000 * ones((6, 56))
        self.last_jacnorm = 1.
        self.last_projectors = None
        self.nb_recomp = 0

    def compute_actuated_projector_righetti(self, q):
        """
        This function implements the actuated projector from

            Righetti, Ludovic, et al. "Optimal distribution of contact forces
            with inverse-dynamics control." The International Journal of
            Robotics Research 32.3 (2013): 280-298.

        with a custom weight matrix on constraint variables.
        """
        weight_mat = eye(6)
        weight_mat[2, 2] = -X ** 2 - Y ** 2 - 2 * mu ** 2  # fz^2
        weight_mat[5, 5] = 0

        J = self.contact.compute_jacobian(q)
        if matnorm(J - self.last_jacobian) < 5e-2 * self.last_jacnorm:
            return self.last_projectors
        elif DEBUG_WRENCH:
            print "recomputing QR decomposition...",
            print matnorm(J - self.last_jacobian),
            print self.nb_recomp,
            self.nb_recomp += 1

        Q, R0 = qr(J.T)
        Qc = Q[:, :6]  # constrained
        Qu = Q[:, 6:]  # unconstrained
        R = R0[:6, :]
        nullmat = R0[6:, :]

        assert weight_mat.shape == (6, 6)
        assert allclose(dot(Q.T, Q), eye(len(q)))
        assert Q.shape == (len(q), len(q))
        assert R0.shape == (len(q), 6)
        assert R.shape == (6, 6)
        assert allclose(nullmat, 0)
        assert allclose(dot(Qc, R), J.T)

        n_act = len(q) - 6  # number of actuated DOF
        I_act = eye(n_act)
        S = hstack([I_act, zeros((n_act, 6))])
        R_inv = inv(R)
        Wc_inside = vstack([
            hstack([alldot(inv(R.T), weight_mat, R_inv), zeros((6, n_act))]),
            hstack([zeros((n_act, 6)), I_act])])
        Wc = alldot(Q, Wc_inside, Q.T)
        W = alldot(S, Wc, S.T)
        W_inv = inv(W)
        QuTSTinv = inv(dot(Qu.T, S.T))

        # Unconstrained-space projector
        #
        #     tau = Pu * (M * qdd + h)
        #     Pu.shape = (n_act, n)
        #
        Pu = dot(QuTSTinv, Qu.T) \
            + alldot((I_act - alldot(QuTSTinv, Qu.T, S.T)), W_inv, S, Wc)

        # Constrained-space projector
        #
        #     lambda = Pc * (M * qdd + h)
        #     Pc.shape = (6, n)
        #
        Pc = alldot(R_inv, Qc.T, eye(len(q)) - dot(S.T, Pu))

        if self.last_projectors:
            print matnorm(Pc - self.last_projectors[0]),
            print matnorm(Pu - self.last_projectors[1])

        self.last_jacobian = J
        self.last_jacnorm = matnorm(J)
        self.last_projectors = (Pc, Pu)
        return (Pc, Pu)

    def compute_actuated_projector(self, q):
        J = self.contact.compute_jacobian(q)
        P, S = zeros((6, 56)), zeros((50, 56))
        P[:, 50:] = eye(6)
        S[:, :50] = eye(50)
        Pc = dot(inv(dot(P, J.T)), P)
        Pu = S - alldot(S, J.T, Pc)
        return (Pc, Pu)

    def compute_abc(self, q, qs, qss):
        a, b, c = [], [], []

        a0, b0, c0 = self.compute_phase_components(q, qs, qss)
        Pc, Pu = self.compute_actuated_projector(q)

        # tau - taumax <= 0
        a_taumax = dot(Pu, a0)
        b_taumax = dot(Pu, b0)
        c_taumax = dot(Pu, c0) - self.taumax
        for dof in self.active_dofs:
            if dof.torque_limit is not None:
                a.append(a_taumax[dof.index])
                b.append(b_taumax[dof.index])
                c.append(c_taumax[dof.index])

        # -tau - taumax <= 0
        a_taumin = -dot(Pu, a0)
        b_taumin = -dot(Pu, b0)
        c_taumin = -dot(Pu, c0) - self.taumax
        for dof in self.active_dofs:
            if dof.torque_limit is not None:
                a.append(a_taumin[dof.index])
                b.append(b_taumin[dof.index])
                c.append(c_taumin[dof.index])

        # Wrench friction cone
        #
        #     C_local * wrench_local <= 0
        #
        # where ``wrench_local`` is expressed in the contact frame located at
        # the center of the rectangular foot contact area, with axes parallel
        # to the edges of the area.
        C_local = array([
            [+1,  0,           -mu,   0,   0,  0],
            [-1,  0,           -mu,   0,   0,  0],
            [0,  +1,           -mu,   0,   0,  0],
            [0,  -1,           -mu,   0,   0,  0],
            [0,   0,            -1,   0,   0,  0],
            [0,   0,            -Y,  +1,   0,  0],
            [0,   0,            -Y,  -1,   0,  0],
            [0,   0,            -X,   0,  +1,  0],
            [0,   0,            -X,   0,  -1,  0],
            [+Y, +X, -mu * (X + Y), -mu, -mu, -1],
            [+Y, -X, -mu * (X + Y), -mu, +mu, -1],
            [-Y, +X, -mu * (X + Y), +mu, -mu, -1],
            [-Y, -X, -mu * (X + Y), +mu, +mu, -1],
            [+Y, +X, -mu * (X + Y), +mu, +mu, +1],
            [+Y, -X, -mu * (X + Y), +mu, -mu, +1],
            [-Y, +X, -mu * (X + Y), -mu, +mu, +1],
            [-Y, -X, -mu * (X + Y), -mu, -mu, +1]])

        R = self.contact.get_link_rotation(q)
        import pylab
        assert pylab.norm(R - eye(3)) <= 1e-1  # tmp
        O = zeros((3, 3))

        # Thanks to Steven Jens Jorgensen for fixing the line below
        # See https://github.com/stephane-caron/analytical-wrench-cone/issues/2
        wrench_rotation = vstack([
            hstack([R.transpose(), O]),
            hstack([O, R.transpose()])])

        # Wrench friction cone (inertial frame)
        #
        #     C * wrench <= 0
        #
        # where ``wrench`` is expressed in the inertial frame located at the
        # center of the rectangular foot contact area, with axes parallel to
        # those of the world frame.
        C = dot(C_local, wrench_rotation)

        a_wrench = alldot(C, Pc, a0)
        b_wrench = alldot(C, Pc, b0)
        c_wrench = alldot(C, Pc, c0)
        a.extend(a_wrench)
        b.extend(b_wrench)
        c.extend(c_wrench)

        return a, b, c

    def test_abc(self, q, qs, qss, sd, sdd=0):
        """Compare (a, b, c) with the DCP model."""
        a, b, c = self.compute_abc(q, qs, qss)
        a, b, c = map(array, [a, b, c])
        xW = a * sdd + b * sd ** 2 + c
        print "SWC: (a sdd + b sd^2 + c <= 0)?", all(xW < 0)

        from dcp_cons import ForceConstraint
        fcons = ForceConstraint(self.hrp, self.hrp.torque_limits,
                                self.hrp.Support.LEFT, self.active_dofs)
        a2, b2, c2 = fcons.compute_abc(q, qs, qss, sd0=sd, sdd0=sdd)
        a2, b2, c2 = map(array, [a2, b2, c2])
        xf = a2 * sdd + b2 * sd ** 2 + c2
        print "DCP: (a sdd + b sd^2 + c <= 0)?", all(xf < 0)
