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

import cvxopt.solvers

from constraint import TOPPConstraint
from dmotions.hrp4.contact import DistributedContactPoints
from numpy import array, dot, zeros, linalg, allclose, eye


cvxopt.solvers.options['show_progress'] = False  # disable cvxopt output


class ForceConstraint(TOPPConstraint):
    def __init__(self, hrp, taumax, support_state, active_dofs):
        self.hrp = hrp
        self.contact = DistributedContactPoints(hrp, support_state)
        self.taumax = taumax
        self.active_dofs = active_dofs

    def compute_abc(self, q, qs, qss, sd0=1., sdd0=0.):
        taumax = self.taumax[:50]
        JT = self.contact.compute_jacobian(q).T
        cdim = JT.shape[1]  # == 3 * nb_contact_points
        pinvJP = linalg.pinv(JT[50:, :])
        nullspace_proj = eye(cdim) - dot(pinvJP, JT[50:, :])

        a0, b0, c0 = self.compute_phase_components(q, qs, qss)

        Sa0, Sb0, Sc0 = map(lambda v: v[:50], [a0, b0, c0])
        fa0, fb0, fc0 = map(lambda v: dot(pinvJP, v[50:]), [a0, b0, c0])

        f0 = fa0 * sdd0 + fb0 * sd0 ** 2 + fc0
        T, fc = self.contact.get_friction_matrix()

        # minimize |z| s.t. (T * f0 + nullspace_proj(z) <= fc)
        cvx_P = cvxopt.matrix(eye(cdim))
        cvx_q = cvxopt.matrix(zeros(cdim))
        cvx_G = cvxopt.matrix(dot(T, nullspace_proj))
        cvx_h = cvxopt.matrix(fc - dot(T, f0))
        cvx_z = cvxopt.solvers.qp(cvx_P, cvx_q, cvx_G, cvx_h)['x']
        z = array(cvx_z).reshape((cdim,))

        z0 = zeros(cdim)
        z1 = z
        z2 = zeros(cdim)

        a_qp = fa0 + dot(nullspace_proj, z0)
        b_qp = fb0 + dot(nullspace_proj, z1)
        c_qp = fc0 + dot(nullspace_proj, z2)

        check = dot(T, a_qp * sdd0 + b_qp * sd0 ** 2 + c_qp) - fc
        # assert (check <= 11e-5).all(), "Error with:" + str(check)
        if not all(check <= 11e-5):
            print "no solution! returning best shot"
        assert allclose(z1 + z2, z)

        # dot(T, f) <= fc
        a_fric = dot(T, a_qp)
        b_fric = dot(T, b_qp)
        c_fric = dot(T, c_qp) - fc
        a, b, c = map(list, [a_fric, b_fric, c_fric])

        # tau <= +taumax
        a_act, b_act, c_act = {}, {}, {}
        a_act['right'] = Sa0 - dot(JT, a_qp)[:50]
        b_act['right'] = Sb0 - dot(JT, b_qp)[:50]
        c_act['right'] = Sc0 - dot(JT, c_qp)[:50] - taumax

        # tau >= -taumax
        a_act['left'] = -a_act['right']
        b_act['left'] = -b_act['right']
        c_act['left'] = -Sc0 + dot(JT, c_qp)[:50] - taumax

        for hs in ['left', 'right']:
            for dof in self.active_dofs:
                if dof.torque_limit is not None:
                    a.append(a_act[hs][dof.index])
                    b.append(b_act[hs][dof.index])
                    c.append(c_act[hs][dof.index])

        return a, b, c
