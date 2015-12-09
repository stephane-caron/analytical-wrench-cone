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


import dmotions

from pylab import hstack, eye, zeros, inv, dot, random, norm


def check_actuated_torques(hrp, q, qd, qdd, tau, J, F_ext):
    """Check actuated torques `tau` computed against the external wrench F_ext
    using Equation (8) from (Mistry, Buchli and Schall, 2010).

    robot -- robot object
    q -- full-body configuration
    qd -- full-body velocity
    qdd -- full-body acceleration
    tau -- active joint torques
    J -- contact Jacobian
    F_ext -- contact wrench

    """
    with hrp.rave:
        hrp.rave.SetDOFValues(q)
        hrp.rave.SetDOFVelocities(qd)
        _, tc, tg = hrp.rave.ComputeInverseDynamics(qdd, returncomponents=True)
    M = hrp.compute_inertia_matrix(hrp.q)
    S = hstack([eye(50), zeros((50, 6))])
    SMS_inv = inv(dot(S, dot(inv(M), S.T)))
    S_bar = dot(SMS_inv, dot(S, inv(M))).T
    v = dot(S_bar.T, (tc + tg - dot(J.T, F_ext)))
    tau_check = dot(SMS_inv, qdd[:50]) + v
    return norm(tau - tau_check) < 1e-5


def test_com_jacobian(dq_norm=1e-3, q=None):
    if q is None:
        q = hrp.dof_llim + random(56) * (hrp.dof_ulim - hrp.dof_llim)
    dq = random(56) * dq_norm
    com = hrp.compute_com(q)
    J_com = hrp.compute_com_jacobian(q)
    expected = com + dot(J_com, dq)
    actual = hrp.compute_com(q + dq)
    assert norm(actual - expected) < 2 * dq_norm ** 2
    return J_com


def test(label, test_fun, *args):
    print "Testing %s...\t" % label,
    test_fun(*args)
    print "OK."


if __name__ == '__main__':
    hrp = dmotions.HRP4()
    for _ in range(10):
        test("COM Jacobian", test_com_jacobian)
