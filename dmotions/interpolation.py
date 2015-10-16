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

import copy
import numpy

from dmotions.trajectory import PolynomialChunk
from dmotions.vector import center_angle_vect
from numpy import poly1d


def poly2_interpolate(q0, q1, qd0=None, qd1=None, **kwargs):
    dofs = xrange(q0.shape[0])
    if qd0 is not None:
        C0, C1, C2 = q0, qd0, q1 - q0 - qd0
    elif qd1 is not None:
        Delta_q = center_angle_vect(q1 - q0)
        C0, C1, C2 = q0, -qd1 + 2 * Delta_q, qd1 - Delta_q
    else:
        raise Exception("please provide either qd0 or qd1")
    q_polynoms = [poly1d([C2[i], C1[i], C0[i]]) for i in dofs]
    traj = PolynomialChunk(1., q_polynoms, **kwargs)
    return traj


def bezier_interpolate(q_init, qd_init, q_dest, qd_dest, **kwargs):
    Delta_q = center_angle_vect(q_dest - q_init)
    q0 = q_init
    q3 = q_init + Delta_q
    q1 = q0 + qd_init / 3.
    q2 = q3 - qd_dest / 3.
    C0 = q0
    C1 = 3 * (q1 - q0)
    C2 = 3 * (q2 - 2 * q1 + q0)
    C3 = -q0 + 3 * q1 - 3 * q2 + q3
    dofs = xrange(len(q0))
    q_polynoms = [poly1d([C3[i], C2[i], C1[i], C0[i]]) for i in dofs]
    traj = PolynomialChunk(1., q_polynoms, **kwargs)
    return traj


def full_to_active(x, active_dofs):
    x_act = numpy.zeros(len(active_dofs))
    for i, dof in enumerate(active_dofs):
        x_act[i] = x[dof.index]
    return x_act


def active_to_full(x, x_ref, active_dofs):
    x_full = numpy.copy(x_ref)
    for i, dof in enumerate(active_dofs):
        x_full[dof.index] = x[i]
    return x_full


def active_bezier_interpolate(q_init, qd_init, q_dest, qd_dest, active_dofs,
                              support_state):
    q_ref = numpy.copy(q_init)
    q_init = full_to_active(q_init, active_dofs)
    q_dest = full_to_active(q_dest, active_dofs)
    qd_init = full_to_active(qd_init, active_dofs)
    qd_dest = full_to_active(qd_dest, active_dofs)
    traj = bezier_interpolate(
        q_init, qd_init, q_dest, qd_dest,
        active_dofs=copy.copy(active_dofs), q_ref=q_ref,
        support_state=support_state)
    return traj
