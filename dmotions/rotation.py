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

from numpy import array, dot, tensordot
from numpy.random import random
from pylab import norm
from math import atan2, asin, cos, sin


vect_to_mat = array([
    [[0,  0,  0], [0,  0, -1], [0, +1, 0]],
    [[0,  0, +1], [0,  0,  0], [-1, 0, 0]],
    [[0, -1,  0], [+1, 0,  0], [0,  0, 0]]])


__quat_to_rot__ = array([[

    # [0, 0]: a^2 + b^2 - c^2 - d^2
    [[+1,  0,  0,  0],
     [.0, +1,  0,  0],
     [.0,  0, -1,  0],
     [.0,  0,  0, -1]],

    # [0, 1]: 2bc - 2ad
    [[.0,  0,  0, -2],
     [.0,  0, +2,  0],
     [.0,  0,  0,  0],
     [.0,  0,  0,  0]],

    # [0, 2]: 2bd + 2ac
    [[.0,  0, +2,  0],
     [.0,  0,  0, +2],
     [.0,  0,  0,  0],
     [.0,  0,  0,  0]]], [

    # [1, 0]: 2bc + 2ad
    [[.0,  0,  0, +2],
     [.0,  0, +2,  0],
     [.0,  0,  0,  0],
     [.0,  0,  0,  0]],

    # [1, 1]: a^2 - b^2 + c^2 - d^2
    [[+1,  0,  0,  0],
     [.0, -1,  0,  0],
     [.0,  0, +1,  0],
     [.0,  0,  0, -1]],

    # [1, 2]: 2cd - 2ab
    [[.0, -2,  0,  0],
     [.0,  0,  0,  0],
     [.0,  0,  0, +2],
     [.0,  0,  0,  0]]], [

    # [2, 0]: 2bd - 2ac
    [[.0,  0, -2,  0],
     [.0,  0,  0, +2],
     [.0,  0,  0,  0],
     [.0,  0,  0,  0]],

    # [2, 1]: 2cd + 2ab
    [[0, +2,  0,  0],
     [0,  0,  0,  0],
     [0,  0,  0, +2],
     [0,  0,  0,  0]],

    # [2, 2]: a^2 - b^2 - c^2 + d^2
    [[+1,  0,  0,  0],
     [.0, -1,  0,  0],
     [.0,  0, -1,  0],
     [.0,  0,  0, +1]]]])

quat_to_rot = __quat_to_rot__.transpose([2, 0, 1, 3])
# quat_to_rot.shape == (4, 3, 3, 4)


def quat_to_rpy(q):
    roll = atan2(
        2 * q[2] * q[3] + 2 * q[0] * q[1],
        q[3] ** 2 - q[2] ** 2 - q[1] ** 2 + q[0] ** 2)
    pitch = -asin(
        2 * q[1] * q[3] - 2 * q[0] * q[2])
    yaw = atan2(
        2 * q[1] * q[2] + 2 * q[0] * q[3],
        q[1] ** 2 + q[0] ** 2 - q[3] ** 2 - q[2] ** 2)
    return array([roll, pitch, yaw])


def rpy_to_quat(roll, pitch, yaw):
    cr, cp, cy = cos(roll / 2), cos(pitch / 2), cos(yaw / 2)
    sr, sp, sy = sin(roll / 2), sin(pitch / 2), sin(yaw / 2)
    return array([
        cr * cp * cy + sr * sp * sy,
        -cr * sp * sy + cp * cy * sr,
        cr * cy * sp + sr * cp * sy,
        cr * cp * sy - sr * cy * sp])


if __name__ == "__main__":
    q = random(4)
    p1 = tensordot(q, dot(quat_to_rot, q), axes=([0, 0]))
    p2_right = tensordot(__quat_to_rot__, q, axes=([3, 0]))
    p2 = tensordot(q, p2_right, axes=([0, 2]))
    assert abs(norm(p1 - p2)) < 1e-10

    q = random(4)
    q = q / norm(q)
    q2 = rpy_to_quat(*quat_to_rpy(q))
    assert norm(q - q2) < 1e-10

    print "All tests OK."
