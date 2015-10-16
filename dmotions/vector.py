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

import numpy.random

from numpy import arccos, dot, fmod, isnan, pi, sqrt


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = arccos(dot(v1_u, v2_u))
    if isnan(angle):
        return 0. if (v1_u == v2_u).all() else pi
    return angle


def angle_between_or_zero(v1, v2):
    try:
        return angle_between(v1, v2)
    except ZeroDivisionError:
        return 0.


def center_angle_vect(q):
    qc = fmod(q, 2 * pi)
    qc += (-2 * pi) * (qc > +pi)
    qc += (+2 * pi) * (qc < -pi)
    return qc


def norm(vector):
    """Two times faster than pylab.norm:

        In [1]: %timeit norm(random.random(42))
        100000 loops, best of 3: 6.77 us per loop

        In [2]: %timeit pylab.norm(random.random(42))
        100000 loops, best of 3: 14.1 us per loop

    """
    return sqrt(dot(vector, vector))


def joints_dist(v1, v2):
    return norm(center_angle_vect(v2 - v1))


def unit_vector(vector):
    n = norm(vector)
    if n < 1e-10:
        raise ZeroDivisionError
    return vector / n


def unit_sphere_random(dim):
    """Uniform sampling from the unit sphere of dimension dim."""
    return unit_vector(numpy.random.normal(size=dim))


def unit_vector_or_zero(vector):
    try:
        return unit_vector(vector)
    except ZeroDivisionError:
        return vector
