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

from numpy import poly1d
from operator import mul
from fractions import Fraction


def B(n, k):
    """Binomial coefficient."""
    return int(reduce(mul, (Fraction(n - i, i + 1) for i in xrange(k)), 1))


def translate_start(P, x0):
    """Returns Q(x) = P(x + x0)."""
    n, a = P.order, P.coeffs
    b = [sum(B(k + n - j, n - j) * a[j - k] * x0 ** k
             for k in xrange(j + 1))
         for j in xrange(n + 1)]
    return poly1d(b)


def bezier_interpolate(P0, Pd0, PT, PdT, T):
    """
    Returns P s.t.
        P(0)  = P0,  P(T)  = PT,
        P'(0) = Pd0, P'(T) = PdT.
    """
    d = P0
    c = Pd0
    b = 3 * (PT - P0) / T ** 2 - (PdT + 2 * Pd0) / T
    a = 2 * (P0 - PT) / T ** 3 + (Pd0 + PdT) / T ** 2
    P = poly1d([a, b, c, d])
    return P
