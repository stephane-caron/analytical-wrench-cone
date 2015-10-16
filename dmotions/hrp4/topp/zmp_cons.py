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

from constraint import TOPPConstraint
from numpy import array, cross, dot, zeros


ZMP_AREA_SCALE = 0.4


class ZMPConstraint(TOPPConstraint):
    def __init__(self, hrp, support_state):
        self.hrp = hrp
        self.support_state = support_state

    def compute_zmp_area(self, q, support):
        Support = self.hrp.Support
        scale = ZMP_AREA_SCALE
        assert scale < 0.95, "The model over-estimates the foot size."
        assert support in [Support.LEFT, Support.RIGHT]
        foot_link = \
            self.hrp.left_foot_link if support == Support.LEFT else \
            self.hrp.right_foot_link
        with self.hrp.rave:
            self.hrp.rave.SetDOFValues(q)
            aabb = foot_link.ComputeAABB()
            length, width = aabb.extents()[0:2]
            pm = -1 if support == Support.LEFT else +1
            y_trans = pm * .5 * (1. - scale) * width
            xmax = aabb.pos()[0] + scale * length
            xmin = aabb.pos()[0] - scale * length
            ymax = aabb.pos()[1] + scale * width + y_trans
            ymin = aabb.pos()[1] - scale * width + y_trans
        return xmin, xmax, ymin, ymax

    def compute_support_matrices(self, q, qd, qss):
        double_support = self.support_state >= 3  # permissif
        if not double_support:
            xmin, xmax, ymin, ymax = self.compute_zmp_area(
                q, self.support_state)
        else:
            xminl, xmaxl, yminl, ymaxl = self.compute_zmp_area(
                q, self.hrp.Support.LEFT)
            xminr, xmaxr, yminr, ymaxr = self.compute_zmp_area(
                q, self.hrp.Support.RIGHT)
            xmin, ymin = min(xminl, xminr), min(yminl, yminr)
            xmax, ymax = max(xmaxl, xmaxr), max(ymaxl, ymaxr)
            assert yminl > ymaxr  # left foot on the left

        A_list = [
            [0, +1, 0],
            [0, -1, 0],
            [-1, 0, 0],
            [+1, 0, 0]]
        B_list = [
            [0, 0, +xmax],
            [0, 0, -xmin],
            [0, 0, +ymax],
            [0, 0, -ymin]]

        def append_segment(x1, y1, x2, y2):
            a, b, c = (x2 - x1), -(y2 - y1), -(x2 * y1 - x1 * y2)
            # append rows for equation (a y + b x + c <= 0)
            A_list.append([-a, b, 0])
            B_list.append([0, 0, -c])

        if double_support:
            if xminl < xminr:  # left foot behind
                append_segment(xminr, yminr, xminl, yminl)
                append_segment(xmaxl, ymaxl, xmaxr, ymaxr)
            else:  # left foot forward
                append_segment(xminr, ymaxr, xminl, ymaxl)
                append_segment(xmaxl, yminl, xmaxr, yminr)

        return array(A_list), array(B_list)

    def compute_abc(self, q, qs, qss):
        A, B = self.compute_support_matrices(q, qs, qss)
        (a, b), com = zeros((2, A.shape[0])), zeros(3)
        with self.hrp.rave:
            self.hrp.rave.SetDOFValues(q)
            self.hrp.rave.SetDOFVelocities(qs)
            for link in self.hrp.links:
                i = link.GetIndex()
                r = link.GetGlobalCOM()
                J = self.hrp.rave.CalculateJacobian(i, r)
                H = self.hrp.rave.ComputeHessianTranslation(i, r)
                rd = dot(J, qs)
                rdd = dot(J, qss) + dot(qs, dot(H, qs))
                m_i = link.GetMass()
                alpha_i = -m_i * (cross(A, r) + B)
                a += dot(alpha_i, rd)
                b += dot(alpha_i, rdd)
                com += m_i * r
        grav, mass = self.hrp.gravity, self.hrp.total_mass
        c = dot(cross(A, com) + mass * B, grav)
        return map(list, [a, b, c])
