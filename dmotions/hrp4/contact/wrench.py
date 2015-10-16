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

import pylab
from numpy import vstack, array


class SurfaceContactWrench(object):

    def __init__(self, hrp, support_state):
        assert support_state in [hrp.Support.LEFT, hrp.Support.RIGHT]
        self.hrp = hrp
        if support_state == hrp.Support.LEFT:
            self.link = hrp.left_foot_link
        else:  # support_state == hrp.Support.RIGHT
            self.link = hrp.right_foot_link

        # temporary: no rotation of the link
        assert pylab.norm(
            self.link.GetTransformPose()[:4] -
            array([1., 0., 0., 0.])) <= 1e-2, \
            str(self.link.GetTransformPose()[:4])

    def get_link_rotation(self, q):
        with self.hrp.rave:
            self.hrp.rave.SetDOFValues(q)
            return self.link.GetTransform()[0:3, 0:3]

    def compute_jacobian(self, q):
        """Compute the jacobian of the contact relation (i.e., the link and
        surface referentials coincidate) in the world reference frame.

        q -- joint-angle values

        """
        with self.hrp.rave:
            self.hrp.rave.SetDOFValues(q)
            i = self.link.GetIndex()
            p = self.hrp.get_contact_point(self.link, surface_coord=(0, 0))
            Jtrans = self.hrp.rave.CalculateJacobian(i, p)
            Jrot = self.hrp.rave.CalculateAngularVelocityJacobian(i)
            return vstack([Jtrans, Jrot])
