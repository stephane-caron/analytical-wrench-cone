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


class TOPPConstraint(object):
    def __init__(self, hrp):
        self.hrp = hrp

    def compute_phase_components(self, q, qs, qss):
        """Compute a(s), b(s), c(s) such that the retimed equation of motion
        becomes

            a(s) * sdd + b(s) * sd^2 + c(s) = S.T * torques + J.T * lambda

        with S the selection matrix, J the jacobian of the constraint equation
        and lambda the vector of constraint variables.
        """
        with self.hrp.rave:
            self.hrp.rave.SetDOFValues(q)
            self.hrp.rave.SetDOFVelocities(qs)
            Mqss, Cqs, g = self.hrp.rave.ComputeInverseDynamics(
                qss, externalforcetorque=None, returncomponents=True)
            Mqs, _, _ = self.hrp.rave.ComputeInverseDynamics(
                qs, externalforcetorque=None, returncomponents=True)
        a, b, c = Mqs, Mqss + Cqs, g
        return a, b, c

    def compute_abc(self, q, qs, qss):
        raise NotImplementedError()
