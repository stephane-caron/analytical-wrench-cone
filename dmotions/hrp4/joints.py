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


class DOF(object):

    def __init__(self, name, index, vel_limit=42, torque_limit=42):
        def float_if_int(x):
            return float(x) if isinstance(x, int) else x
        torque_limit = float_if_int(torque_limit)
        assert isinstance(torque_limit, float) or torque_limit is None
        self.name = name
        self.index = index
        self.joint = None  # if None, set from parent joint
        self.link = None   # if None, set later from Rave robot
        self.llim = None   # if None, set from Rave robot
        self.torque_limit = torque_limit
        self.ulim = None   # if None, set from Rave robot
        self.vel_limit = vel_limit


class Joint(object):

    def __init__(self, name, dofs):
        self.name = name
        self.dofs = dofs
        self.dof_range = [dof.index for dof in dofs]
        self.ulim = None  # set from Rave robot
        self.llim = None  # set from Rave robot
        for dof in self.dofs:
            dof.joint = self

    @property
    def links(self):
        for dof in self.dofs:
            yield dof.link

    @property
    def nb_dofs(self):
        return len(self.dofs)


class RevoluteJoint(Joint):

    def __init__(self, name, index, vel_limit=100., torque_limit=None):
        dof = DOF(name, index, vel_limit, torque_limit)
        super(RevoluteJoint, self).__init__(name, [dof])
