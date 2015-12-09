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


import convert
import TOPP
import time

from dcp_cons import ForceConstraint
from numpy import arange
from wrench_cons import WrenchConstraint
from zmp_cons import ZMPConstraint


TOPP_TAUMAX_RATIO = 1.0

TOPP_VMAX_RATIO = 1.0


class TOPPRetimer(object):
    def __init__(self, hrp, npoints, ConstraintClass):
        self.ConstraintClass = ConstraintClass
        self.npoints = npoints
        self.hrp = hrp
        self.last_inst = None  # debug field
        self.taumax = TOPP_TAUMAX_RATIO * hrp.torque_limits
        self.vmax = TOPP_VMAX_RATIO * hrp.velocity_limits
        from numpy import zeros
        self.vmax = zeros(hrp.velocity_limits.shape)  # disable MVC direct

    def get_constraint(self, traj):
        if self.ConstraintClass is ForceConstraint:
            return ForceConstraint(
                self.hrp, self.taumax, traj.support_state,
                traj.active_dofs)
        elif self.ConstraintClass is WrenchConstraint:
            return WrenchConstraint(
                self.hrp, self.taumax, traj.support_state,
                traj.active_dofs)
        elif self.ConstraintClass is ZMPConstraint:
            return ZMPConstraint(
                self.hrp, traj.support_state)
        return None

    def compute_topp_instance(self, traj):
        abc_list = []
        constraint = self.get_constraint(traj)
        discrtimestep = traj.duration / self.npoints
        T = traj.duration + discrtimestep  # for TOPP
        for s in arange(0., T, discrtimestep):
            q = traj.q_full(s)
            qs = traj.qd_full(s)
            qss = traj.qdd_full(s)
            abc_list.append(constraint.compute_abc(q, qs, qss))
        a, b, c = zip(*abc_list)
        topp_traj = convert.to_topp(traj)
        vmax = traj.to_active(self.vmax)
        return TOPP.QuadraticConstraints(
            topp_traj, discrtimestep, vmax, a, b, c)

    def retime_trajectory(self, traj, sd_beg, sd_end):
        assert traj.active_dofs is not None
        assert traj.q_ref is not None
        assert traj.support_state is not None

        t0 = time.time()
        topp_inst = self.compute_topp_instance(traj)
        self.last_inst = topp_inst  # debug, saved for later
        t1 = time.time()
        print "[TOPP] (a, b, c) computation: %.2f s" % (t1 - t0)
        topp_traj = topp_inst.Reparameterize(sd_beg, sd_end)
        t2 = time.time()
        retimed_traj = convert.from_topp(topp_traj, ref_traj=traj)
        print "[TOPP] retiming:              %.2f s" % (t2 - t1)
        print "[TOPP] new traj. duration:    %.2f s" % topp_traj.duration
        return retimed_traj

    def avp_trajectory(self, traj, sd_beg_min, sd_beg_max):
        topp_inst = self.compute_topp_instance(traj)
        self.last_inst = topp_inst
        return topp_inst.AVP(sd_beg_min, sd_beg_max)

    def plot_last_instance(self, ylim=None):
        import pylab
        self.last_inst.PlotProfiles()
        if ylim is not None:
            pylab.ylim(ylim)
        self.last_inst.PlotAlphaBeta()

    def write_last_instance(self):
        with open("./TOPP-constraintstring.txt", "w") as f:
            f.write(self.last_inst.constraintstring)
        with open("./TOPP-trajstring.txt", "w") as f:
            f.write(self.last_inst.trajstring)
        print "Files written to ./TOPP-*string.txt"
