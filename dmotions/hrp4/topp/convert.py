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

import dmotions
import TOPP


def from_topp(topp_traj, ref_traj):
    active_dofs = ref_traj.active_dofs
    q_ref = ref_traj.q_ref
    support_state = ref_traj.support_state
    return dmotions.trajectory.TrajectoryChunk(
        topp_traj.duration, topp_traj.Eval, topp_traj.Evald,
        qdd_fun=topp_traj.Evaldd, active_dofs=active_dofs, q_ref=q_ref,
        support_state=support_state)


def to_topp(traj):
    def chunk_to_topp(chunk, deg=4):
        topp_polynomials = []
        for q_i in chunk.q_polynoms:
            coeffs = list(q_i.coeffs)
            coeffs.reverse()  # TOPP puts weaker coeffs first
            while len(coeffs) < deg:
                coeffs.append(0.)
            topp_polynomials.append(TOPP.Polynomial(coeffs))
        topp_chunk = TOPP.Chunk(chunk.duration, topp_polynomials)
        return topp_chunk

    chunks = []
    if type(traj) is dmotions.trajectory.PolynomialChunk:
        chunks = [chunk_to_topp(traj)]
    elif type(traj) is dmotions.trajectory.Trajectory:
        chunks = map(chunk_to_topp, traj.chunks)
    return TOPP.PiecewisePolynomialTrajectory(chunks)
