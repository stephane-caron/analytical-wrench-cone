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

import bisect
import dmotions
import numpy

from numpy import poly1d, polyder, zeros, pi
from vector import norm


class NoTrajectoryFound(Exception):

    pass


class TrajectoryError(Exception):

    def __init__(self, msg, traj=None, t=None):
        self.msg = msg
        self.traj = traj
        self.t = None

    def __str__(self):
        if self.t is not None:
            return self.msg + " at time t=%f" % self.t
        return self.msg


class VirtualTrajectory(object):

    def __init__(self):
        self.active_dofs = None
        self.q_ref = None
        self.support_state = None
        self.duration = None
        self.q = None
        self.qd = None
        self.qdd = None
        self.tau = None

    def check_diff(self, m=10, reltol=0.05):
        """Check differential constraints (e.g., dq = qd * dt).

        m -- number of random time instants to test
        reltol -- tolerance in relative error

        """
        tset = numpy.random.random(m) * self.duration
        q, qd, qdd, dt = self.q, self.qd, self.qdd, 1e-5
        for t in tset:
            q_dev = q(t + dt) - q(t) - dt * qd(t)
            qd_dev = qd(t + dt) - qd(t) - dt * qdd(t)
            if norm(q_dev) > reltol * dt * norm(qd(t)):
                raise TrajectoryError("bad velocity", self, t)
            if norm(qd_dev) > reltol * dt * norm(qdd(t)):
                kron = norm(qd_dev), reltol * dt * norm(qdd(t))
                msg = "bad acceleration (%e > %e)" % kron
                raise TrajectoryError(msg, self, t)

    def plot_q(self, dofs=None):
        import pylab
        dofs = xrange(16, 50) if dofs is None else dofs
        trange = pylab.linspace(0., self.duration, 100)
        pylab.plot(trange, [self.q_full(t)[dofs] * 180 / pi for t in trange])

    def plot_qd(self, dofs=None):
        import pylab
        dofs = xrange(16, 50) if dofs is None else dofs
        trange = pylab.linspace(0., self.duration, 100)
        pylab.plot(trange, [self.qd_full(t)[dofs] * 180 / pi for t in trange])


class TrajectoryChunk(VirtualTrajectory):

    def __init__(self, duration, q_fun, qd_fun, qdd_fun=None, tau_fun=None,
                 active_dofs=None, q_ref=None, support_state=None):
        assert (qdd_fun is None) != (tau_fun is None)
        self.active_dofs = active_dofs
        self.q_ref = q_ref
        self.support_state = support_state
        self.dof_zeros = zeros(q_ref.shape[0]) if q_ref is not None else None
        self.duration = duration
        self.q = q_fun
        self.qd = qd_fun
        self.qdd = qdd_fun
        self.tau = tau_fun

    def q_full(self, t):
        from interpolation import active_to_full
        return active_to_full(self.q(t), self.q_ref, self.active_dofs)

    def qd_full(self, t):
        from interpolation import active_to_full
        return active_to_full(self.qd(t), self.dof_zeros, self.active_dofs)

    def qdd_full(self, t):
        from interpolation import active_to_full
        return active_to_full(self.qdd(t), self.dof_zeros, self.active_dofs)

    def to_active(self, x):
        from interpolation import full_to_active
        return full_to_active(x, self.active_dofs)

    def retime(self, s_poly, new_duration):
        assert abs(s_poly(0.)) < 1e-10
        assert abs(s_poly(new_duration) - self.duration) < 1e-10
        s, sd, sdd = s_poly, s_poly.deriv(1), s_poly.deriv(2)

        def q(t):
            return self.q(s(t))

        def qd(t):
            return sd(t) * self.qd(s(t))

        def qdd(t):
            return sdd(t) * self.qd(s(t)) + sd(t) ** 2 * self.qdd(s(t))

        return TrajectoryChunk(
            new_duration, q, qd, qdd_fun=qdd,
            active_dofs=self.active_dofs,
            q_ref=self.q_ref,
            support_state=self.support_state)

    def timescale(self, scaling):
        return self.retime(poly1d([1. / scaling, 0]), scaling * self.duration)


class PolynomialChunk(TrajectoryChunk):

    def __init__(self, duration, q_polynoms, **kwargs):
        qd_polynoms = [polyder(P) for P in q_polynoms]
        qdd_polynoms = [polyder(P) for P in qd_polynoms]
        self.q_polynoms = q_polynoms
        self.qd_polynoms = qd_polynoms
        self.qdd_polynoms = qdd_polynoms
        self._kwargs = kwargs

        def q_fun(t):
            return numpy.array([q(t) for q in q_polynoms])

        def qd_fun(t):
            return numpy.array([qd(t) for qd in qd_polynoms])

        def qdd_fun(t):
            return numpy.array([qdd(t) for qdd in qdd_polynoms])

        super(PolynomialChunk, self).__init__(
            duration, q_fun, qd_fun, qdd_fun=qdd_fun, **kwargs)

    def timescale(self, scaling):
        def timescale_poly(P):
            return P(poly1d([1. / scaling, 0]))

        return PolynomialChunk(
            self.duration * scaling,
            [timescale_poly(q) for q in self.q_polynoms],
            active_dofs=self.active_dofs,
            q_ref=self.q_ref,
            support_state=self.support_state)

    def split(self, tlist):
        out_chunks = []
        t0 = tlist[0]
        out_chunks.append(PolynomialChunk(
            t0, self.q_polynoms, **self._kwargs))
        tlist.append(self.duration)
        for t1 in tlist[1:]:
            duration2 = t1 - t0

            def shift(P):
                return dmotions.poly1d.translate_zero(P, t0)

            q_polynoms2 = map(shift, self.q_polynoms)
            out_chunks.append(PolynomialChunk(
                duration2, q_polynoms2, **self._kwargs))
            t0 = t1
        return out_chunks


class Trajectory(VirtualTrajectory):

    def __init__(self, chunks, active_dofs=None, q_ref=None,
                 support_state=None):
        dtns = [traj.duration for traj in chunks]
        self.active_dofs = active_dofs
        self.q_ref = q_ref
        self.support_state = support_state
        self._kwargs = {
            'active_dofs': active_dofs,
            'q_ref': q_ref,
            'support_state': support_state}
        self.chunks = chunks
        self.cum_durations = [sum(dtns[0:i]) for i in xrange(len(chunks) + 1)]
        self.duration = sum(dtns)
        self.nb_chunks = len(chunks)

    def chunk_at(self, t, return_chunk_index=False):
        i = bisect.bisect(self.cum_durations, t)
        assert i > 0, "The first cumulative time should be zero..."
        chunk_index = min(self.nb_chunks, i) - 1
        t_start = self.cum_durations[chunk_index]
        chunk = self.chunks[chunk_index]
        if return_chunk_index:
            return chunk, (t - t_start), chunk_index
        return chunk, (t - t_start)

    def q(self, t):
        chunk, t2 = self.chunk_at(t)
        return chunk.q(t2)

    def qd(self, t):
        chunk, t2 = self.chunk_at(t)
        return chunk.qd(t2)

    def qdd(self, t):
        chunk, t2 = self.chunk_at(t)
        return chunk.qdd(t2)

    def q_full(self, t):
        chunk, t2 = self.chunk_at(t)
        return chunk.q_full(t2)

    def qd_full(self, t):
        chunk, t2 = self.chunk_at(t)
        return chunk.qd_full(t2)

    def qdd_full(self, t):
        chunk, t2 = self.chunk_at(t)
        return chunk.qdd_full(t2)

    def tau(self, t):
        traj, t2 = self.chunk_at(t)
        return traj.tau(t2)

    def to_active(self, x):
        from interpolation import full_to_active
        if self.active_dofs is not None:
            return full_to_active(x, self.active_dofs)
        raise TrajectoryError("Active DOFs undefined")

    def retime(self, s_poly, new_duration):
        raise NotImplementedError()

    def timescale(self, scaling):
        return Trajectory(
            [chunk.timescale(scaling) for chunk in self.chunks],
            **self._kwargs)

    def split3(self, t1, t2):
        """Split trajectory in three chunks.

        t1 -- first chunk time range is [0, t1]
        t2 -- second chunk time range is [t1, t2]

        """
        chunk1, tc1, i1 = self.chunk_at(t1, return_chunk_index=True)
        chunk2, tc2, i2 = self.chunk_at(t2, return_chunk_index=True)
        if i1 == i2:
            c1, c2, c4 = chunk1.split([tc1, tc2])
            c3 = []
        else:
            c1, c2 = chunk1.split([tc1])
            c3, c4 = chunk2.split([tc2])
        chunks_left = self.chunks[:i1] + [c1]
        chunks_mid = [c2] + self.chunks[i1:i2] + [c3]
        chunks_right = [c4] + self.chunks[i2:]
        traj_left = Trajectory(chunks_left, **self._kwargs)
        traj_mid = Trajectory(chunks_mid, **self._kwargs)
        traj_right = Trajectory(chunks_right, **self._kwargs)
        return traj_left, traj_mid, traj_right

    @staticmethod
    def merge(self, *trajectories):
        """Merge a list of trajectories.

        trajectories -- list of trajectories (given inline)

        For now, the active_dofs and support_state attributes are assumed to be
        the same across all trajectories. See the implementation of TOPPRetimer
        and FootContact for details.

        """
        active_dofs = trajectories[0].active_dofs
        support_state = trajectories[0].support_state
        for traj in trajectories:
            assert set(traj.active_dofs) == set(active_dofs)  # safe and slow
            assert traj.support_state == support_state
        chunks = [chunk for traj in trajectories for chunk in traj.chunks]
        return Trajectory(chunks, active_dofs=active_dofs,
                          q_ref=chunks[0].q_ref, support_state=support_state)
