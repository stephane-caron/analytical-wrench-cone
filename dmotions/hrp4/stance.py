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


from dmotions.hrp4.ik import LinkPointPoseTask
from dmotions.hrp4.ik import COMTask
from dmotions.hrp4.ik import PrioritizedKinematics, IKError


Unset = "Quarante-Deux"


class Stance(object):

    def __init__(self, hrp, lfoot=None, rfoot=None, q_warm=None, com=None):
        assert lfoot is None or lfoot.shape == (7,)
        assert rfoot is None or rfoot.shape == (7,)
        assert com is None or com.shape in [(2,), (3,)]
        q_init = q_warm if q_warm is not None else hrp.q_halfsit
        self.com = com
        self.hrp = hrp
        self.lfoot = lfoot.copy() if lfoot is not None else None
        self.rfoot = rfoot.copy() if rfoot is not None else None
        self.actuated_dofs = hrp.get_dofs(
            'R_LEG', 'CHEST', 'R_ARM', 'L_ARM', 'L_LEG', 'TRANS_Y', 'TRANS_X',
            'TRANS_Z')
        self.q = self.compute_q(q_init)
        if self.com is None:
            self.com = hrp.compute_com(self.q)

    def copy(self, lfoot=Unset, rfoot=Unset, q_warm=Unset, com=Unset):
        if type(lfoot) is str:
            lfoot = self.lfoot
        if type(rfoot) is str:
            rfoot = self.rfoot
        if type(q_warm) is str:
            q_warm = self.q
        if type(com) is str:
            com = self.com
        return Stance(self.hrp, lfoot, rfoot, q_warm, com)

    @staticmethod
    def from_q(hrp, q):
        with hrp.rave:
            hrp.rave.SetDOFValues(q)
            lfoot = hrp.get_contact_pose(hrp.left_foot_link, (0, 0))
            rfoot = hrp.get_contact_pose(hrp.right_foot_link, (0, 0))
        return Stance(hrp, lfoot, rfoot, q)

    def compute_q(self, q_warm):
        ik = PrioritizedKinematics(self.hrp, self.actuated_dofs)

        if self.lfoot is not None:
            ik.append(LinkPointPoseTask, self.hrp.left_foot_link,
                      self.hrp.left_foot_center_local, self.lfoot)

        if self.com is not None:
            ik.append(COMTask, self.com)

        if self.rfoot is not None:
            ik.append(LinkPointPoseTask, self.hrp.right_foot_link,
                      self.hrp.right_foot_center_local, self.rfoot)

        q = ik.enforce(q_warm)
        if self.hrp.self_collides(q):
            raise IKError("self-collides", q)
        elif not ik.converged:
            print "IK error: did not converge"
            raise IKError("did not converge", q)
        return q

    def recompute_q(self):
        q_new = self.compute_q(self.q)
        self.q = q_new
        return q_new
