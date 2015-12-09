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


import cvxopt.solvers
import dmotions
import itertools
import model
import numpy

from dmotions.rave import display_box
from foot_contact import FootContact
from numpy import array, dot
from topp import TOPPRetimer


cvxopt.solvers.options['show_progress'] = False  # disable cvxopt output

mass_inc_batteries = 39.  # [kg] mass including batteries


class HRP4(dmotions.rave.RaveRobotModel):

    """
    Robot model for the HRP4 series.
    """

    chains = model.chains
    left_foot_center_local = array([0.014, +0.01, -0.093])
    right_foot_center_local = array([0.014, -0.01, -0.093])

    class Support(object):
        FLY = 0     # like an eagle
        LEFT = 1    # left foot on the floor
        RIGHT = 2   # right foot on the floor
        DOUBLE = 3  # both feet on the floor

    def __init__(self, topp_npoints=None, topp_constraint=None):
        super(HRP4, self).__init__('dmotions/hrp4/env.xml')
        self.db = None
        for dof in self.dofs:
            dof.link = self.rave.GetLink(dof.name + '_LINK')
        self._adj_links = self.rave.GetAdjacentLinks()
        self.body_link = self.rave.GetLink("BODY")
        self.left_foot_link = self.rave.GetLink("L_FOOT_LINK")
        self.right_foot_link = self.rave.GetLink("R_FOOT_LINK")
        self.left_hand_link = self.rave.GetLink("L_WRIST_R_LINK")
        self.right_hand_link = self.rave.GetLink("R_WRIST_R_LINK")
        self.topp = None

        if topp_npoints is not None:
            assert topp_constraint is not None
            self.topp = TOPPRetimer(self, topp_npoints, topp_constraint)

        # note that we don't count the fingers as actuated joints
        actuated_dofs = [dof for dof in self.dofs if 16 <= dof.index < 50]
        self.actuated_dofs = sorted(actuated_dofs, key=lambda dof: dof.index)
        self.nb_actuated_dof = len(actuated_dofs)

        max_trans = .2
        self.dof_llim[50:53] = array([-max_trans, -max_trans, -max_trans])
        self.dof_ulim[50:53] = array([+max_trans, +max_trans, +max_trans])

        self.mass_diff = mass_inc_batteries - self.total_mass

    def set_dof_values(self, q):
        self.rave.SetDOFValues(q)
        self.display_com(q)
        self.display_floor_com(q)

    def display(self):
        if not self.display_on:
            self.display_on = True
            self.env.SetViewer('qtcoin')
            viewer = self.env.GetViewer()
            cam_trans = numpy.array([
                [0,  0, -1, 1.1],
                [1,  0,  0, 0.0],
                [0, -1,  0, 0.3],
                [0,  0,  0, 1.0]])
            cam_trans[:, 3] *= 3  # step back for HRP4
            viewer.SetBkgndColor([.4, .7, .5])
            viewer.SetCamera(cam_trans)

    def get_support_feet(self, support_state):
        support_feet = []
        if support_state & self.Support.LEFT:
            support_feet.append(self.left_foot_link)
        if support_state & self.Support.RIGHT:
            support_feet.append(self.right_foot_link)
        return support_feet

    def get_contact_pose(self, link, surface_coord):
        """

        Get the pose (in the absolute referential) of a point on the surface of
        a contacting link.

        link -- contacting link
        surface_coord -- (x, y) coordinates of the surface point

        """
        p = array([surface_coord[0], surface_coord[1], 0.])
        if link == self.left_foot_link:
            p += self.left_foot_center_local
        elif link == self.right_foot_link:
            p += self.right_foot_center_local
        else:
            raise NotImplemented("Unknown contacting link")
        pose = link.GetTransformPose()
        pose[4:] += dot(link.GetTransform()[0:3, 0:3], p)
        return pose

    def get_contact_point(self, link, surface_coord):
        """Get the absolute coordinates of a point on a contacting surface."""
        return self.get_contact_pose(link, surface_coord)[4:]

    def foot_floor_center(self, q, foot_link):
        with self.rave:
            self.rave.SetDOFValues(q)
            aabb = foot_link.ComputeAABB()
            return aabb.pos()[0:2]

    def check_links_collision(self, link1, link2):
        i1, i2 = link1.GetIndex(), link2.GetIndex()
        i1, i2 = (i1, i2) if i1 < i2 else (i2, i1)
        if (i1, i2) in self._adj_links:
            return False
        return self.env.CheckCollision(link1, link2)

    def joint_collides(self, q, joint, against_joints=[]):
        groumpf = [j.links for j in against_joints]
        links = list(itertools.chain.from_iterable(groumpf))
        with self.rave:
            self.rave.SetDOFValues(q)
            for link in links:
                if link in joint.links:
                    continue
                for joint_link in joint.links:
                    if self.check_links_collision(link, joint_link):
                        return True
        return False

    def compute_foot_jacobian(self, foot_link, q):
        with self.rave:
            self.rave.SetDOFValues(q)
            index = foot_link.GetIndex()
            assert index in [13, 20]
            com = foot_link.GetGlobalCOM()
            J = self.rave.CalculateJacobian(index, com)
            return J

    def compute_foot_hessian(self, foot_link, q):
        with self.rave:
            self.rave.SetDOFValues(q)
            index = foot_link.GetIndex()
            assert index in [13, 20]
            com = foot_link.GetGlobalCOM()
            H = self.rave.ComputeHessianTranslation(index, com)
            return H

    def display_com(self, q):
        com = self.compute_com(q)
        display_box(self.env, com, box_id="COM", thickness=0.04)

    def display_floor_com(self, q):
        com = self.compute_com(q)
        com[2] = 0.
        box_id = "ZMP"  # krooooooooooon
        display_box(self.env, com, box_id=box_id, thickness=0.04)

    def display_zmp(self, q, qd, qdd):
        zmp = self.compute_zmp(q, qd, qdd)
        zmp[2] = 0.
        display_box(self.env, zmp, box_id="ZMP", thickness=0.02)

    def play_trajectory(self, traj, callback=None, dt=3e-2, start=0.,
                        stop=None, nowait=False, com=True, zmp=True,
                        contact=False, slowdown=1.):
        def default_callback(t, q, qd, qdd, com=com, zmp=zmp, contact=contact):
            if com:
                self.display_com(q)
            if zmp:
                self.display_zmp(q, qd, qdd)
            if contact:
                support_state = traj.support_state
                self.display_contact_forces(q, qd, qdd, support_state)

        callback = callback if callback is not None else default_callback
        super(HRP4, self).play_trajectory(traj, callback, dt, start=start,
                                          stop=stop, nowait=nowait,
                                          slowdown=slowdown)

    def maintain_foot_contact(self, traj):
        foot_contact = FootContact(self, traj)
        return foot_contact.compensate_trajectory()

    def compute_zmp_waist(self, q, qd, qdd):
        """Compute the ZMP coordinates in the robot's WAIST frame."""
        zmp_global = self.compute_zmp(q, qd, qdd)
        with self.rave:
            self.rave.SetDOFValues(q)
            T = self.body_link.GetTransform()
            R, p = T[0:3, 0:3], T[:3, 3]
            zmp_local = numpy.dot(R.T, zmp_global - p)
        return zmp_local

    def export_openhrp(self, traj, fname, dt=5e-3):
        # NB: this function cannot be multi-threaded due to OpenRAVE calls.
        path = "openhrp/motions/" + fname
        print "Writing %s..." % (path + '.pos')
        with open(path + '.pos', 'w') as pos_fh:
            def callback(t, q, qd, qdd):
                dof_str = ' '.join(map(str, q[16:50]))
                pos_fh.write("%f %s\n" % (t, dof_str))
            with self.rave:  # don't play the trajectory
                self.play_trajectory(traj, callback=callback, dt=dt)
        if False:  # don't write ZMP for now
            print "Writing %s..." % (path + '.zmp')
            with open(path + '.zmp', 'w') as zmp_fh:
                def callback(t, q, qd, qdd):
                    zmp = self.compute_zmp_waist(q, qd, qdd)
                    zmp_str = ' '.join(map(str, zmp))
                    zmp_fh.write("%f %s\n" % (t, zmp_str))
                with self.rave:  # don't display the trajectory
                    self.play_trajectory(traj, callback=callback, dt=dt)
        print "Done."

    def check_dof_values(self, q):
        def humhum(v):
            v[39] = True
            v[40] = True
            v[48] = True
            v[49] = True
            return v[16:50]

        llim = humhum(self.dof_llim <= q)
        ulim = humhum(q <= self.dof_ulim)
        return llim.all() and ulim.all()
