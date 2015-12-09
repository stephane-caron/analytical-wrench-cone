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


import cvxopt
import cvxopt.solvers
import pylab

cvxopt.solvers.options['show_progress'] = False


X = .5 * 0.224
Y = .5 * 0.130
NB_TRIALS = 10000


def check_on_random_instance():
    """

    Check our criterion with an LP solver on a random instance of the problem.
    Returns the random point, the criterion's outcome on this point (true or
    false) and whether an LP solution was found (positive or negative). If the
    criterion is correct, true == positive and false == negative.

    """

    K1, K2, K3, C1, C2 = 2 * pylab.random(5) - 1
    px, py = X / (X + Y), Y / (X + Y)

    D4max = .5 * min(2, 1 + C1, 1 + C2, 2 + C1 + C2)
    D4min = .5 * max(-1 + C1, C1 + C2, -1 + C2, 0)
    D4 = D4min + (D4max - D4min) * pylab.random()
    D1, D2, D3 = .5 * (1 + C1) - D4, -.5 * (C1 + C2) + D4, .5 * (1 + C2) - D4

    c = cvxopt.matrix(pylab.array([[1.]] * 8))   # score vector
    G = cvxopt.matrix(pylab.array([
        [+1, 0., 0., 0., 0., 0., 0., 0.],
        [-1, 0., 0., 0., 0., 0., 0., 0.],
        [0., +1, 0., 0., 0., 0., 0., 0.],
        [0., -1, 0., 0., 0., 0., 0., 0.],
        [0., 0., +1, 0., 0., 0., 0., 0.],
        [0., 0., -1, 0., 0., 0., 0., 0.],
        [0., 0., 0., +1, 0., 0., 0., 0.],
        [0., 0., 0., -1, 0., 0., 0., 0.],
        [0., 0., 0., 0., +1, 0., 0., 0.],
        [0., 0., 0., 0., -1, 0., 0., 0.],
        [0., 0., 0., 0., 0., +1, 0., 0.],
        [0., 0., 0., 0., 0., -1, 0., 0.],
        [0., 0., 0., 0., 0., 0., +1, 0.],
        [0., 0., 0., 0., 0., 0., -1, 0.],
        [0., 0., 0., 0., 0., 0., 0., +1],
        [0., 0., 0., 0., 0., 0., 0., -1]]))
    h = cvxopt.matrix(pylab.array([[1.]] * 16))  # h - G x >= 0
    A = cvxopt.matrix(pylab.array([
        [D1, D2, D3, D4, 0, 0, 0, 0],
        [0, 0, 0, 0, D1, D2, D3, D4],
        [-py * D1, +py * D2, +py * D3, -py * D4,
         +px * D1, +px * D2, -px * D3, -px * D4]]))
    b = cvxopt.matrix(pylab.array([K1, K2, K3]))
    sol = cvxopt.solvers.lp(c, G, h, A, b)

    K3min = -1 + py * abs(K1 - C1) + px * abs(K2 - C2)
    K3max = 1 - py * abs(K1 + C1) - px * abs(K2 + C2)

    is_true = K3min <= K3 <= K3max
    is_positive = sol['x'] is not None
    return is_true, is_positive, (K1, K2, K3, C1, C2)


def plot_samples(true_positives, true_negatives, false_positives):
    ax = pylab.subplot(111, projection='3d')
    ax.cla()
    if true_positives:
        K1, K2, K3, C1, C2 = zip(*true_positives)
        ax.scatter(K1, C1, K2, c='g')
    if true_negatives:
        K1, K2, K3, C1, C2 = zip(*true_negatives)
        ax.scatter(K1, C1, K2, c='r')
    if false_positives:
        K1, K2, K3, C1, C2 = zip(*false_positives)
        ax.scatter(K1, C1, K2, c='r')
    ax.set_xlabel("K1")
    ax.set_ylabel("C1")
    ax.set_zlabel("K2")
    # ax.view_init(elev=90, azim=90)  # X-Y plane
    # ax.view_init(elev=0, azim=180)  # Y-Z plane
    ax.view_init(elev=0, azim=270)  # X-Z plane
    pylab.show()


if __name__ == "__main__":
    true_positives = []
    true_negatives = []
    false_positives = []
    false_negatives = []

    for _ in xrange(NB_TRIALS):
        is_true, is_positive, inst = check_on_random_instance()
        l = true_positives if is_true and is_positive \
            else true_negatives if is_true and not is_positive \
            else false_positives if is_positive and not is_true \
            else false_negatives
        l.append(inst)

    print "Points in the 3D plot ought to be greeen."
    print ""
    print "Success rate (should be 100%%): %.2f%%" % (
        (len(true_positives) + len(false_negatives)) * 100. / NB_TRIALS)
    print "%d false-positives" % len(false_positives)
    print "%d true-negatives" % len(true_negatives)

    plot_samples(true_positives, true_negatives, false_positives)
