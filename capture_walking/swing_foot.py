#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017-2018 Stephane Caron <stephane.caron@lirmm.fr>
#
# This file is part of capture-walking
# <https://github.com/stephane-caron/capture-walking>.
#
# capture-walking is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# capture-walking is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with capture-walking. If not, see <http://www.gnu.org/licenses/>.

import pymanoid
import toppra

from numpy import array, dot, eye, linspace, vstack, zeros
from openravepy import InterpolateQuatSlerp as quat_slerp
from pymanoid.gui import draw_line, draw_point, draw_trajectory
from pymanoid.interp import interpolate_cubic_hermite
from pymanoid.optim import solve_qp


def factor_cubic_hermite_curve(p0, n0, p1, n1):
    """
    Let `H` denote the Hermite curve (parameterized by `\\lambda` and `\\mu`)
    such that:

    .. math::

        \\begin{array}{rcl}
            H(0) & = & p_0 \\\\
            H'(0) & = & \\lambda n_0 \\\\
            H(1) & = & p_1 \\\\
            H'(1) & = & \\mu n_1 \\\\
        \\end{array}

    This function returns the three factors of `H` corresponding to the symbols
    :math:`\\lambda`, :math:`\\mu` and :math:`1` (constant part).

    Parameters
    ----------
    p0 : (3,) array
        Initial position.
    n0 : (3,) array
        Initial tangent vector.
    p1 : (3,) array
        Final position.
    n1 : (3,) array
        Final tangent vector.

    Returns
    -------
    H_lambda : function
        Factor of :math:`\\lambda` in `H`.
    H_mu : function
        Factor of :math:`\\mu` in `H`.
    H_cst : function
        Part of `H` that depends neither on :math:`\\lambda` nor :math:`\\mu`.
    """
    def H_lambda(s):
        return s * (1 + s * (s - 2)) * n0

    def H_mu(s):
        return s ** 2 * (s - 1) * n1

    def H_cst(s):
        return p0 + s ** 2 * (3 - 2 * s) * (p1 - p0)

    return H_lambda, H_mu, H_cst


class SwingFootTracker(pymanoid.Process):

    """
    Swing foot interpolation and tracking.

    Parameters
    ----------
    start_contact : pymanoid.Contact
        Initial contact.
    end_contact : pymanoid.Contact
        Target contact.
    foot_target : pymanoid.Body
        IK target to be updated during swing foot motion.
    """

    def __init__(self, start_contact, end_contact, foot_target):
        self.default_clearances = [0.05, 0.1]
        self.dt = 0.03
        self.duration = None
        self.end_contact = end_contact
        self.foot_target = foot_target
        self.foot_vel = zeros(3)
        self.handles = []
        self.max_accel = 10. * array([1., 1., 1.])  # in [m] [s]^{-2}
        self.nb_steps = 10
        self.path = None
        self.playback_time = None
        self.start_contact = start_contact
        self.traj_len = None
        self.traj_points = None

    def reset(self, start_contact, end_contact, foot_target):
        """
        Reset for a new pair of contacts.

        Parameters
        ----------
        start_contact : pymanoid.Contact
            Initial contact.
        end_contact : pymanoid.Contact
            Target contact.
        foot_target : pymanoid.Body
            IK target to be updated during swing foot motion.
        """
        self.start_contact = start_contact
        self.end_contact = end_contact
        self.foot_target = foot_target
        #
        self.interpolate()

    def interpolate_hermite(self):
        """
        Interpolate optimized Hermite curve between the two contacts.
        """
        n0 = self.start_contact.n
        n1 = self.end_contact.n
        p0 = self.start_contact.p
        p1 = self.end_contact.p
        start_clearance, end_clearance = self.default_clearances
        if hasattr(self.end_contact, 'landing_clearance'):
            end_clearance = self.end_contact.landing_clearance
        if hasattr(self.end_contact, 'landing_tangent'):
            n1 = self.end_contact.landing_tangent
        if hasattr(self.start_contact, 'takeoff_clearance'):
            start_clearance = self.start_contact.takeoff_clearance
        if hasattr(self.start_contact, 'takeoff_tangent'):
            n0 = self.start_contact.takeoff_tangent
        # H(s) = H_lambda(s) * lambda + H_mu(s) * mu + H_cst(s)
        H_lambda, H_mu, H_cst = factor_cubic_hermite_curve(p0, n0, p1, n1)
        s0 = 1. / 4
        a0 = dot(H_lambda(s0), n0)
        b0 = dot(H_mu(s0), n0)
        c0 = dot(H_cst(s0) - p0, n0)
        h0 = start_clearance
        # a0 * lambda + b0 * mu + c0 >= h0
        s1 = 3. / 4
        a1 = dot(H_lambda(s1), n1)
        b1 = dot(H_mu(s1), n1)
        c1 = dot(H_cst(s1) - p1, n1)
        h1 = end_clearance
        # a1 * lambda + b1 * mu + c1 >= h1
        P = eye(2)
        q = zeros(2)
        G = array([[-a0, -b0], [-a1, -b1]])
        h = array([c0 - h0, c1 - h1])
        x = solve_qp(P, q, G, h)
        # H = lambda s: H_lambda(s) * x[0] + H_mu(s) * x[1] + H_cst(s)
        self._H = lambda s: H_lambda(s) * x[0] + H_mu(s) * x[1] + H_cst(s)
        poly = interpolate_cubic_hermite(p0, x[0] * n0, p1, x[1] * n1)
        self.path = toppra.PolynomialInterpolator(array(poly.coeffs).T)

    def retime(self):
        """
        Retime interpolated Hermite curve using TOPP-RA.
        """
        ss = linspace(0, 1, self.nb_steps + 1)
        a_lim = vstack((-self.max_accel, self.max_accel)).T
        pc_acc = toppra.create_acceleration_path_constraint(
            self.path, ss, a_lim)
        constraints = [pc_acc]
        solver = toppra.qpOASESPPSolver(constraints)
        us, xs = solver.solve_topp()
        # self.solver = solver
        # self.retimed = toppra.compute_trajectory_gridpoints(self.path,
        # solver.ss, us, xs)
        t, q, _, _ = toppra.compute_trajectory_points(
            self.path, solver.ss, us, xs, dt=self.dt)
        return t, q

    def interpolate(self):
        """
        Interpolate complete swing foot trajectory.
        """
        self.duration = None
        self.traj_len = None
        self.traj_points = None
        self.interpolate_hermite()
        t, q = self.retime()
        self.playback_time = 0.
        self.duration = t[-1]
        self.traj_len = len(q)
        self.traj_points = q
        self.draw()

    @property
    def progression(self):
        """
        Progression index between zero and one.
        """
        return min(1., 1.1 * self.playback_time / self.duration)

    @property
    def time_to_heel_strike(self):
        """
        Time until the swing foot touches down the target contact.
        """
        if self.duration is None:
            return None
        return max(0., self.duration - self.playback_time)

    def draw(self, details=False):
        """
        Draw swing foot trajectory.
        """
        new_handles = []
        if details:  # draw control points
            H0, H1 = self._H(1. / 4), self._H(3. / 4)
            p0, n0 = self.start_contact.p, self.start_contact.n
            p1, n1 = self.end_contact.p, self.end_contact.n
            H0_proj = H0 - dot(H0 - p0, n0) * n0
            H1_proj = H1 - dot(H1 - p1, n1) * n1
            new_handles.append(draw_point(H0, pointsize=0.015))
            new_handles.append(draw_point(H1, pointsize=0.015))
            new_handles.append(draw_point(H0_proj, pointsize=0.007))
            new_handles.append(draw_point(H1_proj, pointsize=0.007))
            new_handles.append(draw_line(H0, H0_proj))
            new_handles.append(draw_line(H1, H1_proj))
        path = self.path.eval(linspace(0., 1., 20))
        new_handles.extend(draw_trajectory(path, color='g', pointsize=0))
        self.handles += new_handles

    def on_tick(self, sim):
        """
        Main function called at each control cycle.

        Parameters
        ----------
        sim : Simulation
            Current simulation instance.
        """
        self.playback_time += sim.dt
        i = min(int(self.playback_time / self.dt) + 1, self.traj_len - 1)
        s = min(1., 1.1 * self.playback_time / self.duration)
        quat = quat_slerp(self.start_contact.quat, self.end_contact.quat, s)
        self.foot_target.set_pos(self.traj_points[i])
        self.foot_target.set_quat(quat)
