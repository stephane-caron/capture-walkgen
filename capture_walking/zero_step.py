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

from numpy import arange, arctan, array, cos, cross, dot, sin, tan, vstack
from pymanoid.body import Point
from pymanoid.gui import draw_point, draw_trajectory
from pymanoid.misc import normalize, warn
from pymanoid.models import InvertedPendulum
from time import time

from .capture import CaptureController


class MockFrame(object):

    def __init__(self):
        self.R = None
        self.b = None
        self.phi = None
        self.t = None
        self.theta = None

    def update(self, t, b, n, phi, theta):
        self.R = vstack([t, b, n]).T
        self.b = b
        self.t = t
        self.phi = phi
        self.theta = theta


class ZeroStepController(CaptureController):

    """
    Balance controller based on predictive control with boundedness condition.

    Parameters
    ----------
    pendulum : pymanoid.InvertedPendulum
        State estimator of the inverted pendulum.
    nb_steps : integer
        Number of discretization steps for the preview trajectory.
    target_height : scalar
        Desired altitude in the stationary regime.
    cop_gain : scalar
        CoP feedback gain (must be > 1).

    Notes
    -----
    This implementation works in a local frame, as reported in the ICRA 2018
    paper <https://scaron.info/research/icra-2018.html>. Computations are a bit
    simpler in the world frame as done in the `OneStepController`.
    """

    def __init__(self, pendulum, nb_steps, target_height, cop_gain):
        assert cop_gain > 1.05, "CoP gain has to be strictly > 1"
        super(ZeroStepController, self).__init__(
            pendulum, nb_steps, target_height)
        self.comp_time = None
        self.contact = pendulum.contact
        self.cop_gain = cop_gain
        self.frame = MockFrame()  # local frame
        self.local_state = Point([0., 0., 0.], visible=False)  # in local frame
        self.world_state = pendulum.com.copy(visible=False)

    def set_contact(self, contact):
        """
        Update the supporting contact.

        Parameters
        ----------
        contact : pymanoid.Contact
            New contact to use for stabilization.
        """
        self.contact = contact
        self.solution = None

    def update_state(self):
        """
        Update state information and surface frame.
        """
        self.world_state.set_pos(self.pendulum.com.p)
        self.world_state.set_vel(self.pendulum.com.pd)
        delta = self.world_state.p - self.contact.p
        e_z = array([0., 0., 1.])
        e_x = -normalize(delta - dot(delta, e_z) * e_z)
        e_y = cross(e_z, e_x)
        R = vstack([e_x, e_y, e_z])  # from world to local frame
        p = dot(R, delta)  # in local frame
        pd = dot(R, self.world_state.pd)  # in local frame
        dot_ez_n = dot(e_z, self.contact.n)
        phi = arctan(-dot(e_x, self.contact.n) / dot_ez_n)
        theta = arctan(-dot(e_y, self.contact.n) / dot_ez_n)
        t = cos(phi) * e_x + sin(phi) * e_z
        b = cos(theta) * e_y + sin(theta) * e_z
        self.local_state.set_pos(p)
        self.local_state.set_vel(pd)
        self.frame.update(t, b, self.contact.n, phi, theta)

    def update_capture_pb(self):
        """
        Prepare the CaptureProblem for the current state.
        """
        W, H = self.contact.shape
        b_h = dot(self.frame.b, self.contact.b) / cos(self.frame.theta)
        b_w = dot(self.frame.b, self.contact.t) / cos(self.frame.theta)
        t_h = dot(self.frame.t, self.contact.b) / cos(self.frame.phi)
        t_w = dot(self.frame.t, self.contact.t) / cos(self.frame.phi)
        A = array([
            [+t_w, +b_w],
            [-t_w, -b_w],
            [+t_h, +b_h],
            [-t_h, -b_h]])
        b = array([W, W, H, H])
        u = b / self.cop_gain - dot(A, self.local_state.p[:2])
        v = dot(A, self.local_state.pd[:2])
        # Property: u * omega_i >= v
        init_omega_min, init_omega_max = self.wrap_omega_lim(u, v)
        x, y, z = self.local_state.p
        xd, yd, zd = self.local_state.pd
        zbar = z - tan(self.frame.phi) * x - tan(self.frame.theta) * y
        zbar_deriv = zd - tan(self.frame.phi) * xd - tan(self.frame.theta) * yd
        self.capture_pb.init_zbar = zbar
        self.capture_pb.init_zbar_deriv = zbar_deriv
        self.capture_pb.set_init_omega_lim(init_omega_min, init_omega_max)
        self.capture_pb.target_height = self.target_height

    def compute_controls(self):
        """
        Compute pendulum controls for the current simulation step.

        Returns
        -------
        cop : (3,) array
            COP coordinates in the world frame.
        push : scalar
            Leg push `lambda >= 0`.
        """
        self.comp_time = None
        self.solution = None
        start_time = time()
        self.update_state()
        self.update_capture_pb()
        self.solution = self.capture_pb.solve()
        self.comp_time = time() - start_time
        com, comd = self.local_state.p, self.local_state.pd
        r_i = self.cop_gain * (com + comd / self.solution.omega_i)
        alpha_i = r_i[0] / cos(self.frame.phi)
        beta_i = r_i[1] / cos(self.frame.theta)
        cop = self.contact.p + dot(self.frame.R, [alpha_i, beta_i, 0.])
        return cop, self.solution.lambda_i

    def draw_solution(self, color='m'):
        """
        Draw the center-of-mass trajectory.

        Returns
        -------
        handles : list of openravepy.GraphHandle
            OpenRAVE graphical handles. Must be stored in some variable,
            otherwise the drawn object will vanish instantly.
        """
        if self.capture_pb is None or self.solution is None:
            warn("no solution to draw")
            return []
        if self.solution.lambda_ is None:
            self.solution.compute_lambda()
        if self.solution.switch_times is None:
            self.solution.compute_switch_times()
        pendulum = InvertedPendulum(
            self.world_state.p, self.world_state.pd,
            self.contact, visible=False)
        max_time = self.solution.switch_times[-1] * 1.1
        omega_i = self.solution.omega_i
        k = self.cop_gain
        r_i = k * (self.local_state.p + self.local_state.pd / omega_i)
        alpha_i = r_i[0] / cos(self.frame.phi)
        beta_i = r_i[1] / cos(self.frame.theta)
        points = []
        for i in xrange(self.capture_pb.nb_steps):
            t_i = self.solution.switch_times[i]
            t_next = max_time
            if i < self.capture_pb.nb_steps - 1:
                t_next = self.solution.switch_times[i + 1]
            pendulum.set_lambda(
                self.solution.lambda_[self.capture_pb.nb_steps - i - 1])
            dt = (t_next - t_i) / 10.
            for t in arange(t_i, t_next, dt):
                s = self.solution.s_from_t(t)
                omega_t = self.solution.omega_from_t(t)
                alpha = alpha_i * (s * omega_t / omega_i) ** (k - 1)
                beta = beta_i * (s * omega_t / omega_i) ** (k - 1)
                cop = self.contact.p + dot(self.frame.R, [alpha, beta, 0.])
                pendulum.set_cop(cop)
                pendulum.integrate(dt)
            points.append(pendulum.com.p)
        target = self.contact.p + array([0., 0., self.target_height])
        handles = draw_trajectory(points, color=color)
        handles.append(draw_point(target, color='b', pointsize=0.01))
        return handles
