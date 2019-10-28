#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017-2018 Stephane Caron <stephane.caron@lirmm.fr>
#
# This file is part of capture-walkgen
# <https://github.com/stephane-caron/capture-walkgen>.
#
# capture-walkgen is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# capture-walkgen is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# capture-walkgen. If not, see <http://www.gnu.org/licenses/>.

import pymanoid

from numpy import array, cross, dot, hstack, linspace, sqrt
from pymanoid.gui import draw_trajectory
from pymanoid.models import InvertedPendulum
from pymanoid.sim import e_z

from .capture_problem import CaptureProblem
from .interval import find_interval_bounds


class OneStepController(object):

    """
    Stepping controller based on predictive control with boundedness condition.

    Parameters
    ----------
    pendulum : pymanoid.InvertedPendulum
        State estimator of the inverted pendulum.
    nb_steps : integer
        Number of discretization steps for the preview trajectory.
    target_height : scalar
        Desired altitude at the end of the step.
    """

    def __init__(self, pendulum, nb_steps, target_height):
        self.__sqrt_lambda_max = sqrt(pendulum.lambda_max)
        self.__sqrt_lambda_min = sqrt(pendulum.lambda_min)
        self.capture_pb = CaptureProblem(
            pendulum.lambda_min, pendulum.lambda_max, nb_steps)
        self.com = pendulum.com.copy(visible=False)
        self.pendulum = pendulum
        self.solution = None
        self.support_contact = None
        self.target_contact = None
        self.target_height = target_height

    def set_contacts(self, support_contact, target_contact):
        """
        Update support and target contacts.

        Parameters
        ----------
        support_contact : pymanoid.Contact
            Contact used during the takeoff phase.
        target_contact : pymanoid.Contact
            Contact used during the landing phase.
        """
        v_x = cross(support_contact.b, e_z)
        v_y = cross(support_contact.t, e_z)
        X_n = support_contact.shape[0] * support_contact.n[2]
        Y_n = support_contact.shape[1] * support_contact.n[2]
        # Contact area constraints: F * r_{xy} <= p  (see paper for details)
        F_area = array([+v_x, -v_x, +v_y, -v_y])  # note that F[:, 2] is zero
        p_area = array([X_n, X_n, Y_n, Y_n]) + dot(F_area, support_contact.p)
        v_ineq = p_area - dot(F_area, target_contact.p)
        self.F_area = F_area
        self.p_area = p_area
        self.solution = None
        self.support_contact = support_contact
        self.target_contact = target_contact
        self.v_ineq = v_ineq

    def compute_alpha_intervals(self, alpha_min=0.1):
        """
        Compute intervals of feasible alpha values for current state.

        Parameters
        ----------
        alpha_min : scalar, optional
            Restrict solutions to alpha's greater than this threshold.

        Returns
        -------
        intervals : list of pairs
            List of disjoint intervals `(alpha_min, alpha_max)`.
        """
        # (u - alpha * v) * omega_i >= w
        u = self.p_area - dot(self.F_area, self.com.p)
        v = self.v_ineq
        w = dot(self.F_area, self.com.pd)

        # extend (u, v, w) with sqrt(lambda_min) < omega_i < sqrt(lambda_max)
        u = hstack([u, [+1., -1.]])
        v = hstack([v, [0., 0.]])
        w = hstack([w, [+self.__sqrt_lambda_min, -self.__sqrt_lambda_max]])

        row_ids = set(range(len(u)))
        roots = [u[i] / v[i] for i in row_ids if abs(u[i]) < abs(v[i])]
        roots.sort()
        roots = [alpha_min] + [r for r in roots if alpha_min < r < 1.] + [1.]

        def refine_alpha_interval(alpha_min, alpha_max):
            alpha_sample = .5 * (alpha_min + alpha_max)
            lb = set(i for i in row_ids if u[i] - alpha_sample * v[i] >= 0)
            ub = row_ids - lb
            v_w = [v[i] * w[j] - v[j] * w[i] for i in lb for j in ub]
            u_w = [u[i] * w[j] - u[j] * w[i] for i in lb for j in ub]
            # v_w * alpha >= u_w
            return find_interval_bounds(v_w, u_w, alpha_min, alpha_max)

        alpha_intervals = []
        for root_id in range(len(roots) - 2, -1, -1):
            alpha_min = roots[root_id]
            alpha_max = roots[root_id + 1]
            alpha_min, alpha_max = refine_alpha_interval(alpha_min, alpha_max)
            if alpha_min is not None:
                alpha_min += 1e-2
                alpha_max -= 1e-2
                if alpha_min < alpha_max:
                    alpha_intervals.append((alpha_min, alpha_max))
        return alpha_intervals

    def update_capture_problem(self, alpha):
        """
        Prepare the CaptureProblem for the current state.

        Parameters
        ----------
        alpha : scalar
            Contact switch indicator.
        """
        # u * omega_i >= v
        u = (1. - alpha) * self.p_area + dot(
            self.F_area, alpha * self.target_contact.p - self.com.p)
        v = dot(self.F_area, self.com.pd)
        init_omega_min, init_omega_max = find_interval_bounds(
            u, v, self.__sqrt_lambda_min, self.__sqrt_lambda_max)
        if init_omega_min is None:
            raise RuntimeError("no feasible CoP")
        r_alpha = alpha * self.target_contact.p \
            + (1. - alpha) * self.support_contact.p
        zbar = dot(self.com.p - r_alpha, self.support_contact.n)
        zbar /= self.support_contact.n[2]
        init_zbar_deriv = dot(self.com.pd, self.support_contact.n)
        init_zbar_deriv /= self.support_contact.n[2]
        self.capture_pb.init_zbar = zbar
        self.capture_pb.init_zbar_deriv = init_zbar_deriv
        self.capture_pb.set_init_omega_lim(init_omega_min, init_omega_max)
        self.capture_pb.target_height = self.target_height

    def pick_solution(self, alpha_intervals, time_to_heel_strike=None):
        best_alpha = None
        best_cost = 1e5
        best_solution = None
        for (alpha_min, alpha_max) in alpha_intervals:
            for alpha in linspace(alpha_max, alpha_min, 5):
                try:
                    self.update_capture_problem(alpha)
                except RuntimeError as e:
                    pymanoid.error("RuntimeError: %s" % str(e))
                    continue
                solution = self.capture_pb.solve()
                if solution is None:
                    continue
                solution.compute_lambda()
                time_cost = 0.
                if time_to_heel_strike is not None:
                    solution.compute_switch_times()
                    phi_switch = (alpha * solution.omega_i)**2
                    s_switch = solution.s_from_phi(phi_switch)
                    t_switch = solution.t_from_s(s_switch)
                    time_cost = max(0., time_to_heel_strike - t_switch)
                total_cost = 1000 * time_cost + solution.var_cost
                if total_cost < best_cost:
                    best_alpha = alpha
                    best_cost = total_cost
                    best_solution = solution
        self.update_capture_problem(best_alpha)
        self.alpha = best_alpha
        self.solution = best_solution

    def compute_controls(self, time_to_heel_strike=None):
        """
        Compute pendulum controls for the current state.

        Parameters
        ----------
        time_to_heel_strike : scalar
            When set, make sure that the contact switch happens after this
            time.

        Returns
        -------
        cop : (3,) array
            CoP coordinates in the world frame.
        push : scalar
            Leg stiffness :math:`\\lambda \\geq 0`.
        """
        self.solution = None
        self.com.set_pos(self.pendulum.com.p)
        self.com.set_vel(self.pendulum.com.pd)
        alpha_intervals = self.compute_alpha_intervals()
        self.pick_solution(alpha_intervals, time_to_heel_strike)
        alpha, omega_i = self.alpha, self.solution.omega_i
        pos_proj = self.com.p - e_z * self.capture_pb.init_zbar
        vel_proj = self.com.pd - e_z * self.capture_pb.init_zbar_deriv
        r_f = self.target_contact.p
        r_i = (pos_proj + vel_proj / omega_i - alpha * r_f) / (1. - alpha)
        self.solution.r_f = r_f
        self.solution.r_i = r_i
        return r_i, self.solution.lambda_i

    def draw_solution(self, color='m'):
        """
        Draw the solution trajectory.

        Returns
        -------
        handles : list of openravepy.GraphHandle
            OpenRAVE graphical handles. Must be stored in some variable,
            otherwise the drawn object will vanish instantly.
        """
        if self.capture_pb is None or self.solution is None:
            pymanoid.warn("no solution to draw")
            return []
        if self.solution.lambda_ is None:
            self.solution.compute_lambda()
        if self.solution.switch_times is None:
            self.solution.compute_switch_times()
        omega_i = self.solution.omega_i
        phi_switch = (self.alpha * omega_i)**2
        s_switch = self.solution.s_from_phi(phi_switch)
        t_switch = self.solution.t_from_s(s_switch)
        pendulum = InvertedPendulum(
            self.com.p, self.com.pd, self.support_contact,
            visible=False)
        max_time = self.solution.switch_times[-1] * 1.1
        points = []
        pendulum.set_contact(self.support_contact)
        pendulum.set_cop(self.solution.r_i)
        for j in range(self.capture_pb.nb_steps):
            t_j = self.solution.switch_times[j]
            lambda_j = self.solution.lambda_[self.capture_pb.nb_steps - j - 1]
            t_next = max_time
            if j < self.capture_pb.nb_steps - 1:
                t_next = self.solution.switch_times[j + 1]
            pendulum.set_lambda(lambda_j)
            if t_j <= t_switch < t_next:
                pendulum.integrate(t_switch - t_j)
                points.append(pendulum.com.p)
                pendulum.set_contact(self.target_contact)
                pendulum.set_cop(self.target_contact.p)
                pendulum.integrate(t_next - t_switch)
            else:  # no contact switch
                pendulum.integrate(t_next - t_j)
            points.append(pendulum.com.p)
        return draw_trajectory(points, color=color)
