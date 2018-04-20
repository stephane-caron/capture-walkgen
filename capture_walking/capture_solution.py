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

from bisect import bisect_right
from numpy import array, cosh, log, sinh, sqrt, tanh


class CaptureSolution(object):

    """
    Solution to a capture optimization problem.

    Parameters
    ----------
    phi_1_n : array
        Vector of optimization variables returned by call to solver.
    capture_pb : CaptureProblem
        Original capture problem.
    optimal_found : bool
        Did the solver converge to this solution?
    """

    def __init__(self, phi_1_n, capture_pb, optimal_found=None):
        phi = array([0.] + list(phi_1_n))
        delta = capture_pb.delta
        n = capture_pb.nb_steps
        omega_i = sqrt(phi[n])
        self.delta = delta
        self.lambda_ = None
        self.lambda_i = (phi[n] - phi[n - 1]) / delta[n - 1]
        self.nb_steps = n
        self.omega_i = omega_i
        self.optimal_found = optimal_found
        self.phi = phi
        self.s = capture_pb.s
        self.switch_times = None

    def __str__(self):
        return "phi = %s;" % str(list(self.phi))

    @property
    def var_cost(self):
        """
        Compute a squared-error cost in stiffness variations.
        """
        return sum(
            (self.lambda_[i + 1] - self.lambda_[i])**2
            for i in xrange(self.nb_steps))

    def compute_lambda(self):
        """
        Compute the full vector of stiffness values.
        """
        delta, phi, n = self.delta, self.phi, self.nb_steps
        lambda_ = [(phi[i + 1] - phi[i]) / delta[i] for i in xrange(n)]
        self.lambda_ = array(lambda_ + [lambda_[-1]])

    def compute_switch_times(self):
        """
        Compute the times :math:`t_j` where :math:`s(t_j) = s_j`.
        """
        switch_times = [0.]
        switch_time = 0.
        for j in xrange(self.nb_steps - 1, 0, -1):
            sqrt_lambda_j = sqrt(self.lambda_[j])
            num = sqrt(self.phi[j + 1]) + sqrt_lambda_j * self.s[j + 1]
            denom = sqrt(self.phi[j]) + sqrt_lambda_j * self.s[j]
            duration = log(num / denom) / sqrt_lambda_j
            switch_time += duration
            switch_times.append(switch_time)
        self.switch_times = switch_times

    def find_switch_time_before(self, t):
        """
        Find a switch time :math:`t_j` such that :math:`t_j \leq t < t_{j+1}`.

        Parameters
        ----------
        t : scalar
            Time in [s]. Must be positive.

        Returns
        -------
        j : integer
            Switch-time index between 0 and n - 1.
        t_j : scalar
            Switch time in [s].
        """
        j = bisect_right(self.switch_times, t) - 1 if t > 0 else 0
        assert self.switch_times[j] <= t
        assert j == len(self.switch_times) - 1 or t < self.switch_times[j + 1]
        return j, self.switch_times[j]

    def lambda_from_s(self, s):
        """
        Compute the leg stiffness :math:`\\lambda(s)` for a given path index.

        Parameters
        ----------
        s : scalar
            Path index between 0 and 1.

        Returns
        -------
        lambda_ : scalar
            Leg stiffness :math:`\\lambda(s)`.

        """
        j = bisect_right(self.s, s) - 1 if s > 0 else 0
        assert self.s[j] <= s and (j == self.nb_steps or s < self.s[j + 1])
        return self.lambda_[j]

    def lambda_from_t(self, t):
        """
        Compute the leg stiffness :math:`\\lambda(t)` to apply at time `t`.

        Parameters
        ----------
        t : scalar
            Time in [s]. Must be positive.

        Returns
        -------
        lambda_ : scalar
            Leg stiffness :math:`\\lambda(t)`.

        """
        return self.lambda_from_s(self.s_from_t(t))

    def omega_from_s(self, s):
        """
        Compute :math:`\\omega(s)` for a given path index.

        Parameters
        ----------
        s : scalar
            Path index between 0 and 1.

        Returns
        -------
        omega : scalar
            Value of :math:`\\omega(s)`.
        """
        if s < 1e-3:
            return sqrt(self.lambda_[0])
        j = bisect_right(self.s, s) - 1 if s > 0 else 0
        assert self.s[j] <= s and (j == self.nb_steps or s < self.s[j + 1])
        # integral from 0 to s of f(u) = 2 * u * lambda(u)
        f_integral = self.phi[j] + self.lambda_[j] * (s ** 2 - self.s[j] ** 2)
        return sqrt(f_integral) / s

    def omega_from_t(self, t):
        """
        Compute the value of :math:`\\omega(t)`.

        Parameters
        ----------
        t : scalar
            Time in [s]. Must be positive.

        Returns
        -------
        omega : scalar
            Value of :math:`\\omega(t)`.
        """
        j, t_j = self.find_switch_time_before(t)
        omega_ = sqrt(self.phi[self.nb_steps - j]) / self.s[self.nb_steps - j]
        lambda_ = self.lambda_[self.nb_steps - j - 1]
        sqrt_lambda = sqrt(lambda_)
        x = sqrt_lambda * (t - t_j)
        z = sqrt_lambda / omega_
        return sqrt_lambda * (1. - z * tanh(x)) / (z - tanh(x))

    def s_from_phi(self, phi):
        """
        Invert the function :math:`s \\mapsto \\varphi(s)`.

        Parameters
        ----------
        phi : scalar
            Value of the function :math:`\\varphi(s) = s \\omega(s)`.

        Returns
        -------
        s : scalar
            Index `s` such that `phi(s) = phi`.

        Notes
        -----
        Given the index `j` such that :math:`\\varphi_j \\leq \\varphi <
        \\varphi_{j+1}`, the important formula behind this function is:

        .. math::

            \\varphi(s) = \\sqrt{\\varphi_j + \\lambda_j (s^2 - s_j^2)}

        See the paper for derivation details.
        """
        assert 0. < phi < self.phi[-1], "no solution"
        j = bisect_right(self.phi, phi) - 1 if phi > 0 else 0
        assert self.phi[j] <= phi
        assert j == self.nb_steps or phi < self.phi[j + 1]
        return sqrt(self.s[j]**2 + (phi - self.phi[j]) / self.lambda_[j])

    def s_from_t(self, t):
        """
        Compute the path index corresponding to a given time.

        Parameters
        ----------
        t : scalar
            Time in [s]. Must be positive.

        Returns
        -------
        s : scalar
            Path index `s(t)`.
        """
        j, t_j = self.find_switch_time_before(t)
        s_start = self.s[self.nb_steps - j]
        omega_ = sqrt(self.phi[self.nb_steps - j]) / self.s[self.nb_steps - j]
        lambda_ = self.lambda_[self.nb_steps - j - 1]
        sqrt_lambda = sqrt(lambda_)
        x = sqrt_lambda * (t - t_j)
        return s_start * (cosh(x) - omega_ / sqrt_lambda * sinh(x))

    def t_from_s(self, s):
        """
        Compute the time corresponding to a given path index.

        Parameters
        ----------
        s : scalar
            Path index `s` between 0 and 1.

        Returns
        -------
        t : scalar
            Time `t(s) > 0` in [s].

        Notes
        -----
        Given the index `j` such that :math:`s_j \\leq s < s_{j+1}`, the
        important formula behind this function is:

        .. math::

            t(s) = t_{j+1} + \\frac{1}{\\sqrt{\\lambda_j}} \\log\\left(
                \\frac{
                    \\sqrt{\\varphi_{i+1}} + \\sqrt{\\lambda_j} s_{j+1}}{
                    \\sqrt{\\varphi_{i+1} - \\lambda_j (s_{j+1}^2 - s^2)}
                + \\sqrt{\\lambda_j} s} \\right)

        See the paper for a derivation of this formula.
        """
        assert 0. < s < 1.
        j = bisect_right(self.s, s) - 1 if s > 0 else 0
        assert self.s[j] <= s and j < self.nb_steps and s < self.s[j + 1]
        s_next = self.s[j + 1]
        t_next = self.switch_times[self.nb_steps - (j + 1)]
        sqrt_lambda_j = sqrt(self.lambda_[j])
        num = sqrt(self.phi[j + 1]) + s_next * sqrt_lambda_j
        denom = sqrt(self.phi[j + 1] - self.lambda_[j] * (s_next**2 - s**2))
        denom += s * sqrt_lambda_j
        return t_next + log(num / denom) / sqrt(self.lambda_[j])
