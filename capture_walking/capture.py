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

from bisect import bisect_right
from casadi import sqrt as casadi_sqrt
from numpy import array, cosh, log, sinh, sqrt, tanh
from pymanoid.misc import AvgStdEstimator
from pymanoid.optim import NonlinearProgram
from pymanoid.sim import gravity_const

from .utils import find_interval_bounds

DEFAULT_NLP_SOLVER = "ipopt"

try:
    import CaptureProblemSolver as cps
    DEFAULT_NLP_SOLVER = "cps"
except ImportError:
    pymanoid.warn("Could not import cps solver")
    cps = None


class CaptureProblem(object):

    """
    Optimization problem over a time-varying pendulum frequency.

    Parameters
    ----------
    lambda_min : scalar
        Minimum leg stiffness (positive).
    lambda_max : scalar
        Maximum leg stiffness (positive).
    nb_steps : integer
        Number of segments where :math:`\\lambda(t)` is constant.
    """

    def __init__(self, lambda_min, lambda_max, nb_steps):
        s_list = [j * 1. / nb_steps for j in xrange(nb_steps)] + [1.]
        s_sq = [s ** 2 for s in s_list]
        delta = [s_sq[j + 1] - s_sq[j] for j in xrange(nb_steps)]
        cps_pb = None
        lssol_available = False
        if cps is not None:
            raw_pb = cps.RawProblem()
            raw_pb.delta = delta
            raw_pb.lambda_max = lambda_max
            raw_pb.lambda_min = lambda_min
            cps_pb = cps.Problem(raw_pb)
            try:
                sqp = cps.SQP(cps_pb.size)
                sqp.useOptimizedLSSolver(True)
                lssol_available = True
            except:
                pass
        self.__cps_ls_solver = "optimized"
        self.__nlp_solver = DEFAULT_NLP_SOLVER
        self.__lssol_available = lssol_available
        self.cps_nb_nls_iter = AvgStdEstimator()
        self.cps_nb_nls_iter_frac = AvgStdEstimator()
        self.cps_nb_sqp_iter = AvgStdEstimator()
        self.cps_nb_sqp_solved = 0
        self.cps_pb = cps_pb
        self.cps_status_count = {}
        self.cps_total_nls_iter = 0
        self.cps_total_sqp_iter = 0
        self.delta = delta
        self.init_omega_max = None
        self.init_omega_min = None
        self.init_zbar = None
        self.init_zbar_deriv = None
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.nb_steps = nb_steps
        self.record_solver_times = False
        self.s = s_list
        self.s_sq = s_sq
        self.solver_times = []
        self.solver_times_succ = []
        self.target_height = None

    def cps_precompute(self):
        """
        Precompute QR decompositions for all nullspace-projected Jacobians used
        by the least squares sub-solver of cps.
        """
        assert self.__cps_ls_solver == "optimized"
        assert self.__nlp_solver == "cps"
        self.cps_pb.precompute()

    def set_cps_ls_solver(self, solver_name):
        """
        Change the least squares solver used in cps.

        Parameters
        ----------
        solver_name : string
            Solver name between "optimized" (taylored optimization) and "lssol"
            (generic solver).
        """
        assert solver_name in ["optimized", "lssol"]
        assert solver_name != "lssol" or self.__lssol_available
        self.__cps_ls_solver = solver_name

    def set_nlp_solver(self, solver_name):
        """
        Change the nonlinear solver.

        Paramters
        ---------
        solver_name : string
            Solver name between "cps" (taylored SQP optimization) and "ipopt"
            (generic nonlinear programming).
        """
        assert solver_name in ["cps", "ipopt"]
        self.__nlp_solver = solver_name

    def set_init_omega_lim(self, init_omega_min, init_omega_max):
        if init_omega_max > sqrt(self.lambda_max):
            init_omega_max = sqrt(self.lambda_max)
            pymanoid.warn("init_omega_max capped to sqrt(lambda_max)")
        if init_omega_min < sqrt(self.lambda_min):
            pymanoid.warn("init_omega_min capped to sqrt(lambda_min)")
            init_omega_min = sqrt(self.lambda_min)
        self.init_omega_max = init_omega_max
        self.init_omega_min = init_omega_min

    def __str__(self):
        s = "delta = %s;\n" % str(list(self.delta))
        s += "g = %.32f;\n" % gravity_const
        s += "init_omega_max = %.32f;\n" % self.init_omega_max
        s += "init_omega_min = %.32f;\n" % self.init_omega_min
        s += "init_zbar = %.32f;\n" % self.init_zbar
        s += "init_zbar_deriv = %.32f;\n" % self.init_zbar_deriv
        s += "lambda_max = %.32f;\n" % self.lambda_max
        s += "lambda_min = %.32f;\n" % self.lambda_min
        s += "s = %s;\n" % str(list(self.s))
        s += "target_height = %.32f;" % self.target_height
        return s

    def solve_with_cps(self):
        self.cps_pb.set_init_zbar(self.init_zbar)
        self.cps_pb.set_init_zbar_deriv(self.init_zbar_deriv)
        self.cps_pb.set_init_omega_max(self.init_omega_max)
        self.cps_pb.set_init_omega_min(self.init_omega_min)
        self.cps_pb.set_target_height(self.target_height)
        sqp = cps.SQP(self.cps_pb.size)
        if self.__lssol_available:
            sqp.useOptimizedLSSolver(self.__cps_ls_solver == "optimized")
        if self.record_solver_times:
            status = sqp.timedSolve(self.cps_pb)
            self.solver_times.append(sqp.solveTime() * 1e-6)
        else:
            status = sqp.solve(self.cps_pb)
        if __debug__:
            if self.__lssol_available:
                self.log_cps_stats(sqp, status)
            if status == cps.SolverStatus.LineSearchFailed:
                pymanoid.warn("cps line search failed")
            elif status == cps.SolverStatus.MaxIteration:
                pymanoid.warn("cps reached maximum number of iterations")
            elif status == cps.SolverStatus.NoLinearlyFeasiblePoint:
                pymanoid.warn("cps problem not linearly feasible")
            elif status == cps.SolverStatus.NumericallyEquivalentIterates:
                pymanoid.warn("cps returned with num. equivalent iterates")
        if self.record_solver_times:
            self.solver_times_succ.append(sqp.solveTime() * 1e-6)
        if status == cps.SolverStatus.Fail and self.__cps_ls_solver == "lssol":
            sqp.useOptimizedLSSolver(True)
            pymanoid.error("LSSOL failed, re-solving with optimized solver")
            status = sqp.solve(self.cps_pb)
            sqp.useOptimizedLSSolver(False)
        if status == cps.SolverStatus.Fail:
            msg = "cps solver failed on the following problem"
            raise RuntimeError("{}:\n\n{}".format(msg, self))
        elif status == cps.SolverStatus.NoLinearlyFeasiblePoint:
            return None
        return CaptureSolution(sqp.x(), capture_pb=self, optimal_found=True)

    def log_cps_stats(self, sqp, status):
        if status not in self.cps_status_count:
            self.cps_status_count[status] = 0
        self.cps_status_count[status] += 1
        nls_iter = sqp.numberOfNonLineSearchIterations()
        nb_iter = sqp.numberOfIterations()
        self.cps_nb_sqp_iter.add(nb_iter)
        self.cps_nb_sqp_solved += 1
        self.cps_total_nls_iter += nls_iter
        self.cps_total_sqp_iter += nb_iter
        self.cps_nb_nls_iter.add(sqp.numberOfNonLineSearchIterations())
        if nb_iter > 0:
            self.cps_nb_nls_iter_frac.add(float(nls_iter) / nb_iter)

    def print_cps_stats(self):
        print "Number of SQP solved:", self.cps_nb_sqp_solved
        print "Total number of SQP iterations:", self.cps_total_sqp_iter
        print "Total number of NonLineSearch SQP iterations:", \
            self.cps_total_nls_iter
        print "Average number of iterations per SQP:", self.cps_nb_sqp_iter
        print "Average number of NLS iterations per SQP:", self.cps_nb_nls_iter
        print "Average (# NLS iter) / (# iter) per SQP:", \
            self.cps_nb_nls_iter_frac

    def solve_with_ipopt(self):
        nlp = NonlinearProgram()
        g = gravity_const
        bc_integral_expr = 0.
        phi_i = 0.  # phi_0 = 0
        phi_1 = None
        lambda_cost = 0.
        lambda_f = g / self.target_height
        lambda_guess = lambda_f
        lambda_prev = lambda_f
        for j in xrange(self.nb_steps):
            phi_lb = self.s_sq[j + 1] * self.lambda_min
            phi_ub = self.s_sq[j + 1] * self.lambda_max
            if j == self.nb_steps - 1:
                if self.init_omega_min is not None:
                    phi_lb = max(phi_lb, self.init_omega_min ** 2)
                if self.init_omega_max is not None:
                    phi_ub = min(phi_ub, self.init_omega_max ** 2)
            phi_next = nlp.new_variable(
                'phi_%d' % (j + 1),  # from phi_1 to phi_n
                dim=1,
                init=[self.s_sq[j + 1] * lambda_guess],
                lb=[phi_lb],
                ub=[phi_ub])
            if phi_1 is None:
                phi_1 = phi_next
            bc_integral_expr += self.delta[j] / (
                casadi_sqrt(phi_next) + casadi_sqrt(phi_i))
            lambda_i = (phi_next - phi_i) / self.delta[j]
            lambda_cost += ((lambda_i - lambda_prev)) ** 2
            lambda_prev = lambda_i
            nlp.add_constraint(
                phi_next - phi_i,
                lb=[self.delta[j] * self.lambda_min],
                ub=[self.delta[j] * self.lambda_max])
            phi_i = phi_next
        phi_n = phi_next
        bc_cvx_obj = bc_integral_expr \
            - (self.init_zbar / g) * casadi_sqrt(phi_n)
        nlp.add_equality_constraint(bc_cvx_obj, self.init_zbar_deriv / g)
        nlp.add_equality_constraint(phi_1, self.delta[0] * lambda_f)
        nlp.extend_cost(lambda_cost)
        nlp.create_solver()
        phi_1_n = nlp.solve()
        if self.record_solver_times:
            self.solver_times.append(nlp.solve_time)
        if not nlp.optimal_found:
            return None
        if self.record_solver_times:
            self.solver_times_succ.append(nlp.solve_time)
        return CaptureSolution(
            phi_1_n, capture_pb=self, optimal_found=nlp.optimal_found)

    def solve(self):
        """
        Solve the capture problem.

        Returns
        -------
        solution : CaptureSolution
            Solution to the problem, if any.
        """
        if self.__nlp_solver == "ipopt":
            return self.solve_with_ipopt()
        return self.solve_with_cps()


class CaptureSolution(object):

    def __init__(self, phi_1_n, capture_pb, optimal_found=None):
        phi = array([0.] + list(phi_1_n))
        delta = capture_pb.delta
        n = capture_pb.nb_steps
        omega_i = sqrt(phi[n])
        self.delta = delta
        self.nb_steps = n
        self.phi = phi
        self.lambda_ = None
        self.lambda_i = (phi[n] - phi[n - 1]) / delta[n - 1]
        self.omega_i = omega_i
        self.optimal_found = optimal_found
        self.s = capture_pb.s
        self.switch_times = None

    def __str__(self):
        return "phi = %s;" % str(list(self.phi))

    @property
    def var_cost(self):
        """
        Compute an L2 cost in lambda variations.
        """
        return sum(
            (self.lambda_[i + 1] - self.lambda_[i])**2
            for i in xrange(self.nb_steps))

    def compute_lambda(self):
        """
        Compute the full vector of lambda values.
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
        Invert the function :math:`s \\mapsto \\phi(s)`.

        Parameters
        ----------
        phi : scalar
            Value of the function :math:`\\phi(s) = s \\omega(s)`.

        Returns
        -------
        s : scalar
            Index `s` such that `phi(s) = phi`.

        Notes
        -----
        Given the index `j` such that :math:`\\phi_j \\leq \\phi < \\phi_{j+1}`,
        the important formula behind this function is:

        .. math::

            \\phi(s) = \\sqrt{\\phi_j + \\lambda_j (s^2 - s_j^2)}

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
                    \\sqrt{\\phi_{i+1}} + \\sqrt{\\lambda_j} s_{j+1}}{
                    \\sqrt{\\phi_{i+1} - \\lambda_j (s_{j+1}^2 - s^2)}
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


class CaptureController(object):

    """
    Controller based on predictive control with boundedness condition.

    Parameters
    ----------
    pendulum : pymanoid.InvertedPendulum
        State estimator of the inverted pendulum.
    nb_steps : integer
        Number of discretization steps for the preview trajectory.
    target_height : scalar
        Desired altitude in the stationary regime.
    """

    def __init__(self, pendulum, nb_steps, target_height):
        self.capture_pb = CaptureProblem(
            pendulum.lambda_min, pendulum.lambda_max, nb_steps)
        self.pendulum = pendulum
        self.solution = None
        self.sqrt_lambda_max = sqrt(pendulum.lambda_max)
        self.sqrt_lambda_min = sqrt(pendulum.lambda_min)
        self.target_height = target_height

    def wrap_omega_lim(self, u, v):
        """
        Convert a set of constraints `u * omega_i >= v` into lower/upper bounds.

        Parameters
        ----------
        u : (n,) array
            Vector of coefficients such that `u * omega_i >= v`.
        v : (n,) array
            Vector of coefficients such that `u * omega_i >= v`.

        Returns
        -------
        init_omega_min : scalar or None
            Lower bound on the initial value of omega, or None if no solution.
        init_omega_max : scalar
            Upper bound on the initial value of omega, or None if no solution.
        """
        init_omega_min, init_omega_max = find_interval_bounds(
            u, v, self.sqrt_lambda_min, self.sqrt_lambda_max)
        if init_omega_min is None:
            raise RuntimeError("no feasible CoP")
        return (init_omega_min, init_omega_max)
