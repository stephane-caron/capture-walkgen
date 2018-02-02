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

from casadi import sqrt as casadi_sqrt
from numpy import sqrt
from pymanoid.optim import NonlinearProgram
from pymanoid.sim import gravity_const

from .capture_solution import CaptureSolution


try:
    import CaptureProblemSolver as cps
    DEFAULT_NLP_SOLVER = "cps"
except ImportError:
    pymanoid.warn("Could not import CaptureProblemSolver")
    DEFAULT_NLP_SOLVER = "ipopt"
    cps = None


class CaptureProblem(object):

    """
    Capture optimization problem.

    Parameters
    ----------
    lambda_min : scalar
        Minimum leg stiffness (positive).
    lambda_max : scalar
        Maximum leg stiffness (positive).
    nb_steps : integer
        Number of segments where :math:`\\lambda(t)` is constant.

    Attributes
    ----------
    cps_pb : cps.Problem
        Internal problem for CaptureProblemSolver.
    delta : array
        Vector of squared differences.
    init_omega_max : scalar
        Upper bound on the initial IPM damping.
    init_omega_min : scalar
        Lower bound on the initial IPM damping.
    init_zbar : scalar
        Initial CoM height above contact.
    init_zbar_deriv : scalar
        Initial derivative of CoM height above contact.
    lambda_max : scalar
        Upper bound on IPM stiffness.
    lambda_min : scalar
        Lower bound on IPM stiffness.
    nb_steps : int
        Number of spatial discretization steps.
    nlp_solver : string
        Internal nonlinear solver to be used.
    s : list
        Partition of the interval [0, 1] used for spatial discretization.
    s_sq : list
        List of squared values of the aforementioned partition.
    target_height : scalar
        Target CoM height.
    """

    def __init__(self, lambda_min, lambda_max, nb_steps):
        s_list = [j * 1. / nb_steps for j in xrange(nb_steps)] + [1.]
        s_sq = [s ** 2 for s in s_list]
        delta = [s_sq[j + 1] - s_sq[j] for j in xrange(nb_steps)]
        cps_pb = None
        if cps is not None:
            raw_pb = cps.RawProblem()
            raw_pb.delta = delta
            raw_pb.lambda_max = lambda_max
            raw_pb.lambda_min = lambda_min
            cps_pb = cps.Problem(raw_pb)
        self.cps_pb = cps_pb
        self.delta = delta
        self.init_omega_max = None
        self.init_omega_min = None
        self.init_zbar = None
        self.init_zbar_deriv = None
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.nb_steps = nb_steps
        self.nlp_solver = DEFAULT_NLP_SOLVER
        self.s = s_list
        self.s_sq = s_sq
        self.target_height = None

    def precompute(self):
        """
        Precompute QR decompositions for all nullspace-projected Jacobians used
        by the least squares sub-solver of cps.
        """
        assert self.nlp_solver == "cps"
        self.cps_pb.precompute()

    def set_init_omega_lim(self, init_omega_min, init_omega_max):
        """
        Set minimum and maximum values for the initial IPM damping omega.

        Parameters
        ----------
        init_omega_min : scalar
            Lower bound.
        init_omega_max : scalar
            Upper bound.
        """
        if init_omega_max > sqrt(self.lambda_max):
            init_omega_max = sqrt(self.lambda_max)
            pymanoid.warn("init_omega_max capped to sqrt(lambda_max)")
        if init_omega_min < sqrt(self.lambda_min):
            pymanoid.warn("init_omega_min capped to sqrt(lambda_min)")
            init_omega_min = sqrt(self.lambda_min)
        self.init_omega_max = init_omega_max
        self.init_omega_min = init_omega_min

    def solve(self):
        """
        Solve the capture problem.

        Returns
        -------
        solution : CaptureSolution
            Solution to the problem, if any.
        """
        if self.nlp_solver == "ipopt":
            return self.solve_with_ipopt()
        return self.solve_with_cps()

    def solve_with_cps(self):
        """
        Solve problem using `CaptureProblemSolver
        <https://github.com/jrl-umi3218/CaptureProblemSolver>`_.

        Returns
        -------
        solution : CaptureSolution
            Solution to the problem, if any.
        """
        self.cps_pb.set_init_zbar(self.init_zbar)
        self.cps_pb.set_init_zbar_deriv(self.init_zbar_deriv)
        self.cps_pb.set_init_omega_max(self.init_omega_max)
        self.cps_pb.set_init_omega_min(self.init_omega_min)
        self.cps_pb.set_target_height(self.target_height)
        sqp = cps.SQP(self.cps_pb.size)
        status = sqp.solve(self.cps_pb)
        if status == cps.SolverStatus.Fail:
            msg = "cps solver failed on the following problem"
            raise RuntimeError("{}:\n\n{}".format(msg, self))
        elif status == cps.SolverStatus.NoLinearlyFeasiblePoint:
            return None
        return CaptureSolution(sqp.x(), capture_pb=self, optimal_found=True)

    def solve_with_ipopt(self):
        """
        Solve problem using IPOPT via `CasADi <http://casadi.org>`_.

        Returns
        -------
        solution : CaptureSolution
            Solution to the problem, if any.
        """
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
        if not nlp.optimal_found:
            return None
        return CaptureSolution(
            phi_1_n, capture_pb=self, optimal_found=nlp.optimal_found)
