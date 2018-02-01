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

from numpy import linspace, sqrt
from pylab import grid, legend, plot, step, xlabel, ylim
from pymanoid.misc import warn
from pymanoid.sim import gravity


def plot_s(capture_sol):
    """
    Plot :math:`\\lambda(s)` and :math:`\\omega(s)^2` curves.

    Parameters
    ----------
    capture_sol : CaptureSolution
        Solution to a boundedness OCP.
    """
    if capture_sol.lambda_ is None:
        return warn("Call solver's compute_lambda() first")

    def subsample(s):
        s2 = [(s[i] + s[i + 1]) / 2. for i in xrange(len(s) - 1)]
        s2.extend(s)
        s2.sort()
        return s2

    s_more = subsample(subsample(capture_sol.s))
    omega_f = sqrt(-gravity[2] / capture_sol.z_f)
    omega_ = [capture_sol.omega_from_s(s) ** 2 for s in s_more]
    step(capture_sol.s, capture_sol.lambda_, 'b-', where='post', marker='o')
    step([0., 1.], [capture_sol.omega_i_min ** 2] * 2, 'g', linestyle='--',
         where='post')
    step([0., 1.], [capture_sol.omega_i_max ** 2] * 2, 'g', linestyle='--',
         where='post')
    step([0., 1.], [omega_f ** 2] * 2, 'k', linestyle='--',
         where='post')
    plot(s_more, omega_, 'r-', marker='o')
    legend(('$\\lambda(s)$', '$\\omega(s)^2$'), loc='upper left')
    xlabel('$s$', fontsize=18)
    ymin = 0.9 * min(min(capture_sol.lambda_), min(omega_))
    ymax = 1.1 * max(max(capture_sol.lambda_), max(omega_))
    ylim((ymin, ymax))
    grid()


def plot_t(capture_sol):
    """
    Plot :math:`\\lambda(t)` and :math:`\\omega(t)^2` curves.

    Parameters
    ----------
    capture_sol : CaptureSolution
        Solution to a boundedness OCP.
    """
    if capture_sol.lambda_ is None:
        return warn("Call solver's compute_lambda() first")
    times = list(capture_sol.switch_times) + [2 * capture_sol.switch_times[-1]]
    lambda_ = list(capture_sol.lambda_[-1::-1])
    trange = linspace(0, max(times), 20)
    omega_ = [capture_sol.omega_from_t(t) ** 2 for t in trange]
    omega_f = sqrt(-gravity[2] / capture_sol.z_f)
    step(times, lambda_, marker='o', where='pre')
    step([0., 2. * capture_sol.switch_times[-1]],
         [capture_sol.omega_i_min ** 2] * 2, 'g',
         linestyle='--', where='post')
    step([0., 2. * capture_sol.switch_times[-1]],
         [capture_sol.omega_i_max ** 2] * 2, 'g',
         linestyle='--', where='post')
    step([0., 2. * capture_sol.switch_times[-1]], [omega_f ** 2] * 2, 'k',
         linestyle='--', where='post')
    plot(trange, omega_, 'r-', marker='o')
    legend(('$\\lambda(t)$', '$\\omega(t)^2$'))
    xlabel('$t$', fontsize=18)
    ymin = 0.9 * min(min(capture_sol.lambda_), min(omega_))
    ymax = 1.1 * max(max(capture_sol.lambda_), max(omega_))
    ylim((ymin, ymax))
    grid()


def succ(a, b):
    if abs(a) < 1e-10:
        return 100. if abs(b) < 1e-10 else 0.
    return 100 * (1. - abs(a - b) / abs(a))


def debug_capture_pb(capture_pb, capture_sol):
    """
    Print debug information on a boundedness problem and its solution.

    Parameters
    ----------
    capture_pb : CaptureProblem
        Problem at hand.
    capture_sol : CaptureSolution
        Solution corresponding to the problem.
    """
    assert capture_sol.phi is not None
    g = -gravity[2]
    delta, phi, n = capture_pb.delta, capture_sol.phi, capture_pb.nb_steps
    out_bc_integral = sum(
        delta[i] / (sqrt(phi[i+1]) + sqrt(phi[i])) for i in xrange(n))
    out_bc_cvx_obj = out_bc_integral - (capture_pb.init_zbar / g) * sqrt(phi[n])
    capture_pb_omega_f = sqrt(g / capture_pb.z_f)
    out_omega_f = sqrt(phi[1] / delta[0])
    print "Terminal condition: %.1f%%" % succ(capture_pb_omega_f, out_omega_f)
    print "Boundedness condition: %.1f%%" % succ(
        capture_pb.init_zbar_deriv / g, out_bc_cvx_obj)
