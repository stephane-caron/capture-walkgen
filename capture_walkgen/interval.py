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


def find_interval_bounds(u, v, x_min, x_max):
    """
    Apply a set of inequalities `u * x >= v` to the interval `x_min <= x <=
    x_max`, resulting in a new interval `x_new_min <= x <= x_new_max`.

    Parameters
    ----------
    u : (n,) array
        Vector of coefficients such that `u * omega_i >= v`.
    v : (n,) array
        Vector of coefficients such that `u * omega_i >= v`.
    x_min : scalar
        Lower bound of the initial interval.
    x_max : scalar
        Upper bound of the initial interval.

    Returns
    -------
    x_new_min : scalar or None
        Lower bound of the new interval, None if this interval is empty.
    x_new_max : scalar or None
        Upper bound of the new interval, None if this interval is empty.
    """
    for i in range(len(u)):
        if u[i] > 1e-3:
            x_min = max(x_min, v[i] / u[i])
        elif u[i] < -1e-3:
            x_max = min(x_max, v[i] / u[i])
        elif v[i] > 0:  # u[i] is almost 0., so v[i] must be negative
            return (None, None)
    if x_min > x_max:
        return (None, None)
    return (x_min, x_max)
