# This file is part of the PyCEST package.
# Copyright (C) 2021  Corentin Martens
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Contact: corentin.martens@ulb.be


import inspect
import numpy as np
from numpy import ndarray
from scipy.optimize import curve_fit


def z_spectrum_fit(dw, z, func, params, bounds, p_0=None,
                   fit_options={'max_nfev': 10000, 'ftol': 1e-12, 'xtol': 1e-12, 'verbose': 0}):

    """Z-spectrum fitting routine from experimental data.

    Given experimental measurements z of the Z-magnetisation of a proton pool A at offset frequencies dw, fits all
    parameters of a Z-spectrum model function func (see PyCEST.modelling) that are not specified by params, provided
    value bounds bounds and an optional initial guess p_0. Fitting options can be specified through fit_options passed
    as kwargs to scipy.optimize.curve_fit.

    Parameters
    ----------
    dw : ndarray
        The saturation frequency offsets [Hz].
    z : ndarray
        The measured Z-magnetization of pool 1 (typically water) [T].
    func : callable
        The z-spectrum model function.
    params : dict
        A dictionary gathering the non-fitted arguments of func with their respective values. All unspecified arguments,
        even optional, will be fitted except the first unspecified argument which must correspond to dw.
    bounds : 2-tuple of array_like
        The parameter bounds. The lower (first tuple element) and upper (second tuple element) values must be specified
        in the same order as in the signature of func and provided as ndarrays for ndarrays arguments.
    p_0 : ndarray, optional
        An initial guess for the solution. The values must be specified in the same order as in the signature of func
        and provided as ndarrays for ndarrays arguments. Default: None (mid-bound values are used).
    fit_options : dict, optional
        The fitting options for scipy.optimize.leastsq or scipy.optimize.least_squares (see scipy.optimize.curve_fit).
        Default: {'max_nfev': 10000, 'ftol': 1e-12, 'xtol': 1e-12, 'verbose': 0}.

    Returns
    -------
    p : ndarray
        The fitted values of the parameters.
    z_ : ndarray
        The estimated Z-magnetization of pool 1 [T].

    """

    fit_params = [p for p in inspect.getfullargspec(func)[0] if p not in params.keys()]
    print(f'PyCEST.fitting.z_spectrum_fit: Info: xdata is \'{fit_params[0]}\', optimizing {fit_params[1:]}.')

    assert len(bounds[0]) == len(fit_params)-1 and len(bounds[1]) == len(fit_params)-1

    # Numpy array inputs are flattened for scipy.optimize.curve_fit.
    n = []        # Keeps a track of the original ndarray length or 0 if not an ndarray.
    b = [[], []]  # Flattened bounds.

    for lb, ub in zip(*bounds):
        b[0] += list(lb) if isinstance(lb, np.ndarray) else [lb]
        b[1] += list(ub) if isinstance(ub, np.ndarray) else [ub]
        n.append(len(lb) if isinstance(lb, np.ndarray) else 0)

    bounds = b

    # args = ', '.join(fit_params)
    # kwargs = ', '.join([f'{p}={p}' for p in fit_params])
    # function_str = f'def function({args}): return func({kwargs}, **params)'
    # exec(function_str, locals())
    # function = locals()['function']

    # Doesn't work since scipy.optimize.curve_fit uses getfullargspec to determine the number of parameters to fit,
    # which are hidden behind args => provide a p_0 of suitable shape to solve the issue.
    def function(xdata, *args):

        params[fit_params[0]] = xdata
        index = 0

        for i in range(1, len(fit_params)):
            p = args[index] if n[i-1] == 0 else np.array(args[index:index+n[i-1]])  # Rebuilds a ndarray if necessary.
            params[fit_params[i]] = p
            index += max(n[i-1], 1)

        return np.squeeze(func(**params))  # Non-steady state Z-spectrum model functions in PyCEST.modelling return one
                                           # Z-spectrum per specified time (which should be unique here).

    if p_0 is None:
        p_0 = np.array([(lb+ub)/2.0 if np.isfinite(lb) and np.isfinite(ub) else 1.0 for lb, ub in zip(*bounds)])

    try:
        r, _ = curve_fit(function, dw, z, p0=p_0, bounds=bounds, **fit_options)

    except RuntimeError:
        r = p_0
        print('Warning: Unable to fit model. Initial values are returned instead.')

    p = []
    index = 0

    # Reformatting of the fitted parameter values.
    for i in range(1, len(fit_params)):
        p.append(r[index] if n[i-1] == 0 else np.array(r[index:index+n[i-1]]))
        index += max(n[i-1], 1)

    z_ = function(dw, *r)

    return p, z_
