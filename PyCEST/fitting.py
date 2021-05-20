import inspect
import numpy as np
from numpy import ndarray, inf
from scipy.optimize import curve_fit


def z_spectrum_fit(dw, z, func, params, p_0=None, bounds=None,
                   fit_options={'max_nfev': 1000, 'ftol': 1e-12, 'xtol': 1e-12, 'verbose': 0}):

    """Z-spectrum fitting routine from experimental data.

    Given experimental measurements z of the Z-magnetisation of a proton pool A at offset frequencies dw, fits all
    parameter of a Z-spectrum generation function func (see PyCEST.simulation) that are not specified by params,
    provided an optional initial guess p_0 and optional bounds bounds. Fitting options can be specified through
    fit_options passed as kwargs to scipy.optimize.curve_fit.

    Parameters
    ----------
    dw : ndarray
        The saturation frequency offsets [Hz].
    z : ndarray
        The measured Z-magnetization of pool 1 (typically water) [T].
    func : callable
        The z-spectrum generation function.
    params : dict
        A parameter dictionary for func. All unspecified parameters of func, even optional, will be fitted except the
        first one which must correspond to dw.
    p_0 : ndarray, optional
        An initial guess for the solution (see scipy.optimize.curve_fit). Must have length equal to the number of
        variables to fit. Default: None (mid-bound values are used).
    bounds : 2-tuple of array_like, optional
        The parameter bounds (see scipy.optimize.curve_fit). The dimensions of each array_like of the tuple dimensions
        must be 1 or be equal to the number of variables to fit. Default: None.
    fit_options : dict, optional
        The fitting options for scipy.optimize.leastsq or scipy.optimize.least_squares (see scipy.optimize.curve_fit).
        Default: {'max_nfev': 1000, 'ftol': 1e-12, 'xtol': 1e-12, 'verbose': 0}.

    Returns
    -------
    p : ndarray
        The fitted values of the parameters.

    """

    fit_params = [p for p in inspect.getfullargspec(func)[0] if p not in params.keys()]
    print(f'PyCEST.fitting.z_spectrum_fit: Info: xdata is \'{fit_params[0]}\', optimizing {fit_params[1:]}.')

    # args = ', '.join(fit_params)
    # kwargs = ', '.join([f'{p}={p}' for p in fit_params])
    # function_str = f'def function({args}): return func({kwargs}, **params)'
    # exec(function_str, locals())
    # function = locals()['function']

    # Doesn't work since scipy.optimize.curve_fit uses getfullargspec to determine the number of parameters to fit,
    # which are hidden behind args => provide a p_0 of suitable shape to solve the issue.

    def function(xdata, *args):  # TODO: Handle ndarray inputs for n pool models.

        params[fit_params[0]] = xdata

        for i in range(len(fit_params)-1):
            params[fit_params[i+1]] = args[i]

        return np.squeeze(func(**params))  # Non-steady state Z-spectrum model functions in PyCEST.simulations
                                           # return one Z-spectrum per specified time (which should be unique here).

    if bounds is None:
        bounds = ((-inf,)*len(fit_params), (inf,)*len(fit_params))

    if p_0 is None:
        p_0 = np.array([(lb+ub)/2.0 if np.isfinite(lb) and np.isfinite(ub) else 1.0 for lb, ub in zip(*bounds)])

    try:
        p, _ = curve_fit(function, dw, z, p0=p_0, bounds=bounds, **fit_options)

    except RuntimeError:
        p = p_0
        print('Warning: Unable to fit model. Initial values are returned instead.')

    return p
