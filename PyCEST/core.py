import numpy as np


def list_as_numpy(func):

    """Decorator used to convert list arguments into numpy arrays.

    Parameters
    ----------
    func : callable
        The function to decorate.

    Returns
    -------
    function : callable
        The decorated function.

    """

    def function(*args, **kwargs):
        args = [np.array(arg) if isinstance(arg, list) else arg for arg in args]
        kwargs = {k: np.array(v) if isinstance(v, list) else v for k, v in kwargs.items()}
        return func(*args, **kwargs)

    return function
