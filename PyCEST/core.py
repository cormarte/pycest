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
