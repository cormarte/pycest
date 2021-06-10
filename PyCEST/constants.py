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


""" The gyromagnetic ratio of 1H nuclei [rad /s /T] """
PROTON_GYROMAGNETIC_RATIO = 2.0*np.pi*42577478.92

""" The proton concentration of pure water [M.] """
PROTON_WATER_CONCENTRATION = 2.0*1000.0/18.01528
