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


import multiprocessing
import numpy as np
import sympy
from numpy import ndarray
from PyCEST.constants import PROTON_GYROMAGNETIC_RATIO, PROTON_WATER_CONCENTRATION
#from PyCEST.core import list_as_numpy
from PyCEST.utils import ppm_to_frequency
from scipy.linalg import expm


def __bloch_mcconell_continuous_wave_2_pools_analytical(t, w_1, m_0, dw_a, k_a, r1_a, k_1a, k_2a, dw_b, k_b, r1_b, k_1b,
                                                        k_2b):

    """Internal solving routine for continuous wave 2 pools Bloch McConnell simulations for a single time point and a
    single offset frequency.

    From Murase. Open J. Appl. Sci. 7(1):1-14. 2017.

    Parameters
    ----------
    t : float
        The saturation time [s].
    w_1 : float
        The nutation frequency [rad/s].
    m_0 : ndarray
        The initial magnetization vector with stacked (x, y, z) components of pool A and B [T].
    dw_a : float
        The absolute difference between the saturation frequency and the resonance frequency of pool A [rad/s].
    k_a : float
        The exchange rate from pool A to pool B [Hz].
    r1_a : float
        The R1 value of pool A [Hz].
    k_1a : float
        The sum of k_a and r1_a [Hz].
    k_2a : float
        The sum of k_a and r2_a [Hz].
    dw_b : float
        The absolute difference between the saturation frequency and the resonance frequency of pool B [rad/s].
    k_b : float
        The exchange rate from pool B to pool A [Hz].
    r1_b : float
        The R1 value of pool B [Hz].
    k_1b : float
        The sum of k_b and r1_b [Hz].
    k_2b : float
        The sum of k_b and r2_b [Hz].

    Returns
    -------
    m_az : ndarray
        The final Z-magnetization of pool A at time t [T].

    """

    A = np.array([[-k_2a,  dw_a,    0,    k_b,     0,    0,            0],
                  [-dw_a, -k_2a,   w_1,     0,   k_b,    0,            0],
                  [    0,  -w_1, -k_1a,     0,     0,  k_b,  r1_a*m_0[2]],
                  [  k_a,     0,     0, -k_2b,  dw_b,    0,            0],
                  [    0,   k_a,     0, -dw_b, -k_2b,  w_1,            0],
                  [    0,     0,   k_a,     0,  -w_1, -k_1b, r1_b*m_0[5]],
                  [    0,     0,      0,    0,     0,     0,           0]])

    m_az = np.matmul(expm(A*t), m_0)[2]

    return m_az


def __bloch_mcconell_continuous_wave_2_pools_init(dw, b_1, b_0, db_0, w_c, m_az_0, w_a, t1_a, t2_a, c_a, w_b, t1_b,
                                                  t2_b, c_b, k_b):

    """Parameter initialization routine for continuous wave 2 pools Bloch McConnell simulations.

    Parameters
    ----------
    dw : ndarray
        The saturation frequency offsets [Hz].
    b_1 : float
        The amplitude of the continuous RF pulse B1 [T].
    b_0 : float
        The reference static field value [T].
    db_0 : float
        The reference static field offset value [T].
    w_c : float
        The center imaging frequency [Hz].
    m_az_0 : float
        The initial Z-magnetization of pool A [T].
    w_a : float
        The resonance frequency of pool A [ppm].
    t1_a : float
        The T1 value of pool A [s].
    t2_a : float
        The T2 value of pool A [s].
    c_a : float
        The concentration of exchangeable protons of pool A [M.].
    w_b : float
        The resonance frequency of pool B [ppm].
    t1_b : float
        The T1 value of pool B [s].
    t2_b : float
        The T2 value of pool B [s].
    c_b : float
        The concentration of exchangeable protons of pool B [M.].
    k_b : float
        The exchange rate from pool B to pool A [Hz].

    Returns
    -------
    w_1 : float
        The nutation frequency [rad/s].
    m_0 : ndarray
        The initial magnetization vector with stacked (x, y, z) components of pool A and B [T].
    dw_a : ndarray
        The absolute difference between the saturation frequencies and the resonance frequency of pool A [rad/s].
    k_a : float
        The exchange rate from pool A to pool B [Hz].
    r1_a : float
        The R1 value of pool A [Hz].
    k_1a : float
        The sum of k_a and r1_a [Hz].
    k_2a : float
        The sum of k_a and 1/t2_a [Hz].
    dw_b : ndarray
        The absolute difference between the saturation frequencies and the resonance frequency of pool B [rad/s].
    r1_b : float
        The R1 value of pool B [Hz].
    k_1b : float
        The sum of k_b and r1_b [Hz].
    k_2b : float
        The sum of k_b and 1/t2_b [Hz].

    """

    b_0 = b_0 + db_0
    w_0 = PROTON_GYROMAGNETIC_RATIO*b_0
    w_1 = PROTON_GYROMAGNETIC_RATIO*b_1

    if w_c is None:
        w_c = w_0
        print('Warning: the central imaging frequency was not set. 1H resonance frequency will be used.')
    else:
        w_c = 2.0*np.pi*w_c
    dw = 2.0*np.pi*dw
    w = w_c+dw

    w_a = ppm_to_frequency(w_a, w_0)
    dw_a = w_a-w
    k_a = (c_b/c_a)*k_b  # For mass balance, see Eq. [9] in Woessner et al. Magn. Reson. Med. 53:790-9. 2005.
    r1_a = 1.0/t1_a
    r2_a = 1.0/t2_a
    k_1a = r1_a + k_a
    k_2a = r2_a + k_a

    m_bz_0 = (c_b/c_a)*m_az_0  # Equilibrium magnetization is proportional to the proton density.
    w_b = ppm_to_frequency(w_b, w_0)
    dw_b = w_b-w
    r1_b = 1.0/t1_b
    r2_b = 1.0/t2_b
    k_1b = r1_b + k_b
    k_2b = r2_b + k_b

    m_0 = np.array([0, 0, m_az_0, 0, 0, m_bz_0, 1])

    return w_1, m_0, dw_a, k_a, r1_a, k_1a, k_2a, dw_b, r1_b, k_1b, k_2b


def __bloch_mcconell_continuous_wave_n_pools_analytical(t, w_1, m_0, dw, k, r1, r2):

    """Internal solving routine for continuous wave n pools Bloch McConnell simulations for a single time point and a
    single offset frequency.

    From Murase. Open J. Appl. Sci. 7(1):1-14. 2017.

    Parameters
    ----------
    t : float
        The saturation time [s].
    w_1 : float
        The nutation frequency [rad/s].
    m_0 : ndarray
        The initial magnetization vector with stacked (x, y, z) components of each pool [T].
    dw : ndarray
        The absolute difference between the saturation frequency and the resonance frequency of each pool [rad/s].
    k : ndarray
        The exchange rates from pools 1 to 2:n (n-1 first elements) and 2:n to 1 (n-1 last elements) [Hz].
    r1 : ndarray
        The R1 values of pool 1:n [Hz].
    r2 : ndarray
        The R2 values of pool 1:n [Hz].

    Returns
    -------
    m_az : ndarray
        The final Z-magnetization of pool 1 at time t [T].

    """

    n = len(r1)

    # R matrix assembly
    R = np.zeros((3*n, 3*n))

    for i in range(n):
        R[3*i, 3*i] = -r2[i]
        R[3*i, 3*i+1] = dw[i]
        R[3*i+1, 3*i] = -dw[i]
        R[3*i+1, 3*i+1] = -r2[i]
        R[3*i+1, 3*i+2] = w_1
        R[3*i+2, 3*i+1] = -w_1
        R[3*i+2, 3*i+2] = -r1[i]

    # K matrix assembly
    K = np.zeros((n, n))
    K[0, 0] = -np.sum(k[:n-1])

    for i in range(1, n):
        K[0, i] = k[n-2+i]
        K[i, 0] = k[i-1]
        K[i, i] = -k[n-2+i]

    K = np.kron(K, np.identity(3))

    # E matrix assembly
    E = R + K

    # C matrix assembly
    C = np.zeros((3*n+1,))

    for i in range(n):
        C[3*i+2] = r1[i]*m_0[3*i+2]

    # A matrix assembly
    A = np.zeros((3*n+1, 3*n+1))
    A[:-1, :-1] = E
    A[:, -1] = C

    m_az = np.matmul(expm(A*t), m_0)[2]

    return m_az


def __bloch_mcconell_continuous_wave_n_pools_init(dw, b_1, b_0, db_0, w_c, m_az_0, w, t1, t2, c, k):

    """Parameter initialization routine for continuous wave n pools Bloch McConnell simulations.

    Parameters
    ----------
    dw : list or ndarray
        The saturation frequency offsets [Hz].
    b_1 : float
        The amplitude of the continuous RF pulse B1 [T].
    b_0 : float, optional
        The reference static field value [T].
    db_0 : float, optional
        The reference static field offset value [T].
    w_c : float, optional
        The center imaging frequency [Hz].
    m_az_0 : float, optional
        The initial Z-magnetization of pool 1 [T].
    w : list or ndarray
        The resonance frequencies of the n pools [ppm].
    t1 : list or ndarray
        The T1 values of the n pools [s].
    t2 : list or ndarray
        The T2 values of the n pools [s].
    c : list or ndarray
        The concentrations of exchangeable protons of the n pools [M.].
    k : list or ndarray
        The exchange rates from pools 2:n to pool 1 [Hz].

    Returns
    -------
    w_1 : float
        The nutation frequency [rad/s].
    m_0 : ndarray
        The initial magnetization vector with stacked (x, y, z) components of each pool [T].
    dw : ndarray
        The absolute difference between the saturation frequencies (axis 0) and the resonance frequency of each pool
        (axis 1) [rad/s].
    k : ndarray
        The exchange rates from pools 1 to 2:n (n-1 first elements) and 2:n to 1 (n-1 last elements) [Hz].
    r1 : ndarray
        The R1 values of pool 1:n [Hz].
    r2 : ndarray
        The R2 values of pool 1:n [Hz].

    """

    b_0 = b_0 + db_0
    w_0 = PROTON_GYROMAGNETIC_RATIO*b_0
    w_1 = PROTON_GYROMAGNETIC_RATIO*b_1

    if w_c is None:
        w_c = w_0
        print('Warning: the central imaging frequency was not set. 1H resonance frequency will be used.')
    else:
        w_c = 2.0*np.pi*w_c
    dw = 2.0*np.pi*dw
    w_i = w_c + dw

    m_0 = np.zeros((3*len(w)+1,))
    m_0[-1] = 1.0

    for i in range(len(w)):
        m_0[3*i+2] = (c[i]/c[0])*m_az_0  # Equilibrium magnetization is proportional to the proton density.

    w = np.array([ppm_to_frequency(w[i], w_0) for i in range(len(w))])
    dw = np.array([w-w_i[i] for i in range(len(w_i))])
    k = np.concatenate((np.array([(c[i+1]/c[0])*k[i] for i in range(len(k))]), k))   # For mass balance, see Eq. [19]-[20] in Woessner et al. Magn. Reson. Med. 53:790-9. 2005.
    r1 = 1.0/t1
    r2 = 1.0/t2

    return w_1, m_0, dw, k, r1, r2


#@list_as_numpy
def bloch_mcconell_continuous_wave_2_pools_analytical(t, dw, b_1, t1_a, t2_a, w_b, t1_b, t2_b, c_b, k_b, b_0=3.0,
                                                      db_0=0.0, w_c=None, m_az_0=1.0, w_a=0.0,
                                                      c_a=PROTON_WATER_CONCENTRATION):

    """Computes the Z-magnetization of a proton pool A at times t after saturation by a continuous wave with amplitude
    b_1 and frequency w_c + dw for two exchangeable proton pools A and B immersed in a static field with amplitude b_0 +
    db_0 using the Bloch-McConnell equations.

    Pool A (typically water) has initial z-magnetization m_az_0, resonance frequency w_a, T1 value t1_a, T2 value t2_a,
    and exchangeable proton concentration c_a. Pool B (metabolite) has resonance frequency w_b, T1 value t1_b, T2 value
    t2_b, exchangeable proton concentration c_b, and exchange rate k_b with pool A.

    Parameters
    ----------
    t : ndarray
        The saturation times [s].
    dw : ndarray
        The saturation frequency offsets [Hz].
    b_1 : float
        The amplitude of the continuous RF pulse B1 [T].
    t1_a : float
        The T1 value of pool A [s].
    t2_a : float
        The T2 value of pool A [s].
    w_b : float
        The resonance frequency of pool B [ppm].
    t1_b : float
        The T1 value of pool B [s].
    t2_b : float
        The T2 value of pool B [s].
    c_b : float
        The concentration of exchangeable protons of pool B [M.].
    k_b : float
        The exchange rate from pool B to pool A [Hz].
    b_0 : float, optional
        The reference static field value [T]. Default: 3.0.
    db_0 : float, optional
        The reference static field offset value [T]. Default: 0.0.
    w_c : float, optional
        The center imaging frequency [Hz]. Default: PyCEST.constants.PROTON_GYROMAGNETIC_RATIO*b_0.
    m_az_0 : float, optional
        The initial Z-magnetization of pool A [T]. Default: 1.0.
    w_a : float, optional
        The resonance frequency of pool A [ppm]. Default: 0.0.
    c_a : float, optional
        The concentration of exchangeable protons of pool A [M.]. Default: PyCEST.constants.PROTON_WATER_CONCENTRATION.

    Returns
    -------
    m_az : ndarray
        The final Z-magnetization of pool A for the specified times (axis 0) and offset frequencies (axis 1) [T].

    """

    w_1, m_0, dw_a, k_a, r1_a, k_1a, k_2a, dw_b, r1_b, k_1b, k_2b = \
        __bloch_mcconell_continuous_wave_2_pools_init(dw, b_1, b_0, db_0, w_c, m_az_0, w_a, t1_a, t2_a, c_a, w_b, t1_b,
                                                      t2_b, c_b, k_b)

    m_az = np.zeros((len(t), len(dw)), dtype=np.float64)

    for i in range(len(t)):
        for j in range(len(dw)):
            m_az[i, j] = __bloch_mcconell_continuous_wave_2_pools_analytical(t[i], w_1, m_0, dw_a[j], k_a, r1_a, k_1a,
                                                                             k_2a, dw_b[j], k_b, r1_b, k_1b, k_2b)

    return m_az


def bloch_mcconell_continuous_wave_2_pools_fdm(t, dw, b_1, t1_a, t2_a, w_b, t1_b, t2_b, c_b, k_b, b_0=3.0, db_0=0.0,
                                               w_c=None, m_az_0=1.0, w_a=0.0, c_a=PROTON_WATER_CONCENTRATION, dt=1e-8):

    """Computes the Z-magnetization of a proton pool A at time t after saturation by a continuous wave with amplitude
    b_1 and frequency w_c + dw for two exchangeable proton pools A and B immersed in a static field with amplitude b_0 +
    db_0 using the Bloch-McConnell equations. This implementation uses a finite difference approach.

    Pool A (typically water) has initial z-magnetization m_az_0, resonance frequency w_a, T1 value t1_a, T2 value t2_a,
    and exchangeable proton concentration c_a. Pool B (metabolite) has resonance frequency w_b, T1 value t1_b, T2 value
    t2_b, exchangeable proton concentration c_b, and exchange rate k_b with pool A.

    Parameters
    ----------
    t : float
        The saturation time [s].
    dw : list or ndarray
        The saturation frequency offsets [Hz].
    b_1 : float
        The amplitude of the continuous RF pulse B1 [T].
    t1_a : float
        The T1 value of pool A [s].
    t2_a : float
        The T2 value of pool A [s].
    w_b : float
        The resonance frequency of pool B [ppm].
    t1_b : float
        The T1 value of pool B [s].
    t2_b : float
        The T2 value of pool B [s].
    c_b : float
        The concentration of exchangeable protons of pool B [M.].
    k_b : float
        The exchange rate from pool B to pool A [Hz].
    b_0 : float, optional
        The reference static field value [T]. Default: 3.0.
    db_0 : float, optional
        The reference static field offset value [T]. Default: 0.0.
    w_c : float, optional
        The center imaging frequency [Hz]. Default: PyCEST.constants.PROTON_GYROMAGNETIC_RATIO*b_0.
    m_az_0 : float, optional
        The initial Z-magnetization of pool A [T]. Default: 1.0.
    w_a : float, optional
        The resonance frequency of pool A [ppm]. Default: 0.0.
    c_a : float, optional
        The concentration of exchangeable protons of pool A [M.]. Default: PyCEST.constants.PROTON_WATER_CONCENTRATION.
    dt : float, optional
        The time step [s]. Default: 1e-8.

    Returns
    -------
    m_az : ndarray
        The final Z-magnetization of pool A for the specified times (axis 0) and offset frequencies (axis 1) [T].

    """

    w_1, m_0, dw_a, k_a, r1_a, k_1a, k_2a, dw_b, r1_b, k_1b, k_2b = \
        __bloch_mcconell_continuous_wave_2_pools_init(dw, b_1, b_0, db_0, w_c, m_az_0, w_a, t1_a, t2_a, c_a, w_b, t1_b,
                                                      t2_b, c_b, k_b)

    n = int(np.ceil(t/dt))

    m_ax = np.zeros((len(dw),))
    m_ay = np.zeros((len(dw),))
    m_az = m_0[2]*np.ones((len(dw),))
    m_bx = np.zeros((len(dw),))
    m_by = np.zeros((len(dw),))
    m_bz = m_0[5]*np.ones((len(dw),))

    for i in range(n - 1):

        m_ax_t = -k_2a*m_ax + dw_a*m_ay + k_b*m_bx
        m_ay_t = -dw_a*m_ax - k_2a*m_ay + w_1*m_az + k_b*m_by
        m_az_t = -w_1*m_ay - k_1a*m_az + k_b*m_bz + r1_a*m_0[2]

        m_bx_t = k_a*m_ax - k_2b*m_bx + dw_b*m_by
        m_by_t = k_a*m_ay - dw_b*m_bx - k_2b*m_by + w_1*m_bz
        m_bz_t = k_a*m_az - w_1*m_by - k_1b*m_bz + r1_b*m_0[5]

        m_ax = m_ax + dt*m_ax_t
        m_ay = m_ay + dt*m_ay_t
        m_az = m_az + dt*m_az_t
        m_bx = m_bx + dt*m_bx_t
        m_by = m_by + dt*m_by_t
        m_bz = m_bz + dt*m_bz_t

    # m_ay = np.zeros((n, len(dw)))
    # m_az = np.zeros((n, len(dw)))
    # m_bx = np.zeros((n, len(dw)))
    # m_by = np.zeros((n, len(dw)))
    # m_bz = np.zeros((n, len(dw)))

    # m_ax = np.zeros((n, len(dw)))
    # m_ay = np.zeros((n, len(dw)))
    # m_az = np.zeros((n, len(dw)))
    # m_bx = np.zeros((n, len(dw)))
    # m_by = np.zeros((n, len(dw)))
    # m_bz = np.zeros((n, len(dw)))
    #
    # m_az[0, :] = m_0[2]
    # m_bz[0, :] = m_0[5]

    # for i in range(n-1):
    #
    #     m_ax_t = -k_2a*m_ax[i] + dw_a*m_ay[i] + k_b*m_bx[i]
    #     m_ay_t = -dw_a*m_ax[i] - k_2a*m_ay[i] + w_1*m_az[i] + k_b*m_by[i]
    #     m_az_t = -w_1*m_ay[i] - k_1a*m_az[i] + k_b*m_bz[i] + r1_a*m_az_0
    #
    #     m_bx_t = k_a*m_ax[i] - k_2b*m_bx[i] + dw_b*m_by[i]
    #     m_by_t = k_a*m_ay[i] - dw_b*m_bx[i] - k_2b*m_by[i] + w_1*m_bz[i]
    #     m_bz_t = k_a*m_az[i] - w_1*m_by[i] - k_1b*m_bz[i] + r1_b*m_bz_0
    #
    #     m_ax[i+1] = m_ax[i] + dt*m_ax_t
    #     m_ay[i+1] = m_ay[i] + dt*m_ay_t
    #     m_az[i+1] = m_az[i] + dt*m_az_t
    #     m_bx[i+1] = m_bx[i] + dt*m_bx_t
    #     m_by[i+1] = m_by[i] + dt*m_by_t
    #     m_bz[i+1] = m_bz[i] + dt*m_bz_t

    return m_az


#@list_as_numpy
def bloch_mcconell_continuous_wave_2_pools_analytical_parallel(t, dw, b_1, t1_a, t2_a, w_b, t1_b, t2_b, c_b, k_b,
                                                               b_0=3.0, db_0=0.0, w_c=None, m_az_0=1.0, w_a=0.0,
                                                               c_a=PROTON_WATER_CONCENTRATION, threads=16):

    """Computes the Z-magnetization of a proton pool A at times t after saturation by a continuous wave with amplitude
    b_1 and frequency w_c + dw for two exchangeable proton pools A and B immersed in a static field with amplitude b_0 +
    db_0 using the Bloch-McConnell equations. This implementation is parallelized over a number of threads threads.

    Pool A (typically water) has initial z-magnetization m_az_0, resonance frequency w_a, T1 value t1_a, T2 value t2_a,
    and exchangeable proton concentration c_a. Pool B (metabolite) has resonance frequency w_b, T1 value t1_b, T2 value
    t2_b, exchangeable proton concentration c_b, and exchange rate k_b with pool A.

    Parameters
    ----------
    t : ndarray
        The saturation times [s].
    dw : ndarray
        The saturation frequency offsets [Hz].
    b_1 : float
        The amplitude of the continuous RF pulse B1 [T].
    t1_a : float
        The T1 value of pool A [s].
    t2_a : float
        The T2 value of pool A [s].
    w_b : float
        The resonance frequency of pool B [ppm].
    t1_b : float
        The T1 value of pool B [s].
    t2_b : float
        The T2 value of pool B [s].
    c_b : float
        The concentration of exchangeable protons of pool B [M.].
    k_b : float
        The exchange rate from pool B to pool A [Hz].
    b_0 : float, optional
        The reference static field value [T]. Default: 3.0.
    db_0 : float, optional
        The reference static field offset value [T]. Default: 0.0.
    w_c : float, optional
        The center imaging frequency [Hz]. Default: PyCEST.constants.PROTON_GYROMAGNETIC_RATIO*b_0.
    m_az_0 : float, optional
        The initial Z-magnetization of pool A [T]. Default: 1.0.
    w_a : float, optional
        The resonance frequency of pool A [ppm]. Default: 0.0.
    c_a : float, optional
        The concentration of exchangeable protons of pool A [M.]. Default: PyCEST.constants.PROTON_WATER_CONCENTRATION.
    threads : int, optional
        The number of parallel threads. Default: 16.

    Returns
    -------
    m_az : ndarray
        The final Z-magnetization of pool A for the specified times (axis 0) and offset frequencies (axis 1) [T].

    """

    w_1, m_0, dw_a, k_a, r1_a, k_1a, k_2a, dw_b, r1_b, k_1b, k_2b = \
        __bloch_mcconell_continuous_wave_2_pools_init(dw, b_1, b_0, db_0, w_c, m_az_0, w_a, t1_a, t2_a, c_a, w_b, t1_b,
                                                      t2_b, c_b, k_b)

    m_az = np.zeros((len(t), len(dw)), dtype=np.float64)

    pool = multiprocessing.Pool(processes=threads)
    results = []

    for i in range(len(t)):
        for j in range(len(dw_a)):
            results.append(pool.apply_async(__bloch_mcconell_continuous_wave_2_pools_analytical, args=(t[i], w_1, m_0,
                                                                                                       dw_a[j], k_a,
                                                                                                       r1_a, k_1a, k_2a,
                                                                                                       dw_b[j], k_b,
                                                                                                       r1_b, k_1b, k_2b)
                                            ))
    pool.close()
    pool.join()

    index = 0

    for i in range(len(t)):
        for j in range(len(dw_a)):
            m_az[i, j] = results[index].get()
            index += 1

    return m_az


#@list_as_numpy
def bloch_mcconell_continuous_wave_2_pools_steady_state(dw, b_1, t1_a, t2_a, w_b, t1_b, t2_b, c_b, k_b, b_0=3.0,
                                                        db_0=0.0, w_c=None, m_az_0=1.0, w_a=0.0,
                                                        c_a=PROTON_WATER_CONCENTRATION):

    """Computes the steady state Z-magnetization of a proton pool A after saturation by a continuous wave with amplitude
    b_1 and frequency w_c + dw for two exchangeable proton pools A and B immersed in a static field with amplitude b_0 +
    db_0 using the Bloch-McConnell equations.

    Pool A (typically water) has initial z-magnetization m_az_0, resonance frequency w_a, T1 value t1_a, T2 value t2_a,
    and exchangeable proton concentration c_a. Pool B (metabolite) has resonance frequency w_b, T1 value t1_b, T2 value
    t2_b, exchangeable proton concentration c_b, and exchange rate k_b with pool A.

    Parameters
    ----------
    dw : ndarray
        The saturation frequency offsets [Hz].
    b_1 : float
        The amplitude of the continuous RF pulse B1 [T].
    t1_a : float
        The T1 value of pool A [s].
    t2_a : float
        The T2 value of pool A [s].
    w_b : float
        The resonance frequency of pool B [ppm].
    t1_b : float
        The T1 value of pool B [s].
    t2_b : float
        The T2 value of pool B [s].
    c_b : float
        The concentration of exchangeable protons of pool B [M.].
    k_b : float
        The exchange rate from pool B to pool A [Hz].
    b_0 : float, optional
        The reference static field value [T]. Default: 3.0.
    db_0 : float, optional
        The reference static field offset value [T]. Default: 0.0.
    w_c : float, optional
        The center imaging frequency [Hz]. Default: PyCEST.constants.PROTON_GYROMAGNETIC_RATIO*b_0.
    m_az_0 : float, optional
        The initial Z-magnetization of pool A [T]. Default: 1.0.
    w_a : float, optional
        The resonance frequency of pool A [ppm]. Default: 0.0.
    c_a : float, optional
        The concentration of exchangeable protons of pool A [M.]. Default: PyCEST.constants.PROTON_WATER_CONCENTRATION.

    Returns
    -------
    m_az : ndarray
        The steady state Z-magnetization of pool A for the specified offset frequencies [T].

    """

    w_1, m_0, dw_a, k_a, r1_a, k_1a, k_2a, dw_b, r1_b, k_1b, k_2b = \
        __bloch_mcconell_continuous_wave_2_pools_init(dw, b_1, b_0, db_0, w_c, m_az_0, w_a, t1_a, t2_a, c_a, w_b, t1_b,
                                                      t2_b, c_b, k_b)

    # Analytical expression returned by PyCEST.simulation.bloch_mcconell_continuous_wave_2_pools_steady_state_sym()
    m_az = (dw_a**2*dw_b**2*k_1b*m_az_0*r1_a + dw_a**2*dw_b**2*k_b*m_0[5]*r1_b + dw_a**2*k_1b*k_2b**2*m_az_0*r1_a
            + dw_a**2*k_2b**2*k_b*m_0[5]*r1_b + dw_a**2*k_2b*m_az_0*r1_a*w_1**2 + 2*dw_a*dw_b*k_1b*k_a*k_b*m_az_0*r1_a
            + 2*dw_a*dw_b*k_a*k_b**2*m_0[5]*r1_b + dw_a*dw_b*k_b*m_0[5]*r1_b*w_1**2 + dw_b**2*k_1b*k_2a**2*m_az_0*r1_a
            + dw_b**2*k_2a**2*k_b*m_0[5]*r1_b + k_1b*k_2a**2*k_2b**2*m_az_0*r1_a - 2*k_1b*k_2a*k_2b*k_a*k_b*m_az_0*r1_a
            + k_1b*k_a**2*k_b**2*m_az_0*r1_a + k_2a**2*k_2b**2*k_b*m_0[5]*r1_b + k_2a**2*k_2b*m_az_0*r1_a*w_1**2
            - 2*k_2a*k_2b*k_a*k_b**2*m_0[5]*r1_b - k_2a*k_2b*k_b*m_0[5]*r1_b*w_1**2 - k_2a*k_a*k_b*m_az_0*r1_a*w_1**2
            + k_a**2*k_b**3*m_0[5]*r1_b + k_a*k_b**2*m_0[5]*r1_b*w_1**2) / \
           (dw_a**2*dw_b**2*k_1a*k_1b - dw_a**2*dw_b**2*k_a*k_b + dw_a**2*k_1a*k_1b*k_2b**2 + dw_a**2*k_1a*k_2b*w_1**2
            - dw_a**2*k_2b**2*k_a*k_b + 2*dw_a*dw_b*k_1a*k_1b*k_a*k_b - 2*dw_a*dw_b*k_a**2*k_b**2
            - 2*dw_a*dw_b*k_a*k_b*w_1**2 + dw_b**2*k_1a*k_1b*k_2a**2 + dw_b**2*k_1b*k_2a*w_1**2
            - dw_b**2*k_2a**2*k_a*k_b + k_1a*k_1b*k_2a**2*k_2b**2 - 2*k_1a*k_1b*k_2a*k_2b*k_a*k_b
            + k_1a*k_1b*k_a**2*k_b**2 + k_1a*k_2a**2*k_2b*w_1**2 - k_1a*k_2a*k_a*k_b*w_1**2
            + k_1b*k_2a*k_2b**2*w_1**2 - k_1b*k_2b*k_a*k_b*w_1**2 - k_2a**2*k_2b**2*k_a*k_b + 2*k_2a*k_2b*k_a**2*k_b**2
            + 2*k_2a*k_2b*k_a*k_b*w_1**2 + k_2a*k_2b*w_1**4 - k_a**3*k_b**3 - 2*k_a**2*k_b**2*w_1**2 - k_a*k_b*w_1**4)

    return m_az


def bloch_mcconell_continuous_wave_2_pools_steady_state_sym():

    """Returns an analytical expression for the steady state Z-magnetization of a proton pool A after saturation by a
    continuous wave with nutation frequency w_1 and absolute frequency difference dw_a and dw_b with two proton pools A
    and B using the Bloch-McConnell equations.

    Pool A (typically water) has initial z-magnetization m_az_0, R1 value r1_a, R2 value r2_a, and exchange rate k_a
    with pool B. k_1a = k_a + r1_a and k_2a = k_a + r2_a. Pool B (metabolite) has initial z-magnetization m_bz_0, R1
    value r1_b, R2 value r2_b, and exchange rate k_b with pool A. k_1b = k_b + r1_b and k_2b = k_b + r2_b.

    From Murase. Open J. Appl. Sci. 7(1):1-14. 2017.

    Returns
    -------
    m_az : str
        The analytical expression for the steady state Z-magnetization of pool A.

    """

    m_ax = sympy.Symbol('m_ax')
    m_ay = sympy.Symbol('m_ay')
    m_az = sympy.Symbol('m_az')
    m_bx = sympy.Symbol('m_bx')
    m_by = sympy.Symbol('m_by')
    m_bz = sympy.Symbol('m_bz')

    w_1 = sympy.Symbol('w_1')

    m_az_0 = sympy.Symbol('m_az_0')
    dw_a = sympy.Symbol('dw_a')
    r1_a = sympy.Symbol('r1_a')
    k_a = sympy.Symbol('k_a')
    k_1a = sympy.Symbol('k_1a')
    k_2a = sympy.Symbol('k_2a')

    m_bz_0 = sympy.Symbol('m_bz_0')
    dw_b = sympy.Symbol('dw_b')
    r1_b = sympy.Symbol('r1_b')
    k_b = sympy.Symbol('k_b')
    k_1b = sympy.Symbol('k_1b')
    k_2b = sympy.Symbol('k_2b')

    solution = sympy.solve((-k_2a*m_ax + dw_a*m_ay + k_b*m_bx,
                            -dw_a*m_ax - k_2a*m_ay + w_1*m_az + k_b*m_by,
                            -w_1*m_ay - k_1a*m_az + k_b*m_bz + r1_a*m_az_0,
                            k_a*m_ax - k_2b*m_bx + dw_b*m_by,
                            k_a*m_ay - dw_b*m_bx - k_2b*m_by + w_1*m_bz,
                            k_a*m_az - w_1*m_by - k_1b*m_bz + r1_b*m_bz_0),
                           (m_ax, m_ay, m_az, m_bx, m_by, m_bz))

    return solution[m_az]


def bloch_mcconell_continuous_wave_3_pools_steady_state_sym():

    """Returns an analytical expression for the steady state Z-magnetization of a proton pool A after saturation by a
    continuous wave with nutation frequency w_1 and absolute frequency difference dw_a, dw_b, and dw_c with three proton
    pools A, B, and C using the Bloch-McConnell equations.

    Pool A (typically water) has initial z-magnetization m_az_0, R1 value r1_a, R2 value r2_a, and exchange rates k_ab
    with pool B and k_ac with pool C. k_1a = k_ab + k_ac + r1_a and k_2a = k_ab + k_ac + r2_a. Pool B (metabolite) has
    initial z-magnetization m_bz_0, R1 value r1_b, R2 value r2_b, and exchange rate k_ba with pool A. k_1b = k_ba + r1_b
    and k_2b = k_ba + r2_b. Pool B (metabolite) has initial z-magnetization m_cz_0, R1 value r1_c, R2 value r2_c, and
    exchange rate k_ca with pool A. k_1c = k_ca + r1_c and k_2c = k_ca + r2_c.

    From Murase. Open J. Appl. Sci. 7(1):1-14. 2017.

    Returns
    -------
    m_az : str
        The analytical expression for the steady state Z-magnetization of pool A.

    """

    m_ax = sympy.Symbol('m_ax')
    m_ay = sympy.Symbol('m_ay')
    m_az = sympy.Symbol('m_az')
    m_bx = sympy.Symbol('m_bx')
    m_by = sympy.Symbol('m_by')
    m_bz = sympy.Symbol('m_bz')
    m_cx = sympy.Symbol('m_cx')
    m_cy = sympy.Symbol('m_cy')
    m_cz = sympy.Symbol('m_cz')

    w_1 = sympy.Symbol('w_1')

    m_az_0 = sympy.Symbol('m_az_0')
    dw_a = sympy.Symbol('dw_a')
    r1_a = sympy.Symbol('r1_a')
    k_ab = sympy.Symbol('k_ab')
    k_ac = sympy.Symbol('k_ac')
    k_1a = sympy.Symbol('k_1a')
    k_2a = sympy.Symbol('k_2a')

    m_bz_0 = sympy.Symbol('m_bz_0')
    dw_b = sympy.Symbol('dw_b')
    r1_b = sympy.Symbol('r1_b')
    k_ba = sympy.Symbol('k_ba')
    k_1b = sympy.Symbol('k_1b')
    k_2b = sympy.Symbol('k_2b')

    m_cz_0 = sympy.Symbol('m_cz_0')
    dw_c = sympy.Symbol('dw_c')
    r1_c = sympy.Symbol('r1_c')
    k_ca = sympy.Symbol('k_ca')
    k_1c = sympy.Symbol('k_1c')
    k_2c = sympy.Symbol('k_2c')

    solution = sympy.solve((-k_2a*m_ax + k_ba*m_bx + k_ca*m_cx - dw_a*m_ay,
                            k_ab*m_ax - k_2b*m_bx - dw_b*m_by,
                            k_ac*m_ax - k_2c*m_cx - dw_c*m_cy,
                            dw_a*m_ax - k_2a*m_ay + k_ba*m_by + k_ca*m_cy - w_1*m_az,
                            dw_b*m_bx - k_2b*m_by + k_ab*m_ay - w_1*m_bz,
                            dw_c*m_cx - k_2c*m_cy + k_ac*m_ay - w_1*m_cz,
                            -k_1a*m_az + k_ba*m_bz + k_ca*m_cz + w_1*m_ay + m_az_0*r1_a,
                            -k_1b*m_bz + k_ab*m_az + w_1*m_by + m_bz_0*r1_b,
                            -k_1c*m_cz + k_ac*m_az + w_1*m_cy + m_cz_0*r1_c),
                           (m_ax, m_ay, m_az, m_bx, m_by, m_bz, m_cx, m_cy, m_cz))

    return solution[m_az]


#@list_as_numpy
def bloch_mcconell_continuous_wave_n_pools_analytical(t, dw, b_1, w, t1, t2, c, k, b_0=3.0, db_0=0.0, w_c=None,
                                                      m_az_0=1.0):

    """Computes the Z-magnetization of a reference proton pool at times t after saturation by a continuous wave with
    amplitude b_1 and frequency w_c + dw for n exchangeable proton pools in star-type configuration immersed in a static
    field with amplitude b_0 + db_0 using the Bloch-McConnell equations.

    The reference pool 1 (index 0) has an initial z-magnetization m_az_0. The pools have resonance frequencies w, T1
    values t1, T2 values t2, and exchangeable proton concentrations c. Pools 2:n have exchange rates k with pool 1.

    Parameters
    ----------
    t : ndarray
        The saturation times [s].
    dw : ndarray
        The saturation frequency offsets [Hz].
    b_1 : float
        The amplitude of the continuous RF pulse B1 [T].
    w : ndarray
        The resonance frequencies of the n pools [ppm].
    t1 : ndarray
        The T1 values of the n pools [s].
    t2 : ndarray
        The T2 values of the n pools [s].
    c : ndarray
        The concentrations of exchangeable protons of the n pools [M.].
    k : ndarray
        The exchange rates from pools 2:n to pool 1 [Hz].
    b_0 : float, optional
        The reference static field value [T]. Default: 3.0.
    db_0 : float, optional
        The reference static field offset value [T]. Default: 0.0.
    w_c : float, optional
        The center imaging frequency [Hz]. Default: PyCEST.constants.PROTON_GYROMAGNETIC_RATIO*b_0.
    m_az_0 : float, optional
        The initial Z-magnetization of pool 1 [T]. Default: 1.0.

    Returns
    -------
    m_az : ndarray
        The final Z-magnetization of pool 1 for the specified times (axis 0) and offset frequencies (axis 1) [T].

    """

    w_1, m_0, dw, k, r1, r2 = __bloch_mcconell_continuous_wave_n_pools_init(dw, b_1, b_0, db_0, w_c, m_az_0, w, t1, t2,
                                                                            c, k)
    m_az = np.zeros((len(t), len(dw)))

    for i in range(len(t)):
        for j in range(len(dw)):
            m_az[i, j] = __bloch_mcconell_continuous_wave_n_pools_analytical(t[i], w_1, m_0, dw[j], k, r1, r2)

    return m_az


#@list_as_numpy
def bloch_mcconell_continuous_wave_n_pools_analytical_parallel(t, dw, b_1, w, t1, t2, c, k, b_0=3.0, db_0=0.0, w_c=None,
                                                               m_az_0=1.0, threads=16):

    """Computes the Z-magnetization of a reference proton pool at times t after saturation by a continuous wave with
    amplitude b_1 and frequency w_c + dw for n exchangeable proton pools in star-type configuration immersed in a static
    field with amplitude b_0 + db_0 using the Bloch-McConnell equations. This implementation is parallelized over a
    number of threads threads.

    The reference pool 1 (index 0) has an initial z-magnetization m_az_0. The pools have resonance frequencies w, T1
    values t1, T2 values t2, and exchangeable proton concentrations c. Pools 2:n have exchange rates k with pool 1.

    Parameters
    ----------
    t : ndarray
        The saturation times [s].
    dw : ndarray
        The saturation frequency offsets [Hz].
    b_1 : float
        The amplitude of the continuous RF pulse B1 [T].
    w : ndarray
        The resonance frequencies of the n pools [ppm].
    t1 : ndarray
        The T1 values of the n pools [s].
    t2 : ndarray
        The T2 values of the n pools [s].
    c : ndarray
        The concentrations of exchangeable protons of the n pools [M.].
    k : ndarray
        The exchange rates from pools 2:n to pool 1 [Hz].
    b_0 : float, optional
        The reference static field value [T]. Default: 3.0.
    db_0 : float, optional
        The reference static field offset value [T]. Default: 0.0.
    w_c : float, optional
        The center imaging frequency [Hz]. Default: PyCEST.constants.PROTON_GYROMAGNETIC_RATIO*b_0.
    m_az_0 : float, optional
        The initial Z-magnetization of pool 1 [T]. Default: 1.0.
    threads : int, optional
        The number of parallel threads. Default: 16.

    Returns
    -------
    m_az : ndarray
        The final Z-magnetization of pool 1 for the specified times (axis 0) and offset frequencies (axis 1) [T].

    """

    w_1, m_0, dw, k, r1, r2, = __bloch_mcconell_continuous_wave_n_pools_init(dw, b_1, b_0, db_0, w_c, m_az_0, w, t1, t2,
                                                                             c, k)
    pool = multiprocessing.Pool(processes=threads)
    results = []

    for i in range(len(t)):
        for j in range(len(dw)):
            results.append(pool.apply_async(__bloch_mcconell_continuous_wave_n_pools_analytical, args=(t[i], w_1, m_0,
                                                                                                       dw[j], k, r1, r2)
                                            ))
    pool.close()
    pool.join()

    m_az = np.zeros((len(t), len(dw)))
    index = 0

    for i in range(len(t)):
        for j in range(len(dw)):
            m_az[i, j] = results[index].get()
            index += 1

    return m_az


def lorentzian_n_pools(dw, w, a, s, b_0=3.0, db_0=0.0, w_c=None):

    """Computes an approximation of the Z-magnetization of a reference proton pool after saturation by an RF pulse with
    frequency w_c + dw for n exchangeable proton pools immersed in a static field with amplitude b_0 + db_0 using a
    Lorentzian model.

    The pools have resonance frequencies w and are modelled by means of Lorentzian functions with amplitudes a and FWHM
    s.

    Parameters
    ----------
    dw : ndarray
        The saturation frequency offsets [Hz].
    w : ndarray
        The resonance frequencies of the n pools [ppm].
    a : ndarray
        The peak amplitudes of the n pools [T].
    s : ndarray
        The peak FWHM of the n pools [Hz].
    b_0 : float, optional
        The reference static field value [T]. Default: 3.0.
    db_0 : float, optional
        The reference static field offset value [T]. Default: 0.0.
    w_c : float, optional
        The center imaging frequency [Hz]. Default: None (PyCEST.constants.PROTON_GYROMAGNETIC_RATIO*b_0 is used).

    Returns
    -------
    m_az : ndarray
        The final Z-magnetization of pool 1 for the specified frequencies [T].

    """

    b_0 = b_0 + db_0
    w_0 = PROTON_GYROMAGNETIC_RATIO*b_0

    if w_c is None:
        w_c = w_0
        print('Warning: the central imaging frequency was not set. 1H resonance frequency will be used.')
    else:
        w_c = 2.0*np.pi*w_c
    dw = 2.0*np.pi*dw
    w_i = w_c + dw
    w = np.array([ppm_to_frequency(w[i], w_0) for i in range(len(w))])

    m = len(w_i)
    n = len(a)

    w_i = np.transpose(np.tile(w_i, (n, 1)))
    w = np.tile(w, (m, 1))
    a = np.tile(a, (m, 1))
    s = np.tile(s, (m, 1))

    m_az = 1.0-np.sum(a/(1.0 + 4.0*np.square((w_i-w)/s)), axis=1)

    return m_az


