import matplotlib.pyplot as plt
import numpy as np
from PyCEST.constants import PROTON_WATER_CONCENTRATION
from PyCEST.fitting import z_spectrum_fit
from PyCEST.simulation import bloch_mcconell_continuous_wave_2_pools_steady_state, \
    bloch_mcconell_continuous_wave_n_pools_analytical_parallel


def z_spectrum_plot_example():

    """Examples of Z-spectrum plots.

    See Khlebnikov et al. Sci. Rep. 9:1089. 2019.

    """

    # Imaging parameters
    t = np.arange(0.0, 31.0, 1.0)     # The saturation times [s].
    dw = np.arange(-896, 897)         # The saturation frequency offsets [Hz].
    b_1 = 1.0e-6                      # The amplitude of the continuous RF pulse B1 [T].
    b_0 = 3.0                         # The reference static field value [T].
    db_0 = 0.000898                   # The reference static field offset value [T]. Typically return by a B0 map.
    w_c = 127770676.0                 # The center imaging frequency [Hz]. Typically found in the DICOM header.

    # Pool A: Water
    m_az_0 = 1.0                      # The initial Z-magnetization of pool A [T].
    w_a = 0.0                         # The resonance frequency of pool A [ppm].
    t1_a = 4.0                        # The T1 value of pool A @3T [s].
    t2_a = 2.0                        # The T2 value of pool A @3T [s].
    c_a = PROTON_WATER_CONCENTRATION  # The concentration of exchangeable protons of pool A (water) [M.].

    # Pool B: Gln
    w_b = 2.87                        # The resonance frequency of pool B [ppm].
    t1_b = 4.0                        # The T1 value of pool B @3T [s]. Not very sensitive.
    t2_b = 13.8e-3                    # The T2 value of pool B @3T [s].
    c_b = 0.2                         # The concentration of exchangeable protons of pool B [M.].
    k_b = 49.0                        # The exchange rate from pool B to pool A [Hz].

    # Let us compute the steady state Z-spectrum
    z_spectrum = bloch_mcconell_continuous_wave_2_pools_steady_state(dw, b_1, t1_a, t2_a, w_b, t1_b, t2_b, c_b, k_b,
                                                                     b_0=b_0, db_0=db_0, w_c=w_c, m_az_0=m_az_0,
                                                                     w_a=w_a, c_a=c_a)

    plt.figure()
    plt.plot(dw, z_spectrum, 'k')

    # Let us now compute the dynamic Z-spectrum using the n-pool analytical implementation
    z_spectrum = bloch_mcconell_continuous_wave_n_pools_analytical_parallel(t, dw, b_1, [w_a, w_b], [t1_a, t1_b],
                                                                            [t2_a, t2_b], [c_a, c_b], [k_b], b_0=b_0,
                                                                            db_0=db_0, w_c=w_c, m_az_0=m_az_0,
                                                                            threads=16)

    plt.plot(dw, z_spectrum[5, :], 'g')   # Steady-state is not reached yet after a 5 s saturation.
    plt.plot(dw, z_spectrum[-1, :], 'r')  # Steady-state is now reached after a 30 s saturation.
    plt.show()


def z_spectrum_fit_example():

    z = np.load('./data/example/z.npy')    # The measured Z-magnetization of pool A [T]. Loaded from example data.
    dw = np.load('./data/example/dw.npy')  # The saturation frequency offsets [Hz]. Loaded from example data.
    b_1 = 1.0e-6                           # The amplitude of the continuous RF pulse B1 [T].
    b_0 = 3.0                              # The reference static field value [T].
    db_0 = 0.000898                        # The true reference static field offset value [T]. Will be fitted here.
    w_c = 127770676.0                      # The center imaging frequency [Hz]. Typically found in the DICOM header.

    # Pool A: Water
    m_az_0 = 1.0                           # The initial Z-magnetization of pool A [T].
    w_a = 0.0                              # The resonance frequency of pool A [ppm].
    t1_a = 4.0                             # The T1 value of pool A @3T [s].
    t2_a = 2.0                             # The T2 value of pool A @3T [s].
    c_a = PROTON_WATER_CONCENTRATION       # The concentration of exchangeable protons of pool A (water) [M.].

    # Pool B: Gln
    w_b = 2.87                             # The true resonance frequency of pool B [ppm]. Will be fitted here.
    t1_b = 4.0                             # The T1 value of pool B @3T [s]. Not very sensitive.
    t2_b = 13.8e-3                         # The T2 value of pool B @3T [s].
    c_b = 0.2                              # The true concentration of exchangeable protons of pool B [M.]. Will be
                                           # fitted here.
    k_b = 49.0                             # The true exchange rate from pool B to pool A [Hz]. Will be fitted here.

    # The non-fitted parameters of PyCEST.simulation.bloch_mcconell_continuous_wave_2_pools_steady_state. dw (the first
    # unspecified argument) will be used as xdata. The other unspecified arguments w_b, c_b, k_b and db_0 will be
    # fitted.
    params = {'b_1': b_1,
              't1_a': t1_a,
              't2_a': t2_a,
              't1_b': t1_b,
              't2_b': t2_b,
              'b_0': b_0,
              'w_c': w_c,
              'm_az_0': m_az_0,
              'w_a': w_a,
              'c_a': c_a}

    # The lower and upper bounds for w_b, c_b, k_b and db_0.
    bounds = ((2.7, 0.01, 10.0, 8e-4), (3.0, 1.0, 1000.0, 1e-3))

    p = z_spectrum_fit(dw, z, bloch_mcconell_continuous_wave_2_pools_steady_state, params, bounds=bounds)
    print(f'Fitted values: {p}, True values: {[w_b, c_b, k_b, db_0]}')


if __name__ == '__main__':

    #z_spectrum_plot_example()
    z_spectrum_fit_example()
