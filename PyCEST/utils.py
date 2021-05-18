def exchange_rate(k_0, k_a, k_b, ph, pk_w=14.0):

    """Computes the proton exchange rate from the spontaneous, acid, and base catalysis constants for a given pH value.

    From Khlebnikov et al. Sci. Rep. 9:1089. 2019.

    Parameters
    ----------
    k_0 : float
        The spontaneous catalysis exchange rate constant [Hz].
    k_a : float
        The acid catalysis exchange rate constant [Hz].
    k_b : float
        The base catalysis exchange rate constant [Hz].
    ph : float
        The pH value.
    pk_w: float, optional
        The ionisation constant of water at the considered temperature. Default: 14.

    Returns
    -------
    k : float
        The proton exchange rate [Hz].

    """

    k = k_0 + k_a*10.0**(-ph) + k_b*10.0**(ph-pk_w)

    return k


def frequency_to_ppm(w, w_0):

    """Computes the relative frequency shift in ppm from the absolute and reference frequencies.

    Parameters
    ----------
    w : float
        The absolute frequency [Hz].
    w_0 : float
        The reference frequency [Hz].

    Returns
    -------
    ppm : float
        The relative frequency shift [ppm].

    """

    ppm = (w-w_0)/(w_0*1e-6)

    return ppm


def ppm_to_frequency(ppm, w_0):

    """Computes the absolute frequency in Hz from the relative frequency shift in ppm and the reference frequency.

    Parameters
    ----------
    ppm : float
        The relative frequency shift [ppm].
    w_0 : float
        The reference frequency [Hz].

    Returns
    -------
    w : float
        The absolute frequency [Hz].

    """

    w = w_0*(1.0 + ppm*1e-6)

    return w
