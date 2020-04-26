""" Computation of the fourier coefficients on sliding windows.
"""


import numpy as np
from scipy import signal


__all__ = ['time_to_freq', 'slepian_window']


# TODO: verifier si on drop pas une fenetre de trop...


def slepian_window(tau):
    """ return the 'window function' for a slepian with parameter tau
    """
    def _slepian(N):
        return slepian(N, tau)
    return _slepian


def time_to_freq(data, sampling_freq, freq, Nper, overlap, window=None):
    """ Compute the fourier coefficients on sliding windows.

    Parameters
    ----------
    data : list of arrays
        the time sequences to transform
    sampling_freq : scalar
        the sampling frequency of the time sequences
    freq : scalar
        the frequency of the Fourier transform
    Nper : int
        number of period in each window
        window length  =  Nper / freq * sampling_freq
    overlap : scalar in [0;0.5]
        the overlap ration between windows
    window : array or function
        window function to apply before Fourier transform
        None means no windowing (default)

    Returns
    -------
    freq_data : list of arrays
        processing of each array of the data list
    discrete_window_data: tuple of integer
        data of the sliding discrete window
        discrete_window_data = (Nw, Lw, shift)
        Nw: number of windows
        Lw: length of the window
        shift: index shift beetwen windows

    """
    length = set(len(e) for e in data)
    assert len(length) == 1, "all data must have the same length"
    Ntot = length.pop()

    nf = np.true_divide(freq, sampling_freq)
    Nw, Lw, shift = discrete_window(Ntot, nf, Nper, overlap)

    if callable(window):
        window = window(Lw)
    assert window is None or window.shape == (Lw,)

    result = [
        ft(sliding_window(y, Nw, Lw, shift).T, freq, sampling_freq, window)
        for y in data
    ]

    return result, (Nw, Lw, shift)


def discrete_window(size, normalized_freq, Nper, overlap):
    """ parameters for a discrete sliding window

    return nb_window, size_window, shift

    """
   # assert 0 <= overlap <= .5
   # assert 0 < normalized_freq <= .5
    assert 0 <= overlap <= 1
    assert 0 < normalized_freq <= .5
    # Lw, _ = divmod(Nper, normalized_freq)
    Lw = np.ceil(Nper / float(normalized_freq))
    shift = int(Lw * (1-overlap))
    Nw, _ = divmod(size - Lw, shift)
    if Nw < 1:
        raise ValueError(
            "Number of window < 1. "
            "normalized_freq(=%g) is too small or Nper(=%g) is too big "
            "for size(=%g)."
            % (normalized_freq, Nper, size)
        )
    return int(Nw), int(Lw), shift


def sliding_window(arr, nb_window, size_window, shift):
    """ sliding window over arr

    arr must be 1d
    result is 2d: result[i] is the i-th window

    >>> sliding_window(np.arange(15), 4, 3, 2)
    array([[0, 1, 2],
           [2, 3, 4],
           [4, 5, 6],
           [6, 7, 8]])

    """
    assert arr.ndim == 1, "array must be 1D"
    min_ = (nb_window - 1) * shift + size_window
    assert len(arr) >= min_, "array is too small (min=%d)" % min_
    size = arr.itemsize
    return np.lib.stride_tricks.as_strided(
        arr, shape=(nb_window, size_window),
        strides=(shift*size, size)
    )


def slepian(N, tau, N_MAX=1000):
    """ return the slepian window of size N with main lobe end at +/- tau

    the result is normalized by its mean.

    """
    ##  NOTES
    ##  MemoryError si N grand
    ##  signal.slepian is bugged and need a factor 2 on its second arg
    ##  the factor 4 is here to get the half band width
    ##  scipy 1.1 introduce signal.windows.dpss which is more standard


    if N <= N_MAX:
        res = signal.windows.dpss(N, tau)
        res /= res.mean()
    else:
        a = float(N) / float(N_MAX)
        ref = slepian(N_MAX, tau)
        res = np.interp(np.arange(N), a * np.arange(N_MAX), ref)
    return res


def ft(y, freq, sampling_freq=1., window=None):
    """ return the complex Fourier coefficient of y at freq

    y.shape = (N_time,)
    or
    y.shape = (N_time, N_signal)

    """
    N = len(y)
    pulsation = 2 * np.pi * freq / sampling_freq
    x = np.exp((-1j * pulsation) * np.arange(N))
    if window is not None:
        x *= window
    return x.dot(y) / N * 2
