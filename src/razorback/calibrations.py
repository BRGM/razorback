""" a collection of calibration functions.
"""


import os
import re
import numpy as np
import scipy.interpolate

from .data import get_data_file


__all__ = ['metronix']


METRONIX_DATA_PATH = "metronix_calibration"


def metronix(filename, sampling_rate, chopper_on_limit=512.):
    """ return calibration function for metronix devices

    Parameters
    ----------
    filename: string
        data file name
        resolved with data.get_data_file(filename, calibrations.METRONIX_DATA_PATH)
    sampling_rate: float
    chopper_on_limit: float [optional]
        threshold for activating chopper

    Returns
    -------
    calibration function: f(freq) -> calib

    Notes
    -----
    The chopper on/off switch is based on the sampling rate and the chopper limit:

        - sampling_rate <= chopper_on_limit -> chopper on
        - sampling_rate >  chopper_on_limit -> chopper off

    """
    def start_stop(lines, mark):
        pattern = r'\s+'.join(re.split(r'\s+', mark.lower()))
        start = next(i for (i, l) in enumerate(lines, 1)
                     if re.match(pattern, l.lower().strip()))
        stop = [i for (i, l) in enumerate(lines[start:], start)
                if not l.strip()]
        stop = (stop or [None])[0]
        return start, stop

    def version(filename):
        pattern = r".*(MFS\d\d).*"
        #
        name, _ = os.path.splitext(os.path.basename(filename))
        m = re.match(pattern, name)
        if m is None:
            raise ValueError("cannot find version of calibration file '%s'"
                             % filename)
        return m.group(1)

    def cal_mp(freq, module, phase):
        return freq * module * np.exp(1j * phase * np.pi/180.)

    def calibration(table, filename, chopper):
        freq = table[:, 0]
        calib = cal_mp(freq, table[:, 1], table[:, 2])
        tabuled = scipy.interpolate.interp1d(freq, calib, copy=False)

        vers = version(filename)
        freq_min = freq[0]
        mod_min = table[0, 1]
        alpha = {
            'MFS06': 4.0,
            'MFS07': 32.0,
        }.get(vers, None)
        if alpha is None:
            raise ValueError(
                "unknown version %r of calibration file %r"
                % (vers, filename)
            )

        def calib_func(f):
            if f < freq_min and chopper:
                phase = np.angle(f + 1j * alpha, deg=True)
                return cal_mp(f, mod_min, phase)
            return tabuled(f)

        return calib_func

    filename = get_data_file(filename, METRONIX_DATA_PATH)
    with open(filename, 'r') as file:
        lines = file.readlines()

    chopper = sampling_rate <= chopper_on_limit
    mark = 'Chopper On' if chopper else 'Chopper Off'
    start, stop = start_stop(lines, mark)
    calib = calibration(np.loadtxt(lines[start:stop]), filename, chopper)
    return calib


def phoenix(cts_file, level):
    """ return calibration functions for phoenix devices

    Parameters
    ----------
    cts_file: string
        cts file name
    level: integer
        sample rate level / file number
        corresponds to the 'n' of '.TSn' (eg: TS2, TS3, TS4)

    Returns
    -------
    list of calibration functions (one by channel): f(freq) -> calib

    """
    with open(cts_file) as f:
        (
            date, serial_num, fieldtype, nb_channels
        ) = filter(None, map(str.strip, f.readline().strip().split(',')))
    n = int(nb_channels)
    cols = range(2 + 2 * n)
    try:
        data = np.loadtxt(cts_file, delimiter=',', skiprows=1, usecols=cols)
    except Exception:
        data = np.loadtxt(cts_file, delimiter=',', skiprows=2, usecols=cols)
    data = data[data[:, 1] == level]
    assert data.size, f"no level {level} in {cts_file:r}"
    freq = data[:, 0]
    values = data[:, 2::2] + 1j * data[:, 3::2]
    calibs = [
        scipy.interpolate.interp1d(freq, values[:, i], copy=False)
        for i in range(n)
    ]
    return calibs
