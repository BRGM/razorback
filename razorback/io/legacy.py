""" some legacy function for reading data files
"""


import os

import numpy as np
import scipy.io


from .. import calibrations, signalset

__all__ = ['load_signal_from_mat']


# TODO: brut_data() is obsolete
def brut_data(fname, path='', ext=''):
    fname = fname + ext
    path = os.path.abspath(path)
    fname = os.path.join(path, fname)
    return np.loadtxt(fname)


def load_signal_from_mat(
    fname,
    E_names=('Ex', 'Ey'),
    B_names=('Hx', 'Hy'),
    Bremote_names=('Hxr', 'Hyr'),
    calib_template='tical_%s',
    remote_tag='Bremote',
):
    """ return a SignalSet loaded from a matlab file

    The matlab file should contains:

      - 'SampleFreqMob' : the sampling rate
      - the names of the metronix calibrations for magnetic signals
      - the raw signals, each in one array

    """
    # TODO: improve interface

    def load(db, name):
        return np.hstack(list(db[name].flat)).reshape(-1)

    db = scipy.io.loadmat(fname)

    sampling_rate = db['SampleFreqMob']
    data = [load(db, n) for n in E_names+B_names+Bremote_names]
    tags = {'E': [0, 1], 'B': [2, 3], remote_tag: [4, 5]}
    calib_E = [1., 1.]
    calib_B = [
        calibrations.metronix(db[calib_template % n][0], sampling_rate)
        for n in B_names+Bremote_names
    ]
    calib = calib_E + calib_B

    signal = signalset.SyncSignal(data, sampling_rate, calibrations=calib)
    return signalset.SignalSet(tags, signal)
