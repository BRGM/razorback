""" function to handle 'ats' files
"""


import os
import struct
import glob

import numpy as np

from ..signalset import SyncSignal
from .binary_file_array import BinaryFileArray_1D


__all__ = ['load_ats']


def load_ats_from(path, calibrations=None):
    """ return a SyncSignal from ats directory
    """
    path = os.path.abspath(path)
    filenames = glob.glob(os.path.join(path, '*.ats'))
    return load_ats(filenames, calibrations)


def load_ats(filenames, calibrations=None, lazy=False):
    """ return a SyncSignal from multiples ats files

    """
    # TODO: check for metronix calibration in headers

    assert not isinstance(filenames, str), "'filenames' should be a list of str, not a single str."
    
    samples = [read_ats_sample(f, lazy) for f in filenames]
    signals = [x for (x, _, _) in samples]
    sampling_rates = set(x for (_, x, _) in samples)
    starts = set(x for (_, _, x) in samples)

    assert len(starts) == 1
    assert len(sampling_rates) == 1

    start = starts.pop()
    sampling_rate = sampling_rates.pop()

    return SyncSignal(signals, sampling_rate, start, calibrations)


def read_ats_sample(filename, lazy=False):
    """ return (sample_array, sampling_rate, start_time)

    sampling_rate in Hz
    start_time in second (POSIX time)
    """
    header = read_ats_header(filename)
    sampling_rate = header['sampling_rate']
    start = header['start']
    header_length = header['header_length']
    sample_length = header['sample_length']
    lsbval = header['lsbval']

    bf_arr = BinaryFileArray_1D(
        filename=filename,
        offset=header_length,
        size=sample_length,
        dtype=np.int32,
    )
    if lazy:
        import dask.array as da
        # TODO: chunks could be smaller, eg when file is too big for memory
        chunks = sample_length
        arr = da.from_array(bf_arr, chunks, name=filename, fancy=False)
    else:
        arr = bf_arr[:]
    arr = lsbval * arr

    return arr, sampling_rate, start


def read_ats_header(filename):
    """ return ats file header as dict
    """
    byte_format = [
        ('header_length', 'h'),
        ('header_version', 'h'),
        ('sample_length', 'i'),
        ('sampling_rate', 'f'),
        ('start', 'i'),
        ('lsbval', 'd'),
        ('GMToffset', 'i'),
        ('Res1', 'i'),
        ('serial_number_ADU06', 'H'),
        ('serial_number_ADC_board', 'h'),
        ('channel_number', 'b'),
        ('Res2', 'b'),
        ('channel_type', '2s'),
        ('sensor_type', '6s'),
        ('sensor_serial_number', 'h'),
        ('x1', 'f'),
        ('y1', 'f'),
        ('z1', 'f'),
        ('x2', 'f'),
        ('y2', 'f'),
        ('z2', 'f'),
        ('E_field_dipole_length', 'f'),
        ('angle', 'f'),
        ('rho_probe', 'f'),
        ('DC_offset_voltage', 'f'),
        ('internal_gain_ampli', 'f'),
        ('Res3', 'i'),
        ('ADU_Lat', 'i'),
        ('ADU_Long', 'i'),
        ('ADU_Elev', 'i'),
        ('Lat_Long_TYPE', '1s'),
        ('add_coordinates', '1s'),
        ('ref_meridian', 'h'),
        ('xcoor', 'd'),
        ('ycoor', 'd'),
        ('gps_clock_status', '1s'),
        ('accuracy_GPS', 'b'),
        ('offset_UTC', 'h'),
        ('Res4a', 'i'),
        ('Res4b', 'i'),
        ('Res4c', 'i'),
        ('survey_header_filename', '12s'),
        ('type_of_meas', '4s'),
        ('logfile', '12s'),
        ('result_selftest', '2s'),
        ('Res5', 'h'),
        ('number_of_calib_freq', 'h'),
        ('length_of_freq_entry', 'h'),
        ('version_calib', 'h'),
        ('start_addres', 'h'),
        ('Res6', 'q'),
        ('cal_filename_ADU06', '12s'),
        ('datetime_calib', 'i'),
        ('cal_sensor_filename', '12s'),
        ('datetime_calib_sens', 'i'),
        ('powerline1', 'f'),
        ('powerline2', 'f'),
        ('Res7', 'q'),
        ('CSAMT_Tx_freq', 'f'),
        ('CSAMT_TS_blocks', 'h'),
        ('CSAMT_stacks', 'h'),
        ('CSAMT_blk_length', 'i'),
        ('Res8', 'i'),
        ('Client', '16s'),
        ('Contractor', '16s'),
        ('Area', '16s'),
        ('SurveyID', '16s'),
        ('Operator', '16s'),
        ('Res9', '112s'),
        ('Weather', '64s'),
        ('Comments', '512s'),
    ]

    try:
        with open(filename, 'rb') as f:
            b_length = f.read(2)
            (length,) = struct.unpack('H', b_length)
            f.seek(0)
            b_header = f.read(length)
    except Exception:
        raise Exception(f"unable to read header of ats file: {filename}")

    fmt = ''.join(f for (_, f) in byte_format)
    header = dict(
        (name, data) for ((name, _), data)
        in zip(byte_format, struct.unpack_from(fmt, b_header))
    )

    # some corrections
    header['ADU_Lat'] *= 1e-3 / 3600  # in Degres (from msec)
    header['ADU_Long'] *= 1e-3 / 3600  # in Degres
    header['ADU_Elev'] *= 1e-2  # in m

    return header
