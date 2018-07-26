

import struct
import datetime
import numpy as np

from ..signalset import SyncSignal, SignalSet, Tags
from .binary_file_array import FileArrayProxy


__all__ = ['load_tsn']


class BinaryFileArray_24bits_to_i4(FileArrayProxy):
    def __init__(self, filename, offset, size, nb_channels):
        super().__init__(filename, offset, (nb_channels, size), '<i4')

    def extract(self, index):
        filename, offset = self.source
        dtype = self.dtype
        nb_channels, size = self.shape
        index_channel, idx = index

        ## lecture d'un entier sur 24 bits = 3 bytes
        ## on le prends comme un i4 avec le byte faible à 0
        ## puis on "décale" de 8 bits (= division par 256)
        n = nb_channels * (idx.stop - idx.start)
        bytesize = 3  # 24 bits
        offset += nb_channels * bytesize * idx.start
        arr4 = np.empty((n, 4), dtype='b', order='C')
        arr4[:, :1] = 0
        with open(filename, 'rb') as f:
            f.seek(offset)
            arr4[:, 1:] = np.fromfile(f, dtype='b', count=n*bytesize
                                     ).reshape(n, bytesize)
        arr = arr4.view('<i4') >> 8
        assert len(arr) == n
        arr.shape = (-1, nb_channels)
        return arr[::idx.step, index_channel].T


def load_tsn(filename, calibrations=None, lazy=False):
    """ return a SignalSet from a .TSn file

    channel tags are ['Ex', 'Ey', 'Hx', 'Hy', Hz']

    """
    tags = Tags(5, Ex=0, Ey=1, Hx=2, Hy=3, Hz=4)
    signals = [
        SyncSignal(arr, sample_rate, start, calibrations)
        for arr, sample_rate, start in read_tsn_samples(filename, lazy)
    ]
    return SignalSet(tags, *signals)


def read_tsn_samples(filename, lazy=False):
    """ return (sample_array, sampling_rate, start_time)

    sampling_rate shape is (nb_channels, nb_time)
    sampling_rate in Hz
    start_time in second (POSIX time)
    """

    for header in read_tsn_headers(filename):
        sampling_rate = header['sample_rate']
        start = header['utc_timestamp']
        header_length = header['tag_length']
        sample_length = header['number_of_scans']
        nb_channels = header['number_of_channels']
        offset = header['offset']

        bf_arr = BinaryFileArray_24bits_to_i4(
            filename=filename,
            offset=offset + header_length,
            size=sample_length,
            nb_channels=nb_channels,
        )
        if lazy:
            import dask.array as da
            # TODO: chunks could be smaller, eg when file is too big for memory
            chunks = sample_length
            arr = da.from_array(bf_arr, chunks, name=filename, fancy=False)
        else:
            arr = bf_arr[:]
        yield arr, sampling_rate, start


def read_tsn_headers(filename):
    res = []
    offset = 0
    with open(filename, 'rb') as f:
        while f:
            header = _read_tsn_header(f, offset)
            if not header:
                break
            nb_data = header['number_of_channels'] * header['number_of_scans']
            data_size = 3  # bytes
            offset += data_size * nb_data
            offset += header['tag_length']
            res.append(header)
    return res


def _read_tsn_header(file, offset=0):
    byte_format = [
        ('utc_second', 'B'),
        ('utc_minute', 'B'),
        ('utc_hour', 'B'),
        ('utc_day', 'B'),
        ('utc_month', 'B'),
        ('utc_year', 'B'),
        ('utc_day_of_week', 'B'),
        ('utc_century', 'B'),
        ('serial_number', 'H'),
        ('number_of_scans', 'H'),
        ('number_of_channels', 'B'),
        ('tag_length', 'B'),
        ('status', 'B'),
        ('saturation', 'B'),
        ('_reserved_future', 'B'),
        ('sample_length', 'B'),
        ('sample_rate', 'H'),
        ('sample_rate_unit', 'B'),
        ('clock_status', 'B'),
        ('clock_error_mus', 'I'),
        # ('_reserved_0', '6c'),  # FIXME trop grand !
        ('_reserved_0', '4c'),
    ]

    file.seek(offset)
    b_header = file.read(32)

    if not b_header:
        return None

    assert len(b_header) == 32

    fmt = ''.join(f for (_, f) in byte_format)
    header = dict(
        (name, data) for ((name, _), data)
        in zip(byte_format, struct.unpack_from(fmt, b_header))
    )

    assert header['tag_length'] == 32

    header['offset'] = offset

    header['sample_rate'] /= {
        0: 1.0,
        1: 60.0,
        2: 3600.0,
        3: 86400.0,
    }[header['sample_rate_unit']]

    header['utc_timestamp'] = datetime.datetime(
        header['utc_century'] * 100 + header['utc_year'],
        header['utc_month'],
        header['utc_day'],
        header['utc_hour'],
        header['utc_minute'],
        header['utc_second'],
        tzinfo=datetime.timezone.utc,
    ).timestamp()

    return header
