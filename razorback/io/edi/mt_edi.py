import collections
from . import core
from .core import Block
from ...mt_proc import MTProc, Result, FTParam, ProcParam, Run, Station, Sensor


__all__ = ['load_edi', 'write_edi']


def build_edi_data():
    data = collections.defaultdict(lambda: collections.defaultdict(list))
    data['']['INFO'] = ''
    return data


def load_edi(filename):
    """ load edi file into a MTProc object.
    """
    pass


def write_edi(filename, mtproc):
    """ write a MTProc object into a edi file.
    """

    # TODO  everything!

    data = build_edi_data()
    p = mtproc

    ## MTSECT
    ##
    n = len(p.result.frequencies)
    rot = [p.result.rotation] * n
    ## FREQ
    data['MTSECT']['FREQ'].append(Block({}, p.result.frequencies))
    ## Z
    data['MTSECT']['ZROT'].append(Block({}, rot))
    data['MTSECT']['ZXXR'].append(Block({'ROT': 'ZROT'}, zxxr))
    data['MTSECT']['ZXXI'].append(Block({'ROT': 'ZROT'}, zxxi))
    data['MTSECT']['ZXYR'].append(Block({'ROT': 'ZROT'}, zxyr))
    data['MTSECT']['ZXYI'].append(Block({'ROT': 'ZROT'}, zxyi))
    data['MTSECT']['ZYXR'].append(Block({'ROT': 'ZROT'}, zyxr))
    data['MTSECT']['ZYXI'].append(Block({'ROT': 'ZROT'}, zyxi))
    data['MTSECT']['ZYYR'].append(Block({'ROT': 'ZROT'}, zyyr))
    data['MTSECT']['ZYYI'].append(Block({'ROT': 'ZROT'}, zyyi))

    core.write(filename, data)
