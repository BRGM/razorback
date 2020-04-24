""" helper functions to handle data shipped with the package
"""


import pathlib


def default_data_path(target):
    " returns the default (internal) path to the data directory "
    return pathlib.Path(__file__).parent / 'data' / target


def data_path(target):
    " returns the path to the data directory "
    local = pathlib.Path().resolve() / 'data' / target
    if local.is_dir():
        return local
    return default_data_path(target)


def get_data_file(filename, dirname):
    """ returns valid path to data file

    Raises ValueError if not found.

    It seeks for a file names `filename` in different locations (in that order):
        1/ `data_path(dirname)`
        2/ `default_data_path(dirname)`

    """
    data_dirs = [data_path(dirname), default_data_path(dirname)]
    for directory in data_dirs:
        datafile = directory / filename
        if datafile.exists():
            return datafile
    raise ValueError(f"'{filename}' file not found in '{dirname}' data directory")
