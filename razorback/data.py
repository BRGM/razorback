""" helper functions to handle data shipped with the package
"""


import os


def data_path():
    " get the path to the data directory "
    return os.path.abspath(os.path.join(os.path.dirname(__file__), 'data/'))


class DataPath(object):
    """
    """
    
    def __init__(self):
        self.paths = []

    def register(self, path):
        path = os.path.abspath(path)
        assert os.path.isdir(path)
        if path not in self.paths:
            self.paths.append(path)

    def get(self, filename):
        for path in self.paths:
            res = os.path.join(path, filename)
            if os.path.exists(res):
                return res
        return None  # nothing found
