"""
"""


import os
import warnings
import itertools
from collections import namedtuple

import numpy as np


__all__ = []


class BaseArrayProxy(object):
    """ base class for use with dask.array.from_array(..., fancy=False)

    subclasses should provide the method self.extract(index):
        - index: a tuple of slices, same size than self.shape
        - return: array corresponding to index

    """

    def __init__(self, source, shape, dtype, **kwds):
        dtype = np.dtype(dtype)
        shape = tuple(shape)
        assert all(isinstance(e, int) and e > 0 for e in shape)
        super().__init__(**kwds)
        self.source = source
        self.shape = shape
        self.dtype = dtype
        self.ndim = len(shape)

    def __array__(self, dtype=None, **kwargs):
        x = self[:]
        if dtype and x.dtype != dtype:
            x = x.astype(dtype)
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return x

    def __getitem__(self, index):
        slice_index, final_index = self._pure_slice_index(index)
        res = self.extract(slice_index)
        return res[final_index]

    def _pure_slice_index(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        if len(index) > len(self.shape):
            raise IndexError('too many indices')
        if len(index) < len(self.shape):
            index = index + (slice(None),) * (len(self.shape) - len(index))
        res = itertools.starmap(self._to_slice, zip(index, self.shape))
        slice_index, final_index = map(tuple, zip(*res))
        return slice_index, final_index

    def _to_slice(self, index, dim):
        """ convert index into a (idx_slice, idx_final) pair

        the pair of index is such that arr[index] == arr[idx_slice][idx_final]
        idx_slice is slice with integer values for start, stop and step
        idx_final is any kind of index

        """
        ## NOTE: self is not use here
        # case: slice
        if isinstance(index, slice):
            if index.step is None or index.step > 0:
                return slice(*index.indices(dim)), slice(None)
            else:
                # TODO optimize  negative step
                return slice(None), index
        # case: int
        elif int(index) == index:
            index = int(index)
            if index < 0:
                index = index + dim
            if not 0 <= index < dim:
                raise IndexError(
                    f"index {index} not in range for dimension {dim}"
                )
            return slice(index, index+1, 1), 0
        else:
            raise TypeError()

    def extract(self, index):
        raise NotImplementedError("should be overloaded in subclasses")


FileSource = namedtuple('FileSource', 'path offset')


class FileArrayProxy(BaseArrayProxy):
    """

    self.source = FileSource(...)

    """

    def __init__(self, filename, offset, shape, dtype, **kwds):
        filename = os.path.abspath(filename)
        if not os.path.isfile(filename):
            raise ValueError("no such file: {filename:r}")
        source = FileSource(path=filename, offset=offset)
        super().__init__(source=source, shape=shape, dtype=dtype)


class BinaryFileArray_1D(FileArrayProxy):
    def __init__(self, filename, offset, size, dtype, **kwds):
        dtype = np.dtype(dtype)
        size = self._get_real_size(filename, offset, size, dtype)
        assert isinstance(size, int) and size > 0
        shape = (size,)
        super().__init__(filename, offset, shape, dtype)

    @staticmethod
    def _get_real_size(filename, offset, size, dtype):
        # return size
        new_size, exact = divmod(os.stat(filename).st_size - offset, dtype.itemsize)
        if new_size < size:
            warnings.warn(
                f"file {filename} with offset {offset} and dtype {dtype}"
                f" has only {new_size} values (expected {size}).")
            return new_size
        else:
            return size

    def extract(self, index):
        (idx,) = index
        (size,) = self.shape
        filename, offset = self.source

        with open(filename, 'rb') as f:
            f.seek(offset + self.dtype.itemsize * idx.start)
            length = idx.stop - idx.start
            arr = np.fromfile(f, dtype=self.dtype, count=length)
            if len(arr) != length:
                # warnings.warn(f"while reading file {str(self.source)!r} : get {len(arr)} values but header indicates {length}")
                raise Exception(
                    f"while reading file {filename} with offset {offset}:"
                    f" get {len(arr)} values but was expected {length}"
                )
        return arr[::idx.step]


class DummyArrayProxy(BaseArrayProxy):
    def extract(self, index):
        class Dummy:
            def __getitem__(self, final_index):
                return index, final_index
        return Dummy()
