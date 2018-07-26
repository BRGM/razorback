""" Some prefilter function working with mestimator.transfer_function()
"""


import numpy as np


__all__ = ['bcoher_filter', 'BCoherFilter']


class BCoherFilter(object):
    """ the 'bcoher' filter
    """

    def __init__(self, coher_min, coher_max=1., size=10):
        self._coher_min = coher_min
        self._coher_max = coher_max
        self._size = size

    def __call__(self, e, b):
        coher = self.value(e, b)
        (ivid,) = np.where((coher < self._coher_min) | (coher > self._coher_max))
        return ivid

    def value(self, e, b):
        size = self._size
        coeff_det = self.coeff_determination
        breshape = lambda arr: arr[:len(arr)-(len(arr) % size)].reshape(
            (divmod(len(arr), size)[0], size) + (-1,) * (arr.ndim-1)
        )
        coher = np.zeros(len(e))
        queue = size + (len(e) % size)
        tmp = coeff_det(breshape(e)[:-1], breshape(b)[:-1])
        breshape(coher)[:-1] = tmp[:, None]
        coher[-queue:] = coeff_det(e[None, -queue:], b[None, -queue:])
        return coher

    def coeff_determination(self, x, y):
        yc = y.conjugate()
        A = np.einsum("ilj,ilk->ijk", yc, y)
        b = np.einsum("ikj,ik->ij", yc, x)
        z = np.linalg.solve(A, b)
        residual = np.einsum("ijk,ik->ij", y, z) - x
        norm2 = lambda arr: np.sum(np.abs(arr) ** 2, axis=1)
        res = np.sqrt(1 - norm2(residual) / norm2(x))
        return res


bcoher_filter = BCoherFilter
