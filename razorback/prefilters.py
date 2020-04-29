""" Some prefilter function working with mestimator.transfer_function()
"""


import numpy as np


__all__ = ['cod_filter', 'CoefficientOfDeterminationFilter']


class CoefficientOfDeterminationFilter(object):
    """ Filter based on the coefficient of determination

    """

    def __init__(self, cod_min, cod_max=1., size=10):
        self._cod_min = cod_min
        self._cod_max = cod_max
        self._size = size

    def __call__(self, e, b):
        cod = self.value(e, b)
        (ivid,) = np.where((cod < self._cod_min) | (cod > self._cod_max))
        return ivid

    def value(self, e, b):
        size = self._size
        coeff_det = self.coeff_determination
        breshape = lambda arr: arr[:len(arr)-(len(arr) % size)].reshape(
            (divmod(len(arr), size)[0], size) + (-1,) * (arr.ndim-1)
        )
        cod = np.zeros(len(e))
        queue = size + (len(e) % size)
        tmp = coeff_det(breshape(e)[:-1], breshape(b)[:-1])
        breshape(cod)[:-1] = tmp[:, None]
        cod[-queue:] = coeff_det(e[None, -queue:], b[None, -queue:])
        return cod

    def coeff_determination(self, x, y):
        yc = y.conjugate()
        A = np.einsum("ilj,ilk->ijk", yc, y)
        b = np.einsum("ikj,ik->ij", yc, x)
        z = np.linalg.solve(A, b)
        residual = np.einsum("ijk,ik->ij", y, z) - x
        norm2 = lambda arr: np.sum(np.abs(arr) ** 2, axis=1)
        res = np.sqrt(1 - norm2(residual) / norm2(x))
        return res


cod_filter = CoefficientOfDeterminationFilter
