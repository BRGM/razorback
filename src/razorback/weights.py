""" Weight functions for working with mestimator.transfer_function()
"""


import numpy as np
from scipy import linalg

from .mestimator import H, dotdiag


__all__ = [
    'Weights', 'Huber', 'Thomson', 'BoundedInfluenceStep',
    'mest_weights', 'least_square', 'bi_weights',
]


ignore_overflow = np.errstate(over='ignore')
ignore_invalid = np.errstate(invalid='ignore')
ignore_divide = np.errstate(divide='ignore')


class Weights(object):
    """ pretty weights sequence
    """
    def __init__(self, weights, name=None):
        self._weights = tuple(weights)
        self.__name__ = str(name) if name else None

    def __repr__(self):
        if self.__name__:
            return str(self.__name__)
        lname = (getattr(e, '__name__', str(e)) for e in self)
        return '(%s)' % ', '.join(lname)

    def __iter__(self):
        return iter(self._weights)

    def __getitem__(self, idx):
        return self._weights[idx]

    def __len__(self):
        return len(self._weights)


class Huber(object):
    def __init__(self, alpha=1.5):
        self._alpha = alpha

    def __str__(self):
        return 'Huber(%g)' % self._alpha

    def __call__(self, it, residual, e, b, invalid_idx):
        x = vnorm(residual)
        scale = mad_scale(x, is_absolute=True)
        with ignore_invalid:
            result = np.minimum(1, (self._alpha * scale) / x)
        return result


class Thomson(object):
    def __init__(self, xi=None):
        assert xi is None or xi > 0
        self._xi = xi
        self._data = {}

    def __str__(self):
        if self._xi is None:
            return 'Thomson()'
        return 'Thomson(%g)' % self._xi

    def __call__(self, it, residual, e, b, invalid_idx):
        xi = self._xi
        if xi is None:
            xi = (2 * np.log(2 * len(residual))) ** .5

        x = vnorm(residual)
        if it == 0:
            scale = self._data['scale'] = mad_scale(x, is_absolute=True)
        else:
            scale = self._data['scale']
        x /= scale
        return expexp(x, xi)


mest_weights = Weights([None, Huber(), Thomson()], "M_estimator_weights")
least_square = Weights([None], "least_square_weights")


class BoundedInfluenceStep(object):
    def __init__(self, weight_func, lower, khi):
        self._weight_func = weight_func
        self._lower = lower
        self._khi = khi
        self._prev = {}

    def __str__(self):
        return 'BoundedInfluenceStep(%s, %g, %g)' % (
            getattr(self._weight_func, '__name__', str(self._weight_func)),
            self._lower, self._khi
        )

    def __call__(self, it, residual, e, b, invalid_idx):
        _prev = self._prev
        if it == 0:
            _prev['it'] = -1
            _prev['bi_weight'] = 1.
            _prev['trace_bi_weight'] = len(b)
            _prev['hat_weight'] = 1.

        assert it == 1 + _prev.get('it', -np.inf)

        _, p = b.shape
        weight = self._weight_func(it, residual, e, b, invalid_idx)
        weight[invalid_idx] = 0.

        A = H(b).dot(dotdiag(_prev['bi_weight'], b))
        h = _prev['bi_weight'] * np.sum(linalg.solve(A, H(b)) * b.T, axis=0).real
        y = _prev['trace_bi_weight'] * h / p
        with ignore_divide:
            hat_weight = _prev['hat_weight'] * expexp(y, self._khi) * expexp(np.log(y), np.log(self._lower))

        bi_weight = weight * hat_weight

        _prev['it'] = it
        _prev['bi_weight'] = bi_weight
        _prev['trace_bi_weight'] = np.sum(bi_weight)
        _prev['hat_weight'] = hat_weight

        return bi_weight


def bi_weights(reject_prob, n_step, dim=2):
    """
    reject_prob : rejection probability
                    smaller value means less leverage point filtering
    n_step      : number of intermediate steps
    dim         : dimension (dim=2 for MT)
    """
    from scipy.special import gammaincinv
    lower = gammaincinv(dim, .5 * reject_prob) / dim
    upper = gammaincinv(dim, 1-.5 * reject_prob) / dim

    res = [BoundedInfluenceStep(Huber(), lower*2**-i, upper*2**i)
           for i in range(n_step)]
    res = [None] + res[::-1]
    res.append(BoundedInfluenceStep(Thomson(), lower, upper))
    return Weights(res, 'bi_weights(%g, %g, %g)' % (reject_prob, n_step, dim))


def mad_scale(residual, is_absolute=False, sigma_mad=0.44845):
    x = residual if is_absolute else vnorm(residual)
    scale = median_absolute_deviation(x, is_absolute) / sigma_mad
    return scale


def median_absolute_deviation(arr, is_absolute=False):
    if not is_absolute:
        arr = vnorm(arr)
    med = np.nanmedian(arr)
    with ignore_invalid:
        abs_dev = vnorm(arr - med)
    return np.nanmedian(abs_dev)


def expexp(x, a):
    C = np.exp(-a ** 2)
    with ignore_overflow:
        result = np.exp(C - np.exp(a * (x - a)))
    return result


def m_expexp(x, a):
    C = np.exp(-np.exp(-1./a))
    with ignore_overflow:
        result = (1. - np.exp(-np.exp((x - a) / a))) / C
    return result


def vnorm(x, ord=None):
    """ vectorized norm

    Compute the norm of each x[i], using np.linalg.norm().

    Parameters
    ----------
    x : array_like
        shape is (N, ...)
    ord : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
        see np.linalg.norm()

    Returns
    -------
    norms : ndarray
        shape is (N,)
    """
    if np.ndim(x) == 1 and ord is None:
        return abs(x)

    count = len(x)
    dtype = float
    it = (np.linalg.norm(v, ord=ord) for v in x)
    return np.fromiter(it, dtype, count)
