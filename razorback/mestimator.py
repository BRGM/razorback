""" Implementation of the M-estimate method.
"""


import sys
import numpy as np
from scipy import linalg

from .errors import NonConvergence


try:
    from functools import reduce
except ImportError:
    pass  # reduce is builtin in py2


__all__ = ['transfer_function', 'transfer_error']


def transfer_error(outputs, inputs, transfer, invalid_idx):
    """

    from Egbert, Booker (1986)

    """
    inputs = np.array(inputs, copy=False)
    outputs = np.array(outputs, copy=False)
    assert inputs.ndim == 2
    assert outputs.ndim == 2
    assert inputs.shape[1] == outputs.shape[1]
    assert len(invalid_idx) == len(outputs)

    n = outputs.shape[1]
    p = transfer.shape[1]
    r = np.sqrt(n-p) / n
    iiinv = np.sqrt(np.abs(np.diag(
        np.linalg.inv(inputs.dot(inputs.conj().T))
    )))

    error = []
    for line, T, ivid in zip(outputs, transfer, invalid_idx):
        nres = np.linalg.norm(line - T.dot(inputs))
        err = (nres / r / (n-len(ivid))) * iiinv
        error.append(err)

    return np.array(error)


def transfer_function(
    outputs, inputs,
    weights=(None,),
    init=None, invalid_idx=None,
    **options
):
    """ estimate the transfer function (tensor) beetwen inputs and outputs

    transfer, invalid_idx = transfer_function(outputs, inputs, ...)

        outputs = transfer.dot(inputs)

    Estimation is made using the M-estimate method.

    Parameters
    ----------
    outputs: list of P array of shape (N,)
    inputs: list of Q array of shape (N,)
    weights: list of weighting function
    init: array
        initial guess for transfer
    invalid_idx: None or 1d-array or list of P 1d-array
    options: options passed to m_estimate()
        - tol
        - maxit
        - eps

    Returns
    -------
    transfer: array of shape (P, Q)
        outputs = transfer.dot(inputs)
    invalid_idx: list of P 1d-array
        indices rejected from inputs and outputs during the estimation

    """
    # acc = lambda (t_p, iv_p), (t, iv): (t, iv_p + [iv])
    def acc(prev, new):
        (t_p, iv_p), (t, iv) = prev, new
        return (t, iv_p + [iv])

    if invalid_idx is None or not len(invalid_idx):
        invalid_idx = [()] * len(outputs)
    if not hasattr(invalid_idx[0], '__getitem__'):
        invalid_idx = [invalid_idx] * len(outputs)

    inputs = np.array(inputs, copy=False)
    outputs = np.array(outputs, copy=False)
    assert inputs.ndim == 2
    assert outputs.ndim == 2
    assert inputs.shape[1] == outputs.shape[1]
    assert len(invalid_idx) == len(outputs)

    if init is None:
        init = [None] * len(outputs)
    assert len(init) == len(outputs)

    transfer = []
    ivids = []
    for i, line in enumerate(outputs):
        ivid_0 = invalid_idx[i]
        iter_est = chain_m_estimate(line, inputs.T, weights, init[i], ivid_0, **options)
        tt, l_ivid = reduce(acc, iter_est, (None, []))
        transfer.append(tt)
        ivids.append(merge_invalid_indices(l_ivid))

    transfer = np.array(transfer)
    return transfer, ivids


def merge_invalid_indices(l_ivid):
    _max = lambda it: max(it) if len(it) else 0
    Nmax = sum(map(_max, l_ivid)) + 1
    keep = np.arange(Nmax)
    for li in l_ivid:
        keep = np.delete(keep, li)
    return np.delete(np.arange(Nmax), keep)


def chain_m_estimate(
    e, b, weightings,
    z0=None, invalid_idx=None,
    **options
):
    """ chains M-estimation with a sequence of weightings

    Iterator of pair (z_est, invalid_idx) returned by m_estimate().

    At each step, the previous z_est and invalid_idx are given as starting
    point for the new call of m_estimate().
    z0 and invalid_idx parameters are used is used for the call of m_estimate().

    options are m_estimate() options:
        - tol
        - maxit
        - eps

    See m_estimate() for details.
    """
    z = z0
    ii = invalid_idx

    yield z, ii

    for it, weight_func in enumerate(weightings, start=1):
        e = eliminate(e, ii)
        b = eliminate(b, ii)

        if len(e) == 0:
            msg = (
                "All indices have been excluded, "
                "step %d cannot be processed."
            ) % it
            raise ValueError(msg)

        try:
            z, ii = m_estimate(e, b, z, weight_func, None, **options)
        except NonConvergence as err:
            name = getattr(weight_func, 'func_name', '1')
            msg = "while processing step %d (weighting=%s)." % (it, name)
            err.args += (msg,)
            # raise err, None, sys.exc_info()[2]
            raise

        yield z, ii


def m_estimate(
    e, b, z0, weight_func, invalid_idx,
    tol=1e-2, maxit=100, eps=1e-14,
):
    """ M-estimate of z for e=b.z

    weight_func = None -> least square estimate
    invalid_idx = None -> no invalid indices, equivalent to []

    Parameters
    ----------
    e : array of shape (N,)
    b : array of shape (N, 2)
    weight_func : None or
                  function returning array of shape (N,) or None
                  None means identity
    invalid_idx : array of indices (<N)
                  indices of e and b ignored by the estimation
    tol : float
          tolerance for convergence
          tested again the relative change of the weighted residual square sum
    maxit : int
            maximum iteration
    eps : float
          indices of the weight elements below eps are rejected
          and stored in the returned invalid_idx array

    Returns
    -------
    z_estimate : array (2,) estimating z
    invalid_idx_update  : array of rejected indices (see eps)
                          include indices of invalid_idx

    """
    if invalid_idx is None:
        invalid_idx = []
    invalid_idx = np.array(invalid_idx, dtype=int, copy=False)

    # do least square if no weight function
    if weight_func is None:
        if len(invalid_idx):
            weight = np.ones(len(e))
            weight[invalid_idx] = 0
        else:
            weight = None
        return wlse(e, b, weight), invalid_idx

    z = z0
    residual = e - b.dot(z)
    wresid_prev = wresid = None

    for it in range(maxit):
        weight, invalid_idx = _eval_weight(
            it, residual, e, b, invalid_idx, weight_func, eps
        )
        z = wlse(e, b, weight)
        residual = e - b.dot(z)

        wresid_prev = wresid
        wresid = H(residual).dot(dotdiag(weight, residual))
        wresid = np.max(abs(wresid))
        if wresid_prev is not None:
            change = abs((wresid - wresid_prev) / wresid_prev)
            if change < tol:
                break
    else:
        raise NonConvergence("failed to converge (maxit=%d)." % maxit)

    return z, invalid_idx


def _eval_weight(it, residual, e, b, invalid_idx, weight_func, eps):
    residual[invalid_idx] = np.nan
    weight = weight_func(it, residual, e, b, invalid_idx)
    if weight is not None:
        assert weight.ndim == 1
        weight[invalid_idx] = 0.
        (invalid_idx,) = np.nonzero(np.isclose(0, weight, 0, eps))
    return weight, invalid_idx


def eliminate(arr, idx):
    " friendly wrapper around np.delete "
    if idx is None or len(idx) == 0:
        return arr
    return np.delete(arr, idx, axis=0)


def H(arr):
    "return Hermitian transpose of arr"
    return arr.T.conjugate()


def dotdiag(d, x):
    "return diag(d).dot(x)"
    if d is None:
        return x

    if np.ndim(d) == 0:
        return d * x

    assert d.ndim == 1, "diagonal array dimension must be 1"
    assert x.ndim in (1, 2), "x array dimension must be 1 or 2"
    d = d.reshape((-1,) + (1,) * (x.ndim - 1))
    return d * x


def wlse(x, y, weight=None):
    """ weighted least square estimate

    estimate z such as: x ~= y.dot(z)

    z_est = (y* . w . y)^-1 . y* . w . x
    with w = diag(weight)

    Parameters
    ----------
    x : array of shape (N, P)
    y : array of shape (N, Q)
    weight : array of shape (N,) or None
             None means identity

    Returns
    -------
    z_est : array of shape (Q, P)
    """
    A = H(y).dot(dotdiag(weight, y))
    rhs = H(y).dot(dotdiag(weight, x))
    return linalg.solve(A, rhs)
