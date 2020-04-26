""" high level functions to compute transfer function estimates
"""


from __future__ import print_function

import os
import re
import string
import fnmatch

from collections import namedtuple

import itertools

import numpy as np

from .mestimator import transfer_function, transfer_error, merge_invalid_indices
from .fourier_transform import slepian_window


__all__ = ['impedance', 'compute_prefilter', 'tags_from_path']


ImpedanceResult = namedtuple('ImpedanceResult', 'impedance invalid_time error')


def impedance_mass_proc(
    data, remote_names,
    l_freq, l_interval,
    impedance_opts,
):
    """ massive processing with multiple remote
    all combination of remote

    TODO: doc

    data: SignalSet
        tags:
            'E' -> local elec
            'B' -> local magn
            others -> remotes

    remote_names: list of str
        tag names for the remotes magn

    """

    E = data.tags.E
    B = data.tags.B

    remote_combination = list(itertools.product(*[
        (None, e) for e in range(len(remote_names))
    ]))

    ptl_z, ptl_err = [], []
    l_rcomb = []
    for k, rindices in enumerate(remote_combination):
        rcomb = [data.tags[remote_names[e]]
                 for e in rindices if e is not None]
        Bremote = sum(rcomb, ())

        #indices = range(data.nb_channels)
        #data = data.select(*indices, E=E, B=B, Bremote=Bremote)

        #data = data.select_channels(E+B+Bremote)

        options = dict(impedance_opts)
        if Bremote:
            options['remote'] = 'Bremote'
            data.tags['Bremote'] = Bremote

        tl_z, tl_err = [], []
        for i, interval in enumerate(l_interval):
            n, m = map(len, [remote_combination, l_interval])
            print('%.1f%%' % ((100. * (k*m+i) / (n*m))))
            rdata = data.extract_t(*interval)
            #z, l_ivt, l_err = impedance(rdata, l_freq, **options)
            z, l_ivt, l_err = impedance(rdata, l_freq, **options)
            tl_z.append(z)
            tl_err.append(l_err)

        ptl_z.append(tl_z)
        ptl_err.append(tl_err)
        l_rcomb.append(rindices)

    return ptl_z, ptl_err, l_rcomb


def impedance(
    data, l_freq,
    weights=(None,), prefilter=None,
    fourier_opts=None,
    remote=None, remote_weights=None, remote_prefilter=None,
    tag_elec='E', tag_mag='B',
    mest_opts=None,
    real_pb=False,
):
    """
    TODO

    data: SignalSet
        tags:
            'E' -> local elec
            'B' -> local magn
            other -> remote

    prefilter: None or function(outputs[i], inputs) -> invalid_idx

    [TODO: section outdated]
    remote: str or dict
            if str then it is the same as {'name': remote_tag}.
            if dict then keys are 'name', 'weights' and 'prefilter'.
            'name' : the str tag of remote fields in data
            'weights' : the weighting function to use on stage 1
                        if missing, weights is used instead
            'prefilter' : the prefilter to use on stage 1
                          if missing, prefilter is used instead

    fourier_opts defaults : dict(Nper=8, overlap=.71, window=slepian_window(4))

    """
    default_fourier_opts = dict(Nper=8, overlap=.71, window=slepian_window(4))
    fourier_opts = fourier_opts if fourier_opts else {}
    fourier_opts = dict(itertools.chain(default_fourier_opts.items(), fourier_opts.items()))

    mest_opts = dict(mest_opts) if mest_opts else {}

    if real_pb:
        _transfer_function = transfer_function_real_prob
    else:
        _transfer_function = transfer_function

    if remote:
        remote_name = remote
        remote_weights = remote_weights or weights
        remote_prefilter = remote_prefilter or prefilter

    l_z, l_ivt, l_err = [], [], []
    for freq in l_freq:
        print(f"starting frequency {freq:g}")
        fail_at_first_stage = False
        coeffs, (l_Nw, l_Lw, l_shift) = data.fourier_coefficients(freq, **fourier_opts)
        e = [coeffs[i] for i in data.tags[tag_elec]]
        b = [coeffs[i] for i in data.tags[tag_mag]]
        ## First stage
        if remote:
            br = [coeffs[i] for i in data.tags[remote_name]]
            remote_ivid = apply_prefilter(b, br, remote_prefilter, None)
            try:
                T, ivT = _transfer_function(b, br, remote_weights,
                                            invalid_idx=remote_ivid, **mest_opts)
            # TODO: catch only NonConvergence and (pas assez de poids ???)
            except Exception as ex:
                print(ex)
                fail_at_first_stage = True
                # raise
            else:
                be = T.dot(br)
                ivid_1 = len(e) * [merge_invalid_indices(ivT)]
        else:
            be = b
            ivid_1 = None
        ## Second stage
        fail_at_second_stage = False
        if fail_at_first_stage:
            fail_at_second_stage = True
        else:
            ivid_2 = apply_prefilter(e, be, prefilter, ivid_1)
            try:
                z, ivid = _transfer_function(e, be, weights,
                                             invalid_idx=ivid_2, **mest_opts)
            # TODO: catch only NonConvergence and (pas assez de poids ???)
            except Exception as ex:
                print(ex)
                fail_at_second_stage = True
                # raise
        if fail_at_second_stage:
            z, ivid, ivt = np.array([[np.nan]*len(b)]*len(e)), None, ()
        else:
            ivt = []
            for ivid_line in ivid:
                ivt_line = np.empty(len(ivid_line))
                start = 0
                for s, (Nw, Lw, shift) in zip(data.signals, zip(*(l_Nw, l_Lw, l_shift))):
                    mask = ivid_line < Nw
                    ii = ivid_line[mask]
                    ivid_line = ivid_line[~mask] - Nw
                    res = (shift * ii + 0.5 * Lw) / s.sampling_rate + s.start
                    ivt_line[start:start+len(res)] = res
                    start += len(res)
                ivt.append(ivt_line)
            ivt = tuple(ivt)


        l_z.append(z)
        l_ivt.append(ivt)
        ## error estimate
        if fail_at_second_stage:
            err = np.nan * z
        else:
            err = transfer_error(e, be, z, ivid)
        l_err.append(err)

    # return np.array(l_z), l_ivt, np.array(l_err)
    return ImpedanceResult(np.array(l_z), l_ivt, np.array(l_err))


def prefilter_values(
    data, l_freq,
    prefilter,
    fourier_opts=None,
    remote=None,
    tag_elec='E', tag_mag='B',
    real_pb=False,
):
    default_fourier_opts = dict(Nper=8, overlap=.71, window=slepian_window(4))
    fourier_opts = fourier_opts if fourier_opts else {}
    fourier_opts = dict(itertools.chain(default_fourier_opts.items(), fourier_opts.items()))

    if real_pb:
        _transfer_function = transfer_function_real_prob
    else:
        _transfer_function = transfer_function

    l_values = []
    l_times = []
    for freq in l_freq:
        coeffs, (l_Nw, l_Lw, l_shift) = data.fourier_coefficients(freq, **fourier_opts)
        e = [coeffs[i] for i in data.tags[tag_elec]]
        b = [coeffs[i] for i in data.tags[tag_mag]]

        inputs = np.transpose(b)
        l_values.append([prefilter.value(line, inputs) for line in e])

        times = [ (shift * np.arange(Nw) + 0.5 * Lw) / s.sampling_rate + s.start
                  for s, (Nw, Lw, shift) in zip(data.signals, zip(*(l_Nw, l_Lw, l_shift))) ]
        l_times.append(np.concatenate(times))

    return l_values, l_times


def prefilter_values_mass_proc(
    data, remote_names,
    l_freq, l_interval,
    prefilter
):
    """
    """

    E = data.tags.E
    B = data.tags.B

    remote_combination = list(itertools.product(*[
        (None, e) for e in range(len(remote_names))
    ]))

    ptl_values = []
    ptl_times = []
    for k, rindices in enumerate(remote_combination):
        rcomb = [data.tags[remote_names[e]]
                 for e in rindices if e is not None]
        Bremote = sum(rcomb, ())

        options = dict()
        if Bremote:
            options['remote'] = 'Bremote'
            data.tags['Bremote'] = Bremote

        tl_values = []
        tl_times = []
        for i, interval in enumerate(l_interval):
            n, m = map(len, [remote_combination, l_interval])
            print('%.1f%%' % ((100. * (k*m+i) / (n*m))))
            rdata = data.extract_t(*interval)
            l_values, l_times = prefilter_values(rdata, l_freq, prefilter, **options)
            tl_values.append(l_values)
            tl_times.append(l_times)

        ptl_values.append(tl_values)
        ptl_times.append(tl_times)

    return ptl_values, ptl_times


def apply_prefilter(outputs, inputs, prefilter, invalid_idx):
    """ return new invalid_idx augmented by prefilter result

    prefilter: None or function(outputs[i], inputs) -> invalid_idx
    """
    if invalid_idx is None or not len(invalid_idx):
        invalid_idx = [()] * len(outputs)
    if not hasattr(invalid_idx[0], '__getitem__'):
        invalid_idx = [invalid_idx] * len(outputs)
    assert len(invalid_idx) == len(outputs)

    if prefilter is None:
        return list(invalid_idx)

    return [np.union1d(ivid, prefilter(line, np.transpose(inputs)))
            for line, ivid in zip(outputs, invalid_idx)]



def transfer_function_real_prob(outputs, inputs, **kwargs):
    q, n = np.shape(inputs)

    new_in = np.empty((2*q, 2*n))
    r_in, i_in = np.real(inputs), np.imag(inputs)
    new_in[0::2, 0::2] = r_in
    new_in[0::2, 1::2] = i_in
    new_in[1::2, 0::2] = -i_in
    new_in[1::2, 1::2] = r_in

    new_out = []
    for line in outputs:
        new_line = np.empty((2*n,))
        r_out, i_out = np.real(line), np.imag(line)
        new_line[0::2] = r_out
        new_line[1::2] = i_out
        new_out.append(new_line)

    new_z, ivid = transfer_function(new_out, new_in, **kwargs)

    z = new_z[..., 0::2] + 1j * new_z[..., 1::2]

    return z, ivid


def compute_prefilter(data, freq, prefilter, remote=None, fourier_opts=None):
    """
    """
    default_fourier_opts = dict(Nper=8, overlap=.71, window=slepian_window(4))
    fourier_opts = fourier_opts if fourier_opts else {}
    fourier_opts = dict(default_fourier_opts.items() + fourier_opts.items())

    if remote:
        inputs = data.select(*data.tags[remote])
        outputs = data.select(*data.tags.B)
    else:
        inputs = data.select(*data.tags.B)
        outputs = data.select(*data.tags.E)

    c_in, (Nw, Lw, shift) = inputs.fourier_coefficients(freq, **fourier_opts)
    c_out, (Nw, Lw, shift) = outputs.fourier_coefficients(freq, **fourier_opts)
    c_in = np.array(c_in, copy=False)
    c_out = np.array(c_out, copy=False)

    filter_value = np.array([prefilter.value(line, c_in.T) for line in c_out])
    times = (shift * np.arange(Nw) + 0.5 * Lw) /data.sampling_rate + data.start
    return times, filter_value


def tags_from_path(names, pattern, *tag_tpls):
    """ yield (name, tags) for each name in names

    name is skiped if it does not match the pattern.
    tags is a list of strings formed from tag_tpls by using the matching fields in name

    fields are marked with curly brackets by '{identifier}',
    identifier must be a valid identifier in python grammar.

    Some special characters are available:

        - '*'     matches everything in one directory path level
        - '**'    matches everything across directory path level
        - '?'     matches any single character
        - [seq]   matches any character in seq
        - [!seq]  matches any character not in seq


    Examples:

    >>> g = tags_from_path(['rep/A/X.txt', 'rep/B/Y.txt'], 'rep/{a}/{x}.txt', '{a}_{x}')
    >>> list(g)
    [('rep/A/X.txt', ['A_X']), ('rep/B/Y.txt', ['B_Y'])]

    more complex patterns are possible, like:

    >>> g = tags_from_path(names, 'path_{a}/to_{b}/my_{c}/file_{d}.txt', '{a}_{b}_{c}_{d}')
    >>> g = tags_from_path(names, 'path_{a}/to_{b}/my_{c}/file_{d}.txt', '{a}_{d}')
    >>> g = tags_from_path(names, 'path_{a}/*/*/file_{d}.txt', '{a}_{d}')
    >>> g = tags_from_path(names, 'path_{a}/**/file_{d}.txt', '{a}_{d}')
    >>> g = tags_from_path(names, '**_{d}.txt', '{d}')


    using tags_from_path() to build an inventory from a directory tree:

    >>> root = 'the/main/directory/'
    >>> pattern = '**/Set?/site{site}/{type}/meas*/*_T{channel}_BL*.ats'
    >>> tag_tpl =  'site{site}_{channel}_{type}'
    >>> files = (os.path.join(r, f) for r, _, fs in os.walk(root) if fs for f in fs)
    >>> inv = Inventory(
    ...     SignalSet({tag:0 for tag in tags}, rb.io.ats.load_ats([name], calibrations=None, lazy=True))
    ...     for name, tags in tags_from_path(files, pattern, tag_tpl)
    ... )

    """
    rep = _prepare_pattern(pattern)
    keys = list(_get_fields(pattern))

    for tpl in tag_tpls:
        if not set(keys).issuperset(_get_fields(tpl)):
            raise ValueError("pattern fields and tag_tpls fields must match")

    match = re.compile(rep).match
    for name in names:
        m = match(os.path.normpath(name))
        if m:
            values = m.groups()
            new_names = [tpl.format(**dict(zip(keys, values))) for tpl in tag_tpls]
            yield name, new_names


def _prepare_pattern(pattern):
    """ convert pattern from tags_from_path() to regex with groups
    """
    flag = '[$@flag@$]'
    singlestar = '[$@singlestar@$]'
    doublestar = '[$@doublestar@$]'
    p0 = '([_0-9A-Za-z]+?)'

    pattern = os.path.normpath(pattern)

    rep = re.sub('{.*?}', flag, pattern)
    rep = re.sub('\*\*', doublestar, rep)
    rep = re.sub('\*', singlestar, rep)
    rep = fnmatch.translate(rep)
    rep = re.sub(re.escape(flag), p0, rep)

    sub = '.*?'
    rep = re.sub(re.escape(doublestar), sub, rep)
    sub = '[^%s]*?' % re.escape(re.escape(os.path.sep))
    rep = re.sub(re.escape(singlestar), sub, rep)

    return rep


def _get_fields(pattern):
    """ get fields in a format string

    see str.format()
    if the field name is not an identifier, raise error
    """
    F = string.Formatter()
    for (literal_text, field_name, format_spec, conversion) in F.parse(pattern):
        if format_spec or conversion:
            raise ValueError("just '{%}' please" % field_name)
        if field_name is None:
            continue
        if not field_name:
            raise ValueError("no empty field please")
        if not re.match('[_a-zA-Z][_a-zA-Z0-9]*', field_name):
            raise ValueError("'{%s}' is not valid" % field_name)
        yield field_name
