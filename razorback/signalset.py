""" Classes to handle signals (raw data + meta data)
"""


import warnings
import itertools
from collections import Counter
from collections.abc import MutableMapping
from functools import reduce
from datetime import datetime
import fnmatch

import numpy as np

from .fourier_transform import time_to_freq


__all__ = ['SignalSet', 'SyncSignal', 'Tags', 'Inventory']


try:
    basestring
except NameError:
    basestring = str


def _tupleit(value):
    try:
        return tuple(value)
    except TypeError:
        return (value,)


class TagsBase(MutableMapping):
    " Base class for Tags "
    def __init__(self, indices, dct):
        self.__dict__.update(_tags={}, _indices=frozenset(indices))
        self.update(dct)

    @property
    def indices(self):
        return self._indices

    def __eq__(self, other):
        return (self._tags == getattr(other, '_tags', None)
                and self.indices == getattr(other, 'indices', None))

    __req__ = __eq__

    def __iter__(self):
        return iter(self._tags)

    def __len__(self):
        return len(self._tags)

    def __delitem__(self, key):
        del self._tags[key]

    def __getitem__(self, key):
        return self._tags[key]

    def __setitem__(self, key, value):
        assert isinstance(key, basestring)
        assert key[:1].isalpha()
        value = _tupleit(value)
        assert value, "at least one index must be given"
        if not all(v in self.indices for v in value):
            msg = "values %s should be in indices %s"
            raise ValueError(msg % (value, self.indices))
        self._tags[key] = value

    def __getattribute__(self, name):
        try:
            tags = super(TagsBase, self).__getattribute__('_tags')
        except AttributeError:
            tags = None
        if tags and (name in tags):
            return tags[name]
        return super(TagsBase, self).__getattribute__(name)

    def __setattr__(self, name, value):
        if name.startswith('_') or name in ['indices']:
            return super(TagsBase, self).__setattr__(name, value)
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            super(TagsBase, self).__delattr__(name)


class Tags(TagsBase):
    """
    Tags(n, name1=idx1, name2=idx2, ...)
    Tags(name1=idx1, name2=idx2, ...)

    Tags handle mappings between names and groups of indices.

    A Tags object behaves like a dict but its keys must be legal string names
    and its values are tuples of integers in a fixed range.
    New entries of a Tags object are converted in tuples if needed
    then their content is checked to belong in the fixed range.

    Item access can be emulated through attribute access:

      - tags.name1         -->  tags['name1']
      - tags.name1 = idx1  -->  tags['name1'] = idx1


    Tags also implements the union (|) operator according to the SignalSet class:

      - tags | signal  ->  SignalSet(tags, signal)


    See Also
    --------
    SignalSet

    """
    def __init__(self, n=None, **dct):
        if n is None:
            idx = sum(map(_tupleit, dct.values()), ())
            n = (1 + max(idx)) if idx else 0
        super(Tags, self).__init__(range(n), dct)

    def __or__(self, signal):
        return SignalSet(self, signal)

    def __repr__(self):
        items = sorted(self.items(), key=lambda x: (len(x[1]), x[0]))
        tags = ', '.join('%s=%s' % e for e in items)
        return '%s(%s, %s)' % (type(self).__name__, len(self.indices), tags)

    def __str__(self):
        return str(self._tags)

    # filter = fnmatch.filter
    def filter(self, *patterns):
        return tuple(set().union(*(fnmatch.filter(self, p) for p in patterns)))

    def filter_get(self, *patterns):
        return tuple(sorted(set().union(*(map(self.get, self.filter(*patterns))))))


class SignalSet(object):
    """ SignalSet(tags, signal1, signal2, ...)

    A tagged group of several synchronous signals.

    A SignalSet is meant to gather distinct runs of the same channels.
    Each run is stored in a SyncSignal and channels are handled with Tags.

    It is required that there is no time overlap between runs.

    tags may be a Tags object or a dict-like object that will be converted as Tags.


    **Building SignalSet from others**

        * SignalSet can be combined in two ways:

          - join(): same channels, union of time intervals
          - merge(): union of channels, intersection of time intervals

        * SignalSet can be reduced in three ways:

          - select_channels: same times, reduce channels
          - select_runs: same channels, reduce runs
          - extract_t(): same channels, reduce times (can have less runs)


    **Syntactic sugar**

        The merge/join operations can be performed with the intersection(&)/union(|) operators::

          - s1 & s2  -->  s1.merge(s2)
          - s1 | s2  -->  s1.join(s2)

        A fully tagged subset of channels can be obtained through attribute access using tag names::

          - s.Ex  -->  s.select_channels('Ex')

        Selecting channels/runs can be performed through indexing::

          - s[:]                                      -->  s
          - s['Ex']                                   -->  s.select_channels('Ex')
          - s[['Ex', 'Ey']]                           -->  s.select_channels(['Ex', 'Ey'])
          - s[:, 0]                                   -->  s.select_runs(0)
          - s[:, [0, 3]]                              -->  s.select_runs([0, 3])
          - s[:, s.sampling_rates == 1024]            -->  s.select_runs(s.sampling_rates == 1024)
          - s[['Ex', 'Ey'], s.sampling_rates <= 512]  -->  filtering on channels and sampling_rates


    Parameters
    ----------
    tags : Tags or dict
        will be converted as Tags, used to name one or more channels

    signal*i*: SyncSignal or SignalSet
        a group of synchronous signals,
        channels are identified by indices (SyncSignal) or by tags (SignalSet).


    See Also
    --------
    SyncSignal, Tags


    Example
    -------

    A SignalSet with 5 runs and 4 channels::

                    run 0    run 1    run 2    run 3   run 4

        channel 0   ~~~~~~~  ~~~~~~~  ~~~~~~~  ~~~~~~~ ~~~~~~~
        channel 1   ~~~~~~~  ~~~~~~~  ~~~~~~~  ~~~~~~~ ~~~~~~~
        channel 2   ~~~~~~~  ~~~~~~~  ~~~~~~~  ~~~~~~~ ~~~~~~~
        channel 3   ~~~~~~~  ~~~~~~~  ~~~~~~~  ~~~~~~~ ~~~~~~~

    There can't be any missing signals: each channel has a signal in each run.
    Each run is a SyncSignal containing 4 signals.

    Channels can be found by their indices but the tags allow to name groups of channels.
    If tags is set up as::

        tags = Tags(4, Ex=0, Ey=1, Hx=2, Hy=3, E=(0, 1), H=(2, 3))

    then one can retrieve one channel or a subset of channels with::

        s.Ex  ->  new SignalSet with only channel 0
        s.H   ->  new SignalSet with only channel 2 and 3

    """
    def __init__(self, tags, *signals):
        signals = [getattr(s, '_reorder_data', lambda t: s)(tags)
                   for s in signals]

        signals = sum((s.signals for s in signals), ())

        lengths = set(len(s.data) for s in signals)
        assert len(lengths) <= 1, "all signals must have the same length"

        l = sorted((s.interval for s in signals), key=lambda e: e[0])
        assert all(l[i][1]<=l[i+1][0] for i in range(len(l)-1)), "signals must not overlap"

        self._signals = tuple(sorted(signals, key=lambda s: s.start))
        self._tags = None
        self.tags = tags

    def __str__(self):
        ## signals
        titles = 'sampling', 'start', 'stop'
        sizes = 10, 19, 19
        sep = '  '
        date = lambda sec: '%s' % datetime.utcfromtimestamp(np.round(sec))
        rjust = lambda w, s: [_w.rjust(_s) for _w, _s in zip(w, s)]
        titles = rjust(titles, sizes)
        sizes = [len(t) for t in titles]
        title = sep.join(titles)
        sigs = [sep.join(rjust(['%10.5g' % s.sampling_rate, date(s.start),
                                date(s.stop)], sizes)) for s in self.signals]
        line = sep.join('-' * n for n in sizes)
        sig = '\n'.join([line, title] + sigs + [line])
        #
        ## tags
        pre, post, sep, width = 'tags: {', '}', ', ', 52
        items = sorted(self.tags.items(), key=lambda x: (len(x[1]), x[0]))
        items = ['%r: %s' % e for e in items]
        content = ['']
        for elem in items:
            if len(content[-1]) + len(sep) + len(elem) > width - len(pre):
                content.append('')
            content[-1] += elem + sep
        content = ('\n' + ' ' * len(pre)).join(content)
        content = content[:-len(sep)]
        tags = '%s%s%s' % (pre, content, post)
        #
        ## first line
        num = lambda n, w: '%d %s%s' % (n, w, 's'[:n>1])
        first_line = '%s: %s, %s' % (type(self).__name__,
                                     num(self.nb_channels, 'channel'),
                                     num(self.nb_runs, 'run'))
        #
        return "%s\n%s\n%s" % (first_line, tags, sig)

    def __repr__(self):
        res = '%s(%s, %%s)' % (type(self).__name__, self.tags)
        res = res % ', '.join(map(repr, self.signals))
        return res

    def __getattr__(self, name):
        try:
            return self.select_channels(name)
        except ValueError:
            return super(SignalSet, self).__getattribute__(name)

    @property
    def nb_channels(self):
        "number of channels"
        if not self.signals:
            if self.tags is None:
                return 0
            return len(self.tags.indices)
        return len(self.signals[0].data)

    @property
    def nb_runs(self):
        "number of runs"
        return len(self.signals)

    def __getitem__(self, idx):
        """
        s[channels] -> s.select_channels(channels)
        s[:, runs] -> s.select_runs(runs)
        s[channels, runs] -> s.select_channels(channels).select_runs(runs)
        """
        if not isinstance(idx, tuple):
            idx = (idx,)

        if not idx:
            return self
        elif len(idx) == 1:
            channels, = idx
            return self.select_channels(channels)
        elif len(idx) == 2:
            channels, runs = idx
            return self.select_channels(channels).select_runs(runs)
        else:
            raise IndexError('too many indices for SignalSet (1 or 2)')

    def _reorder_data(self, tags):
        """ return the equivalent SignalSet with given tags

        The new SignalSet signals list contains equivalent signals
        with reordered data.

        """
        if self.tags == tags:
            return self

        msg = 'incompatible tags: %s and %s' % (tags, self.tags)

        assert set(tags) == set(self.tags), msg
        assert all(len(tags[t]) == len(self.tags[t]) for t in tags), msg

        idx_new = sum((tags[t] for t in tags), ())
        idx_self = sum((self.tags[t] for t in tags), ())

        inv = {i: set() for i in idx_new}
        for si, ni in zip(idx_self, idx_new):
            inv[ni].add(si)
        assert all(len(v) == 1 for v in inv.values()), msg
        inv = {k: v.pop() for k, v in inv.items()}

        idx = range(self.nb_channels)
        signals = [
            SyncSignal(
                data=[s.data[inv[i]] for i in idx],
                calibrations=[s.calibrations[inv[i]] for i in idx],
                sampling_rate=s.sampling_rate,
                start=s.start,
            )
            for s in self.signals
        ]

        return type(self)(tags, *signals)

    @property
    def signals(self):
        " tuple of SyncSignal objects: the sorted sequence of runs"
        return self._signals

    @property
    def sampling_rates(self):
        " sampling rates of the runs "
        return np.array([s.sampling_rate for s in self.signals])

    @property
    def sizes(self):
        " sizes of the runs "
        return np.array([s.size for s in self.signals])

    @property
    def intervals(self):
        " intervals of the runs "
        return np.array([s.interval for s in self.signals])

    @property
    def starts(self):
        " starts of the runs "
        return np.array([s.start for s in self.signals])

    @property
    def stops(self):
        " stops of the runs "
        return np.array([s.stop for s in self.signals])

    @property
    def tags(self):
        " tags mapping names to channel indices "
        return self._tags

    @tags.setter
    def tags(self, value):
        n = self.nb_channels
        vidx = getattr(value, 'indices', None)
        if n and vidx is not None and not set(vidx).issubset(range(n)):
            msg = "incompatible tags"
            raise ValueError(msg)
        self._tags = Tags(n or None, **value)

    def select_channels(self, channels):
        """ return a SignalSet containing the selected channels

        `channels` can be one identifier or a sequence of identifiers.
        An identifier can be a tag name (str) or an index (integer).
        `channels` can also be a slice.

        """
        tagkeys = list(self.tags)  # a list to protect vs unhashable type
        msg = "unknown channel identifier: %r"

        if channels in tagkeys:
            channels = tuple(self.tags[channels])
        elif isinstance(channels, slice):
            channels = tuple(range(self.nb_channels)[channels])
        elif isinstance(channels, basestring):
            raise ValueError(msg % channels)
        else:
            channels = _tupleit(channels)

        indices = []
        for cid in channels:
            if cid in tagkeys:
                indices.extend(self.tags[cid])
            else:
                if cid not in self.tags.indices:
                    raise ValueError(msg % cid)
                indices.append(cid)

        indices = sorted(set().union(*map(_tupleit, indices)))
        idxmap = {e: i for i, e in enumerate(indices)}
        keep = set(indices).issuperset
        tags = {k: map(idxmap.get, _tupleit(v))
                for k, v in self.tags.items() if keep(v)}
        tags = Tags(len(indices), **tags)
        signals = [s.select(*indices) for s in self.signals]
        return SignalSet(tags, *signals)

    def select_runs(self, runs):
        """ return a SignalSet containing the selected runs

        `runs` can be an index (integer), an array of index,
        a slice or an array of bool.
        In each case, the behavior is similar to indexing a 1d numpy array.

        """
        indices = np.arange(self.nb_runs)[runs].flat
        return SignalSet(self.tags, *(self.signals[i] for i in indices))

    def join(self, *others):
        """ return a SignalSet joining different signal sets of the same channels

        alias: a | b -> a.join(b)

        Join is based on tag names: all the signal sets must have the same tag names.
        The resulting time intervals are the union of the given signal set intervals.

        It is possible to join with a SyncSignal (which has no tags), in this case
        the order of the data are supposed to match with the SignalSet tags.

        """
        return type(self)(self.tags, self, *others)

    __or__ = __ror__ = join

    def merge(self, *others):
        """ return a SignalSet merging channels of given signal sets

        alias: a & b -> a.merge(b)

        Merge is based on tag names: the given signal sets can't have common tag names.

        The resulting intervals are the common parts of all the given signal set intervals.

        Signal sets sharing an interval must be synchronous.

        """
        assert all(isinstance(o, type(self)) for o in others), "wrong type"
        seqs = (self,) + others

        fcount = sum((Counter(s.tags.keys()) for s in seqs), Counter())
        name_conflict = [f for f, n in fcount.items() if n > 1]
        assert not name_conflict, name_conflict

        shift = reduce(lambda c, s: c+[c[-1]+s.nb_channels], seqs, [0])
        shift, N = shift[:-1], shift[-1]
        tags = {t: [i+c for i in ii] for c, s in zip(shift, seqs)
                                     for t, ii in s.tags.items()}
        tags = Tags(N, **tags)

        def rec(d, f, seq):
            "recursively find common intervals"
            if not seq:
                yield d, f
                return
            for dh, fh in seq[0]:
                if fh <= d: continue
                if f <= dh: return
                for res in rec(max(d, dh), min(f, fh), seq[1:]): yield res
        
        seq_intervals = [s.intervals for s in seqs]
        start, stop = min(map(np.min, seq_intervals)), max(map(np.max, seq_intervals))
        intervals = list(rec(start, stop, seq_intervals))

        signals = []
        for start, stop in intervals:
            ss = [s.extract_t(start, stop) for s in seqs]
            assert all(len(s.signals) == 1 for s in ss)
            ss = [s.signals[0] for s in ss]
            sr = {s.sampling_rate for s in ss}
            # TODO: choose a strategy: permissive or strict
            if len(sr) != 1:
                continue
            # assert len(sr) == 1, "incompatible sampling rates"
            assert all(s.start == start for s in ss), (
                'same sampling and time overlap but clock not synchronized')
            sampling_rate = sr.pop()
            signals.append(SyncSignal(
                data=sum((s.data for s in ss), ()),
                calibrations=sum((s.calibrations for s in ss), ()),
                sampling_rate=sampling_rate,
                start=start,
            ))

        return type(self)(tags, *signals)

    __and__ = __rand__ = merge

    def extract_t(self, start, stop, exclude=False):
        """ return a SignalSet with reduced time intervals

        If exclude=False, all times outside the given interval are skipped.
        If exclude=True, all times inside the given interval are skipped.

        """
        if exclude:
            left = self.extract_t(self.starts.min(), start)
            right = self.extract_t(stop, self.stops.max())
            return left.join(right)

        signals = [s.extract_t(start, stop) for s in self.signals]
        signals = [s for s in signals if s.size > 1]
        return type(self)(self.tags, *signals)

    def fourier_coefficients(self, freq, Nper, overlap, window):
        """ compute the fourier coefficients at freq of sliding windows

        coeffs, discrete_window_data = signal.fourier_coefficients(freq, Nper, overlap, window)

        The result include the calibration functions specified in the signal set:

            calibrated_coeff = coeff / calib


        Parameters
        ----------
        freq : scalar
            the frequency of the Fourier transform
        Nper : int
            number of period in each window
            window length  =  Nper / freq * sampling_freq
        overlap : float
            must be in [0 ; 0.5]
            the overlap ratio between windows
        window : array or function
            window function to apply before Fourier transform
            None means no windowing

        Returns
        -------
        coeffs : list of arrays
            fourier coefficients of each signal of the signal set
        discrete_window_data: tuple of list of integer
            data of the sliding discrete window
            discrete_window_data = (l_Nw, l_Lw, l_shift)
            l_Nw: number of windows for each run
            l_Lw: length of the window for each run
            l_shift: index shift beetwen windows for each run

        """

        l_coeffs, l_windata = zip(*(
            ss.fourier_coefficients(freq, Nper, overlap, window)
            for ss in self.signals
        ))

        # TODO: a verifier
        coeffs = map(np.hstack, zip(*l_coeffs))
        coeffs = list(coeffs)

        l_Nw, l_Lw, l_shift = zip(*l_windata)

        return coeffs, (l_Nw, l_Lw, l_shift)

    def merge_consecutive_runs(self):
        """ return a new SignalSet where consecutive runs are merged into one.

        EXPERIMENTAL

        !!! calibrations taken from the first run

        """
        warnings.warn("SignalSet.merge_consecutive_runs() is experimental")
        res = SignalSet(self.tags)
        for sampling in np.unique(self.sampling_rates):
            ss = self.select_runs(self.sampling_rates == sampling)
            is_consecutive = np.isclose(ss.sizes[:-1]/sampling, ss.starts[1:] - ss.starts[:-1])
            for start, stop in _group_indices(is_consecutive):
                group = ss.select_runs(slice(start, stop))
                data = [np.concatenate([v.data[i] for v in group.signals])
                        for i in range(group.nb_channels)]
                res |= SyncSignal(data, sampling, group.starts[0], group.signals[0].calibrations)
        return res


def _group_indices(bool_arr):
    """
    bool_arr = [0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, ...]
    -> [(2, 5), (6, 7), (10, ?), ...]
    """
    assert np.ndim(bool_arr) == 1
    start, in_group = None, False
    for i, b in enumerate(bool_arr):
        if in_group == b:
            continue
        elif in_group:
            yield (start, i+1)
            in_group = False
        else:
            start = i
            in_group = True
    if in_group:
        yield (start, None)


class SyncSignal(object):
    """ SyncSignal(signals, sampling_rate, start=0, calibrations=None)

    A group of synchronous signals.

    Signals are said synchronous when they have the same sampling rate,
    and the same start and stop times.

    Parameters
    ----------
    signals: list of arrays
        the time signals, they all must have the same length
    sampling_rate: float
        frequency of sampling (Hz)
    start: float
        starting time (seconds)
        [default = 0.0]
    calibrations: list of functions
        must have the same length than signals
        [default = None] no calibration needed

    """

    def __init__(self, data, sampling_rate, start=0., calibrations=None):
        # assert np.ndim(data) == 2, "data must be a sequence of 1d array of the same length"
        data = tuple(data)
        #
        if not calibrations:
            calibrations = [1.] * len(data)
        calibrations = tuple(
            cal if callable(cal) else (lambda cc: (lambda f: cc))(cal)
            for cal in calibrations
        )
        assert len(data) == len(calibrations)
        #
        lengths = set(map(len, data))
        assert len(lengths) == 1, "all data must have the same length"
        size = lengths.pop()
        #
        self._data = data
        self._calibrations = calibrations
        self._sampling_rate = float(sampling_rate)
        self._start = float(start)
        self._size = size

    def __str__(self):
        date = lambda sec: datetime.utcfromtimestamp(np.round(sec))
        model = [
            ("{type}", ''),
            ("nb of channels", "{len}"),
            ("signal size", "{self.size}"),
            ("sampling rate", "{self.sampling_rate} Hz"),
            ("start", "{start}"),
            ("stop", "{stop}"),
        ]
        n = max(len(e[0]) for e in model)
        line = "{0:<%d} :   {1}" % n
        tpl = '\n  - '.join(line.format(l, r) if r else l for (l, r) in model)
        return tpl.format(
            self=self,
            type=type(self).__name__,
            len=len(self.data),
            start=date(self.start),
            stop=date(self.stop),
        )

    def __repr__(self):
        res = ('{name}([{len:d}x{self.size:d}],'
               ' sampling_rate={self.sampling_rate:.2g},'
               ' start={self.start:.2g}, calibrations=[...])')
        return res.format(name=type(self).__name__, len=len(self.data), self=self)

    @property
    def signals(self):
        """ tuple of synchronous signal groups handled

        here it's just (self,)
        """
        return (self,)

    @property
    def data(self):
        " tuple of signal data "
        return self._data

    @property
    def size(self):
        " length of each signal data "
        return self._size

    @property
    def sampling_rate(self):
        " sampling rate of the signal "
        return self._sampling_rate

    @property
    def start(self):
        " start time of the signal "
        return self._start

    @property
    def calibrations(self):
        " tuple of calibration functions for each signal channel "
        return self._calibrations

    @property
    def stop(self):
        " stop time of the signal "
        return self._index_to_time(self.size - 1)

    @property
    def interval(self):
        " the time interval of the signal set "
        return (self.start, self.stop)

    def select(self, *indices):
        " return a SyncSignal composed of selected channels "
        return type(self)(
            data=[self.data[i] for i in indices],
            calibrations=[self.calibrations[i] for i in indices],
            sampling_rate=self.sampling_rate,
            start=self.start,
        )

    def _time_to_index(self, time):
        " convert time to sample index "
       # TODO: to test
        index = (time - self.start) * self.sampling_rate
        #assert index.is_integer()
        return int(index)

    def _index_to_time(self, index):
        " convert sample index to time "
        if index < 0:
            index += self.size
        return self.start + index / self.sampling_rate

    def extract_i(self, i_start, i_stop, strict=True, include_last=False):
        """ new SyncSignal from (i_start, i_stop) indices
        """
        i_start = 0 if i_start is None else i_start
        if include_last and i_stop is not None:
            i_stop += 1
        i_stop = self.size if i_stop is None else i_stop
        if strict:
            assert i_start >= 0
            assert i_start < self.size
            assert i_stop >= 0
            assert i_stop <= self.size
        new_start = self._index_to_time(i_start)
        return type(self)(
            data=[e[i_start:i_stop] for e in self.data],
            sampling_rate=self.sampling_rate,
            start=new_start,
            calibrations=self.calibrations,
        )

    def extract_t(self, start, stop, strict=False):
        """ return SyncSignal reduced on (start, stop) time interval

        If 'strict=True', start and stop time must be in the original
        interval or an error is raised.
        """
        if strict:
            assert self.start <= start <= self.stop
            assert self.start <= stop <= self.stop
        else:
            start = max(self.start, start)
            stop = max(self.start, stop)
            start = min(self.stop, start)
            stop = min(self.stop, stop)
        i_start = self._time_to_index(start)
        i_stop = self._time_to_index(stop)
        return self.extract_i(i_start, i_stop, strict, include_last=True)

    def fourier_coefficients(self, freq, Nper, overlap, window):
        """ compute the fourier coefficients at freq of sliding windows

        coeffs, discrete_window_data = signal.fourier_coefficients(freq, Nper, overlap, window)

        The result include the calibration functions specified in the signal set:

            calibrated_coeff = coeff / calib


        Parameters
        ----------
        freq : scalar
            the frequency of the Fourier transform
        Nper : int
            number of period in each window
            window length  =  Nper / freq * sampling_freq
        overlap : float
            must be in [0 ; 0.5]
            the overlap ratio between windows
        window : array or function
            window function to apply before Fourier transform
            None means no windowing

        Returns
        -------
        coeffs : list of arrays
            fourier coefficients of each signal of the signal set
        discrete_window_data: tuple of integer
            data of the sliding discrete window
            discrete_window_data = (Nw, Lw, shift)
            Nw: number of windows
            Lw: length of the window
            shift: index shift beetwen windows

        """
        coeffs, discrete_window_data = time_to_freq(
            self.data, self.sampling_rate, freq, Nper, overlap, window
        )
        coeffs = [c / calib(freq)
                  for (c, calib) in zip(coeffs, self.calibrations)]
        return coeffs, discrete_window_data


class Inventory(object):
    """ Inventory(signalsets)

    A container for SignalSet objects:

      - New signalsets can be added with append() and extend().
      - New inventories are obtained by filtering an inventory
        with extract_t(), select_channels() and filter().
      - A SignalSet gathering the inventory content can be created with pack().


    Parameters
    ----------
    signalsets : list of SignalSet objects

    See Also
    --------
    SignalSet

    """

    _type = SignalSet

    def __init__(self, content=()):
        self._content = []
        self.extend(content)

    def __repr__(self):
        return '%s(%s)' % (type(self).__name__, self._content)

    def __iter__(self):
        return iter(self._content)

    def __getitem__(self, idx):
        return self._content[idx]

    def __len__(self):
        return len(self._content)

    def __bool__(self):
        return bool(self._content)

    __nonzero__ = __bool__

    def __add__(self, other):
        assert isinstance(other, type(self)), "wrong type"
        return type(self)(itertools.chain(self, other))

    __radd__ = __add__

    @classmethod
    def _assert_valid_element(cls, obj):
        if not isinstance(obj, cls._type):
            raise ValueError("can only append %s objects (%s given)"
                             % (cls._type.__name__, type(obj).__name__))

    def append(self, obj):
        """ append one element

        raise ValueError if obj is not a SignalSet.
        """
        self._assert_valid_element(obj)
        self._content.append(obj)

    def extend(self, seq):
        """ append all elements of a sequence

        raise ValueError if elements of seq are not SignalSet.
        """
        for e in list(seq):
            self.append(e)

    @property
    def tags(self):
        " set of all the available tags (str) "
        # TODO: valid consistency -> len(e.tags[t]) is the same for all e
        return {t for e in self for t in e.tags}

    @property
    def sampling_rates(self):
        return (s.sampling_rates for s in self)

    @property
    def starts(self):
        return (s.starts for s in self)

    @property
    def stops(self):
        return (s.stops for s in self)

    @property
    def intervals(self):
        return (s.intervals for s in self)

    def extract_t(self, start, stop, exclude=False):
        """ return a new Inventory by applying extract_t() on the content
        """
        content = (e.extract_t(start, stop, exclude) for e in self)
        content = (e for e in content if e.signals)
        return type(self)(content)

    def select_channels(self, *keys):
        """ return a new Inventory by applying select_channels() on the content
        """
        indices = ((e, set().union(*(e.tags.get(k, ()) for k in keys)))
                   for e in self)
        content = (e.select_channels(idx) for e, idx in indices if idx)
        return type(self)(content)

    def select_runs(self, runs):
        runs = list(runs)
        assert len(runs) == len(self)
        content = (s.select_runs(r) for s, r in zip(self, runs))
        content = (s for s in content if s.nb_runs)
        return type(self)(content)

    def filter(self, *patterns):
        """ return a new Inventory with channels matching one of the patterns

        Patterns are Unix shell style:

          - '*'       matches everything
          - '?'       matches any single character
          - '[seq]'   matches any character in seq
          - '[!seq]'  matches any char not in seq

        It's basically using fnmatch.filter() with select_channels()

        """
        keys = set().union(*(fnmatch.filter(self.tags, p) for p in patterns))
        return self.select_channels(*keys)

    def _join_merge(self):
        if not self:
            return type(self)._type({})

        group_keys = dict()
        for k, t in (((e for e in self if t in e.tags), t) for t in self.tags):
            group_keys.setdefault(frozenset(k), []).append(t)
        keys = map(set, group_keys.values())
        keys = sorted(map(sorted, keys))

        join = type(self)._type.join
        content = [join(*self.select_channels(*k)) for k in keys]

        a, others = content[0], content[1:]
        return a.merge(*others) if others else a

    def pack(self, keep_multi_tags=True):
        """ return a SignalSet that contains as much as possible of the content.

        The rules to build the signalset are:

          - All the channels defined in the inventory must be present.
          - All the tags defined in the inventory must be present,
            unless `keep_multi_tags` is False, then only the tags targeting single channels
            are preserved.

        Since the result is a SignalSet, it will be composed of disjoint synchronous runs,
        each run covering all the channels.

        None is returned if no group of signals can satisfied the rules above.

        """
        if not keep_multi_tags:
            tags = {t for s in self for t in self.tags
                    if len(s.tags.get(t, ())) == 1}
            content = itertools.chain(*map(self.select_channels, tags))
            inventory = type(self)(content)
            return inventory.pack(True)
        freqs, tags = list(self.sampling_rates), self.tags
        sigs = [self.select_runs(f == f0 for f in freqs)._join_merge()
                for f0 in set(itertools.chain(*freqs))]

        sigs = [s for s in sigs if s.nb_runs and tags.issubset(s.tags)]
        return type(self)._type.join(*sigs) if sigs else None

        # sigs = [s for s in sigs if tags.issubset(s.tags)]
        # return type(self)._type(*sigs)

    merge = _join_merge
    # merge = pack
