import numpy as np
import pytest
from razorback.signalset import Tags, SyncSignal, SignalSet


@pytest.fixture
def tags():
    return Tags(a=0, b=1, c=2, BC=(1, 2))


@pytest.fixture
def data():
    n = 3
    return np.arange(n*10).reshape(n, 10)


@pytest.fixture
def three_signals(tags, data):
    sy1 = SyncSignal(data, 1.0, start=0)
    sy2 = SyncSignal(-data[:, :8], 0.5, start=50)
    sy3 = SyncSignal(10*data, 1.0, start=20)

    s1 = tags | sy1
    s2 = Tags(a=2, b=0, c=1, BC=(0, 1)) | sy2
    s3 = SignalSet(tags, sy3)

    return s1, s2, s3


def test_base(tags, data, three_signals):
    s1, s2, s3 = three_signals

    s = Tags(a=0, b=1, c=2) | SyncSignal(data, 1.0)
    s.tags.BC = 1, 2
    a = s.select_channels([2, 0])

    assert np.all(s.signals[0].data[0] == a.signals[0].data[0])
    assert np.all(s.signals[0].data[0] == a.a.signals[0].data[0])
    assert np.all(s.signals[0].data[2] == a.signals[0].data[1])
    assert np.all(s.signals[0].data[2] == a.c.signals[0].data[0])

    ab = SignalSet.merge(s.a, s.b)
    assert np.all(ab.signals[0].data[0] == s.a.signals[0].data[0])
    assert np.all(ab.signals[0].data[1] == s.b.signals[0].data[0])

    assert np.all(((s1 | s2) | s3).intervals == (s1 | (s2 | s3)).intervals)
    assert np.all((s1|s2).a.signals[0].data[0] == s1.a.signals[0].data[0])
    assert np.all((s2|s1).a.signals[0].data[0] == s1.a.signals[0].data[0])
    assert np.all((s1|s2).a.signals[1].data[0] == s2.a.signals[0].data[0])
    assert np.all((s2|s1).a.signals[1].data[0] == s2.a.signals[0].data[0])


def test_indexing(tags, three_signals):
    s1, s2, s3 = three_signals
    ss = SignalSet(tags, s1, s2, s3)

    sa = ss['a']
    assert np.all(sa.intervals == ss.intervals)
    assert sa.tags == Tags(1, a=0)

    sb = ss['b']
    assert np.all(sb.intervals == ss.intervals)
    assert sb.tags == Tags(1, b=0)

    sbc = ss[['b', 'c']]
    assert np.all(sbc.intervals == ss.intervals)
    assert sbc.tags == Tags(2, b=0, c=1, BC=(0, 1))

    sb2 = ss[['b', 2]]
    assert np.all(sb2.intervals == ss.intervals)
    assert sb2.tags == Tags(2, b=0, c=1, BC=(0, 1))

    assert ss[:].tags == ss.tags

    assert np.all(ss[:, ss.sampling_rates == 1].sampling_rates == [1, 1])
    assert np.all(ss[:, ss.sampling_rates == 1].starts == [0, 20])

    assert np.all(ss[:, ss.sampling_rates == 0.5].sampling_rates == [0.5])
    assert np.all(ss[:, ss.sampling_rates == 0.5].starts == [50])

    assert np.all(ss.extract_t(5, 25).intervals == [[5, 9], [20, 25]])

    import fnmatch
    channels = fnmatch.filter(ss.tags, '[bc]')
    sbc = ss[channels]
    assert np.all(sbc.intervals == ss.intervals)
    assert sbc.tags == Tags(2, b=0, c=1, BC=(0, 1))


def test_simple_fourier():

    t = np.linspace(0, 100, 30)
    dt = t[1] - t[0]
    x = np.cos(t*2*np.pi)
    y = np.sin(t*2*np.pi)

    tags = Tags(x=0, y=1)
    sy = SyncSignal([x, y], 1/dt)
    sig = tags | sy
    sigc = tags | sy.extract_i(0, 15) | sy.extract_i(15-1, 30)

    c1, w1 = sig.fourier_coefficients(.5/dt, 1, 0, None)
    c2, w2 = sigc.fourier_coefficients(.5/dt, 1, 0, None)

    assert np.allclose(c1[0][:6], c2[0][:6])
    assert np.allclose(c1[0][-7:], c2[0][-7:])
    assert np.allclose(c1[1][:6], c2[1][:6])
    assert np.allclose(c1[1][-7:], c2[1][-7:])
