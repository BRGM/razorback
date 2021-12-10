import datetime
import dask.array as da
import numpy as np

import razorback as rzb


class Count(rzb.io.binary_file_array.BaseArrayProxy):
    """ a fake data array proxy class
    """
    def extract(self, index):
        s, = index
        return np.arange(s.start, s.stop, s.step, dtype=self.dtype)


def fake_signal(tag, period, sampling_rate, start):
    "build a SignalSet with fake data array"
    size = int(period * sampling_rate)
    arr = da.from_array(Count(None, (size,), float), "auto", fancy=False)
    signal = rzb.SyncSignal([arr], sampling_rate, start)
    return rzb.SignalSet({tag: 0}, signal)


def date(*args, **kwds):
    """ convert datetime info in timestamp (UTC)
    date(year, month, day[, hour[, minute[, second[, microsecond]]]])
    """
    return datetime.datetime(*args, **kwds, tzinfo=datetime.timezone.utc).timestamp()


def period(*args, **kwds):
    """ convert timedelta info in seconds
    period(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)
    """
    return datetime.timedelta(*args, **kwds).total_seconds()


def test_merge_1_runs_of_1_channel():
    s = fake_signal('Ex', period(days=7), 4, date(2000, 1, 1))
    np.testing.assert_allclose(
        s.intervals,
        [[date(2000, 1, 1), date(2000, 1,  8) - 1/4]]
    )
    np.testing.assert_allclose(
        s.merge_consecutive_runs().intervals,
        [[date(2000, 1, 1), date(2000, 1,  8) - 1/4]]
    )


def test_merge_2_runs_of_1_channel():
    s = ( fake_signal('Ex', period(days=7), 4, date(2000, 1, 1))
        | fake_signal('Ex', period(days=2), 4, date(2000, 1, 8))
    )
    np.testing.assert_allclose(
        s.intervals,
        [[date(2000, 1, 1), date(2000, 1,  8) - 1/4],
         [date(2000, 1, 8), date(2000, 1, 10) - 1/4]]
    )
    np.testing.assert_allclose(
        s.merge_consecutive_runs().intervals,
        [[date(2000, 1, 1), date(2000, 1, 10) - 1/4]]
    )


def test_merge_4_runs_of_1_channel():
    s = ( fake_signal('Ex', period(days=7), 4, date(2000, 1,  1))
        | fake_signal('Ex', period(days=2), 4, date(2000, 1,  8))
        | fake_signal('Ex', period(days=1), 4, date(2000, 1, 10))
        | fake_signal('Ex', period(days=3), 4, date(2000, 1, 11))
    )
    np.testing.assert_allclose(
        s.intervals,
        [[date(2000, 1,  1), date(2000, 1,  8) - 1/4],
         [date(2000, 1,  8), date(2000, 1, 10) - 1/4],
         [date(2000, 1, 10), date(2000, 1, 11) - 1/4],
         [date(2000, 1, 11), date(2000, 1, 14) - 1/4]]
    )
    np.testing.assert_allclose(
        s.merge_consecutive_runs().intervals,
        [[date(2000, 1, 1), date(2000, 1, 14) - 1/4]]
    )


def test_merge_1_and_2_runs_of_1_channel():
    s = ( fake_signal('Ex', period(days=7), 4, date(2000, 1,  1))
        | fake_signal('Ex', period(days=1), 4, date(2000, 1, 10))
        | fake_signal('Ex', period(days=3), 4, date(2000, 1, 11))
    )
    np.testing.assert_allclose(
        s.intervals,
        [[date(2000, 1,  1), date(2000, 1,  8) - 1/4],
         [date(2000, 1, 10), date(2000, 1, 11) - 1/4],
         [date(2000, 1, 11), date(2000, 1, 14) - 1/4]]
    )
    np.testing.assert_allclose(
        s.merge_consecutive_runs().intervals,
        [[date(2000, 1,  1), date(2000, 1,  8) - 1/4],
         [date(2000, 1, 10), date(2000, 1, 14) - 1/4]]
    )


def test_merge_2_and_1_runs_of_1_channel():
    s = ( fake_signal('Ex', period(days=7), 4, date(2000, 1,  1))
        | fake_signal('Ex', period(days=1), 4, date(2000, 1,  8))
        | fake_signal('Ex', period(days=3), 4, date(2000, 1, 11))
    )
    np.testing.assert_allclose(
        s.intervals,
        [[date(2000, 1,  1), date(2000, 1,  8) - 1/4],
         [date(2000, 1,  8), date(2000, 1,  9) - 1/4],
         [date(2000, 1, 11), date(2000, 1, 14) - 1/4]]
    )
    np.testing.assert_allclose(
        s.merge_consecutive_runs().intervals,
        [[date(2000, 1,  1), date(2000, 1,  9) - 1/4],
         [date(2000, 1, 11), date(2000, 1, 14) - 1/4]]
    )


def test_merge_2_and_3_runs_of_1_channel_with_2_sampling():
    s = ( fake_signal('Ex', period(days=7), 4, date(2000, 1,  1))
        | fake_signal('Ex', period(days=2), 4, date(2000, 1,  8))
        | fake_signal('Ex', period(days=1), 2, date(2000, 1, 10))
        | fake_signal('Ex', period(days=3), 2, date(2000, 1, 11))
        | fake_signal('Ex', period(days=4), 2, date(2000, 1, 14))
    )
    np.testing.assert_allclose(
        s.intervals,
        [[date(2000, 1,  1), date(2000, 1,  8) - 1/4],
         [date(2000, 1,  8), date(2000, 1, 10) - 1/4],
         [date(2000, 1, 10), date(2000, 1, 11) - 1/2],
         [date(2000, 1, 11), date(2000, 1, 14) - 1/2],
         [date(2000, 1, 14), date(2000, 1, 18) - 1/2]]
    )
    np.testing.assert_allclose(
        s.merge_consecutive_runs().intervals,
        [[date(2000, 1,  1), date(2000, 1, 10) - 1/4],
         [date(2000, 1, 10), date(2000, 1, 18) - 1/2]]
    )


def test_merge_1_and_3_and_1_runs_of_1_channel_with_2_sampling():
    s = ( fake_signal('Ex', period(days=7), 4, date(2000, 1,  1))
        | fake_signal('Ex', period(days=2), 2, date(2000, 1,  8))
        | fake_signal('Ex', period(days=1), 2, date(2000, 1, 10))
        | fake_signal('Ex', period(days=3), 2, date(2000, 1, 11))
        | fake_signal('Ex', period(days=4), 4, date(2000, 1, 14))
    )
    np.testing.assert_allclose(
        s.intervals,
        [[date(2000, 1,  1), date(2000, 1,  8) - 1/4],
         [date(2000, 1,  8), date(2000, 1, 10) - 1/2],
         [date(2000, 1, 10), date(2000, 1, 11) - 1/2],
         [date(2000, 1, 11), date(2000, 1, 14) - 1/2],
         [date(2000, 1, 14), date(2000, 1, 18) - 1/4]]
    )
    np.testing.assert_allclose(
        s.merge_consecutive_runs().intervals,
        [[date(2000, 1,  1), date(2000, 1,  8) - 1/4],
         [date(2000, 1,  8), date(2000, 1, 14) - 1/2],
         [date(2000, 1, 14), date(2000, 1, 18) - 1/4]]
    )


def test_merge_1_and_1_and_2_and_1_runs_of_1_channel_with_2_sampling():
    s = ( fake_signal('Ex', period(days=7), 4, date(2000, 1,  1))
        | fake_signal('Ex', period(days=1), 2, date(2000, 1,  8))
        | fake_signal('Ex', period(days=1), 2, date(2000, 1, 10))
        | fake_signal('Ex', period(days=3), 2, date(2000, 1, 11))
        | fake_signal('Ex', period(days=4), 4, date(2000, 1, 14))
    )
    np.testing.assert_allclose(
        s.intervals,
        [[date(2000, 1,  1), date(2000, 1,  8) - 1/4],
         [date(2000, 1,  8), date(2000, 1,  9) - 1/2],
         [date(2000, 1, 10), date(2000, 1, 11) - 1/2],
         [date(2000, 1, 11), date(2000, 1, 14) - 1/2],
         [date(2000, 1, 14), date(2000, 1, 18) - 1/4]]
    )
    np.testing.assert_allclose(
        s.merge_consecutive_runs().intervals,
        [[date(2000, 1,  1), date(2000, 1,  8) - 1/4],
         [date(2000, 1,  8), date(2000, 1,  9) - 1/2],
         [date(2000, 1, 10), date(2000, 1, 14) - 1/2],
         [date(2000, 1, 14), date(2000, 1, 18) - 1/4]]
    )


def test_merge_1_runs_of_2_channel():
    s = fake_signal('Ex', period(days=7), 4, date(2000, 1, 1))
    s &= rzb.SignalSet({'Ey': 0}, *s.signals)
    assert s.nb_channels == 2
    np.testing.assert_allclose(
        s.intervals,
        [[date(2000, 1, 1), date(2000, 1,  8) - 1/4]]
    )
    np.testing.assert_allclose(
        s.merge_consecutive_runs().intervals,
        [[date(2000, 1, 1), date(2000, 1,  8) - 1/4]]
    )


def test_merge_2_runs_of_2_channel():
    s = ( fake_signal('Ex', period(days=7), 4, date(2000, 1, 1))
        | fake_signal('Ex', period(days=2), 4, date(2000, 1, 8))
    )
    s &= rzb.SignalSet({'Ey': 0}, *s.signals)
    assert s.nb_channels == 2
    np.testing.assert_allclose(
        s.intervals,
        [[date(2000, 1, 1), date(2000, 1,  8) - 1/4],
         [date(2000, 1, 8), date(2000, 1, 10) - 1/4]]
    )
    np.testing.assert_allclose(
        s.merge_consecutive_runs().intervals,
        [[date(2000, 1, 1), date(2000, 1, 10) - 1/4]]
    )


def test_merge_4_runs_of_2_channel():
    s = ( fake_signal('Ex', period(days=7), 4, date(2000, 1,  1))
        | fake_signal('Ex', period(days=2), 4, date(2000, 1,  8))
        | fake_signal('Ex', period(days=1), 4, date(2000, 1, 10))
        | fake_signal('Ex', period(days=3), 4, date(2000, 1, 11))
    )
    s &= rzb.SignalSet({'Ey': 0}, *s.signals)
    assert s.nb_channels == 2
    np.testing.assert_allclose(
        s.intervals,
        [[date(2000, 1,  1), date(2000, 1,  8) - 1/4],
         [date(2000, 1,  8), date(2000, 1, 10) - 1/4],
         [date(2000, 1, 10), date(2000, 1, 11) - 1/4],
         [date(2000, 1, 11), date(2000, 1, 14) - 1/4]]
    )
    np.testing.assert_allclose(
        s.merge_consecutive_runs().intervals,
        [[date(2000, 1, 1), date(2000, 1, 14) - 1/4]]
    )


def test_merge_1_and_2_runs_of_2_channel():
    s = ( fake_signal('Ex', period(days=7), 4, date(2000, 1,  1))
        | fake_signal('Ex', period(days=1), 4, date(2000, 1, 10))
        | fake_signal('Ex', period(days=3), 4, date(2000, 1, 11))
    )
    s &= rzb.SignalSet({'Ey': 0}, *s.signals)
    assert s.nb_channels == 2
    np.testing.assert_allclose(
        s.intervals,
        [[date(2000, 1,  1), date(2000, 1,  8) - 1/4],
         [date(2000, 1, 10), date(2000, 1, 11) - 1/4],
         [date(2000, 1, 11), date(2000, 1, 14) - 1/4]]
    )
    np.testing.assert_allclose(
        s.merge_consecutive_runs().intervals,
        [[date(2000, 1,  1), date(2000, 1,  8) - 1/4],
         [date(2000, 1, 10), date(2000, 1, 14) - 1/4]]
    )


def test_merge_2_and_1_runs_of_2_channel():
    s = ( fake_signal('Ex', period(days=7), 4, date(2000, 1,  1))
        | fake_signal('Ex', period(days=1), 4, date(2000, 1,  8))
        | fake_signal('Ex', period(days=3), 4, date(2000, 1, 11))
    )
    s &= rzb.SignalSet({'Ey': 0}, *s.signals)
    assert s.nb_channels == 2
    np.testing.assert_allclose(
        s.intervals,
        [[date(2000, 1,  1), date(2000, 1,  8) - 1/4],
         [date(2000, 1,  8), date(2000, 1,  9) - 1/4],
         [date(2000, 1, 11), date(2000, 1, 14) - 1/4]]
    )
    np.testing.assert_allclose(
        s.merge_consecutive_runs().intervals,
        [[date(2000, 1,  1), date(2000, 1,  9) - 1/4],
         [date(2000, 1, 11), date(2000, 1, 14) - 1/4]]
    )


def test_merge_2_and_3_runs_of_2_channel_with_2_sampling():
    s = ( fake_signal('Ex', period(days=7), 4, date(2000, 1,  1))
        | fake_signal('Ex', period(days=2), 4, date(2000, 1,  8))
        | fake_signal('Ex', period(days=1), 2, date(2000, 1, 10))
        | fake_signal('Ex', period(days=3), 2, date(2000, 1, 11))
        | fake_signal('Ex', period(days=4), 2, date(2000, 1, 14))
    )
    s &= rzb.SignalSet({'Ey': 0}, *s.signals)
    assert s.nb_channels == 2
    np.testing.assert_allclose(
        s.intervals,
        [[date(2000, 1,  1), date(2000, 1,  8) - 1/4],
         [date(2000, 1,  8), date(2000, 1, 10) - 1/4],
         [date(2000, 1, 10), date(2000, 1, 11) - 1/2],
         [date(2000, 1, 11), date(2000, 1, 14) - 1/2],
         [date(2000, 1, 14), date(2000, 1, 18) - 1/2]]
    )
    np.testing.assert_allclose(
        s.merge_consecutive_runs().intervals,
        [[date(2000, 1,  1), date(2000, 1, 10) - 1/4],
         [date(2000, 1, 10), date(2000, 1, 18) - 1/2]]
    )


def test_merge_1_and_3_and_1_runs_of_2_channel_with_2_sampling():
    s = ( fake_signal('Ex', period(days=7), 4, date(2000, 1,  1))
        | fake_signal('Ex', period(days=2), 2, date(2000, 1,  8))
        | fake_signal('Ex', period(days=1), 2, date(2000, 1, 10))
        | fake_signal('Ex', period(days=3), 2, date(2000, 1, 11))
        | fake_signal('Ex', period(days=4), 4, date(2000, 1, 14))
    )
    s &= rzb.SignalSet({'Ey': 0}, *s.signals)
    assert s.nb_channels == 2
    np.testing.assert_allclose(
        s.intervals,
        [[date(2000, 1,  1), date(2000, 1,  8) - 1/4],
         [date(2000, 1,  8), date(2000, 1, 10) - 1/2],
         [date(2000, 1, 10), date(2000, 1, 11) - 1/2],
         [date(2000, 1, 11), date(2000, 1, 14) - 1/2],
         [date(2000, 1, 14), date(2000, 1, 18) - 1/4]]
    )
    np.testing.assert_allclose(
        s.merge_consecutive_runs().intervals,
        [[date(2000, 1,  1), date(2000, 1,  8) - 1/4],
         [date(2000, 1,  8), date(2000, 1, 14) - 1/2],
         [date(2000, 1, 14), date(2000, 1, 18) - 1/4]]
    )


def test_merge_1_and_1_and_2_and_1_runs_of_2_channel_with_2_sampling():
    s = ( fake_signal('Ex', period(days=7), 4, date(2000, 1,  1))
        | fake_signal('Ex', period(days=1), 2, date(2000, 1,  8))
        | fake_signal('Ex', period(days=1), 2, date(2000, 1, 10))
        | fake_signal('Ex', period(days=3), 2, date(2000, 1, 11))
        | fake_signal('Ex', period(days=4), 4, date(2000, 1, 14))
    )
    s &= rzb.SignalSet({'Ey': 0}, *s.signals)
    assert s.nb_channels == 2
    np.testing.assert_allclose(
        s.intervals,
        [[date(2000, 1,  1), date(2000, 1,  8) - 1/4],
         [date(2000, 1,  8), date(2000, 1,  9) - 1/2],
         [date(2000, 1, 10), date(2000, 1, 11) - 1/2],
         [date(2000, 1, 11), date(2000, 1, 14) - 1/2],
         [date(2000, 1, 14), date(2000, 1, 18) - 1/4]]
    )
    np.testing.assert_allclose(
        s.merge_consecutive_runs().intervals,
        [[date(2000, 1,  1), date(2000, 1,  8) - 1/4],
         [date(2000, 1,  8), date(2000, 1,  9) - 1/2],
         [date(2000, 1, 10), date(2000, 1, 14) - 1/2],
         [date(2000, 1, 14), date(2000, 1, 18) - 1/4]]
    )


