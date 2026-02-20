"""
E1  aaaaaaaa     
H1  aaaaaaaa     
E2           cccc
H2      bbbb cccc
"""

import numpy as np
from razorback.signalset import SignalSet, SyncSignal, Tags, Inventory


def test_simple():

    sync = lambda t0, t1, rate=1, n=1: SyncSignal([np.arange(1+(t1-t0)*rate)]*n, rate, t0)

    inv = Inventory([
        Tags(4, Ex1=0, Ey1=1, Hx1=2, Hy1=3, E1=(0, 1), H1=(2, 3)) | sync(0, 20, n=4),
        Tags(2, Hx2=0, Hy2=1, H2=(0, 1)) | sync(10, 20, n=2),
        Tags(4, Ex2=0, Ey2=1, Hx2=2, Hy2=3, E2=(0, 1), H2=(2, 3)) | sync(20, 30, n=4),
    ])


    res = inv.select_channels('H1').merge()
    assert sorted(res.tags.keys()) == ['H1', 'Hx1', 'Hy1']
    assert np.allclose(res.intervals, [[  0.,  20.]])

    res = inv.select_channels('H1', 'H2').merge()
    assert sorted(res.tags.keys()) == ['H1', 'H2', 'Hx1', 'Hx2', 'Hy1', 'Hy2']
    assert np.allclose(res.intervals, [[10., 20.]])

    res = inv.select_channels('H1', 'E2').merge()
    assert sorted(res.tags.keys()) == ['E2', 'Ex2', 'Ey2', 'H1', 'Hx1', 'Hy1']
    assert np.allclose(res.intervals, [])

    res = inv.select_channels('E2', 'H2').merge()
    assert sorted(res.tags.keys()) == ['E2', 'Ex2', 'Ey2', 'H2', 'Hx2', 'Hy2']
    assert np.allclose(res.intervals, [[20., 30.]])
