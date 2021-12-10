import numpy as np
import razorback as rzb


def make_data(factor=2, nsr=1e-12, size=5000):
    time = np.linspace(0, 10, size)
    inputs = np.sin((time[-1]-time)**2)
    outputs = factor * inputs
    inputs += np.random.normal(scale=nsr, size=inputs.shape)
    outputs += np.random.normal(scale=nsr, size=outputs.shape)
    data = rzb.SignalSet({'B': 0, 'E': 1}, rzb.SyncSignal([inputs, outputs], 1, 0))
    return data


def test_scalar_factor():
    data = make_data(2)
    freqs = np.logspace(np.log10(1e-1), np.log10(0.5), 5) * data.sampling_rates[0]

    res = rzb.utils.impedance(data, freqs)
    np.testing.assert_allclose(res.impedance, 2)

    res = rzb.utils.impedance(data, freqs, weights=rzb.weights.mest_weights)
    np.testing.assert_allclose(res.impedance, 2)

    res = rzb.utils.impedance(data, freqs, weights=rzb.weights.bi_weights(0.1, 3, 1))
    np.testing.assert_allclose(res.impedance, 2)

    res = rzb.utils.impedance(data, freqs, remote='B')
    np.testing.assert_allclose(res.impedance, 2)


def test_fail_freq_to_big():
    data = make_data(2)
    freqs = [data.sampling_rates[0]]
    res = rzb.utils.impedance(data, freqs)
    np.testing.assert_allclose(res.impedance, np.nan)
    np.testing.assert_allclose(res.error, np.nan)
    assert res.invalid_time == [()]


def test_fail_freq_to_small():
    data = make_data(2)
    freqs = [1e-5 * data.sampling_rates[0]]
    res = rzb.utils.impedance(data, freqs)
    np.testing.assert_allclose(res.impedance, np.nan)
    np.testing.assert_allclose(res.error, np.nan)
    assert res.invalid_time == [()]
