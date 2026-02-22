import numpy as np
import razorback as rzb
from test_impedance import make_data


def test_cod():
    data = make_data(2)
    freqs = np.logspace(np.log10(1e-1), np.log10(0.5), 5) * data.sampling_rates[0]

    rzb.utils.impedance(data, freqs, prefilter=rzb.prefilters.cod_filter(0.8))
