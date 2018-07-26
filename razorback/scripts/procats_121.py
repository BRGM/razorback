""" a script that computes the transfer function between 2 ats files using M-estimate method

"""

from __future__ import print_function

import argparse
import warnings
import numpy as np

from .. import weights, mestimator, calibrations, fourier_transform, io, signalset

__all__ = []


# TODO: voir a etendre les calibration metronix avec du atan
#       cf D:/Documents/smai/Travail/PROJETS/optmt/dev/razorback/scem/calc_mag_cal2.m


def main(argv=None):
    args = build_parser().parse_args(argv)
    coeffs, errors = process(**vars(args))
    response = ' '.join('%s %s' % pair for pair in zip(coeffs, errors))
    print(response)


class WeigthsAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(WeigthsAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, value, option_string=None):
        setattr(namespace, self.dest, self.convert(value))

    def convert(self, value):
        globals_ = {name: getattr(weights, name) for name in weights.__all__}
        try:
            return eval(value, globals_)
        except:
            raise argparse.ArgumentError(self, 'failed to evaluate "%s"' % value)


def ranged(type, min=-np.inf, max=np.inf):
    def convert(string):
        value = type(string)
        if min <= value <= max:
            return value
        msg = "%s is out of range [%s, %s]" % (string, min, max)
        raise argparse.ArgumentTypeError(msg)
    convert.__name__ = type.__name__
    return convert


def build_parser(*args, **kwargs):
    ## some help message template
    idx_help = """
%s index interval to work with
START and STOP indices are included
Use negative for STOP to indicate the end of the sample
See also the --IDXbase option
[default is all sample]
""".strip()

    cal_help = """
%s calibration: scalar or metronix file path
[default is 1]
    """.strip()

    description = """
M-Estimate of transfer function between 2 ats files.

Output: T1 err1 [T2 err2 ...]
    """.strip()

    path_help = "path to %s ats file"

    ## build & populate the parser
    parser = argparse.ArgumentParser(
        *args,
        description=description,
        formatter_class=argparse.RawTextHelpFormatter,
        **kwargs
    )

    parser.add_argument(
        '--freq', nargs='+', type=ranged(float, 0), required=True,
        help="list of frequencies to process"
    )
    parser.add_argument(
        '--TX', metavar='ATS_FILE', required=True, help=path_help % 'TX'
    )
    parser.add_argument(
        '--RX', metavar='ATS_FILE', required=True, help=path_help % 'RX'
    )
    parser.add_argument(
        '--TXcal', default=1., help=cal_help % 'TX'
    )
    parser.add_argument(
        '--RXcal', default=1., help=cal_help % 'TX'
    )
    parser.add_argument(
        '--TXidx', nargs=2, metavar=('START', 'STOP'), type=int, default=(None, None),
        help=idx_help % 'TX'
    )
    parser.add_argument(
        '--RXidx', nargs=2, metavar=('START', 'STOP'), type=int, default=(None, None),
        help=idx_help % 'RX'
    )
    parser.add_argument(
        '--IDXbase', type=int, choices=(0, 1), default=1,
        help="""
Index base used to handle --TXidx and --RXidx options
[default is 1]
        """.strip()
    )
    parser.add_argument(
        '--weights', metavar='"WEIGTHS EXPRESSION"',
        action=WeigthsAction, default=weights.least_square,
        help="""
weighting strategy used for M-estimation
recommanded values are:
- "least_square"
  just the least square algorithm (no weights)
- "mest_weights"
  robust M-estimator algorithm
- "bi_weights(REJECT_PROB, N_STEP)"
  bounded influence algorithm
  typical parameter values are
    REJECT_PROB ~ 0.01 (must be in [0, 1])
    N_STEP ~ 3 (more if convergence problems)
        """.strip()
    )
    parser.add_argument(
        '--FTper', metavar='NPER', type=ranged(float, 1), default=8,
        help="Number of period per sliding window [default is 8]"
    )
    parser.add_argument(
        '--FToverlap', metavar='OVERLAP', type=ranged(float, 0, 1), default=0.7,
        help="Overlap ratio between sliding window [default is 0.7]"
    )

    # return parser.parse_args(argv)
    return parser


def process(
    TX, RX, TXcal, RXcal, freq, weights,
    TXidx, RXidx, IDXbase,
    FTper, FToverlap,
):
    default_mest_opts = dict()

    TXidx = convert_indices(TXidx, IDXbase)
    RXidx = convert_indices(RXidx, IDXbase)

    ## computation options
    window = fourier_transform.slepian_window(4)
    fourier_opts = dict(Nper=FTper, overlap=FToverlap, window=window)
    mest_opts = dict(default_mest_opts)

    ## prepare data
    l_freq = map(float, freq)
    tx = build_signal(TX, TXcal)
    rx = build_signal(RX, RXcal)

    ## reduce to sub-indices
    tx = tx.extract_i(*TXidx, include_last=True)
    rx = rx.extract_i(*RXidx, include_last=True)

    ## check compatibity
    if not (tx.sampling_rate == rx.sampling_rate
            and tx.stop > rx.start
            and rx.stop > tx.start):
        raise ValueError("TX and RX files are not compatible.")

    ## merge tx & rx on their common time interval
    signals = (signalset.Tags(tx=0) | tx) & (signalset.Tags(rx=0) | rx)

    ## compute transfer functions
    l_transf, l_err = [], []
    for freq in l_freq:
        coeffs, _ = signals.fourier_coefficients(freq, **fourier_opts)
        inputs, outputs = [coeffs[0]], [coeffs[1]]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                T, ivid = mestimator.transfer_function(
                    outputs, inputs, weights=weights, **mest_opts
                )
        except Exception as e:
            T, ivid = np.array([[np.nan]]), None
        if np.isnan(T):
            err = np.array([[np.nan]])
        else:
            try:
                err = mestimator.transfer_error(outputs, inputs, T, ivid)
            except Exception as e:
                err = np.array([[np.nan]])
        l_transf.append(T[0, 0])
        l_err.append(err[0, 0])

    return l_transf, l_err


def build_signal(fname, cal_info):
    data, sampling_rate, start = io.ats.read_ats_sample(fname)

    try:
        cal = float(cal_info)
    except ValueError:
        cal = calibrations.metronix(cal_info, sampling_rate, data_dir='')

    return signalset.SyncSignal([data], sampling_rate, start, [cal])


def convert_indices(indices, base):
    f_negative = lambda idx: tuple(None if i < base else i for i in idx)
    f_base = lambda idx: tuple(None if i is None else i - base for i in idx)
    return f_base(f_negative(indices))


__doc__ += '\n\n.. code-block:: none\n\n   '
__doc__ += '\n   '.join(build_parser('procats_121.py').format_help().splitlines())


if __name__ == '__main__':
    main()
