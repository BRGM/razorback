""" helper functions for plotting
"""


import numpy as np
import matplotlib.pyplot as plt


# TODO: add opt2str() and 'title' arg in plot_...()

# TODO: add plot_timelapse()


def plot_one_period(data, l_freq, l_z, l_ivt, apparent_resistivity, time_plot=False):
    # TODO: add title with impedance() options

    plt.figure()
    plt.subplot(211)
    plt.loglog(l_freq, abs(l_z).reshape(-1, 4), '-+')
    plt.legend('xx xy yx yy'.split(), loc='best')
    plt.ylabel('module impedance')
    plt.xlabel('freq (Hz)')
    plt.subplot(212)
    plt.semilogx(l_freq, np.angle(l_z, deg=1).reshape(-1, 4), '-+')
    plt.legend('xx xy yx yy'.split(), loc='best')
    plt.ylabel('phase impedance')
    plt.xlabel('freq (Hz)')

    rho = apparent_resistivity(l_z, l_freq[:, None, None])
    rho_det = apparent_resistivity(np.linalg.det(l_z)**.5, l_freq)

#    plt.figure()
#    plt.loglog(l_freq, abs(rho).reshape(-1, 4), '-+')
#    plt.loglog(l_freq, rho_det, 'k-+')
#    plt.legend('xx xy yx yy det'.split(), loc='best')
#    plt.ylabel('module resistivity')
#    plt.xlabel('freq (Hz)')
    plt.figure()
    plt.subplot(211)
    plt.loglog(l_freq, abs(rho).reshape(-1, 4), '-+')
    plt.loglog(l_freq, rho_det, 'k-+')
    plt.legend('xx xy yx yy det'.split(), loc='best')
    plt.ylabel('module resistivity')
    plt.xlabel('freq (Hz)')
    plt.subplot(212)
    plt.semilogx(l_freq, np.angle(l_z, deg=1).reshape(-1, 4), '-+')
    plt.legend('xx xy yx yy'.split(), loc='best')
    plt.ylabel('phase impedance')
    plt.xlabel('freq (Hz)')


    time = np.arange(data.size) / data.sampling_rate
    log_freq = np.log10(l_freq)
    d_freq = list(0.5 * np.diff(log_freq))
    d_freq = [d_freq[0]] + d_freq + [d_freq[-1]]

    if time_plot:
        plt.figure()
        ax = plt.subplot(711)
        plt.plot(time, data.E[0])
        plt.ylabel('Ex')
        plt.subplot(712, sharex=ax)
        plt.plot(time, data.E[1])
        plt.ylabel('Ey')
        plt.subplot(713, sharex=ax)
        plt.plot(time, data.B[0])
        plt.ylabel('Bx')
        plt.subplot(714, sharex=ax)
        plt.plot(time, data.B[1])
        plt.ylabel('By')
        plt.subplot(715, sharex=ax)
        plt.ylabel('eliminated\ndata\nchannel X\n\nfreq (Hz)')
        plt.yticks(log_freq, ['%.2e' % f for f in l_freq])
        for i, (lf, ivt) in enumerate(zip(log_freq, l_ivt)):
            ivt = ivt[0]
            lc = plt.vlines(ivt, lf-d_freq[i], lf+d_freq[i+1])
            lc.set_linewidth(0.1)
        plt.subplot(716, sharex=ax)
        plt.ylabel('eliminated\ndata\nchannel Y\n\nfreq (Hz)')
        plt.yticks(log_freq, ['%.2e' % f for f in l_freq])
        for i, (lf, ivt) in enumerate(zip(log_freq, l_ivt)):
            ivt = ivt[1]
            lc = plt.vlines(ivt, lf-d_freq[i], lf+d_freq[i+1])
            lc.set_linewidth(0.1)
        plt.subplot(717, sharex=ax)
        plt.xlabel('time (s)')
        plt.ylabel('eliminated\ndata\nchannel X+Y\n\nfreq (Hz)')
        plt.yticks(log_freq, ['%.2e' % f for f in l_freq])
        for i, (lf, ivt) in enumerate(zip(log_freq, l_ivt)):
            ivt = np.union1d(*ivt)
            lc = plt.vlines(ivt, lf-d_freq[i], lf+d_freq[i+1])
            lc.set_linewidth(0.1)

    plt.show()
