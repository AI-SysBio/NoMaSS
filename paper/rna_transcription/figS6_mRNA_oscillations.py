"""
Fig. S6: oscillations of the averaged mRNA count with non-Markovian promoter switching.

Two-state model (Goff <-> Gon, Gon -> Gon + mRNA, mRNA -> 0) with gamma
distributed promoter switching times of shape alpha = alpha_on = alpha_off.
Starting from an OFF promoter and no mRNA, the averaged dynamics oscillate with
period 1/lambda_on + 1/lambda_off before converging to the steady state; the
more regular the switching (larger alpha), the slower the oscillations decay.

A: probability of the ON and OFF promoter states over time, Markovian
   (alpha = 1) versus non-Markovian (alpha = 50).
B: averaged mRNA count for several alpha, with the amplitude factor between
   consecutive peaks.
C: amplitude factor as a function of alpha.

Results ship precomputed in ``results/`` (panel A from the original analysis,
panels B-C generated with NoMaSS); set ``recompute = True`` to regenerate them
(about ten minutes).
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)  # so that `import nomass` works without installing it
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # data and output paths are relative to this script

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import nomass

plt.rcParams.update({'font.size': 14})

LAMBDA_ON = 0.14 / 20
LAMBDA_OFF = 0.55 / 29
LAMBDA_PROD = 3.5 / 5
LAMBDA_DEG = 0.14 / 5
TEND = 15 / min(LAMBDA_ON, LAMBDA_OFF, LAMBDA_PROD, LAMBDA_DEG)
TIMEPOINTS = 500
ALPHA_LIST = [1, 10, 25, 50, 75, 100]
N_SIMULATIONS = 2000
recompute = False


def simulate(alpha, N_simulations=N_SIMULATIONS):
    """Return the time axis and the populations (realisations, time, [Goff, Gon, mRNA])."""
    on = nomass.Reaction_channel('Goff -> Gon', rate=LAMBDA_ON, shape_param=alpha, distribution='Gamma')
    off = nomass.Reaction_channel('Gon -> Goff', rate=LAMBDA_OFF, shape_param=alpha, distribution='Gamma')
    prod = nomass.Reaction_channel('Gon -> Gon + mRNA', rate=LAMBDA_PROD, transfer_identity=True)
    deg = nomass.Reaction_channel('mRNA -> 0', rate=LAMBDA_DEG)
    sim = nomass.Gillespie_simulation({'Goff': 1, 'Gon': 0, 'mRNA': 0}, [on, off, prod, deg], Tend=TEND,
                                      N_simulations=N_simulations, timepoints=TIMEPOINTS, unit='min')
    population = sim.run_simulations(verbose=False)
    return np.linspace(0, TEND, TIMEPOINTS), population[:, :, :3]


def amplitude_factor(time, mrna, smoothing=6):
    """Ratio of the second to the first peak amplitude of the averaged mRNA count.

    Amplitudes are measured relative to the steady-state mean (second half of
    the trace) on a Gaussian-smoothed signal; returns (factor, peak times, peak values).
    """
    smooth = gaussian_filter1d(mrna, smoothing)
    steady = mrna[len(mrna) // 2:].mean()
    # only peaks that rise at least 5% of the steady-state level above their
    # surroundings count as oscillations (a Markovian trace has none)
    peaks, _ = find_peaks(smooth, prominence=0.05 * steady)
    peaks = [p for p in peaks if smooth[p] > 1.05 * steady]
    if len(peaks) < 2:
        return 0.0, time[peaks[:1]], smooth[peaks[:1]]
    factor = (smooth[peaks[1]] - steady) / (smooth[peaks[0]] - steady)
    return factor, time[peaks[:2]], smooth[peaks[:2]]


def main():
    # ---- A: promoter state probabilities (precomputed by the original analysis)
    markov = np.load('results/markov_pop.npy')      # columns: P(Goff), P(Gon), alpha = 1
    nmarkov = np.load('results/nmarkov_pop.npy')    # alpha = 50
    time = np.linspace(0, TEND, markov.shape[0])
    plt.figure(figsize=(5.5, 4))
    plt.plot(time, nmarkov[:, 0], lw=3, color='orange', label='Goff')
    plt.plot(time, nmarkov[:, 1], lw=3, color='green', label='Gon')
    plt.plot(time, markov[:, 0], lw=2, ls='--', color='orange')
    plt.plot(time, markov[:, 1], lw=2, ls='--', color='green')
    plt.plot([], [], 'k--', lw=2, label=r'$\alpha = 1$')
    plt.plot([], [], 'k-', lw=3, label=r'$\alpha = 50$')
    plt.xlim(0, 1500)
    plt.xlabel('Time [min]')
    plt.ylabel('Probability')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()

    # ---- B, C: averaged mRNA traces and amplitude factors
    cache = 'results/oscillation_amplitude.npz'
    if recompute or not os.path.exists(cache):
        traces, factors = {}, {}
        for alpha in ALPHA_LIST:
            print('Simulating alpha = %s ...' % alpha)
            t, population = simulate(alpha)
            traces[alpha] = population[:, :, 2].mean(axis=0)
            factors[alpha] = amplitude_factor(t, traces[alpha])[0]
            print('   amplitude factor %.2f' % factors[alpha])
        np.savez(cache, time=t, alphas=np.array(ALPHA_LIST),
                 traces=np.array([traces[a] for a in ALPHA_LIST]),
                 factors=np.array([factors[a] for a in ALPHA_LIST]))
    data = np.load(cache)
    t, alphas, traces = data['time'], data['alphas'], data['traces']
    factors = np.array([amplitude_factor(t, trace)[0] for trace in traces])
    for alpha, factor in zip(alphas, factors):
        print('alpha = %-4g amplitude factor %.2f' % (alpha, factor))

    show = [100, 25, 10]
    fig, axes = plt.subplots(len(show), 1, figsize=(6, 6), sharex=True)
    for ax, alpha in zip(axes, show):
        trace = traces[list(alphas).index(alpha)]
        factor, t_peaks, v_peaks = amplitude_factor(t, trace)
        ax.plot(t, trace, 'k', lw=1.5)
        ax.axhline(trace[len(trace) // 2:].mean(), color='gray', ls='--')
        if len(t_peaks) == 2:
            ax.annotate('x%.2f' % factor, xy=(t_peaks[1], v_peaks[1]), xytext=(t_peaks[1] - 150, v_peaks[1] * 1.1),
                        color='red', arrowprops=dict(arrowstyle='->', color='red'))
        ax.set_ylabel('mRNA')
        ax.text(0.98, 0.85, r'$\alpha = %d$' % alpha, transform=ax.transAxes, ha='right')
    axes[-1].set_xlabel('Time [min]')
    axes[0].set_xlim(0, 1500)
    fig.tight_layout()
    plt.show()

    plt.figure(figsize=(5, 4))
    plt.plot(alphas, factors, 'k-o')
    plt.ylim(0, 1)
    plt.xlabel(r'shape parameter $\alpha$')
    plt.ylabel('Amplitude factor')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
