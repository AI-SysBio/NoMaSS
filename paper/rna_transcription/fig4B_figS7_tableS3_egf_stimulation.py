"""
Fig. 4B, Fig. S7 and Table S3: mRNA response to EGF stimulation (qPCR time courses).

Two-state model of transcription with a time-dependent promoter activation rate:

    Goff -> Gon          Gamma(alpha), rate lambda_on(t)
    Gon -> Goff          Gamma(alpha), rate lambda_off
    Gon -> Gon + mRNA    Exponential, rate lambda_prod   (transfer_identity: same gene)
    mRNA -> 0            Exponential, rate lambda_deg

The system is simulated to steady state, then the stimulus is applied at t0:

    lambda_on(t) = lambda_A * [ g_sigmoid(t - t0) + g_peak(t - t0) ]
    g_sigmoid(t) = (1 - lambda_B/lambda_A) * sigmoid(-b (t - t00)) + lambda_B/lambda_A
    g_peak(t)    = a1 * (t/tp)^n1 / (1 + (t/tp)^n2),   tp = 4 min, n1 = 2, n2 = 4

(Supplementary Eqs. 22-24; note that the code uses tp = 4 min). Simulated mRNA
counts are averaged over genes and realisations and normalised to their value
at the stimulus, like the measured fold changes of Zeisel et al. (2011).

Parameters are those of Table S3 (``PARAMETERS`` below). The non-Markovian
model (NoMaSS, shape alpha) is compared with the Markovian one (SG, alpha = 1,
no peak term). The score is the mean squared deviation between measured and
simulated fold changes at the measured time points; this is the quantity
reported as "RMSD" in Table S3 and Fig. 4B.

Simulated curves ship precomputed in ``results/`` (non-Markovian model for the
four genes, Markovian model for SLC7A5); set ``recompute = True`` to regenerate
everything with NoMaSS (about a minute per gene and model).
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)  # so that `import nomass` works without installing it
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # data and output paths are relative to this script

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nomass

plt.rcParams.update({'font.size': 14})

# ---- fixed rates [1/min], Table S3
LAMBDA_DEG = 0.028
LAMBDA_OFF = 4 * LAMBDA_DEG      # 0.112
LAMBDA_PROD = 25 * LAMBDA_DEG    # 0.7
TP, N1, N2 = 4.0, 2, 4           # peak term

# ---- per-gene parameters, Table S3: lambda_A, lambda_B / lambda_A, b [1/min], t00 [min], a1, alpha
PARAMETERS = {
    'CREG1':  {'NoMaSS': dict(lamA=LAMBDA_OFF / 4.5, ratio=0.45, b=0.006, t00=0,   a1=-1.2, alpha=40),
               'SG':     dict(lamA=LAMBDA_OFF / 4.5, ratio=0.45, b=0.004, t00=0,   a1=0.0,  alpha=1)},
    'DDB2':   {'NoMaSS': dict(lamA=LAMBDA_OFF / 9,   ratio=0.66, b=0.012, t00=250, a1=1.0,  alpha=60),
               'SG':     dict(lamA=LAMBDA_OFF / 9,   ratio=0.66, b=0.029, t00=214, a1=0.0,  alpha=1)},
    'NFYC':   {'NoMaSS': dict(lamA=LAMBDA_OFF / 18,  ratio=0.64, b=0.008, t00=160, a1=0.55, alpha=15),
               'SG':     dict(lamA=LAMBDA_OFF / 18,  ratio=0.64, b=0.015, t00=150, a1=0.0,  alpha=1)},
    'SLC7A5': {'NoMaSS': dict(lamA=LAMBDA_OFF / 9.6, ratio=0.63, b=0.035, t00=270, a1=1.3,  alpha=26),
               'SG':     dict(lamA=LAMBDA_OFF / 9.6, ratio=0.63, b=0.041, t00=270, a1=0.0,  alpha=1)},
}
GENES = ['SLC7A5', 'DDB2', 'NFYC', 'CREG1']
N_GENES = 100          # genes simulated in parallel in one realisation
N_SIMULATIONS = 30
recompute = False


def activation_factor(t, p):
    """lambda_on(t) / lambda_A as a function of the time since the stimulus (t >= 0)."""
    t = np.asarray(t, dtype=float)
    sigmoid = np.exp(-p['b'] * (t - p['t00'])) / (1 + np.exp(-p['b'] * (t - p['t00'])))
    g_sigmoid = (1 - p['ratio']) * sigmoid + p['ratio']
    g_peak = p['a1'] * (t / TP) ** N1 / (1 + (t / TP) ** N2)
    return g_sigmoid + g_peak


def simulate(p, N_simulations=N_SIMULATIONS):
    """Return (time since stimulus, fold change of the mean mRNA count)."""
    rmin = min(p['lamA'], LAMBDA_OFF, LAMBDA_PROD, LAMBDA_DEG)
    Tend = 30 / rmin
    t0 = Tend / 2

    on = nomass.Reaction_channel('Goff -> Gon', rate=p['lamA'], shape_param=p['alpha'], distribution='Gamma',
                                 name='Gene activation')
    off = nomass.Reaction_channel('Gon -> Goff', rate=LAMBDA_OFF, shape_param=p['alpha'], distribution='Gamma',
                                  name='Gene deactivation')
    prod = nomass.Reaction_channel('Gon -> Gon + mRNA', rate=LAMBDA_PROD, transfer_identity=True,
                                   name='mRNA production')
    deg = nomass.Reaction_channel('mRNA -> 0', rate=LAMBDA_DEG, name='mRNA degradation')

    def stimulus(sim):
        if sim.t >= t0:
            on.rate = p['lamA'] * float(activation_factor(sim.t - t0, p))

    sim = nomass.Gillespie_simulation({'Goff': N_GENES, 'Gon': 0, 'mRNA': 0}, [on, off, prod, deg],
                                      Tend=Tend, N_simulations=N_simulations, timepoints=1000, unit='min',
                                      update_function=stimulus)
    population = sim.run_simulations(verbose=False)
    time = np.linspace(0, Tend, population.shape[1])
    mrna = population[:, :, 2].mean(axis=0)
    i0 = np.argmin(np.abs(time - t0))
    return time - t0, mrna / mrna[i0], i0


def score(data, t_sim, fold_sim):
    """Mean squared deviation at the measured time points (the 'RMSD' of Table S3)."""
    predicted = np.interp(data['mRNA_X'].to_numpy(), t_sim, fold_sim)
    return float(np.mean((data['mRNA_Y'].to_numpy() - predicted) ** 2))


def main():
    curves = {}
    for gene in GENES:
        for model in ('NoMaSS', 'SG'):
            cache = 'results/%s_%s.npy' % (gene, 'NoMaSS' if model == 'NoMaSS' else 'Markovian')
            if recompute or not os.path.exists(cache):
                print('Simulating %s, %s model ...' % (gene, model))
                t, fold, i0 = simulate(PARAMETERS[gene][model])
                if gene == 'SLC7A5' and model == 'NoMaSS':
                    plot_full_trace(t, fold, PARAMETERS[gene][model])      # Fig. S7A-B
                np.save(cache, np.array([t[i0:], fold[i0:]]))
            curves[(gene, model)] = np.load(cache)

    # ---- Fig. 4B and Fig. S7C: measured vs simulated fold changes
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, gene in zip(axes.ravel(), GENES):
        data = pd.read_csv('data/qpcr/%s.csv' % gene)
        ax.scatter(data['mRNA_X'], data['mRNA_Y'], color='blue', label='Measured (qPCR)', zorder=3)
        for model, style, label in (('NoMaSS', '-', 'Simulated (NoMaSS)'), ('SG', '--', 'Simulated (SG)')):
            t, fold = curves[(gene, model)]
            s = score(data, t, fold)
            ax.plot(t, fold, 'k', ls=style, lw=2.5 if model == 'NoMaSS' else 1.5,
                    label='%s, RMSD %.4f' % (label, s))
            print('%-7s %-7s score = %.4f' % (gene, model, s))
        ax.set_xlim(-20, data['mRNA_X'].max() * 1.05)
        ax.set_title(gene)
        ax.set_xlabel('Time [min]')
        ax.set_ylabel('Fold change to t=0')
        ax.legend(fontsize=9)
    fig.tight_layout()
    plt.show()


def plot_full_trace(t, fold, p):
    """Fig. S7A-B: full simulated trace of SLC7A5 and the activation rate lambda_on(t)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    ax1.plot(t, fold, 'k', lw=1.5, label='Simulated (NoMaSS)')
    ax1.axvline(0, color='blue', alpha=0.5)
    ax1.set_ylabel('Fold change to t=0')
    ax1.set_title('SLC7A5')
    ax1.legend(fontsize=10)
    lam = np.where(t >= 0, p['lamA'] * activation_factor(np.maximum(t, 0), p), p['lamA'])
    ax2.plot(t, lam, 'k', lw=1.5)
    ax2.set_ylabel(r'$\lambda_{on}$ [1/min]')
    ax2.set_xlabel('Time [min]')
    ax2.set_ylim(0, None)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
