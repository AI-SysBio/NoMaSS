"""
Fig. S3: growth rate of asymmetric division as a function of the rates and shapes.

    S -> S + Q    division,        Gamma(alpha_div),  rate lambda_div
    Q -> S        differentiation, Gamma(alpha_diff), rate lambda_diff

A: growth rate of the total population over a 19 x 19 grid of
   (lambda_div, lambda_diff) with alpha_div = alpha_diff = 1 (log colour scale).
B: growth rate over a grid of (alpha_div, alpha_diff) with lambda_div = lambda_diff = 1.
The projections averaged over each axis are plotted next to the heatmaps.

The grids ship precomputed in ``results/growth_rates_*.npy``; set
``recompute = True`` to regenerate them (several hours: 2 x 361 simulations,
each stopped when the population reaches ``MAX_POPULATION``).
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)  # so that `import nomass` works without installing it
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # data and output paths are relative to this script

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter
import nomass

plt.rcParams.update({'font.size': 14})

GRID = np.logspace(0, 2, num=19)
N_SIMULATIONS = 5
MAX_POPULATION = 10000
recompute = False


def growth_rate(lambda_div, lambda_diff, alpha_div, alpha_diff):
    """Mean and std over realisations of the fitted exponential growth rate."""
    Tend = 6 / min(lambda_div, lambda_diff)
    division = nomass.Reaction_channel('S -> S + Q', rate=lambda_div, shape_param=alpha_div, distribution='Gamma')
    differentiation = nomass.Reaction_channel('Q -> S', rate=lambda_diff, shape_param=alpha_diff, distribution='Gamma')
    sim = nomass.Gillespie_simulation({'S': 100, 'Q': 0}, [division, differentiation], Tend=Tend,
                                      N_simulations=N_SIMULATIONS, timepoints=1000, max_population=MAX_POPULATION)
    population = sim.run_simulations(verbose=False)[:, :, -1]
    time = np.linspace(0, Tend, population.shape[1])
    # fit log(N) over the middle of the growth phase, before the population cap
    valid = np.where(population.max(axis=0) < 0.9 * MAX_POPULATION)[0]
    window = valid[len(valid) // 2: 6 * len(valid) // 7]
    rates = [np.polyfit(time[window], np.log(population[i, window]), 1)[0] for i in range(population.shape[0])]
    return float(np.mean(rates)), float(np.std(rates))


def compute_grids():
    n = len(GRID)
    g_lambda, g_lambda_std = np.zeros((n, n)), np.zeros((n, n))
    g_alpha, g_alpha_std = np.zeros((n, n)), np.zeros((n, n))
    for i, vi in enumerate(GRID):          # rows: differentiation parameter
        for j, vj in enumerate(GRID):      # columns: division parameter
            print('lambda grid %d/%d' % (i * n + j + 1, n * n))
            g_lambda[i, j], g_lambda_std[i, j] = growth_rate(vj, vi, 1, 1)
    for i, vi in enumerate(GRID):
        for j, vj in enumerate(GRID):
            print('alpha grid %d/%d' % (i * n + j + 1, n * n))
            g_alpha[i, j], g_alpha_std[i, j] = growth_rate(1, 1, vj, vi)
    np.save('results/growth_rates_lambda.npy', g_lambda)
    np.save('results/growth_rates_lambda_std.npy', g_lambda_std)
    np.save('results/growth_rates_alpha.npy', g_alpha)
    np.save('results/growth_rates_alpha_std.npy', g_alpha_std)


def plot_heatmap(rates, symbol, log_colour, smooth=False):
    if smooth:
        rates = gaussian_filter(rates, sigma=1)
    logx = np.log10(GRID)
    fine = np.linspace(logx.min(), logx.max(), 100)
    z = RectBivariateSpline(logx, logx, rates, kx=3, ky=3)(fine, fine)   # rows: diff, columns: div
    ticks = np.linspace(logx.min(), logx.max(), 9)
    labels = ['%.3g' % v for v in 10 ** ticks]

    fig = plt.figure(figsize=(7.5, 7))
    grid = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.05, wspace=0.05)
    ax = fig.add_subplot(grid[1, 0])
    img = ax.imshow(np.log(z) if log_colour else z, origin='lower', cmap='inferno', aspect='auto',
                    extent=[logx.min(), logx.max(), logx.min(), logx.max()])
    ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks(ticks); ax.set_yticklabels(labels)
    ax.set_xlabel(r'$\%s_{div}$' % symbol)
    ax.set_ylabel(r'$\%s_{diff}$' % symbol)
    ax_top = fig.add_subplot(grid[0, 0], sharex=ax)
    ax_top.fill_between(fine, 0, z.mean(axis=0), color='gray', alpha=0.5)
    ax_top.plot(fine, z.mean(axis=0), 'k', lw=2)
    ax_top.set_ylim(z.min(), z.max()); ax_top.tick_params(labelbottom=False); ax_top.set_ylabel('Growth rate', fontsize=10)
    ax_right = fig.add_subplot(grid[1, 1], sharey=ax)
    ax_right.fill_betweenx(fine, 0, z.mean(axis=1), color='gray', alpha=0.5)
    ax_right.plot(z.mean(axis=1), fine, 'k', lw=2)
    ax_right.set_xlim(z.min(), z.max()); ax_right.tick_params(labelleft=False); ax_right.set_xlabel('Growth rate', fontsize=10)
    cbar = fig.colorbar(img, ax=[ax, ax_top, ax_right], location='left', pad=0.15, shrink=0.6)
    if log_colour:
        cticks = np.linspace(np.log(z.min()), np.log(z.max()), 6)
        cbar.set_ticks(cticks); cbar.set_ticklabels(['%.2g' % np.exp(c) for c in cticks])
    cbar.set_label('Growth rate')
    plt.show()


def main():
    if recompute or not os.path.exists('results/growth_rates_lambda.npy'):
        compute_grids()
    plot_heatmap(np.load('results/growth_rates_lambda.npy'), 'lambda', log_colour=True)
    plot_heatmap(np.load('results/growth_rates_alpha.npy'), 'alpha', log_colour=False, smooth=True)


if __name__ == '__main__':
    main()
