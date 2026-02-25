import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from analysis_functions import op
from figure2_orderparam import label_axes

# Style
mpl.style.use('prl_style.mplstyle')

# Base directories — set this to the folder containing your soc_simulation_runs/ and vicsek_simulation_runs/ subdirectories
BASE_DIR = r'X:\\soc_project'  # <-- CHANGE THIS
SOC_DIR = os.path.join(BASE_DIR, 'soc_simulation_runs')
VICSEK_DIR = os.path.join(BASE_DIR, 'vicsek_simulation_runs')

# Target grids (match figure_s4_orderparameter)
DENSITIES = [0.5, 1.0, 1.5]
NOISES = [0.1813799364234218, 0.3627598728468436, 0.5441398092702654]
NOISE_TOL = 1e-2

# (epsilon, gamma) combinations and colors (match figure_s2_avalanchesL)
COMBINATIONS = [(0.3, -0.6), (0.6, -0.6), (0.6, -0.3)]
TAB10 = plt.get_cmap('tab10')
COLOR_MAP = {
    (0.3, -0.6): TAB10(0),
    (0.6, -0.6): TAB10(2),
    (0.6, -0.3): TAB10(3),
}


def _approx_equal(a, b, tol):
    return abs(a - b) <= tol


def load_params(path):
    with open(path, 'r') as f:
        return json.load(f)


def load_observables(run_id, base_dir):
    path = os.path.join(base_dir, f'observables_{run_id}.npz')
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True)


def series_mean_sem(obs):
    """Extract order parameter series from observables and compute (mean, std) after equilibration.

    If 'tc' is present in obs, use it; else fall back to len(series)//10.
    Note: despite the function name, the second return value is the
    standard deviation of the series (not the standard error).
    """
    if obs is None:
        return None, None
    if 'orderparameter' in obs:
        series = obs['orderparameter']
    else:
        data = obs.get('data', None)
        if data is None:
            return None, None
        series = op(data[:, 2, :], axis=1)

    tc = obs.get('tc', None)
    if tc is None:
        tc = len(series) // 10
    elif hasattr(tc, 'item'):
        tc = tc.item()

    if tc is not None and 0 <= tc < len(series):
        series = series[int(tc):]

    if series.size == 0:
        return None, None

    m = float(np.mean(series))
    # Standard deviation over the time series (fluctuations)
    if series.size > 1:
        s = float(np.std(series, ddof=1))
    else:
        s = 0.0
    return m, s


def aggregate_over_runs(values):
    """Aggregate per-run (mean, std_time) into overall (mean, std).

    We report the mean of the per-run means, and a typical fluctuation size
    represented by the RMS of per-run standard deviations. If only one run
    exists, that run's std is used.
    """
    means = [v[0] for v in values if v[0] is not None]
    std_time = [v[1] for v in values if v[0] is not None]
    if len(means) == 0:
        return None, None
    mean_all = float(np.mean(means))
    if len(std_time) >= 2:
        std_all = float(np.sqrt(np.mean(np.square(std_time))))
    else:
        std_all = float(std_time[0] if std_time else 0.0)
    return mean_all, std_all


def gather_soc_stats_for_key(rho, eta):
    """Return dict mapping combo -> (Ls_sorted, means, sems) for given (rho, eta)."""
    # Collect run info: {(combo, L) -> list[(mean, sem_time)]}
    buckets = {}
    for fname in os.listdir(SOC_DIR):
        if not fname.endswith('_params.json'):
            continue
        try:
            run_id = int(fname.split('_')[1])
        except Exception:
            continue
        p = load_params(os.path.join(SOC_DIR, fname))

        p_rho = p.get('density', None)
        p_eta = p.get('noise', None)
        L = p.get('L', None)
        eps = p.get('epsilon', None)
        gam = p.get('gamma', None)
        if None in (p_rho, p_eta, L, eps, gam):
            continue

        # Snap to grid and filter
        rho_ok = _approx_equal(p_rho, rho, 1e-6)
        eta_ok = _approx_equal(p_eta, eta, NOISE_TOL)
        combo = (round(eps, 1), round(gam, 1))
        if not (rho_ok and eta_ok and combo in COMBINATIONS):
            continue

        obs = load_observables(run_id, SOC_DIR)
        m, s = series_mean_sem(obs)
        key = (combo, int(L))
        buckets.setdefault(key, []).append((m, s))

    # Reduce to arrays per combo
    out = {}
    for combo in COMBINATIONS:
        # Collect all L that exist for this combo
        Ls = sorted({L for (c, L) in buckets.keys() if c == combo})
        if not Ls:
            out[combo] = ([], [], [])
            continue
        means = []
        sems = []
        for L in Ls:
            agg = aggregate_over_runs(buckets.get((combo, L), []))
            if agg == (None, None):
                continue
            means.append(agg[0])
            sems.append(agg[1])
        out[combo] = (Ls, means, sems)
    return out


def gather_vm_stats_for_key(rho, eta):
    """Return (Ls_sorted, means, sems) for Vicsek runs at given (rho, eta)."""
    # Map L -> list[(mean, sem_time)]
    buckets = {}
    for fname in os.listdir(VICSEK_DIR):
        if not fname.endswith('_params.json'):
            continue
        try:
            run_id = int(fname.split('_')[1])
        except Exception:
            continue
        p = load_params(os.path.join(VICSEK_DIR, fname))
        p_rho = p.get('density', None)
        p_eta = p.get('noise', None)
        L = p.get('L', None)
        if None in (p_rho, p_eta, L):
            continue
        if not (_approx_equal(p_rho, rho, 1e-6) and _approx_equal(p_eta, eta, NOISE_TOL)):
            continue
        obs = load_observables(run_id, VICSEK_DIR)
        m, s = series_mean_sem(obs)
        buckets.setdefault(int(L), []).append((m, s))

    Ls = sorted(buckets.keys())
    if not Ls:
        return [], [], []
    means = []
    sems = []
    for L in Ls:
        m, s = aggregate_over_runs(buckets[L])
        if m is None:
            continue
        means.append(m)
        sems.append(s)
    return Ls, means, sems


def plot_panels():
    densities = DENSITIES
    noises = NOISES
    display_noises = [0.1, 0.2, 0.3]

    nrows, ncols = 3, 3
    fig_width = plt.rcParams['figure.figsize'][0]
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_width * 1.8, 0.7 * (fig_width * 1.5) * (nrows / 2)),
        sharex=True, sharey=True,
        constrained_layout=True,
    )

    # Iterate grid
    for i, rho in enumerate(densities):
        for j, eta in enumerate(noises):
            ax = axes[i, j]

            # SOC lines for each (epsilon, gamma)
            soc_stats = gather_soc_stats_for_key(rho, eta)
            any_line = False
            for combo in COMBINATIONS:
                Ls, means, sems = soc_stats.get(combo, ([], [], []))
                if Ls and means:
                    ax.errorbar(
                        Ls, means, yerr=sems,
                        label=fr'$\epsilon={combo[0]:.1f}$, $\gamma={combo[1]:.1f}$',
                        color=COLOR_MAP.get(combo, None),
                        marker='o', linestyle='-'
                    )
                    if len(sems) > 0:
                        arr = np.asarray(sems, dtype=float)
                        print(
                            f"SOC (rho={rho}, eta={eta}, combo={combo}) STD: min={arr.min():.3e}, max={arr.max():.3e}, mean={arr.mean():.3e}")
                    any_line = True

            # Vicsek baseline
            vm_Ls, vm_means, vm_sems = gather_vm_stats_for_key(rho, eta)
            if vm_Ls and vm_means:
                ax.errorbar(vm_Ls, vm_means, yerr=vm_sems, label='Vicsek Model',
                            color='grey', marker='s', linestyle='--')
                if len(vm_sems) > 0:
                    arr = np.asarray(vm_sems, dtype=float)
                    print(
                        f"Vicsek (rho={rho}, eta={eta}) STD: min={arr.min():.3e}, max={arr.max():.3e}, mean={arr.mean():.3e}")
                any_line = True

            if not any_line:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

            # Titles and axis labels on edges
            ax.set_title(fr'$\rho={rho:.2f}, \eta={display_noises[j]:.1f}$')
            

            ax.set_ylim(0, 1.05)
            ax.set_xlim(left=0)

    # Legend in the first axis (top left)
    axes[0, 0].legend(loc='best', frameon=True)

    # Global axis labels
    fig.supxlabel(r'System size $L$')
    fig.supylabel(r'Order parameter $\phi$')

    # Label panels (a)–(i)
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']
    label_axes(fig, labels=labels, loc=(0.92, 0.9))

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out_pdf = os.path.join(out_dir, 'figure_s5_orderVSsize.pdf')
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_pdf.replace('.pdf', '.eps'), bbox_inches='tight', dpi=300)
    print(f'Supplement figure saved to {out_pdf}')


def main():
    plot_panels()


if __name__ == '__main__':
    main()
