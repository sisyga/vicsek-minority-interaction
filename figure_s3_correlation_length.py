import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit

# Reuse helpers from figure4
from figure4_correlation_length import (
    load_params,
    get_run_data,
    calculate_correlation,
    find_matching_vicsek_run,
)
from analysis_functions import vel_corr
from figure2_orderparam import label_axes

# Style
mpl.style.use('prl_style.mplstyle')

# Base directory for data — set this to the folder containing your soc_simulation_runs/ and vicsek_simulation_runs/ subdirectories
BASE_DIR = r'X:\\soc_project'  # <-- CHANGE THIS
VM_DIR = os.path.join(BASE_DIR, 'vicsek_simulation_runs')
SOC_DIR = os.path.join(BASE_DIR, 'soc_simulation_runs')

# System sizes considered (as in figure4)
VALID_SIZES = [16, 32, 64, 128]

# Epsilon, gamma combinations (hardcoded)
COMBINATIONS = [
    (0.3, -0.3),
    (0.3, -0.6),
    (0.3, -0.9),
    (0.6, -0.9),
    (0.6, -0.6),
    (0.6, -0.3),
    (0.9, -0.9),
    (0.9, -0.6),
    (0.9, -0.3),
]


def find_correlation_length(rs, corr_values):
    """Distance where correlation crosses zero (linear interpolation)."""
    if np.all(corr_values > 0):
        return np.max(rs)
    try:
        cross_idx = np.where(corr_values <= 0)[0][0]
        if cross_idx == 0:
            return rs[0]
        r1, r2 = rs[cross_idx - 1], rs[cross_idx]
        c1, c2 = corr_values[cross_idx - 1], corr_values[cross_idx]
        r0 = r1 - c1 * (r2 - r1) / (c2 - c1)
        return r0
    except IndexError:
        return np.max(rs)


def st_line(x, m, c):
    return m * x + c


def filter_runs_for_combo(eps, gam, min_noise):
    """Collect run IDs for given (epsilon, gamma) at density=1.0 and noise=min_noise, grouped by L."""
    runs_by_L = {L: [] for L in VALID_SIZES}
    for f in os.listdir(SOC_DIR):
        if not f.endswith('_params.json'):
            continue
        run_id = int(f.split('_')[1])
        params = load_params(os.path.join(SOC_DIR, f))
        L = params.get('L', None)
        if L not in VALID_SIZES:
            continue
        if abs(params.get('density', 0) - 1.0) > 1e-6:
            continue
        if abs(params.get('noise', 0) - min_noise) > 1e-6:
            continue
        if abs(params.get('epsilon', 0) - eps) > 1e-6:
            continue
        if abs(params.get('gamma', 0) - gam) > 1e-6:
            continue
        runs_by_L[L].append(run_id)
    return runs_by_L


def load_or_compute_corr(run_id, data_dir):
    """Return (rs, corr) for a run; compute if missing using figure4 helper."""
    try:
        params, obs = get_run_data(run_id, 'SOC' if 'soc_' in data_dir else 'Vicsek')
    except Exception:
        return None, None
    if 'velocity_fluctuation_correlation' in obs and 'rs' in obs:
        return obs['rs'], obs['velocity_fluctuation_correlation']
    try:
        return calculate_correlation(run_id, data_dir)
    except Exception:
        return None, None


def aggregate_corr_for_L(run_ids, data_dir):
    """Average correlation over runs for the same L."""
    all_rs = []
    all_corr = []
    for rid in run_ids:
        rs, corr = load_or_compute_corr(rid, data_dir)
        if rs is None or corr is None:
            continue
        all_rs.append(rs)
        all_corr.append(corr)
    if not all_corr:
        return None, None
    rs0 = all_rs[0]
    mean_corr = np.mean(all_corr, axis=0)
    return rs0, mean_corr


def vicsek_run_for_L(L, min_noise):
    """Find a Vicsek run ID matching L, density=1.0, noise=min_noise."""
    from figure4_correlation_length import find_matching_vicsek_run
    return find_matching_vicsek_run(L, min_noise)


def main():
    # Determine smallest noise value among SOC runs (as in fig4)
    min_noise = float('inf')
    for f in os.listdir(SOC_DIR):
        if f.endswith('_params.json'):
            p = load_params(os.path.join(SOC_DIR, f))
            noise = p.get('noise', float('inf'))
            if noise < min_noise:
                min_noise = noise

    # Build two figures: (a) correlation functions panel grid, (b) correlation length vs L
    n = len(COMBINATIONS)
    # Arrange panels in a near-square grid (5 combos -> 2x3)
    nrows, ncols = 3, 3
    figA_width = plt.rcParams['figure.figsize'][0]
    figA, axesA = plt.subplots(nrows, ncols, figsize=(figA_width * 1.6, figA_width * 1.2), sharex=True, sharey=True)
    figB, axesB = plt.subplots(nrows, ncols, figsize=(figA_width * 1.6, figA_width * 1.2), sharex=True, sharey=True)

    # Colormap for system sizes
    cmap = plt.cm.viridis
    colors = [cmap(i / (len(VALID_SIZES) - 1)) for i in range(len(VALID_SIZES))]

    panel = 0
    for eps, gam in COMBINATIONS:
        i, j = divmod(panel, ncols)
        axA = axesA[i, j]
        axB = axesB[i, j]
        panel += 1

        # Gather SOC correlations by L
        runs_by_L = filter_runs_for_combo(eps, gam, min_noise)

        soc_corr_lengths = []
        vm_corr_lengths = []

        for idx, L in enumerate(VALID_SIZES):
            # SOC
            rs_soc, corr_soc = aggregate_corr_for_L(runs_by_L.get(L, []), SOC_DIR)
            if rs_soc is not None and corr_soc is not None:
                axA.plot(rs_soc, corr_soc, '-', color=colors[idx], label=fr'$L={L}$')
                cl_soc = find_correlation_length(rs_soc, corr_soc)
                soc_corr_lengths.append((L, cl_soc))

            # Vicsek (matching)
            vm_id = vicsek_run_for_L(L, min_noise)
            if vm_id:
                rs_vm, corr_vm = load_or_compute_corr(vm_id, VM_DIR)
                if rs_vm is not None and corr_vm is not None:
                    axA.plot(rs_vm, corr_vm, '--', color=colors[idx], alpha=0.8)
                    vm_cl = find_correlation_length(rs_vm, corr_vm)
                    vm_corr_lengths.append((L, vm_cl))

        # Axis settings like fig4
        axA.set_xlabel(r'Distance $d$')
        axA.set_ylabel(r'$C(d)$')
        axA.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axA.set_ylim(1e-4, 1.0)
        axA.set_yscale('log')
        axA.set_xlim(-1, 40)
        axA.set_title(fr'$\epsilon={eps}, \gamma={gam}$')
        if i == 0 and j == 0:
            axA.legend(frameon=True)

        # Panel B: correlation length vs L
        if soc_corr_lengths:
            sizes_arr = np.array([cl[0] for cl in soc_corr_lengths])
            lengths_arr = np.array([cl[1] for cl in soc_corr_lengths])
            axB.plot(sizes_arr, lengths_arr, 'ko', label='Minority I.A.')
            if len(sizes_arr) > 1:
                try:
                    popt, _ = curve_fit(lambda x, m, c: m * x + c, sizes_arr, lengths_arr)
                    xs = np.linspace(min(sizes_arr), max(sizes_arr), 100)
                    axB.plot(xs, st_line(xs, *popt), '--', c='k')
                except Exception:
                    pass

        if vm_corr_lengths:
            vm_sizes_arr = np.array([cl[0] for cl in vm_corr_lengths])
            vm_lengths_arr = np.array([cl[1] for cl in vm_corr_lengths])
            axB.plot(vm_sizes_arr, vm_lengths_arr, 'o', c='grey', label='Vicsek model')
            if len(vm_sizes_arr) > 1:
                try:
                    vm_popt, _ = curve_fit(lambda x, m, c: m * x + c, vm_sizes_arr, vm_lengths_arr)
                    xs = np.linspace(min(vm_sizes_arr), max(vm_sizes_arr), 100)
                    axB.plot(xs, st_line(xs, *vm_popt), '--', color='grey')
                except Exception:
                    pass

        axB.set_xlabel(r'System size $L$')
        axB.set_ylabel('Correlation length $d_0$')
        axB.set_title(fr'$\epsilon={eps}, \gamma={gam}$')

    # Clean up empty panels if combos < nrows*ncols
    total_panels = nrows * ncols
    for k in range(panel, total_panels):
        i, j = divmod(k, ncols)
        figA.delaxes(axesA[i, j])
        figB.delaxes(axesB[i, j])

    label_axes(figA, labels=['(a), (b), (c), (d), (e), (f), (g), (h), (i)'], loc=(0.95, 0.9))
    label_axes(figB, labels=['(a), (b), (c), (d), (e), (f), (g), (h), (i)'], loc=(0.95, 0.9))

    figA.tight_layout()
    figB.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(out_dir, exist_ok=True)
    pA = os.path.join(out_dir, 'figure_s3_correlation_length_a.pdf')
    pB = os.path.join(out_dir, 'figure_s3_correlation_length_b.pdf')
    figA.savefig(pA, bbox_inches='tight')
    figA.savefig(pA.replace('.pdf', '.eps'), bbox_inches='tight', dpi=300)
    figB.savefig(pB, bbox_inches='tight')
    figB.savefig(pB.replace('.pdf', '.eps'), bbox_inches='tight', dpi=300)
    print(f'Saved {pA} and {pB}')


if __name__ == '__main__':
    main()
