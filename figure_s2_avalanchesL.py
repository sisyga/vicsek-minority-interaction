import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Reuse helper functions from figure3
from figure3_avalanches import (
    load_params,
    op,
    find_tc,
    identify_avalanches,
    get_run_data,
    find_matching_vicsek_run,
    find_vicsek_reference_run,
)
from figure2_orderparam import label_axes

# Style
mpl.style.use('prl_style.mplstyle')

# Base directories — set this to the folder containing your soc_simulation_runs/ and vicsek_simulation_runs/ subdirectories
BASE_DIR = r'X:\\soc_project'  # <-- CHANGE THIS
VM_DIR = os.path.join(BASE_DIR, 'vicsek_simulation_runs')
SOC_DIR = os.path.join(BASE_DIR, 'soc_simulation_runs')

# Target system sizes for rows (L = 16, 32, 64, 128)
TARGET_SIZES = [16, 32, 64, 128]

# Hardcoded (epsilon, gamma) combinations as used across figures
COMBINATIONS = [(0.3, -0.6), (0.6, -0.6), (0.6, -0.3)]

# Fixed color mapping consistent with other figures (tab10 indices 0, 2, 3)
TAB10 = plt.get_cmap('tab10')
COLOR_MAP = {
    (0.3, -0.6): TAB10(0),
    (0.6, -0.6): TAB10(2),
    (0.6, -0.3): TAB10(3),
}


def plot_ccdf(data, ax, label=None, color=None, alpha=1.0, marker='o', linestyle='-'):
    if data is None or len(data) == 0:
        return None
    return ax.ecdf(
        data,
        complementary=True,
        marker=marker,
        linestyle=linestyle,
        label=label,
        color=color,
        alpha=alpha,
        compress=True,
    )


def calculate_threshold_from_vicsek(soc_params):
    """Find matching Vicsek run and compute threshold from its order parameter."""
    vm_run_id = find_matching_vicsek_run(soc_params)
    if vm_run_id is None:
        return None
    try:
        _, vm_obs = get_run_data(vm_run_id, 'Vicsek')
    except Exception:
        return None

    if 'orderparameter' in vm_obs:
        vm_op = vm_obs['orderparameter']
    else:
        vm_data = vm_obs.get('data', None)
        if vm_data is None:
            return None
        vm_op = op(vm_data[:, 2, :], axis=1)

    tc = vm_obs.get('tc', None)
    if tc is None:
        tc = find_tc(vm_op)
    elif hasattr(tc, 'item'):
        tc = tc.item()

    return np.mean(vm_op[tc:]) - 3 * np.std(vm_op[tc:])


def collect_avalanche_stats_for_L(L, min_noise):
    """Return dict combo -> (sizes, durations, integrated) for given L at density=1.0 and noise=min_noise."""
    results = {combo: (None, None, None) for combo in COMBINATIONS}

    for f in os.listdir(SOC_DIR):
        if not f.endswith('_params.json'):
            continue
        run_id = int(f.split('_')[1])
        params_path = os.path.join(SOC_DIR, f)
        params = load_params(params_path)

        # Filter by L, density ~ 1, and minimum noise
        if params.get('L', 0) != L:
            continue
        if abs(params.get('density', 0) - 1.0) > 1e-6:
            continue
        if abs(params.get('noise', 0) - min_noise) > 1e-12:
            continue

        combo = (round(params.get('epsilon', 0), 1), round(params.get('gamma', 0), 1))
        if combo not in results:
            continue

        # Load SOC OP
        try:
            _, soc_obs = get_run_data(run_id, 'SOC')
        except Exception:
            continue

        if 'orderparameter' in soc_obs:
            soc_op = soc_obs['orderparameter']
        else:
            soc_data = soc_obs.get('data', None)
            if soc_data is None:
                continue
            soc_op = op(soc_data[:, 2, :], axis=1)

        # Threshold from matching Vicsek run
        threshold = calculate_threshold_from_vicsek(params)
        if threshold is None:
            continue

        sizes, durations, integrated = identify_avalanches(soc_op, threshold, return_integrated=True)
        results[combo] = (sizes, durations, integrated)

    return results


def main():
    # Find smallest noise value across SOC runs (same approach as figure3)
    min_noise = float('inf')
    for f in os.listdir(SOC_DIR):
        if f.endswith('_params.json'):
            p = load_params(os.path.join(SOC_DIR, f))
            noise = p.get('noise', float('inf'))
            if noise < min_noise:
                min_noise = noise

    # Figure layout: rows=len(TARGET_SIZES) x 3 cols (size | integrated excursion | duration)
    fig_width = plt.rcParams['figure.figsize'][0] * 1.5
    rows = len(TARGET_SIZES)
    fig_height = fig_width * (rows / 4.0)
    fig, axes = plt.subplots(rows, 3, figsize=(fig_width, fig_height), sharex='col', sharey='row')

    # Axis formatting (match figure3)
    # Left col: size CCDF (max excursion)
    for i in range(rows):
        ax = axes[i, 0]
        ax.set_ylabel(fr'$P(X\geq x)$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(5e-4, 2)
        ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(subs='all', numticks=15))

    # Middle col: integrated excursion CCDF
    for i in range(rows):
        ax = axes[i, 1]
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(5e-4, 5e5)
        # ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(subs='all', numticks=15))
        # only draw every second major x tick label to reduce clutter

    # Right col: duration CCDF
    for i in range(rows):
        ax = axes[i, 2]
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1, 5e5)
        ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(subs='all', numticks=15))

    # Process each system size (row)
    for row, L in enumerate(TARGET_SIZES):
        stats = collect_avalanche_stats_for_L(L, min_noise)
        ax_size = axes[row, 0]
        ax_int = axes[row, 1]
        ax_dur = axes[row, 2]

        for combo in COMBINATIONS:
            sizes, durs, integrated = stats.get(combo, (None, None, None))
            color = COLOR_MAP.get(combo, None)
            label = fr'$\epsilon={combo[0]:.1f}$, $\gamma={combo[1]:.1f}$'
            plot_ccdf(sizes, ax_size, label=label, color=color)
            # Integrated excursion metric
            if integrated is not None:
                plot_ccdf(integrated, ax_int, label=label, color=color)
            plot_ccdf(durs, ax_dur, label=label, color=color)

        # Put L = X as a title of the middle column axis in each row
        ax_int.set_title(fr'$L={L}$')

        # Add Vicsek baseline (grey, dashed) per-row matching L and min_noise
        vm_sizes = vm_durs = vm_integrated = None
        try:
            # Find a Vicsek run matching this L, density=1.0, and the minimal noise used
            for f in os.listdir(VM_DIR):
                if not f.endswith('_params.json'):
                    continue
                vm_params = load_params(os.path.join(VM_DIR, f))
                if (vm_params.get('L', 0) == L and
                        abs(vm_params.get('density', 0) - 1.0) < 1e-6 and
                        abs(vm_params.get('noise', 0) - min_noise) < 1e-12):
                    vm_run_id = int(f.split('_')[1])
                    _, vm_obs = get_run_data(vm_run_id, 'Vicsek')
                    if 'orderparameter' in vm_obs:
                        vm_op = vm_obs['orderparameter']
                    else:
                        vm_data = vm_obs.get('data', None)
                        vm_op = op(vm_data[:, 2, :], axis=1) if vm_data is not None else None
                    if vm_op is not None:
                        tc = vm_obs.get('tc', None)
                        if tc is None:
                            tc = find_tc(vm_op)
                        elif hasattr(tc, 'item'):
                            tc = tc.item()
                        vm_threshold = np.mean(vm_op[tc:]) - 3 * np.std(vm_op[tc:])
                        vm_sizes, vm_durs, vm_integrated = identify_avalanches(vm_op, vm_threshold,
                                                                               return_integrated=True)
                    break
        except Exception:
            pass

        if vm_sizes is not None and len(vm_sizes) > 0:
            plot_ccdf(vm_sizes, ax_size, label='Vicsek Model', color='grey', linestyle='--')
        if vm_integrated is not None and len(vm_integrated) > 0:
            plot_ccdf(vm_integrated, ax_int, label='Vicsek Model', color='grey', linestyle='--')
        if vm_durs is not None and len(vm_durs) > 0:
            plot_ccdf(vm_durs, ax_dur, label='Vicsek Model', color='grey', linestyle='--')

        # Legend only on the top-left panel to avoid clutter (include VM entry on first row)
        if row == 0:
            ax_dur.legend(loc='lower right', frameon=True)

    # Bottom x-labels (place on last row)
    axes[-1, 0].set_xlabel('Avalanche Size')
    axes[-1, 1].set_xlabel('Integrated Excursion')
    axes[-1, 2].set_xlabel('Avalanche Duration')

    # Labels (a–l)
    label_axes(fig, labels=['(a)', '(b)', '(c)',
                            '(d)', '(e)', '(f)',
                            '(g)', '(h)', '(i)',
                            '(j)', '(k)', '(l)'], loc=(0.95, 0.9))

    plt.tight_layout()

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'figure_s2_avalanchesL.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    plt.savefig(out_path.replace('.pdf', '.eps'), bbox_inches='tight', dpi=300)
    print(f'Supplement figure saved to {out_path}')


if __name__ == '__main__':
    main()
