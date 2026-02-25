import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm

# Style
mpl.style.use('prl_style.mplstyle')

# Directories — set BASE_DIR to the folder containing your soc_simulation_runs32/ subdirectory
BASE_DIR = r'X:\\soc_project'  # <-- CHANGE THIS
SOC_DIR = os.path.join(BASE_DIR, 'soc_simulation_runs32')
VICSEK_DIR = os.path.join(BASE_DIR, 'vicsek_simulation_runs')

# Target system size (match figure5)
TARGET_L = 32

# Target grids (only include combos present in Vicsek runs)
DENSITIES = [0.5, 1.0, 1.5]
NOISES = [0.1813799364234218, 0.3627598728468436, 0.5441398092702654]  # ~0.18, 0.36, 0.54
NOISE_TOL = 1e-2


def load_params(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def load_observables(run_id, base_dir):
    path = os.path.join(base_dir, f'observables_{run_id}.npz')
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True)


def _approx_equal(a, b, tol):
    return abs(a - b) <= tol


def find_vicsek_run_by_params(L, density, noise, tol=NOISE_TOL):
    for f in os.listdir(VICSEK_DIR):
        if f.endswith('_params.json'):
            p = load_params(os.path.join(VICSEK_DIR, f))
            if (
                    _approx_equal(p.get('L', 0), L, 1e-6)
                    and _approx_equal(p.get('density', 0), density, 1e-6)
                    and _approx_equal(p.get('noise', 0), noise, tol)
            ):
                try:
                    return int(f.split('_')[1])
                except Exception:
                    pass
    return None


def vicsek_stats(L, density, noise):
    """Return (mean, var, tc) of Vicsek order parameter for the given params, or (np.nan, np.nan, None)."""
    run_id = find_vicsek_run_by_params(L, density, noise)
    if run_id is None:
        return np.nan, np.nan, None
    obs = load_observables(run_id, VICSEK_DIR)
    if obs is None:
        return np.nan, np.nan, None
    if 'orderparameter' in obs:
        op_series = obs['orderparameter']
    else:
        data = obs.get('data', None)
        if data is None:
            return np.nan, np.nan, None
        op_series = np.abs(np.exp(1j * data[:, 2, :]).mean(axis=1))
    tc = obs.get('tc', None)
    if tc is None:
        # simple fallback: use 1/10 of series
        tc = len(op_series) // 10
    elif hasattr(tc, 'item'):
        tc = tc.item()
    series = op_series[tc:] if tc is not None and tc < len(op_series) else op_series
    return float(np.mean(series)), float(np.var(series)), tc


def collect_soc_runs_by_combo():
    """Group SOC runs by (density, noise) for TARGET_L restricted to target grids present in Vicsek.
    Returns dict[(rho, eta)] -> list[(run_id, params)].
    """
    # Precompute the set of valid Vicsek combos
    valid_keys = []
    for rho in DENSITIES:
        for eta in NOISES:
            if find_vicsek_run_by_params(TARGET_L, rho, eta) is not None:
                valid_keys.append((rho, eta))

    groups = {k: [] for k in valid_keys}

    for f in os.listdir(SOC_DIR):
        if not f.endswith('_params.json'):
            continue
        try:
            run_id = int(f.split('_')[1])
        except Exception:
            continue
        p = load_params(os.path.join(SOC_DIR, f))
        if p.get('L', None) != TARGET_L:
            continue
        rho = p.get('density', None)
        eta = p.get('noise', None)
        eps = p.get('epsilon', None)
        gam = p.get('gamma', None)
        if rho is None or eta is None or eps is None or gam is None:
            continue

        # Snap to closest target grid if within tolerance
        rho_snap = next((r for r in DENSITIES if _approx_equal(r, rho, 1e-6)), None)
        eta_snap = next((n for n in NOISES if _approx_equal(n, eta, NOISE_TOL)), None)
        key = (rho_snap, eta_snap)
        if rho_snap is None or eta_snap is None or key not in groups:
            continue
        groups[key].append((run_id, p))

    return groups


def build_grids_for_group(runs, vm_var):
    """Build (grid_mean, grid_ratio, extent, eps_list, gam_list)."""
    eps_vals = sorted({round(r[1]['epsilon'], 10) for r in runs})
    gam_vals = sorted({round(r[1]['gamma'], 10) for r in runs})
    # compute spacing for extent
    deps = eps_vals[1] - eps_vals[0] if len(eps_vals) > 1 else 1.0
    dgam = gam_vals[1] - gam_vals[0] if len(gam_vals) > 1 else 1.0
    extent = [min(gam_vals) - 0.5 * dgam, max(gam_vals) + 0.5 * dgam,
              min(eps_vals) - 0.5 * deps, max(eps_vals) + 0.5 * deps]

    grid_mean = np.full((len(eps_vals), len(gam_vals)), np.nan)
    grid_ratio = np.full((len(eps_vals), len(gam_vals)), np.nan)

    # Preload observables to reduce I/O
    for run_id, p in runs:
        obs = load_observables(run_id, SOC_DIR)
        if obs is None:
            continue
        if 'orderparameter' in obs:
            op_series = obs['orderparameter']
        else:
            data = obs.get('data', None)
            if data is None:
                continue
            op_series = np.abs(np.exp(1j * data[:, 2, :]).mean(axis=1))
        tc = obs.get('tc', None)
        if tc is None:
            tc = len(op_series) // 10
        elif hasattr(tc, 'item'):
            tc = tc.item()
        series = op_series[tc:] if tc is not None and tc < len(op_series) else op_series

        eps = round(p['epsilon'], 10)
        gam = round(p['gamma'], 10)
        i = eps_vals.index(eps)
        j = gam_vals.index(gam)
        m = float(np.mean(series))
        v = float(np.var(series))
        grid_mean[i, j] = m
        grid_ratio[i, j] = (v / vm_var) if vm_var and vm_var != 0 else np.nan

    return grid_mean, grid_ratio, extent, eps_vals, gam_vals


def plot_panels(groups):
    # sort keys for stable layout: densities as rows, noises as columns
    densities = sorted({rho for (rho, _) in groups.keys()})
    noises = sorted({eta for (_, eta) in groups.keys()})

    nrows = len(densities)
    ncols = len(noises)

    # Figure A: mean order parameter
    figA_width = plt.rcParams['figure.figsize'][0]
    figA, axesA = plt.subplots(
        nrows, ncols,
        figsize=(figA_width * 1.5, (figA_width * 1.25) * (nrows / 2)),
        sharex=False, sharey=False,
        constrained_layout=True,
    )
    # Figure B: variance ratio
    figB_width = plt.rcParams['figure.figsize'][0]
    figB, axesB = plt.subplots(
        nrows, ncols,
        figsize=(figB_width * 1.5, (figB_width * 1.25) * (nrows / 2)),
        sharex=False, sharey=False,
        constrained_layout=True,
    )

    imA = None
    imB = None

    for i, rho in enumerate(densities):
        for j, eta in enumerate(noises):
            key = (rho, eta)
            axA = axesA[i, j] if nrows > 1 or ncols > 1 else axesA
            axB = axesB[i, j] if nrows > 1 or ncols > 1 else axesB

            if key not in groups:
                axA.text(0.5, 0.5, 'No data', ha='center', va='center')
                axB.text(0.5, 0.5, 'No data', ha='center', va='center')
                continue

            runs = groups[key]
            vm_mean, vm_var, _ = vicsek_stats(TARGET_L, rho, eta)
            grid_mean, grid_ratio, extent, eps_vals, gam_vals = build_grids_for_group(runs, vm_var)

            # Panel A heatmap
            imA = axA.imshow(
                grid_mean,
                aspect='equal',
                interpolation='nearest',
                cmap='viridis',
                origin='lower',
                extent=extent,
                vmin=0,
                vmax=1,
            )
            # Panel B heatmap
            imB = axB.imshow(
                grid_ratio,
                aspect='equal',
                interpolation='nearest',
                cmap='plasma_r',
                norm=LogNorm(vmin=1, vmax=3000),
                origin='lower',
                extent=extent,
            )

            # Titles and axis labels on edges
            axA.set_title(fr'$\rho={rho:.2f}, \eta={eta:.3f}$')
            axB.set_title(fr'$\rho={rho:.2f}, \eta={eta:.3f}$')
            if i == nrows - 1:
                axA.set_xlabel(r'$\gamma$')
                axB.set_xlabel(r'$\gamma$')
            if j == 0:
                axA.set_ylabel(r'$\epsilon$')
                axB.set_ylabel(r'$\epsilon$')

    # Shared colorbars
    if imA is not None:
        cbarA = figA.colorbar(
            imA,
            ax=axesA.ravel().tolist(),
            orientation='horizontal',
            location='top',
            pad=0.02,
        )
        cbarA.set_label(r'Order parameter')
        cbarA.set_ticks([0, 1])
    if imB is not None:
        cbarB = figB.colorbar(
            imB,
            ax=axesB.ravel().tolist(),
            orientation='horizontal',
            location='top',
            pad=0.02,
        )
        cbarB.set_label(r'Variance Ratio')

    # constrained_layout already active; avoid tight_layout to prevent overlap

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(out_dir, exist_ok=True)
    pA = os.path.join(out_dir, 'figure_s4_orderparameter_a.pdf')
    pB = os.path.join(out_dir, 'figure_s4_orderparameter_b.pdf')
    figA.savefig(pA, bbox_inches='tight')
    figA.savefig(pA.replace('.pdf', '.eps'), bbox_inches='tight', dpi=300)
    figB.savefig(pB, bbox_inches='tight')
    figB.savefig(pB.replace('.pdf', '.eps'), bbox_inches='tight', dpi=300)
    print(f'Saved {pA} and {pB}')


def main():
    groups = collect_soc_runs_by_combo()
    if not groups:
        print('No SOC runs found for L=32 in directory:', SOC_DIR)
        return
    plot_panels(groups)


if __name__ == '__main__':
    main()
