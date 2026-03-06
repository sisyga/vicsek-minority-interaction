import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm  # modified import to include LogNorm

# Import functions from figure3_avalanches.py
from figure3_avalanches import load_params, op, find_tc, identify_avalanches, get_run_data
from figure2_orderparam import label_axes  # added import for labeling axes
import matplotlib as mpl


# Directories and target parameters — set BASE_DIR to the folder containing your soc_simulation_runs32/ subdirectory
BASE_DIR = r'X:\soc_project'  # <-- CHANGE THIS
SOC_DIR = os.path.join(BASE_DIR, 'soc_simulation_runs32')
VICSEK_DIR = os.path.join(BASE_DIR, 'vicsek_simulation_runs')
mpl.style.use('prl_style.mplstyle')


TARGET_L = 32
TARGET_DENSITY = 1.0
TARGET_NOISE = 0.18  # approximate value
NOISE_TOL = 0.01


def find_vicsek_run_by_params(vicsek_dir, L, density, noise):
    """Find a Vicsek run with matching L, density and noise."""
    for f in os.listdir(vicsek_dir):
        if f.endswith('_params.json'):
            params_path = os.path.join(vicsek_dir, f)
            params = load_params(params_path)
            if (abs(params.get('L', 0) - L) < 1e-6 and
                    abs(params.get('density', 0) - density) < 1e-6 and
                    abs(params.get('noise', 0) - noise) < NOISE_TOL):
                run_id = int(f.split('_')[1])
                return run_id
    return None


def compute_vicsek_threshold(v_run_id):
    """Compute the threshold from the Vicsek run data."""
    try:
        _, obs = get_run_data(v_run_id, data_type='Vicsek')
    except Exception as e:
        print(f"Failed loading Vicsek run {v_run_id}: {e}")
        return None
    if 'orderparameter' in obs:
        vm_op = obs['orderparameter']
    else:
        vm_data = obs.get('data', None)
        if vm_data is None:
            print(f"No data available for Vicsek run {v_run_id}")
            return None
        vm_op = op(vm_data[:, 2, :], axis=1)
    tc = obs.get('tc', None)
    if tc is None:
        tc = find_tc(vm_op)
    elif hasattr(tc, 'item'):
        tc = tc.item()
    threshold = np.mean(vm_op[tc:]) - 3 * np.std(vm_op[tc:])
    return threshold


def get_soc_avalanche_sizes(run_id, threshold):
    """Load SOC run data from the specified folder and compute avalanche sizes."""
    params_file = os.path.join(SOC_DIR, f'run_{run_id}_params.json')
    obs_file = os.path.join(SOC_DIR, f'observables_{run_id}.npz')
    if not os.path.exists(params_file) or not os.path.exists(obs_file):
        print(f"Missing files for SOC run {run_id}")
        return None
    # Load SOC parameters and observables
    params = load_params(params_file)
    obs = np.load(obs_file, allow_pickle=True)
    if 'orderparameter' in obs:
        soc_op = obs['orderparameter']
    else:
        soc_data = obs.get('data', None)
        if soc_data is None:
            print(f"No data for SOC run {run_id}")
            return None
        soc_op = op(soc_data[:, 2, :], axis=1)
    avalanche_sizes, _ = identify_avalanches(soc_op, threshold)
    return avalanche_sizes


def get_soc_order_stats(run_id):
    """Load SOC run data and return mean and variance of the order parameter (after tc)."""
    params_file = os.path.join(SOC_DIR, f'run_{run_id}_params.json')
    obs_file = os.path.join(SOC_DIR, f'observables_{run_id}.npz')
    if not os.path.exists(params_file) or not os.path.exists(obs_file):
        print(f"Missing files for SOC run {run_id}")
        return None
    obs = np.load(obs_file, allow_pickle=True)
    if 'orderparameter' in obs:
        soc_op = obs['orderparameter']
    else:
        soc_data = obs.get('data', None)
        if soc_data is None:
            print(f"No data for SOC run {run_id}")
            return None
        soc_op = op(soc_data[:, 2, :], axis=1)
    tc = obs.get('tc', None)
    if tc is None:
        tc = find_tc(soc_op)
    elif hasattr(tc, 'item'):
        tc = tc.item()
    mean_val = np.mean(soc_op[tc:])
    var_val = np.var(soc_op[tc:])
    return mean_val, var_val


def main():
    # Find the corresponding Vicsek run and compute its threshold
    vicsek_run_id = find_vicsek_run_by_params(VICSEK_DIR, TARGET_L, TARGET_DENSITY, TARGET_NOISE)
    if vicsek_run_id is None:
        print("No matching Vicsek run found.")
        return
    v_threshold = compute_vicsek_threshold(vicsek_run_id)
    if v_threshold is None:
        print("Could not compute threshold from Vicsek run.")
        return
    print(f"Vicsek run {vicsek_run_id} threshold: {v_threshold}")

    # Load Vicsek run data and compute order parameter stats after tc
    try:
        _, v_obs = get_run_data(vicsek_run_id, data_type='Vicsek')
    except Exception as e:
        print(f"Failed reloading Vicsek run {vicsek_run_id}: {e}")
        return
    if 'orderparameter' in v_obs:
        v_op = v_obs['orderparameter']
    else:
        v_data = v_obs.get('data', None)
        if v_data is None:
            print(f"No data for Vicsek run {vicsek_run_id}")
            return
        v_op = op(v_data[:, 2, :], axis=1)
    v_tc = v_obs.get('tc', None)
    if v_tc is None:
        v_tc = find_tc(v_op)
    elif hasattr(v_tc, 'item'):
        v_tc = v_tc.item()
    vm_mean = np.mean(v_op[v_tc:])
    vm_var = np.var(v_op[v_tc:])

    # Collect mean and variance stats for each SOC run.
    soc_stats = []  # each element: (epsilon, gamma, mean_op, var_op)
    for f in os.listdir(SOC_DIR):
        if f.endswith('_params.json'):
            run_id = int(f.split('_')[1])
            params_path = os.path.join(SOC_DIR, f)
            params = load_params(params_path)
            if (abs(params.get('L', 0) - TARGET_L) < 1e-6 and
                    abs(params.get('density', 0) - TARGET_DENSITY) < 1e-6 and
                    abs(params.get('noise', 0) - TARGET_NOISE) < NOISE_TOL):
                eps = params.get('epsilon', None)
                gam = params.get('gamma', None)
                if eps is None or gam is None:
                    print(f"Run {run_id}: missing epsilon or gamma.")
                    continue
                stats = get_soc_order_stats(run_id)
                if stats is None:
                    continue
                mean_op, var_op = stats
                soc_stats.append((eps, gam, mean_op, var_op))

    if len(soc_stats) == 0:
        print("No SOC run data available.")
        return

    # Determine unique, sorted epsilon and gamma values.
    epsilons = sorted(list({s[0] for s in soc_stats}))
    gammas = sorted(list({s[1] for s in soc_stats}))

    # Compute extents so that pixel centers correspond to parameter values
    if len(gammas) > 1:
        dgam = gammas[1] - gammas[0]
    else:
        dgam = 1.0
    if len(epsilons) > 1:
        deps = epsilons[1] - epsilons[0]
    else:
        deps = 1.0
    extent = [min(gammas) - 0.5 * dgam, max(gammas) + 0.5 * dgam,
              min(epsilons) - 0.5 * deps, max(epsilons) + 0.5 * deps]

    # Create grids for mean order parameter and variance ratio (SOC var / VM var)
    grid_mean = np.full((len(epsilons), len(gammas)), np.nan)
    grid_ratio = np.full((len(epsilons), len(gammas)), np.nan)
    for eps, gam, mean_op, var_op in soc_stats:
        i = epsilons.index(eps)
        j = gammas.index(gam)
        grid_mean[i, j] = mean_op
        grid_ratio[i, j] = var_op / vm_var if vm_var != 0 else np.nan

    # Create two subplots.
    fig_width = plt.rcParams['figure.figsize'][0]
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(fig_width, fig_width / 2 * 1.2))

    # Subplot a: Mean order parameter heatmap with colormap limits [0, 1]
    im1 = axs[0].imshow(grid_mean, aspect='auto', interpolation='nearest', cmap='viridis',
                        origin='lower', extent=extent,
                        vmin=0, vmax=1)
    axs[0].set_xlabel(r'$\gamma$')
    axs[0].set_ylabel(r'$\epsilon$')
    cbar1 = fig.colorbar(im1, ax=axs[0], ticks=[0, 1], label=r'Order parameter', orientation='horizontal',
                         location='top')
    cbar1.ax.set_xticklabels(['0', '1'])

    # Subplot b: Variance ratio heatmap with logarithmic scale; remove y label
    im2 = axs[1].imshow(grid_ratio, aspect='auto', interpolation='nearest', cmap='plasma_r',
                        norm=LogNorm(vmin=1, vmax=3000),
                        origin='lower', extent=extent)
    axs[1].set_xlabel(r'$\gamma$')
    axs[1].set_ylabel('')
    fig.colorbar(im2, ax=axs[1], label=r'Variance ratio', orientation='horizontal',
                         location='top')

    # Mark specified points on both subplots using crosses with specific tab10 colors: indices 0, 2, and 3.
    points = [(0.3, -0.6), (0.6, -0.6), (0.6, -0.3)]  # (epsilon, gamma)
    tab10 = plt.get_cmap('tab10')
    color_indices = [0, 2, 3]
    for ax in axs:
        for p, idx in zip(points, color_indices):
            eps, gam = p
            ax.plot(gam, eps, marker='o', color=tab10(idx), linestyle='none')

    # Label subfigure axes using the same function as in figure4_correlation_length.py
    label_axes(fig, labels=['(a)', '(b)', '', ''], loc=(0.2, .9))  # added call to label_axes

    # Save figure
    output_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'figure5_heatmap.pdf')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.eps'), bbox_inches='tight', dpi=300)
    print(f"Heatmap figure saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
