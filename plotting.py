import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from matplotlib.colors import LogNorm
import seaborn as sns


def load_parameters(base_dir):
    """Load all parameter files into a pandas DataFrame"""
    params_list = []
    for f in os.listdir(base_dir):
        if f.endswith('_params.json'):
            run_id = int(f.split('_')[1])
            with open(os.path.join(base_dir, f), 'r') as file:
                params = json.load(file)
                params['run_id'] = run_id
                params_list.append(params)
    return pd.DataFrame(params_list)


def load_observable(run_id, base_dir):
    """Load observables for a specific run"""
    run_id = int(run_id)
    obs_file = os.path.join(base_dir, f'observables_{run_id}.npz')
    return np.load(obs_file)


def calculate_avalanche_statistic(obs, stat_type='mean'):
    """Calculate statistics for avalanche sizes"""
    if 'block_minima' in obs and 'threshold' in obs:
        threshold = obs['threshold']
        avalanche_sizes = np.maximum(0, threshold - obs['block_minima'])
        avalanche_sizes = avalanche_sizes[avalanche_sizes > 0]

        if len(avalanche_sizes) > 0:
            if stat_type == 'mean':
                return np.mean(avalanche_sizes)
            elif stat_type == 'max':
                return np.max(avalanche_sizes)
            elif stat_type == 'count':
                return len(avalanche_sizes)
            elif stat_type == 'std':
                return np.std(avalanche_sizes)
    return np.nan


def create_epsilon_gamma_heatmap(base_dir, density=0.5, noise=0.1, L=32, stat_type='mean'):
    """Create heatmap of avalanche statistics vs epsilon and gamma"""
    # Load parameters
    params_df = load_parameters(base_dir)

    # Filter by density, noise, L and only include gamma < 0
    filtered_df = params_df[
        (params_df['density'] == density) &
        (params_df['noise'] == noise) &
        (params_df['L'] == L) &
        (params_df['gamma'] < 0)  # Added condition for negative gamma only
        ]

    if len(filtered_df) == 0:
        print(f"No data found for density={density}, noise={noise}, L={L} with gamma < 0")
        return None

    # Get unique epsilon and gamma values
    epsilon_values = sorted(filtered_df['epsilon'].unique())
    gamma_values = sorted(filtered_df['gamma'].unique())

    # Create result matrix
    result_matrix = np.zeros((len(epsilon_values), len(gamma_values)))
    result_matrix.fill(np.nan)

    # Calculate statistic for each parameter combination
    for _, row in filtered_df.iterrows():
        try:
            eps_idx = epsilon_values.index(row['epsilon'])
            gamma_idx = gamma_values.index(row['gamma'])

            # Load observables using integer run_id
            obs = load_observable(int(row['run_id']), base_dir)

            # Calculate statistic
            stat = calculate_avalanche_statistic(obs, stat_type)
            result_matrix[eps_idx, gamma_idx] = stat
        except Exception as e:
            print(f"Error processing run {row['run_id']}: {e}")

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Determine appropriate colormap
    cmap = 'viridis' if stat_type != 'count' else 'plasma'

    # Use logarithmic scale for count and max stats
    norm = LogNorm() if stat_type in ['count', 'max'] else None

    # Create heatmap
    im = ax.pcolormesh(gamma_values, epsilon_values, result_matrix,
                       cmap=cmap, norm=norm)

    # Add colorbar and labels
    cbar = plt.colorbar(im, ax=ax)

    stat_names = {
        'mean': 'Mean Avalanche Size',
        'max': 'Maximum Avalanche Size',
        'count': 'Avalanche Count',
        'std': 'Avalanche Size Std. Dev.'
    }

    cbar.set_label(stat_names.get(stat_type, stat_type.capitalize()))
    ax.set_xlabel(r'$\gamma$')
    ax.set_ylabel(r'$\varepsilon$')

    title = f"{stat_names.get(stat_type, stat_type.capitalize())} for $\\rho$={density}, $\\eta$={noise:.2f}, L={L}"
    ax.set_title(title)

    # Save plot
    filename = f'avalanche_{stat_type}_heatmap_d{density}_n{noise}_L{L}.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

    return fig, ax


def plot_multiple_parameters(base_dir, fixed_params):
    """Create multiple heatmaps by varying parameters"""
    # Define parameter ranges to explore
    densities = [0.5, 1.0, 1.5] if 'density' not in fixed_params else [fixed_params['density']]
    noises = [0.1, 0.2, 0.3] if 'noise' not in fixed_params else [fixed_params['noise']]
    L_values = [32, 64, 128] if 'L' not in fixed_params else [fixed_params['L']]
    stat_types = ['mean', 'max', 'count']

    for density in densities:
        for noise in noises:
            for L in L_values:
                for stat_type in stat_types:
                    print(f"Creating heatmap for d={density}, n={noise}, L={L}, stat={stat_type}")
                    try:
                        create_epsilon_gamma_heatmap(base_dir, density, noise, L, stat_type)
                    except Exception as e:
                        print(f"Error creating heatmap: {e}")


if __name__ == "__main__":
    # Base directory containing simulation data
    BASE_DIR = r'X:\soc_project\soc_simulation_runs2'

    # Plot a specific heatmap
    create_epsilon_gamma_heatmap(BASE_DIR, density=.5, noise=0.1813799364234218, L=64, stat_type='mean')
    plt.show()
    # Generate multiple plots with different parameters
    # plot_multiple_parameters(BASE_DIR, fixed_params={'L': 64})
