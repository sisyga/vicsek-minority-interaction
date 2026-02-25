import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import os
import json
import matplotlib as mpl
from figure2_orderparam import label_axes

# Use the existing PRL style file
mpl.style.use('prl_style.mplstyle')

# Base directory for data — set this to the folder containing your soc_simulation_runs/ and vicsek_simulation_runs/ subdirectories
BASE_DIR = r'X:\soc_project'  # <-- CHANGE THIS
VM_DIR = os.path.join(BASE_DIR, 'vicsek_simulation_runs')
SOC_DIR = os.path.join(BASE_DIR, 'soc_simulation_runs')

# Import required functions for correlation calculation
from analysis_functions import vel_corr
from interaction_functions import grid_neighbors


def find_correlation_length(rs, corr_values):
    """Find the distance where correlation crosses zero"""
    # Find where correlation becomes zero or changes sign
    if np.all(corr_values > 0):
        return np.max(rs)  # If correlation never crosses zero

    # Find the first index where correlation becomes <= 0
    try:
        cross_idx = np.where(corr_values <= 0)[0][0]

        if cross_idx == 0:
            return rs[0]

        # Linear interpolation to find exact crossing point
        r1, r2 = rs[cross_idx - 1], rs[cross_idx]
        c1, c2 = corr_values[cross_idx - 1], corr_values[cross_idx]

        # Linear interpolation to find exact crossing point
        r0 = r1 - c1 * (r2 - r1) / (c2 - c1)
        return r0
    except IndexError:
        return np.max(rs)  # Fallback if something goes wrong


def st_line(x, m, c):
    """Straight line function for fitting"""
    return m * x + c


def load_params(filepath):
    """Load parameter file and return as dictionary"""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_run_data(run_id, data_type='SOC'):
    """Load data for specific run ID and data type"""
    if data_type == 'SOC':
        dir_path = SOC_DIR
    else:
        dir_path = VM_DIR

    # Load parameters
    params_file = os.path.join(dir_path, f'run_{run_id}_params.json')
    if not os.path.exists(params_file):
        raise FileNotFoundError(f"Parameters file not found: {params_file}")
    params = load_params(params_file)

    # Load observables
    obs_file = os.path.join(dir_path, f'observables_{run_id}.npz')
    if not os.path.exists(obs_file):
        raise FileNotFoundError(f"Observables file not found: {obs_file}")
    obs = np.load(obs_file, allow_pickle=True)

    return params, obs


def load_simulation_data(data_filename):
    """
    Loads simulation data from a memmap file.
    """
    data = np.memmap(data_filename, dtype=np.float16, mode='r')
    return data


def calculate_correlation(run_id, data_dir):
    """
    Calculate velocity fluctuation correlation for a specific run
    when it's not already available in the observables file.
    """
    print(f"Calculating velocity fluctuation correlation for run {run_id}...")

    # Load parameters
    params_file = os.path.join(data_dir, f'run_{run_id}_params.json')
    with open(params_file, 'r') as f:
        params = json.load(f)

    # Load the data file
    data_filename = os.path.join(data_dir, f'run_{run_id}.dat')
    data_memmap = load_simulation_data(data_filename)

    # Reshape the memmap to (tmax, 3, N)
    tmax = params['tmax']
    N = params['N']
    data = data_memmap.reshape((tmax, 3, N))

    # Use a subset of timesteps to reduce computation time
    tc = int(1e4)  # Default equilibration time
    sample_size = 1000
    sample_ids = np.random.choice(len(data[tc:]), size=sample_size, replace=False) + tc
    sampled_data = data[sample_ids]

    # Prepare parameters for correlation calculation
    L = params['L']
    r = params['r']
    dr = r / 10
    rs = np.arange(0, np.sqrt(2) * L / 2 + dr / 2, dr)

    # Calculate correlation
    corr = vel_corr(sampled_data.astype(np.float32), rs, L, fluctuation_correlation=True)  # numba does not like float16

    # Save the results back to the observables file
    obs_file = os.path.join(data_dir, f'observables_{run_id}.npz')
    if os.path.exists(obs_file):
        # Load existing observables
        obs_data = dict(np.load(obs_file))
        # Add new data
        obs_data['velocity_fluctuation_correlation'] = corr
        obs_data['rs'] = rs
        # Save back
        np.savez(obs_file, **obs_data)
    else:
        # Create new observables file
        np.savez(obs_file, velocity_fluctuation_correlation=corr, rs=rs)

    print(f"Calculated and saved velocity fluctuation correlation for run {run_id}")
    return rs, corr


def find_matching_vicsek_run(system_size, min_noise):
    """Find the Vicsek run that matches a given system size and has the minimum noise value"""
    matching_run_id = None

    for f in os.listdir(VM_DIR):
        if f.endswith('_params.json'):
            params = load_params(os.path.join(VM_DIR, f))

            # Check if parameters match
            if (params.get('L', 0) == system_size and
                    abs(params.get('density', 0) - 1.0) < 1e-6 and
                    abs(params.get('noise', 0) - min_noise) < 1e-6):
                # Extract run ID from filename
                run_id = int(f.split('_')[1])
                matching_run_id = run_id
                break

    return matching_run_id


def create_figure4():
    """Create figure 4 with velocity fluctuation correlation analysis"""
    # Create figure and subplots
    fig_width = plt.rcParams['figure.figsize'][0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, 1.7))

    # Valid system sizes for our analysis (same as in figure3)
    valid_system_sizes = [16, 32, 64, 128]

    min_noise = float('inf')
    for f in os.listdir(SOC_DIR):
        if f.endswith('_params.json'):
            params = load_params(os.path.join(SOC_DIR, f))
            noise = params.get('noise', float('inf'))
            if noise < min_noise:
                min_noise = noise

    print(f"Smallest noise value found: {min_noise}")

    # Group runs by system size
    system_size_runs = {}

    # Collect run IDs for each system size
    for f in os.listdir(SOC_DIR):
        if f.endswith('_params.json'):
            run_id = int(f.split('_')[1])
            params = load_params(os.path.join(SOC_DIR, f))

            # Check for required density and noise values
            density = params.get('density', 0)
            noise = params.get('noise', 0)

            if abs(density - 1.0) > 1e-6 or abs(noise - min_noise) > 1e-6:
                continue  # Skip runs that don't match our density and noise criteria

            epsilon = params.get('epsilon', 0)
            gamma = params.get('gamma', 0)
            L = params.get('L', 0)

            # Filter for system size runs
            # Fix epsilon and gamma values and use only specified system sizes
            if (abs(epsilon - 0.3) < 1e-6 and
                    abs(gamma - (-0.6)) < 1e-6 and
                    L in valid_system_sizes):
                if L not in system_size_runs:
                    system_size_runs[L] = []
                system_size_runs[L].append(run_id)

    # Sort system sizes for plotting
    sizes = sorted(list(system_size_runs.keys()))
    print(f"Found runs for system sizes: {sizes}")
    for L in sizes:
        print(f"  L={L}: {len(system_size_runs[L])} runs (IDs: {system_size_runs[L]})")

    # Check if any data was found
    if not sizes:
        print("No matching runs found. Check your data directory and filtering criteria.")
        ax1.text(0.5, 0.5, "No data found", ha='center', va='center')
        ax2.text(0.5, 0.5, "No data found", ha='center', va='center')
        plt.tight_layout()
        return fig

    # Use viridis colormap for system sizes
    cmap = plt.cm.viridis
    colors = [cmap(i / (len(sizes) - 1)) for i in range(len(sizes))]

    # Arrays to store correlation lengths for each system size
    soc_correlation_lengths = []
    vm_correlation_lengths = []

    # Process each system size
    has_data = False  # Flag to track if any data was plotted

    for i, L in enumerate(sizes):
        # Process SOC model data
        run_ids = system_size_runs[L]
        print(f"Processing {len(run_ids)} SOC runs for L={L}")

        # Aggregate correlation data for this system size
        all_rs = []
        all_corrs = []

        for run_id in run_ids:
            try:
                # Load parameters and observables
                params, obs = get_run_data(run_id, 'SOC')

                if 'velocity_fluctuation_correlation' in obs and 'rs' in obs:
                    # If correlation data already exists, use it
                    rs = obs['rs']
                    corr = obs['velocity_fluctuation_correlation']

                    all_rs.append(rs)
                    all_corrs.append(corr)
                    print(f"  Loaded correlation data for SOC run {run_id}")
                else:
                    # If correlation data doesn't exist, calculate it
                    print(f"  Missing correlation data for SOC run {run_id}, calculating...")
                    rs, corr = calculate_correlation(run_id, SOC_DIR)

                    if rs is not None and corr is not None:
                        all_rs.append(rs)
                        all_corrs.append(corr)
                        print(f"  Calculated correlation data for SOC run {run_id}")
                    else:
                        print(f"  Failed to calculate correlation data for SOC run {run_id}")
            except Exception as e:
                print(f"  Error loading/calculating data for SOC run {run_id}: {e}")

        if all_corrs:
            has_data = True  # Set flag that we have data to plot

            # Average over all runs of this system size
            mean_rs = all_rs[0]  # Assume all rs are the same
            mean_corr = np.mean(all_corrs, axis=0)

            # Plot correlation function for SOC model
            ax1.plot(mean_rs, mean_corr, '-', color=colors[i], label=fr'$L = {L}$')

            # Find correlation length (r0 where C(r0) = 0)
            corr_length = find_correlation_length(mean_rs, mean_corr)
            soc_correlation_lengths.append((L, corr_length))
            print(f"SOC L={L}: correlation length = {corr_length}")
        else:
            print(f"No valid correlation data for SOC L={L}")

        # Now find and process matching Vicsek model run
        vm_run_id = find_matching_vicsek_run(L, min_noise)
        if vm_run_id:
            print(f"Found matching Vicsek run {vm_run_id} for L={L}")
            try:
                # Load parameters and observables
                params, obs = get_run_data(vm_run_id, 'Vicsek')

                # Get or calculate Vicsek correlation
                if 'velocity_fluctuation_correlation' in obs and 'rs' in obs:
                    # If correlation data already exists, use it
                    vm_rs = obs['rs']
                    vm_corr = obs['velocity_fluctuation_correlation']
                    print(f"  Loaded correlation data for VM run {vm_run_id}")
                else:
                    # If correlation data doesn't exist, calculate it
                    print(f"  Missing correlation data for VM run {vm_run_id}, calculating...")
                    vm_rs, vm_corr = calculate_correlation(vm_run_id, VM_DIR)

                if vm_rs is not None and vm_corr is not None:
                    # Plot VM correlation function as dotted line with same color
                    ax1.plot(vm_rs, vm_corr, '--', color=colors[i], alpha=0.8)

                    # Find VM correlation length
                    vm_corr_length = find_correlation_length(vm_rs, vm_corr)
                    vm_correlation_lengths.append((L, vm_corr_length))
                    print(f"VM L={L}: correlation length = {vm_corr_length}")
                else:
                    print(f"  Failed to get correlation data for VM run {vm_run_id}")
            except Exception as e:
                print(f"  Error processing Vicsek run {vm_run_id}: {e}")
        else:
            print(f"No matching Vicsek run found for L={L}")

    # Set up correlation function plot
    ax1.set_xlabel(r'Distance $d$')
    ax1.set_ylabel(r'$C(d)$')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax1.set_ylim(1e-4, 1.0)
    ax1.set_yscale('log')
    ax1.set_xlim(-1, 40)

    # Only add legend if we have data
    if has_data:
        ax1.legend(frameon=True)

    # ax1.set_title('(a) Velocity Fluctuation Correlation')

    # Plot correlation length vs system size
    if soc_correlation_lengths:
        sizes_arr = np.array([cl[0] for cl in soc_correlation_lengths])
        lengths_arr = np.array([cl[1] for cl in soc_correlation_lengths])

        ax2.plot(sizes_arr, lengths_arr, 'ko', label='Minority I.A.')

        # Linear fit for SOC correlation length vs. system size
        if len(sizes_arr) > 1:
            try:
                popt, _ = curve_fit(st_line, sizes_arr, lengths_arr)
                fit_sizes = np.linspace(min(sizes_arr), max(sizes_arr), 100)
                fit_lengths = st_line(fit_sizes, *popt)
                ax2.plot(fit_sizes, fit_lengths, '--', c='k',
                         label=f'_Linear fit: slope={popt[0]:.3f}')

                print(f"SOC linear fit: y = {popt[0]:.3f}x + {popt[1]:.3f}")
            except Exception as e:
                print(f"Error in SOC curve fitting: {e}")
    else:
        print("No SOC correlation lengths calculated")

    # Plot Vicsek correlation length vs system size
    if vm_correlation_lengths:
        vm_sizes_arr = np.array([cl[0] for cl in vm_correlation_lengths])
        vm_lengths_arr = np.array([cl[1] for cl in vm_correlation_lengths])

        # Plot VM correlation lengths with grey dotted line and round markers
        ax2.plot(vm_sizes_arr, vm_lengths_arr, 'o', c='grey',
                 label='Vicsek Model')

        # Linear fit for VM correlation length vs. system size
        if len(vm_sizes_arr) > 1:
            try:
                vm_popt, _ = curve_fit(st_line, vm_sizes_arr, vm_lengths_arr)
                vm_fit_sizes = np.linspace(min(vm_sizes_arr), max(vm_sizes_arr), 100)
                vm_fit_lengths = st_line(vm_fit_sizes, *vm_popt)
                ax2.plot(vm_fit_sizes, vm_fit_lengths, '--', color='grey',
                         label=f'_VM fit: slope={vm_popt[0]:.3f}')

                print(f"VM linear fit: y = {vm_popt[0]:.3f}x + {vm_popt[1]:.3f}")
            except Exception as e:
                print(f"Error in VM curve fitting: {e}")
    else:
        print("No Vicsek correlation lengths calculated")

    # Set up correlation length plot
    ax2.set_xlabel(r'System Size $L$')
    ax2.set_ylabel('Correlation Length $d_0$')

    # Add legend if we have data
    if soc_correlation_lengths or vm_correlation_lengths:
        ax2.legend(loc='lower right', frameon=True)
    else:
        ax2.text(0.5, 0.5, "No correlation data available", ha='center', va='center')

    # ax2.set_title('(b) Correlation Length vs System Size')
    label_axes(fig, labels=['(a)', '(b)'], loc=(0.2, .9))
    plt.tight_layout()

    # Save figure
    output_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'figure4_correlation_length.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.eps'), bbox_inches='tight', dpi=300)

    print(f"Figure saved to {output_path}")

    return fig


if __name__ == "__main__":
    fig = create_figure4()
    plt.show()
