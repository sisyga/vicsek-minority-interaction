import numpy as np
import matplotlib.pyplot as plt
import os
import json
from scipy import stats
import matplotlib as mpl
from matplotlib.colors import to_rgba
from analysis_functions import op, autocorrelation
from figure2_orderparam import label_axes

# use the existing PRL style file
mpl.style.use('prl_style.mplstyle')

# Base directory for data — set this to the folder containing your soc_simulation_runs/ and vicsek_simulation_runs/ subdirectories
BASE_DIR = r'X:\soc_project'  # <-- CHANGE THIS
VM_DIR = os.path.join(BASE_DIR, 'vicsek_simulation_runs')
SOC_DIR = os.path.join(BASE_DIR, 'soc_simulation_runs')


def load_params(filepath):
    """Load parameter file and return as dictionary"""
    with open(filepath, 'r') as f:
        return json.load(f)


def find_tc(order_parameter, max_lag=None):
    """Find correlation time from order parameter time series"""
    # Calculate autocorrelation
    acf = autocorrelation(order_parameter, max_lag)

    # Find first crossing of 1/e threshold or use exponential decay fit
    e_threshold = 1 / np.e
    for i, val in enumerate(acf):
        if val < e_threshold:
            return i

    # If no crossing, return a default value (1/10 of the time series)
    return len(order_parameter) // 10


def calculate_critical_threshold(vm_data, tc=None):
    """Calculate critical threshold from Vicsek model data"""
    op_values = op(vm_data[:, 2, :], axis=1)  # Calculate order parameter over time

    # Use provided tc value or calculate if not provided
    if tc is None:
        tc = find_tc(op_values)  # Find correlation time

    # Use only equilibrated data
    equilibrated_op = op_values[tc:]

    # Calculate threshold
    critical_threshold = np.mean(equilibrated_op) - 3 * np.std(equilibrated_op)

    return critical_threshold


def identify_avalanches(order_parameter, threshold, return_integrated=False):
    """
    Identify avalanches in order parameter time series

    Returns:
    - avalanche_sizes: List of avalanche sizes (drop amounts)
    - avalanche_durations: List of avalanche durations
    """
    avalanche_sizes = []
    avalanche_durations = []
    avalanche_integrated = []

    # Find where order parameter drops below threshold
    below_threshold = order_parameter < threshold

    # Find the start and end indices of each avalanche
    avalanche_start_indices = np.where(np.logical_and(below_threshold[1:], ~below_threshold[:-1]))[0] + 1
    avalanche_end_indices = np.where(np.logical_and(~below_threshold[1:], below_threshold[:-1]))[0] + 1

    # Ensure we have matching start and end points
    min_length = min(len(avalanche_start_indices), len(avalanche_end_indices))
    if min_length == 0:
        return ([], [], []) if return_integrated else ([], [])

    # If the first avalanche ends before it starts, skip the first end
    if avalanche_end_indices[0] < avalanche_start_indices[0]:
        avalanche_end_indices = avalanche_end_indices[1:]
        min_length = min(len(avalanche_start_indices), len(avalanche_end_indices))

    # If there are remaining avalanches that start but don't end, truncate
    avalanche_start_indices = avalanche_start_indices[:min_length]
    avalanche_end_indices = avalanche_end_indices[:min_length]

    # Calculate duration, size (max excursion), and optionally integrated excursion of each avalanche
    for start, end in zip(avalanche_start_indices, avalanche_end_indices):
        # Duration is simply the time difference
        duration = end - start

        # Size is maximum deviation from threshold
        if duration > 0:
            segment = order_parameter[start:end]
            size = threshold - np.min(segment)
            if return_integrated:
                vals = threshold - segment
                vals = np.where(vals > 0.0, vals, 0.0)
                integ = np.sum(vals)

            # Only include valid avalanches
            if size > 0 and duration > 0:
                avalanche_sizes.append(size)
                avalanche_durations.append(duration)
                if return_integrated:
                    avalanche_integrated.append(integ)
    if return_integrated:
        return np.array(avalanche_sizes), np.array(avalanche_durations), np.array(avalanche_integrated)
    else:
        return np.array(avalanche_sizes), np.array(avalanche_durations)


def plot_ccdf(data, ax, label=None, color=None, alpha=1.0, marker='o', linestyle='-'):
    """Plot the complementary cumulative distribution function on the given axis"""
    if len(data) == 0:
        return
    
    # Use the new ecdf function with complementary=True to plot the CCDF
    line = ax.ecdf(data, complementary=True, marker=marker, linestyle=linestyle, 
                  label=label, color=color, alpha=alpha, compress=True)
    
    return line


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
    obs = np.load(obs_file)

    return params, obs


def find_matching_vicsek_run(soc_params, vm_dir=VM_DIR):
    """Find a matching Vicsek model run with the same parameters"""
    L = soc_params['L']
    density = soc_params['density']
    noise = soc_params['noise']

    # Check each Vicsek parameters file for a match
    for f in os.listdir(vm_dir):
        if f.endswith('_params.json'):
            vm_params_path = os.path.join(vm_dir, f)
            vm_params = load_params(vm_params_path)

            # Check if parameters match
            if (abs(vm_params.get('L', 0) - L) < 1e-6 and
                    abs(vm_params.get('density', 0) - density) < 1e-6 and
                    abs(vm_params.get('noise', 0) - noise) < 1e-6):
                # Extract run ID from filename
                run_id = int(f.split('_')[1])
                return run_id

    return None


def calculate_avalanche_statistics(run_id):
    """Calculate avalanche statistics for a given SOC run ID"""
    # Load SOC data
    soc_params, soc_obs = get_run_data(run_id, 'SOC')

    # Find matching Vicsek run
    vm_run_id = find_matching_vicsek_run(soc_params)
    if vm_run_id is None:
        print(f"Warning: No matching Vicsek run found for SOC run {run_id}")
        return None, None

    # Load Vicsek data
    _, vm_obs = get_run_data(vm_run_id, 'Vicsek')

    # Calculate order parameter for Vicsek
    if 'orderparameter' in vm_obs:
        vm_op = vm_obs['orderparameter']
    else:
        # Calculate if not available
        vm_data = vm_obs.get('data', None)
        if vm_data is None:
            print(f"Warning: No data available for Vicsek run {vm_run_id}")
            return None, None
        vm_op = op(vm_data[:, 2, :], axis=1)

    # Get correlation time from observables or calculate if not available
    tc = vm_obs.get('tc', None)
    if tc is None:
        tc = find_tc(vm_op)
    else:
        # If tc is provided as a 0D array or similar, extract the scalar value
        if hasattr(tc, 'item'):
            tc = tc.item()

    # Calculate threshold
    threshold = np.mean(vm_op[tc:]) - 3 * np.std(vm_op[tc:])

    # Get SOC order parameter
    if 'orderparameter' in soc_obs:
        soc_op = soc_obs['orderparameter']
    else:
        # Calculate if not available
        soc_data = soc_obs.get('data', None)
        if soc_data is None:
            print(f"Warning: No data available for SOC run {run_id}")
            return None, None
        soc_op = op(soc_data[:, 2, :], axis=1)

    # Calculate avalanche statistics
    avalanche_sizes, avalanche_durations = identify_avalanches(soc_op, threshold)

    return avalanche_sizes, avalanche_durations


def find_vicsek_reference_run(vm_dir=VM_DIR):
    """Find a reference Vicsek run with L=32, density=1.0, and smallest noise"""
    min_noise = float('inf')
    reference_run_id = None

    # First find the smallest noise value
    for f in os.listdir(vm_dir):
        if f.endswith('_params.json'):
            vm_params_path = os.path.join(vm_dir, f)
            vm_params = load_params(vm_params_path)

            # Check L and density
            if vm_params.get('L', 0) == 32 and abs(vm_params.get('density', 0) - 1.0) < 1e-6:
                noise = vm_params.get('noise', float('inf'))
                if noise < min_noise:
                    min_noise = noise

    # Now find a run with the smallest noise
    for f in os.listdir(vm_dir):
        if f.endswith('_params.json'):
            vm_params_path = os.path.join(vm_dir, f)
            vm_params = load_params(vm_params_path)

            if (vm_params.get('L', 0) == 32 and
                    abs(vm_params.get('density', 0) - 1.0) < 1e-6 and
                    abs(vm_params.get('noise', 0) - min_noise) < 1e-6):
                # Extract run ID from filename
                run_id = int(f.split('_')[1])
                return run_id

    return None


def calculate_vicsek_reference_avalanches():
    """Calculate avalanche statistics for a reference Vicsek run"""
    # Find reference Vicsek run
    vm_run_id = find_vicsek_reference_run()
    if vm_run_id is None:
        print("Warning: No reference Vicsek run found")
        return None, None

    print(f"Using Vicsek reference run ID: {vm_run_id}")

    # Load Vicsek data
    try:
        vm_params, vm_obs = get_run_data(vm_run_id, 'Vicsek')
    except FileNotFoundError as e:
        print(f"Error loading Vicsek reference run: {e}")
        return None, None

    # Calculate order parameter for Vicsek
    if 'orderparameter' in vm_obs:
        vm_op = vm_obs['orderparameter']
    else:
        # Calculate if not available
        vm_data = vm_obs.get('data', None)
        if vm_data is None:
            print(f"Warning: No data available for Vicsek run {vm_run_id}")
            return None, None
        vm_op = op(vm_data[:, 2, :], axis=1)

    # Get correlation time from observables or calculate
    tc = vm_obs.get('tc', None)
    if tc is None:
        tc = find_tc(vm_op)
    else:
        # If tc is provided as a 0D array or similar, extract the scalar value
        if hasattr(tc, 'item'):
            tc = tc.item()

    # Calculate threshold (same method as for SOC runs)
    threshold = np.mean(vm_op[tc:]) - 3 * np.std(vm_op[tc:])

    # Calculate avalanche statistics for Vicsek model itself
    avalanche_sizes, avalanche_durations = identify_avalanches(vm_op, threshold)

    return avalanche_sizes, avalanche_durations


def create_figure3():
    """Create figure 3 with avalanche statistics"""
    # Calculate reference Vicsek avalanches
    vicsek_ref_sizes, vicsek_ref_durations = calculate_vicsek_reference_avalanches()

    # Create figure and subplots
    fig_width = plt.rcParams['figure.figsize'][0]
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_width), sharex='col', sharey='row')

    # Plot a) Avalanche size with varying threshold parameters - LOG-LOG SCALE
    ax_a = axes[0, 0]
    ax_a.set_ylabel(fr'$P(X\geq x)$')
    ax_a.set_yscale('log')  # Log scale on y-axis only
    ax_a.set_xscale('log')  # Log scale on x-axis
    ax_a.set_xlim(5e-4, 2)

    # Plot b) Avalanche duration with varying threshold parameters - LOG-LOG SCALE
    ax_b = axes[0, 1]
    # ax_b.set_ylabel('P(Duration ≥ t)')
    ax_b.set_yscale('log')
    ax_b.set_xscale('log')  # Log-log scale
    ax_b.set_xlim(1, 5e5)

    # Plot c) Avalanche size with varying system size
    ax_c = axes[1, 0]
    ax_c.set_xlabel('Avalanche Size')
    ax_c.set_ylabel(fr'$P(X\geq x)$')
    ax_c.set_yscale('log')  # Log scale on y-axis only (matching plot a)
    ax_c.set_xscale('log')  # Log scale on x-axis (matching plot a)
    ax_c.set_xlim(5e-4, 2)

    # Plot d) Avalanche duration with varying system size
    ax_d = axes[1, 1]
    ax_d.set_xlabel('Avalanche Duration')
    # ax_d.set_ylabel('P(Duration ≥ t)')
    ax_d.set_yscale('log')
    ax_d.set_xscale('log')  # Log-log scale (matching plot b)
    ax_d.set_xlim(1, 5e5)


    # Define groups for plotting
    threshold_param_runs = []  # List of run IDs with varying ε and γ but fixed L
    system_size_runs = []  # List of run IDs with varying L but fixed ε and γ

    # Valid system sizes for plots c and d
    valid_system_sizes = [16, 32, 64, 128]

    # Find the smallest noise value in all runs
    min_noise = float('inf')
    for f in os.listdir(SOC_DIR):
        if f.endswith('_params.json'):
            params = load_params(os.path.join(SOC_DIR, f))
            noise = params.get('noise', float('inf'))
            if noise < min_noise:
                min_noise = noise

    print(f"Smallest noise value found: {min_noise}")

    # Collect run IDs for each group
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

            # Filter for threshold param runs (a and b)
            # Only include runs with epsilon in [0.3, 0.6], gamma in [-0.6, -0.3], and L = 32
            if (L == 32 and
                    0.3 <= epsilon <= 0.6 and
                    -0.6 <= gamma <= -0.3):
                threshold_param_runs.append((run_id, params))

            # Filter for system size runs (c and d)
            # Fix epsilon and gamma values and use only specified system sizes
            if (abs(epsilon - 0.3) < 1e-6 and
                    abs(gamma - (-0.6)) < 1e-6 and
                    L in valid_system_sizes):
                system_size_runs.append((run_id, params))

    # Sort by parameter values
    threshold_param_runs.sort(key=lambda x: (x[1].get('epsilon', 0), x[1].get('gamma', 0)))
    system_size_runs.sort(key=lambda x: x[1].get('L', 0))

    # Print debug info
    print(f"Threshold parameter runs (a & b): {len(threshold_param_runs)}")
    for run_id, params in threshold_param_runs:
        print(
            f"  Run {run_id}: L={params.get('L')}, ρ={params.get('density')}, η={params.get('noise'):.4f}, ε={params.get('epsilon')}, γ={params.get('gamma')}")

    print(f"System size runs (c & d): {len(system_size_runs)}")
    for run_id, params in system_size_runs:
        print(
            f"  Run {run_id}: L={params.get('L')}, ρ={params.get('density')}, η={params.get('noise'):.4f}, ε={params.get('epsilon')}, γ={params.get('gamma')}")

    # Use categorical colormap for threshold parameters (plots a & b)
    cmap_threshold = plt.get_cmap('tab10')
    n_threshold = len(threshold_param_runs)
    colors_threshold = [cmap_threshold(i % 10) for i in range(n_threshold)]

    # Define colormap for system sizes (sequential)
    cmap_system = plt.get_cmap('viridis')
    n_system = len(system_size_runs)
    colors_system = [cmap_system(i / (n_system - 1)) for i in range(n_system)]

    # durations_ref = np.logspace(0, 3, 100)  # Log x-axis for durations (plots b & d)

    # Plot varying threshold parameters (plots a & b)
    for i, (run_id, params) in enumerate(threshold_param_runs):
        epsilon = params.get('epsilon', 0)
        gamma = params.get('gamma', 0)

        # Calculate avalanche statistics
        try:
            avalanche_sizes, avalanche_durations = calculate_avalanche_statistics(run_id)

            if avalanche_sizes is not None and len(avalanche_sizes) > 0:
                label = fr"$\epsilon={epsilon:.1f}$, $\gamma={gamma:.1f}$"
                color = colors_threshold[i]

                # Plot size distribution (log-lin scale)
                plot_ccdf(avalanche_sizes, ax_a, label=label, color=color)

                # Plot duration distribution (log-log scale)
                plot_ccdf(avalanche_durations, ax_b, label=label, color=color)
        except Exception as e:
            print(f"Error processing run {run_id}: {str(e)}")

    # Add Vicsek reference model to plots a & b if available
    if vicsek_ref_sizes is not None and len(vicsek_ref_sizes) > 0:
        # Plot Vicsek model reference with dashed grey line
        plot_ccdf(vicsek_ref_sizes, ax_a, label="Vicsek Model", color='grey',
                  linestyle='--', marker='o')
        plot_ccdf(vicsek_ref_durations, ax_b, label="Vicsek Model", color='grey',
                  linestyle='--', marker='o')

    # Plot varying system sizes (plots c & d)
    print(f"Processing {len(system_size_runs)} system size runs:")
    for i, (run_id, params) in enumerate(system_size_runs):
        L = params.get('L', 0)
        print(f"Processing run {run_id} with L={L}...")

        # Calculate avalanche statistics
        try:
            avalanche_sizes, avalanche_durations = calculate_avalanche_statistics(run_id)

            if avalanche_sizes is not None:
                print(f"  Found {len(avalanche_sizes)} avalanches")
                if len(avalanche_sizes) > 0:
                    label = fr"$L={L}$"
                    color = colors_system[i]

                    # Plot size distribution
                    plot_ccdf(avalanche_sizes, ax_c, label=label, color=color)

                    # Plot duration distribution
                    plot_ccdf(avalanche_durations, ax_d, label=label, color=color)
                else:
                    print(f"  Warning: No avalanches found for run {run_id} (L={L})")
            else:
                print(f"  Warning: No avalanche data returned for run {run_id} (L={L})")
        except Exception as e:
            print(f"Error processing run {run_id} (L={L}): {str(e)}")


    # Add legends
    for ax in [ax_a, ax_c]:
        ax.legend(loc='best', frameon=True)
        # enable minor ticks for all axes on the x-axis
        ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(subs='all', numticks=15))

    for ax in [ax_b, ax_d]:
        ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(subs='all', numticks=15))

    label_axes(fig, labels=['(a)', '(b)', '(c)', '(d)'], loc=(0.95, 0.9))

    plt.tight_layout()

    # Save figure
    output_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'figure3_avalanches2.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.eps'), bbox_inches='tight', dpi=300)


    print(f"Figure saved to {output_path}")

    return fig


if __name__ == "__main__":
    create_figure3()
    plt.show()

