import json
import os

import numpy as np
from tqdm.auto import tqdm

import analysis_functions as af
from interaction_functions import grid_neighbors


def compute_observables(data, params, tc=None, existing_observables=None):
    """
    Compute observables for a given simulation run.

    Parameters:
    - data: numpy array of shape (T, 3, N) containing simulation data
    - params: dictionary of simulation parameters
    - tc: equilibration time (if None, uses 1e3)
    - existing_observables: dictionary of already computed observables

    Returns:
    - dict containing computed observables
    """
    if tc is None:
        tc = int(1e4)  # default equilibration time

    # Initialize result dict with existing observables if provided
    result = {} if existing_observables is None else dict(existing_observables)

    # Extract parameters
    L = params['L']
    dt = params['dt']
    r = params['r']
    # data = data.astype(np.float32)

    # Compute order parameter if not already present
    if 'orderparameter' not in result:
        result['orderparameter'] = af.op(data[:, 2], axis=-1)
        print("Computed order parameter")

    # Compute average angle if not already present
    if 'average_angle' not in result:
        result['average_angle'] = af.average_angle(data[:, 2], axis=-1)
        print("Computed average angle")

    # Calculate threshold if needed for other calculations
    if 'threshold' not in result and tc is not None:
        result['threshold'] = np.mean(result['orderparameter'][tc:]) - np.std(result['orderparameter'][tc:])
        result['tc'] = tc

    # Compute velocity correlations if not already present
    if 'velocity_correlation' not in result or 'velocity_fluctuation_correlation' not in result:
        dr = r / 10
        rs = np.arange(0, np.sqrt(2) * L / 2 + dr / 2, dr)
        result['rs'] = rs

        sample_size_velcorr = 5000
        sample_ids = np.random.choice(len(data[tc:]), size=sample_size_velcorr, replace=False) + tc

        if 'velocity_correlation' not in result:
            result['velocity_correlation'] = af.vel_corr(data[sample_ids], rs, L)
            print("Computed velocity correlation")

        if 'velocity_fluctuation_correlation' not in result:
            result['velocity_fluctuation_correlation'] = af.vel_corr(data[sample_ids], rs, L,
                                                                     fluctuation_correlation=True)
            print("Computed velocity fluctuation correlation")

    # Compute block minima if not already present
    if 'block_minima' not in result:
        result['block_minima'] = np.array(af.block_min(result['orderparameter'][tc:])) * dt
        print("Computed block minima")

    # Compute temporal statistics if not already present
    threshold = result['threshold']

    if 'ordered_times' not in result:
        result['ordered_times'] = np.array(af.ordered_time(result['orderparameter'][tc:],
                                                           len(result['orderparameter']) - tc, threshold)) * dt
        print("Computed ordered times")

    if 'return_times' not in result:
        result['return_times'] = np.array(af.return_time(result['orderparameter'][tc:],
                                                         len(result['orderparameter']) - tc, threshold)) * dt
        print("Computed return times")

    # Compute autocorrelation if not already present
    if 'autocorrelation' not in result:
        result['autocorrelation'] = af.autocorrelation(result['orderparameter'][tc:],
                                                       max_lag=int(0.1 * len(result['orderparameter'])))
        print("Computed autocorrelation")

    # Compute velocity autocorrelation if not already present
    if 'velocity_autocorrelation' not in result:
        # Use data after equilibration time and limit max_lag to avoid excessive computation
        result['velocity_autocorrelation'] = af.velocity_autocorrelation(data[tc:], max_lag=tc)
        print("Computed velocity autocorrelation")

    # Compute mean squared displacement using unwrapped trajectories if not already present
    if 'mean_squared_displacement' not in result:
        # Unwrap trajectories first
        unwrapped = af.unwrap_trajectory(data[tc:, :2, :], L)
        result['mean_squared_displacement'] = af.mean_squared_displacement(unwrapped)
        print("Computed mean squared displacement")

    # Compute local order over time if not already present
    if 'local_order' not in result:
        # Generate cell neighbors for the local order calculation
        cell_neighbors = grid_neighbors(L, r)

        # For large simulations, sample a subset of timesteps to reduce computation
        if len(data[tc:]) > 1000:
            sample_size = 1000
            sample_ids = np.random.choice(len(data[tc:]), size=sample_size, replace=False) + tc
            sampled_data = data[sample_ids]
            result['local_order'] = af.local_order_over_time(sampled_data, L, r, cell_neighbors)
            print(f"Computed local order over time for {sample_size} sampled time steps")
        else:
            result['local_order'] = af.local_order_over_time(data[tc:], L, r, cell_neighbors)
            print("Computed local order over time")

    return result


def analyze_simulation_run(run_id, base_dir):
    """
    Analyze a single simulation run and save observables.

    Parameters:
    - run_id: ID of the simulation run
    - base_dir: Directory containing simulation data
    """
    # Define filenames
    observables_file = os.path.join(base_dir, f'observables_{run_id}.npz')
    data_filename = os.path.join(base_dir, f'run_{run_id}.dat')
    params_filename = os.path.join(base_dir, f'run_{run_id}_params.json')

    # Load parameters
    parameters = load_parameters(params_filename)
    print(f"Loaded parameters for run {run_id}: {parameters}")

    # Check if observables file already exists
    existing_observables = {}
    if os.path.exists(observables_file):
        print(f"Found existing observables file for run {run_id}")
        npz_file = np.load(observables_file, allow_pickle=True)
        for key in npz_file.files:
            existing_observables[key] = npz_file[key]
        print(f"Loaded existing observables: {list(existing_observables.keys())}")

    # Check if we need to compute any observables
    all_observables = {'orderparameter', 'rs', 'velocity_correlation', 'velocity_fluctuation_correlation',
                       'block_minima', 'ordered_times', 'return_times', 'autocorrelation', 'threshold', 'tc',
                       'velocity_autocorrelation', 'mean_squared_displacement', 'local_order', 'average_angle'}
    missing_observables = all_observables - set(existing_observables.keys())

    if not missing_observables:
        print(f"All observables already computed for run {run_id}")
        return

    print(f"Need to compute the following observables: {missing_observables}")

    # Load data
    data_memmap = load_simulation_data(data_filename)
    # Reshape the memmap to (tmax, 3, N)
    tmax = parameters['tmax']
    N = parameters['N']
    data = data_memmap.reshape((tmax, 3, N))

    print(f"Loaded data for run {run_id}: shape {data.shape}")

    # Compute observables
    observables = compute_observables(data, parameters, existing_observables=existing_observables)

    # update observables with the already computed ones
    for key in existing_observables:
        if key not in observables:
            observables[key] = existing_observables[key]
    # Save observables
    np.savez(observables_file, **observables)
    print(f"Saved observables for run {run_id}")


def analyze_all_runs(base_dir, skip_existing_output=True):
    """
    Analyze all simulation runs in the given directory.

    Parameters:
    - base_dir: Directory containing simulation data
    """
    # Get list of all data files
    data_files = [f for f in os.listdir(base_dir) if f.startswith('run_') and f.endswith('.dat')]
    run_ids = [int(f.split('_')[1].split('.')[0]) for f in data_files]

    print(f"Found {len(run_ids)} simulation runs to analyze")

    # Analyze each run
    for run_id in tqdm(run_ids):
        result_file = os.path.join(base_dir, f'observables_{run_id}.npz')

        # Check if the result file already exists
        if os.path.exists(result_file) and skip_existing_output:
            print(f"Skipping run {run_id}, results already exist.")
            continue

        analyze_simulation_run(run_id, base_dir)

def load_parameters(filename):
    """
    Loads simulation parameters from a JSON file.

    Parameters:
    - filename: Path to the JSON file.

    Returns:
    - parameters: Dictionary containing simulation parameters.
    """
    with open(filename, 'r') as f:
        parameters = json.load(f)
    return parameters


def load_simulation_data(data_filename):
    """
    Loads simulation data from a memmap file.

    Parameters:
    - data_filename: Path to the memmap file.

    Returns:
    - data: NumPy memmap array containing simulation results.
    """
    # To read, use mode='r'
    data = np.memmap(data_filename, dtype=np.float16, mode='r')

    return data


if __name__ == '__main__':
    # Set the base directory where simulation data is stored
    # BASE_DIR = r'X:\soc_project\vicsek_simulation_runs'  # <-- CHANGE THIS to your data directory
    BASE_DIR = r'X:\soc_project\soc_simulation_runs2'  # <-- CHANGE THIS to your data directory

    # Run analysis
    analyze_all_runs(BASE_DIR)
