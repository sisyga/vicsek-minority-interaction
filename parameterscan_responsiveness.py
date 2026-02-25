import gc
import json
import os

import numpy as np
from tqdm.auto import tqdm

import interaction_functions as ia
import analysis_functions as af


def initialize_ordered_flock_with_defector(N, L):
    """
    Initialize an ordered flock with one defector particle.

    Matches the setup from example_movie.py (Fig 1 d-f):
    - Flock: N-1 particles in a band, all aligned at angle 0 (moving right)
    - Defector: 1 particle to the right of flock, at angle π (moving left)

    Parameters:
    - N: Total number of particles (flock + defector)
    - L: System size

    Returns:
    - initial_state: Array of shape (3, N) with [x_positions, y_positions, angles]
    - defector_idx: Index of the defector particle (always N-1)
    """
    N_flock = N - 1

    # Flock: moving left to right (angle = 0)
    # Positioned in a band in the left-middle part of the box
    xs_flock = np.random.uniform(L/2-5, L/2, N_flock)
    ys_flock = np.random.uniform(L/2-2.5, L/2+2.5, N_flock)
    ths_flock = np.zeros(N_flock)  # Perfectly aligned

    # Defector: moving right to left (angle = pi)
    # Positioned to the right of the flock, in the middle of the band
    xs_defector = np.array([L/2+1])
    ys_defector = np.array([L/2])
    ths_defector = np.array([np.pi])

    # Combine initial states
    xs = np.concatenate((xs_flock, xs_defector))
    ys = np.concatenate((ys_flock, ys_defector))
    ths = np.concatenate((ths_flock, ths_defector))

    initial_state = np.array([xs, ys, ths], dtype=np.float64)
    defector_idx = N - 1  # Defector is always the last particle

    return initial_state, defector_idx


def initialize_memmap_observables(filename, tmax, dtype=np.float32):
    """
    Initialize a memory-mapped file for storing observables time series.

    Parameters:
    - filename: Path to the memmap file
    - tmax: Total number of timesteps
    - dtype: Data type of the array elements

    Returns:
    - mmap: The initialized memmap array of shape (tmax+1, 4)
            [:, 0] = time points (0, 1, 2, ..., tmax)
            [:, 1] = order parameter φ(t)
            [:, 2] = cos(Θ(t))
            [:, 3] = Θ(t)
    """
    # Ensure the directory exists
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Preallocate the memmap array
    # Shape: (tmax+1, 4) for [time, φ, cos(Θ), Θ]
    mmap = np.memmap(filename, dtype=dtype, mode='w+', shape=(tmax+1, 4))

    return mmap


def save_parameters(parameters, filename):
    """
    Save simulation parameters to a JSON file.

    Parameters:
    - parameters: Dictionary containing simulation parameters
    - filename: Path to the JSON file
    """
    # Ensure the directory exists
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    with open(filename, 'w') as f:
        json.dump(parameters, f, indent=4)

    print(f"Parameters saved to {filename}")


def simulation_run_with_perturbation(parameters, mmap, run_id, params_filename, flush_interval=1000):
    """
    Run a single simulation with controlled defector perturbation.

    The defector particle is externally controlled (forced to angle π) for
    duration tau, then released to evolve naturally. We track φ(t) and cos(Θ(t))
    for the full timeline.

    Parameters:
    - parameters: dict containing simulation parameters
    - mmap: Preallocated memmap array to store observables
    - run_id: Unique identifier for the simulation run
    - params_filename: Path to the JSON file for saving parameters
    - flush_interval: Number of timesteps between each memmap flush
    """
    tmax = parameters['tmax']
    N = parameters['N']
    L = parameters['L']
    tau = parameters['tau']
    model = parameters['model']

    # Initialize ordered flock with defector
    current_state, defector_idx = initialize_ordered_flock_with_defector(N, L)

    # Save parameters
    save_parameters(parameters, params_filename)

    # Precompute grid neighbors
    cell_neighbors = ia.grid_neighbors(L, parameters['r'])

    # Select update function based on model type
    if model == 'SOC':
        update_func = ia.update_soc_numba
        param_list = [parameters['dt'], parameters['r'], parameters['noise'],
                      parameters['L'], parameters['speed'], parameters['epsilon'],
                      parameters['gamma'], cell_neighbors]
    else:  # Vicsek
        update_func = ia.update_vicsek_numba
        param_list = [parameters['dt'], parameters['r'], parameters['noise'],
                      parameters['L'], parameters['speed'], cell_neighbors]

    try:
        # Main simulation loop
        for t in tqdm(range(tmax + 1), desc=f"Run {run_id}"):
            # Compute and store observables for current state
            angles = current_state[2]
            order_param = af.op(angles)
            avg_angle = af.average_angle(angles)

            mmap[t, 0] = t
            mmap[t, 1] = order_param
            mmap[t, 2] = np.cos(avg_angle)
            mmap[t, 3] = avg_angle

            # Update state (except on last timestep)
            if t < tmax:
                # Store defector's position before update (for perturbation control)
                if t < tau:
                    defector_old_pos = current_state[:2, defector_idx].copy()

                pos, ths = update_func(current_state, *param_list)
                current_state[:2] = pos
                current_state[2] = ths

                # Override defector angle AND position during controlled perturbation phase
                if t < tau:
                    current_state[2, defector_idx] = np.pi
                    # Recalculate defector position with forced angle π
                    # new_pos = old_pos + speed * dt * [cos(π), sin(π)]
                    current_state[0, defector_idx] = (defector_old_pos[0] + parameters['speed'] * parameters['dt'] * np.cos(np.pi)) % parameters['L']
                    current_state[1, defector_idx] = (defector_old_pos[1] + parameters['speed'] * parameters['dt'] * np.sin(np.pi)) % parameters['L']

            # Periodically flush to disk
            if (t + 1) % flush_interval == 0:
                mmap.flush()

        # Final flush to ensure all data is written
        mmap.flush()
        print(f"Run {run_id}: Simulation completed and data flushed to disk.")

    except Exception as e:
        print(f"Run {run_id}: An error occurred at timestep {t}: {e}")
        mmap.flush()
        raise
    finally:
        # Clean up memory
        del current_state
        gc.collect()


def generate_parameter_list():
    """
    Generate all parameter combinations for the responsiveness scan.

    Parameter space:
    - epsilon: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    - gamma: [-0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8]
    - tau: [1, 2, 5, 10, 20, 50, 100, 200]

    For each (epsilon, gamma, tau) combination, we run:
    - SOC model with those parameters
    - Vicsek model (for baseline comparison)

    Total: 7 × 7 × 8 × 2 = 784 simulations

    Returns:
    - params_list: List of parameter dictionaries
    """
    # Fixed parameters
    base_params = {
        'L': 32,
        'N': 200,
        'r': 1.0,
        'dt': 1.0,
        'speed': 0.5,
        'noise': 0.1 * 2 * np.pi / np.sqrt(12),  # Scaled for low noise
        'tmax': 500
    }

    # Parameter ranges
    epsilon_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    gamma_values = [-0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8]
    tau_values = [1, 2, 5, 10, 20, 50, 100, 200]

    params_list = []

    for tau in tau_values:
        for epsilon in epsilon_values:
            for gamma in gamma_values:
                # SOC model run
                params_soc = base_params.copy()
                params_soc['model'] = 'SOC'
                params_soc['epsilon'] = epsilon
                params_soc['gamma'] = gamma
                params_soc['tau'] = tau
                params_list.append(params_soc)

                # Vicsek model run (for comparison)
                params_vm = base_params.copy()
                params_vm['model'] = 'Vicsek'
                params_vm['tau'] = tau
                params_list.append(params_vm)

    return params_list


def run_parameter_scan(simulation_params_list, base_dir='responsiveness_runs',
                       dtype=np.float32, flush_interval=1000, rewrite=False):
    """
    Run multiple simulations and save results and parameters for each run.

    Parameters:
    - simulation_params_list: List of dicts containing simulation parameters
    - base_dir: Base directory where simulation data and parameters will be stored
    - dtype: Data type of the memmap array elements
    - flush_interval: Number of timesteps between each memmap flush
    - rewrite: If False, skip runs that already exist
    """
    for run_id, params in enumerate(simulation_params_list):
        print(f"Starting simulation run {run_id} with parameters: {params}")

        tmax = params['tmax']

        # Define filenames
        data_filename = os.path.join(base_dir, f'run_{run_id}.dat')
        params_filename = os.path.join(base_dir, f'run_{run_id}_params.json')

        # Check if this run was already done
        if os.path.exists(data_filename) and not rewrite:
            print(f"Run {run_id} already exists. Skipping...")
            continue

        # Initialize memmap for observables
        mmap = initialize_memmap_observables(data_filename, tmax, dtype=dtype)

        # Run simulation and save data
        simulation_run_with_perturbation(
            parameters=params,
            mmap=mmap,
            run_id=run_id,
            params_filename=params_filename,
            flush_interval=flush_interval
        )

        # Ensure the memmap is properly closed and memory is freed
        del mmap
        gc.collect()

        print(f"Completed simulation run {run_id}\n")


def main():
    """
    Main entry point for the responsiveness parameter scan.
    """
    BASE_DIR = 'responsiveness_runs'
    FLUSH_INTERVAL = 1000
    DTYPE = np.float32

    print("Generating parameter list...")
    params_list = generate_parameter_list()
    print(f"Total number of simulations: {len(params_list)}")

    print("\nStarting parameter scan...")
    run_parameter_scan(
        simulation_params_list=params_list,
        base_dir=BASE_DIR,
        dtype=DTYPE,
        flush_interval=FLUSH_INTERVAL,
        rewrite=False
    )

    print("\nParameter scan completed!")


if __name__ == "__main__":
    main()
