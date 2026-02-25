import gc
import json

import numba
import numpy as np
from tqdm.auto import tqdm

import interaction_functions as ia
from parameterscan_analysis import load_parameters, load_simulation_data


def initialize_memmap(filename, tmax, N, dtype=np.float64):
    """
    Initializes a memory-mapped file for storing simulation results.

    Parameters:
    - filename: Path to the memmap file.
    - tmax: Total number of timesteps.
    - N: Number of particles.
    - dtype: Data type of the array elements.

    Returns:
    - mmap: The initialized memmap array.
    """
    # Ensure the directory exists
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Preallocate the memmap array
    mmap = np.memmap(filename, dtype=dtype, mode='w+', shape=(tmax, 3, N))

    return mmap


def save_parameters(parameters, filename):
    """
    Saves simulation parameters to a JSON file.

    Parameters:
    - parameters: Dictionary containing simulation parameters.
    - filename: Path to the JSON file.
    """
    # Ensure the directory exists
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    with open(filename, 'w') as f:
        json.dump(parameters, f, indent=4)

    print(f"Parameters saved to {filename}")


def simulation_run(parameters, initial_condition, mmap, run_id, params_filename, interaction, flush_interval=10000):
    """
    Runs a single simulation and writes each timestep to the memmap array.
    Also saves the simulation parameters to a JSON file.

    Parameters:
    - parameters: dict containing simulation parameters (e.g., 'tmax', 'N').
    - initial_condition: NumPy array of shape (3, N) representing the initial state.
    - mmap: Preallocated memmap array to store simulation results.
    - run_id: Unique identifier for the simulation run.
    - params_filename: Path to the JSON file for saving parameters.
    - flush_interval: Number of timesteps between each memmap flush.
    """
    tmax = parameters['tmax']
    current_state = initial_condition.copy()

    # Save parameters
    save_parameters(parameters, params_filename)
    cell_neighbors = ia.grid_neighbors(parameters['L'], parameters['r'])
    # update_vicsek_numba(state, dt, r, eta, L, speed, cell_neighbors) vicsek interaction call signature
    # update_soc_numba(positions, directions, dt, r, eta, L, speed, eps, gamma, cell_neighbors) soc interaction call signature
    if 'epsilon' not in parameters:
        parameterlist = [parameters['dt'], parameters['r'], parameters['noise'], parameters['L'], parameters['speed'],
                         cell_neighbors]
    else:
        parameterlist = [parameters['dt'], parameters['r'], parameters['noise'], parameters['L'], parameters['speed'],
                         parameters['epsilon'], parameters['gamma'], cell_neighbors]
    try:
        # Write the initial condition
        mmap[0] = current_state

        for i in tqdm(range(1, tmax)):
            # Update the current state using the Vicsek interaction
            current_positions, current_orientations = interaction(current_state, *parameterlist)
            current_state[:2] = current_positions[0], current_positions[1]
            current_state[2] = current_orientations

            # Write the updated state to the memmap array
            mmap[i] = current_state

            # Periodically flush to disk to ensure data integrity
            if (i + 1) % flush_interval == 0:
                mmap.flush()
                # print(f"Run {run_id}, Timestep {i + 1} written to disk.")

        # Final flush to ensure all data is written
        mmap.flush()
        print(f"Run {run_id}: Simulation completed and data flushed to disk.")

    except Exception as e:
        print(f"Run {run_id}: An error occurred at timestep {i}: {e}")
        mmap.flush()
        raise  # Re-raise the exception after flushing
    finally:
        # Clean up memory
        del current_state
        gc.collect()


def get_random_initial_condition(N, L, dtype=np.float64):
    """
    Generates a random initial condition for the Vicsek model.
    :param N:
    :param L:
    :return:
    """
    xs = np.random.rand(N) * L
    ys = np.random.rand(N) * L
    ths = np.random.uniform(0, 2 * np.pi, size=N)
    return np.array([xs, ys, ths], dtype=dtype)


def run_parameter_scan(simulation_params_list, interaction, base_dir='simulation_runs', dtype=np.float64,
                       flush_interval=10000, rewrite=False):
    """
    Runs multiple simulations and saves results and parameters for each run.

    Parameters:
    - simulation_params_list: List of dicts containing simulation parameters.
    - interaction: Function that updates the state of the system, e.g., Vicsek or SOC.
    - base_dir: Base directory where simulation data and parameters will be stored.
    - dtype: Data type of the memmap array elements.
    - flush_interval: Number of timesteps between each memmap flush.
    """
    for run_id, params in enumerate(simulation_params_list):
        print(f"Starting simulation run {run_id} with parameters: {params}")

        N = params['N']
        tmax = params['tmax']
        L = params['L']

        # Define filenames
        data_filename = os.path.join(base_dir, f'run_{run_id}.dat')
        params_filename = os.path.join(base_dir, f'run_{run_id}_params.json')
        # check if this exact run was already done. if yes, print message and skip
        if os.path.exists(data_filename) and not rewrite:
            print(f"Run {run_id} already exists. Skipping...")
            continue

        # Initialize memmap
        mmap = initialize_memmap(data_filename, tmax, N, dtype=dtype)

        # Initialize the initial condition
        initial_condition = get_random_initial_condition(N, L)

        # Run simulation and save data
        simulation_run(parameters=params,
                       initial_condition=initial_condition,
                       interaction=interaction,
                       mmap=mmap,
                       run_id=run_id,
                       params_filename=params_filename,
                       flush_interval=flush_interval)

        # Ensure the memmap is properly closed and memory is freed
        del mmap
        gc.collect()

        print(f"Completed simulation run {run_id}\n")


def test_single_simulation_run(base_dir, parameters, interaction):
    # Configuration for the test run
    run_id = "TEST"
    initial_condition = get_random_initial_condition(parameters['N'], parameters['L'])

    # Define filenames
    data_filename = os.path.join(base_dir, f'run_{run_id}.dat')
    params_filename = os.path.join(base_dir, f'run_{run_id}_params.json')

    # Initialize memmap
    mmap = initialize_memmap(data_filename, parameters['tmax'], parameters['N'], dtype='float64')

    # Run simulation
    simulation_run(parameters=parameters,
                   initial_condition=initial_condition,
                   mmap=mmap,
                   run_id=run_id,
                   params_filename=params_filename,
                   interaction=ia.update_vicsek_numba)

    # Clean up
    del mmap
    gc.collect()

    # Load and verify parameters
    loaded_params = load_parameters(params_filename)
    assert loaded_params == parameters, "Parameters do not match!"
    print("Parameters verified successfully.")

    # Load and verify data
    data_memmap = load_simulation_data(data_filename)
    data = data_memmap.reshape((parameters['tmax'], 3, parameters['N']))

    # Simple verification: check shape
    assert data.shape == (parameters['tmax'], 3, parameters['N']), "Data shape mismatch!"
    print("Data shape verified successfully.")


if __name__ == "__main__":
    # here we perform a parameter scan varying the relevant parameters of the model: noise, density,
    # soc parameters epsilon and gamma, and the system size L
    # we will also scan the vicsek model to have a direct comparison for the relevant noise, density, and system size
    BASE_DIR = r'X:\soc_project\soc_simulation_runs32'  # <-- CHANGE THIS to your output directory
    FLUSH_INTERVAL = 100000
    DTYPE = np.float16  # half precision for more memory efficiency and speed
    densityvalues = [.5, 1, 1.5, 2.0]
    noisevalues = [.5, .1, 1.5, .2, 2.5, .3]
    Lvalues = [32]
    epsilonvalues = [.2, .3, .4, .5, .6, .7, .8]
    gammavalues = [-.2, -.3, -.4, -.5, -.6, -.7, -.8]
    # numba.config.NUMBA_DEBUG = 1
    r = dt = 1.  # standard vicsek params
    speed = .5  # to be varied later
    tmax = int(5e4 + 1)
    numba.set_num_threads(24)
    import os

    os.environ["OMP_NUM_THREADS"] = "24"
    os.environ["NUMBA_NUM_THREADS"] = "24"
    os.environ["MKL_NUM_THREADS"] = "24"

    # create a list of dicts with all the possible combinations of parameters
    vicsek_parameters = []
    soc_parameters = []
    for density in densityvalues:
        for noise in noisevalues:
            for L in Lvalues:
                vicsek_parameters.append({'density': density, 'noise': noise * 2 * np.pi / np.sqrt(12),
                                          # scale noise strength to match variance of uniform distribution, factor 2pi to compare noise values with chaté et al. 2008
                                          'L': L, 'speed': speed, 'r': r, 'dt': dt, 'tmax': tmax,
                                          'N': int(density * L ** 2)})
                for epsilon in epsilonvalues:
                    for gamma in gammavalues:
                        temp_dict = vicsek_parameters[-1].copy()
                        temp_dict |= {'epsilon': epsilon, 'gamma': gamma}
                        soc_parameters.append(temp_dict)

    # print(soc_parameters[107])
    # Test a single simulation run
    # test_single_simulation_run(BASE_DIR, vicsek_parameters[0], ia.update_vicsek_numba)
    # Run simulations and save results
    run_parameter_scan(soc_parameters,
                       ia.update_soc_numba,
                       base_dir=BASE_DIR,
                       dtype=DTYPE,
                       flush_interval=FLUSH_INTERVAL,
                       rewrite=False)
