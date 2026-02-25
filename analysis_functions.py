from math import sqrt
from numba.typed import List
from numba import njit, jit, prange
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import correlate
import matplotlib.pyplot as plt
from interaction_functions import build_cell_list, find_neighbors


def st_line(x, m, c):
    return m * x + c


@njit(fastmath=True, parallel=True)
def autocorrelation_numba(phi_t, max_lag=None):
    """Compute the temporal autocorrelation function using Numba, only up to max_lag."""
    n = len(phi_t)
    if max_lag is None:
        max_lag = n
    else:
        max_lag = min(max_lag, n)

    mean_phi = 0.0
    for i in range(n):
        mean_phi += phi_t[i]
    mean_phi /= n

    var_phi = 0.0
    for i in range(n):
        var_phi += (phi_t[i] - mean_phi) ** 2
    var_phi /= n

    # Initialize the autocorrelation array
    acf = np.zeros(max_lag)

    # Compute autocorrelation for each lag
    for lag in prange(max_lag):
        # For each lag, calculate the correlation for all valid time pairs
        sum_corr = 0.0
        for t in range(n - lag):
            sum_corr += (phi_t[t] - mean_phi) * (phi_t[t + lag] - mean_phi)

        # Normalize by variance and valid number of time points
        acf[lag] = sum_corr / ((n - lag) * var_phi)

    return acf


def autocorrelation(phi_t, max_lag=None):
    """Compute the temporal autocorrelation function of the order parameter."""
    phi_t = np.array(phi_t)

    # Use the Numba-accelerated version for better performance
    return autocorrelation_numba(phi_t, max_lag)


@njit(parallel=True, fastmath=True)
def vel_corr_numba(arr, rs, L):  # TEST OPTIMIZATION
    """
    Parallel velocity correlation with direct bin indexing.
    arr shape: (T, 3, N)
    rs shape: (R,)  # evenly spaced
    L is box side length
    """
    T, _, N = arr.shape
    x_t = arr[:, 0]  # (T, N)
    y_t = arr[:, 1]  # (T, N)
    ths_t = arr[:, 2]  # (T, N)
    R = len(rs)

    Cs_local = np.zeros((N, R))
    norms_local = np.zeros((N, R))

    # Pre-calculate for bin arithmetic
    # r_min = rs[0]
    # r_max = rs[-1]
    # dr = rs[1] - rs[0]  # spacing

    # parallelize over i
    for i in prange(N):
        for j in range(i + 1, N):
            # Distances for each time
            dx = np.minimum(np.abs(x_t[:, i] - x_t[:, j]),
                            L - np.abs(x_t[:, i] - x_t[:, j]))
            dy = np.minimum(np.abs(y_t[:, i] - y_t[:, j]),
                            L - np.abs(y_t[:, i] - y_t[:, j]))
            dist_t = np.hypot(dx, dy)  # shape (T,)
            cs_pair = np.cos(ths_t[:, i] - ths_t[:, j])

            indxs = np.digitize(dist_t, rs) - 1

            Cs_local[i, indxs] += cs_pair
            norms_local[i, indxs] += 1

            # Bin each distance in dist_t
            # for t in range(T):
            #     d = dist_t[t]
            #     # Convert to bin index
            #     idx = int((d - r_min) / dr)  # side='left' style
            #
            #     Cs_local[i, idx] += cs_pair[t]
            #     norms_local[i, idx] += 1

    # Reduce over i to get final Cs, norms
    Cs = np.sum(Cs_local, axis=0)
    norms = np.sum(norms_local, axis=0)
    return Cs, norms


@njit(parallel=True, fastmath=True)
def vel_fluct_corr_numba(arr, rs, L):
    """
    Parallel + direct-binning version of vel_fluct_corr_numba.
    arr shape: (T, 3, N)
    rs shape: (R,) (must be evenly spaced)
    L is the side length of the box.

    Returns:
        Cs, norms: 1D arrays of length R.
    """
    T, _, N = arr.shape  # (time, dof, number_of_particles)
    x_t = arr[:, 0]  # shape (T, N)
    y_t = arr[:, 1]  # shape (T, N)
    ths_t = arr[:, 2]  # shape (T, N)
    v_t = np.exp(1j * ths_t)  # shape (T, N)
    R = len(rs)

    # 1) Compute fluctuation velocities once
    #    v_avg: shape (T,) = average velocity direction (complex) over N
    v_avg = np.sum(v_t, axis=-1) / N
    v_avg = v_avg.reshape(-1, 1)  # shape (T, 1)
    #    u_t: shape (T, N) = fluctuation from the average
    u_t = v_t - v_avg
    ux_t = np.real(u_t)
    uy_t = np.imag(u_t)

    # 2) We'll accumulate results in local arrays for each i
    #    That avoids race conditions when updating Cs/norms in parallel
    Cs_local = np.zeros((N, R))
    norms_local = np.zeros((N, R))

    # For direct bin arithmetic
    # r_min = rs[0]
    # r_max = rs[-1]
    # dr = rs[1] - rs[0]  # spacing

    # 3) Parallel loop over i
    for i in prange(N):
        for j in range(i + 1, N):
            dx = np.minimum(np.abs(x_t[:, i] - x_t[:, j]),
                            L - np.abs(x_t[:, i] - x_t[:, j]))
            dy = np.minimum(np.abs(y_t[:, i] - y_t[:, j]),
                            L - np.abs(y_t[:, i] - y_t[:, j]))
            dist_t = np.hypot(dx, dy)  # shape (T,)

            # Instead of cos(...) of angles, we do dot-product of fluctuation vectors
            cs_pair = ux_t[:, i] * ux_t[:, j] + uy_t[:, i] * uy_t[:, j]
            idxs = np.digitize(dist_t, rs) - 1
            Cs_local[i, idxs] += cs_pair
            norms_local[i, idxs] += 1

            # Accumulate into Cs_local[i,:], norms_local[i,:]
            # for t in range(T):
            #     d = dist_t[t]
            #     # Convert distance to bin index
            #     idx = int((d - r_min) / dr)
            #     Cs_local[i, idx] += cs_pair[t]
            #     norms_local[i, idx] += 1

    # 4) Reduce across i
    Cs = np.sum(Cs_local, axis=0)
    norms = np.sum(norms_local, axis=0)

    return Cs, norms


def vel_corr(arr, rs, L, fluctuation_correlation=False):
    if fluctuation_correlation:
        Cs, norm = vel_fluct_corr_numba(arr, rs, L)
    else:
        Cs, norm = vel_corr_numba(arr, rs, L)
    Cs = np.divide(Cs, norm, out=np.zeros_like(Cs), where=norm > 0)
    Cs = np.ma.masked_where(norm == 0, Cs)
    # Cs = Cs / Cs[0] if fluctuation_correlation else Cs
    return Cs


def op(ths, axis=None):
    """
    Calculate the order parameter for a given set of angles.

    Parameters:
    ths (array-like): Array of angles in radians.

    Returns:
    float: The order parameter, which is a measure of the alignment of the angles.
    """
    if axis is None:
        return abs(np.exp(1j * ths).mean())
        # return np.sqrt((np.sum(np.cos(ths))) ** 2 + (np.sum(np.sin(ths))) ** 2) / len(ths)
    else:
        return np.abs(np.exp(1j * ths).mean(axis=axis))


def average_angle(ths, axis=None):
    """
    Calculate the average orientation, ie, the argument of the complex order parameter.

    Parameters:
    ths (array-like): Array of angles in radians.

    Returns:
    float: The order parameter, which is a measure of the alignment of the angles.
    """
    if axis is None:
        return np.angle(np.exp(1j * ths).mean())
    else:
        return np.angle(np.exp(1j * ths).mean(axis=axis))

@jit(nopython=True)
def block_min(data, block_size=250):
    """
    Calculate the minimum value for each block of data.

    Parameters:
    data (array-like): A 1D array of the order parameter.
    block_size (int): The size of each block.

    Returns:
    list: A list of minimum values for each block.
    """
    return [min(data[i:i + block_size]) for i in range(0, len(data), block_size)]

@jit(nopython=True, nogil=True, fastmath=True)
def return_time(data, tmax, threshold):
    """
    Calculate the return time distribution for a given dataset.

    Parameters:
    data (array-like): A 1D array of the order parameter.
    tmax (int): The maximum time to consider.
    threshold (float): The threshold value to determine return times.

    Returns:
    list: A list of return times.
    """
    return_time = List()
    for t in range(tmax - 1):
        if data[t] < threshold and data[t - 1] >= threshold:
            start = t
        if data[t] < threshold and data[t + 1] >= threshold:
            finish = t
            return_time.append(finish - start + 1)
    return return_time

@jit(nopython=True, nogil=True, fastmath=True)
def ordered_time(data, tmax, threshold):
    """
    Calculate the ordered time distribution for a given dataset.

    Parameters:
    data (array-like): A 1D array of the order parameter.
    tmax (int): The maximum time to consider.
    threshold (float): The threshold value to determine ordered times.

    Returns:
    list: A list of ordered times.
    """
    ordered_time = List()
    for t in range(tmax - 1):
        if data[t] >= threshold and data[t - 1] < threshold:
            start = t
        if data[t] < threshold and data[t + 1] >= threshold:
            finish = t
            ordered_time.append(finish - start + 1)
    return ordered_time


def avalanche_integrated_excursions(order_parameter, threshold):
    """
    Compute the integrated excursion size for each avalanche in a time series.

    An avalanche is defined as a contiguous interval where the order parameter
    stays strictly below the threshold. For each such interval [start, end),
    the integrated excursion is the sum over time of (threshold - order_parameter[t])
    for t in that interval. This differs from the existing avalanche "size"
    (which uses the maximum excursion) by integrating across the full duration.

    Parameters:
    - order_parameter (array-like): 1D array of the order parameter over time.
    - threshold (float): The threshold value (phi_c).

    Returns:
    - np.ndarray: Array of integrated excursions, one per detected avalanche.
    """
    op = np.asarray(order_parameter)
    if op.ndim != 1 or op.size == 0:
        return np.array([])

    below = op < threshold

    # Rising edge: start of avalanche (goes False->True)
    starts = np.where(np.logical_and(below[1:], ~below[:-1]))[0] + 1
    # Falling edge: end of avalanche (goes True->False)
    ends = np.where(np.logical_and(~below[1:], below[:-1]))[0] + 1

    if starts.size == 0 or ends.size == 0:
        return np.array([])

    # If the first end precedes the first start, drop it
    if ends[0] < starts[0]:
        ends = ends[1:]

    # Truncate to matching count
    m = min(starts.size, ends.size)
    starts = starts[:m]
    ends = ends[:m]

    if m == 0:
        return np.array([])

    excursions = []
    for s, e in zip(starts, ends):
        if e > s:
            # Sum over interval [s, e) of (threshold - op[t]) for op[t] < threshold
            vals = threshold - op[s:e]
            # Numerical guard: only positive contributions
            vals = np.where(vals > 0.0, vals, 0.0)
            excursions.append(np.sum(vals))

    return np.array(excursions)


def plot_return_time_distribution(return_times, ax=None, no_bins=None, label=None):
    """
    Plot the return time distribution given an array of return times.

    Parameters:
    return_times (array-like): Array of return times.
    ax (matplotlib.axes.Axes): Existing axes to plot on. If None, a new figure and axes are created.
    no_bins (int): Number of bins for the histogram. Default is 7.
    """
    if ax is None:
        fig, ax = plt.subplots()

    return_times = np.array(return_times)
    if len(return_times) == 0:
        print('No return times found')
        return ax

    count = len(return_times)
    no_bins = np.ceil(np.log2(count) + 1).astype(int) if no_bins is None else no_bins  # use Sturges' rule if no_bins is not specified
    bins = np.logspace(np.log10(2), np.log10(return_times.max()), no_bins)
    freq, edges = np.histogram(return_times, bins=bins, density=True)
    bin_centers = (edges[1:] + edges[:-1]) / 2
    raw_counts, _ = np.histogram(return_times, bins=bins)
    bin_widths = np.diff(edges)
    p = raw_counts / count / bin_widths
    # fact = raw_counts[0] / freq[0] * (edges[1:] - edges[:-1]) / (edges[1] - edges[0])

    err = np.sqrt(p * (1 - p) / count) / bin_widths

    # bin_centerss = np.linspace(min(bin_centers), max(bin_centers), 101)
    # print('Powerlaw exponent =', -1 - alpha)
    ax.errorbar(bin_centers, freq, yerr=err, label=label)
    # ax.plot(bin_centers, raw_counts / fact, 'o-', color='blue')
    return ax, bin_centers

def powerlaw_fit(data):
    """
    Fit a power-law distribution to the data.

    Parameters:
    data (array-like): Data to fit.

    Returns:
    float: The fitted power-law exponent.
    """
    x_m = min(data)
    alpha = len(data) / (np.sum(np.log(data / x_m)))
    return x_m, alpha

def powerlaw(x, x_m, alpha):
    """
    Calculate the power-law distribution. x_m and alpha are obtained from the powerlaw_fit function.
    :param x:
    :param x_m:
    :param alpha:
    :return:
    """
    return alpha * x_m ** alpha / x ** (alpha + 1)


def plot_corr_length(Ls, xis):
    """
    Plots the correlation length and a linear fit.

    Parameters:
    Ls (array-like): System sizes.
    xis (array-like): Correlation lengths.
    """

    def st_line(x, m, c):
        return m * x + c

    # Fit a straight line to the data
    m1, c1 = curve_fit(st_line, Ls, xis)[0]

    # Plot the correlation length data
    plt.plot(Ls, xis, 'o-', label='Correlation length')

    # Plot the linear fit
    plt.plot(Ls, m1 * Ls + c1, '--', label='St line fit', color='red')

    # Add labels and legend
    plt.xlabel("System size (L)")
    plt.ylabel("Correlation length")
    plt.legend()

    # Show the plot
    plt.show()

def local_order_over_time(arr, L, r, cell_neighbors):
    """
    Calculate the local order parameter for each particle over time.

    Parameters:
    arr (ndarray): Array of particle positions and directions.
    L (float): Size of the simulation box.
    r (float): Interaction radius.
    cell_neighbors (ndarray): Precomputed cell neighbors.

    Returns:
    ndarray: Array of local order parameters for each particle over time.
    """
    N = arr.shape[2]
    local_order = np.zeros((arr.shape[0], N))

    for t in range(arr.shape[0]):
        positions = arr[t, :2, :].T  # Transpose here to get (N, 2) shape
        directions = arr[t, 2, :]
        local_order[t] = local_order_parameter(positions, directions, L, r, cell_neighbors)

    return local_order

@jit(nopython=True, nogil=True, parallel=True, fastmath=True)
def local_order_parameter(positions, directions, L, r, cell_neighbors):
    """
    Calculate the local order parameter for each particle.

    Parameters:
    positions (ndarray): Array of particle positions with shape (N, 2).
    directions (ndarray): Array of particle directions with shape (N,).
    L (float): Size of the simulation box.
    r (float): Interaction radius.
    cell_neighbors (ndarray): Precomputed cell neighbors.

    Returns:
    ndarray: Array of local order parameters for each particle.
    """
    N = len(directions)
    vx = np.cos(directions)
    vy = np.sin(directions)
    local_order = np.zeros(N)

    cell_list, cell_particle_counts, cell_indices, n_cells = build_cell_list(positions, L, r)

    # Correctly call find_neighbors with properly ordered parameters
    neighbors, neighbor_counts = find_neighbors(positions, L, cell_list, cell_particle_counts, cell_indices,
                                                cell_neighbors, r)

    for i in prange(N):
        sin_avg = 0.
        cos_avg = 0.

        # Add self to neighbor list for consistency
        sin_avg += vy[i]
        cos_avg += vx[i]
        neighbor_count = 1  # Start with 1 for self

        # Add contributions from all neighbors
        for j in range(neighbor_counts[i]):
            neighbor_idx = neighbors[i, j]
            sin_avg += vy[neighbor_idx]
            cos_avg += vx[neighbor_idx]
            neighbor_count += 1

        # Only normalize if we have neighbors
        if neighbor_count > 1:
            sin_avg /= neighbor_count
            cos_avg /= neighbor_count
            local_order[i] = sqrt(sin_avg ** 2 + cos_avg ** 2)
        else:
            local_order[i] = 1.0  # If no neighbors, particle is perfectly ordered with itself

    return local_order


def calculate_order_parameter_histogram(order_parameters, num_bins=50):
    """
    Calculate the histogram of the order parameter values.

    Parameters:
    order_parameters (ndarray): Array of order parameter values.
    num_bins (int): Number of bins for the histogram.

    Returns:
    tuple: Bin counts and bin edges.
    """
    counts, bin_edges = np.histogram(order_parameters, bins=num_bins)
    return counts, bin_edges


def plot_zipfs_law(counts):
    """
    Plot the histogram counts against their rank to test Zipf's law.

    Parameters:
    counts (ndarray): Array of histogram counts.
    """
    sorted_counts = np.sort(counts)[::-1]
    ranks = np.arange(1, len(sorted_counts) + 1)

    plt.figure()
    plt.loglog(ranks, sorted_counts, 'o-', label='Data')
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.title('Zipf\'s Law')
    plt.legend()
    plt.show()


@njit(fastmath=True)
def unwrap_trajectory(traj, L):
    """
    Unwrap a trajectory with periodic boundary conditions.
    
    Parameters:
    traj: Array of shape (T, 2, N) containing x,y positions over time for N particles
    L: The box size
    
    Returns:
    Unwrapped trajectory of shape (T, 2, N)
    """
    T, dims, N = traj.shape
    unwrapped = np.copy(traj)

    # Offset to add when detecting boundary crossings
    offset = np.zeros((dims, N))

    # Start from the second time step
    for t in range(1, T):
        # Check for boundary crossings (large position changes)
        for d in range(dims):
            # Calculate displacement between consecutive positions
            displacement = traj[t, d, :] - traj[t - 1, d, :]

            # Detect positive crossings (particle jumps from right to left)
            pos_cross = displacement < -L / 2
            # Detect negative crossings (particle jumps from left to right)
            neg_cross = displacement > L / 2

            # Update the offset for particles that crossed boundaries
            offset[d, pos_cross] += L
            offset[d, neg_cross] -= L

            # Apply the cumulative offset to the current position
            unwrapped[t, d, :] = traj[t, d, :] + offset[d, :]

    return unwrapped


@njit(fastmath=True, parallel=True)
def velocity_autocorrelation(arr, max_lag=None):
    """
    Compute the velocity autocorrelation function: C_vv(t) = <v(0)·v(t)>/<v(0)·v(0)>
    
    Parameters:
    arr: Array of shape (T, 3, N) where arr[:, 2, :] contains orientation angles
    max_lag: Maximum time lag to compute (default: half the trajectory length)
    
    Returns:
    Array of velocity autocorrelation values for each time lag
    """
    T, _, N = arr.shape

    if max_lag is None:
        max_lag = T // 2
    else:
        # Ensure max_lag doesn't exceed trajectory length
        max_lag = min(max_lag, T)

    # Extract orientation angles
    thetas = arr[:, 2, :]

    # Compute velocity components (already normalized since cos^2 + sin^2 = 1)
    vx = np.cos(thetas)  # shape (T, N)
    vy = np.sin(thetas)  # shape (T, N)

    # Initialize autocorrelation array
    autocorr = np.zeros(max_lag)

    # For each lag
    for lag in prange(max_lag):
        corr_sum = 0.0
        # Compute dot products for all particles and available time points
        for i in range(N):
            for t in range(T - lag):
                # v(t)·v(t+lag) = vx(t)*vx(t+lag) + vy(t)*vy(t+lag)
                corr_sum += vx[t, i] * vx[t + lag, i] + vy[t, i] * vy[t + lag, i]

        # Normalize by number of particles and time points
        autocorr[lag] = corr_sum / (N * (T - lag))

    # Since velocities are already normalized, <v(0)·v(0)> should be 1,
    # but we'll normalize explicitly for numerical stability
    autocorr = autocorr / autocorr[0] if autocorr[0] > 0 else autocorr
    
    return autocorr


@njit(fastmath=True, parallel=True)
def mean_squared_displacement(traj, max_lag=None):
    """
    Calculate mean squared displacement using unwrapped trajectories.
    
    Parameters:
    traj: Array of shape (T, 2, N) containing unwrapped particle positions over time
    max_lag: Maximum time lag to compute (default: half the trajectory length)
    
    Returns:
    Array of MSD values for each time lag
    """
    T, dims, N = traj.shape

    if max_lag is None:
        max_lag = T // 2
    else:
        max_lag = min(max_lag, T)

    # Initialize MSD array
    msd = np.zeros(max_lag)

    # For each lag
    for lag in prange(max_lag):
        displacement_sq_sum = 0.0
        # Compute displacement for all particles
        for i in range(N):
            for t in range(T - lag):
                # Calculate squared displacement (using unwrapped coordinates)
                for d in range(dims):
                    dx = traj[t + lag, d, i] - traj[t, d, i]
                    displacement_sq_sum += dx * dx

        # Normalize by number of particles and time points
        msd[lag] = displacement_sq_sum / (N * (T - lag))

    return msd


def binder_cumulant(order_param_series):
    """
    Calculate the Binder cumulant U_L = 1 - <phi^4> / (3 * <phi^2>^2).
    Averages are taken over the provided time series.

    Parameters:
    order_param_series (array-like): Time series of the order parameter.

    Returns:
    float: The Binder cumulant. Returns np.nan if <phi^2> is zero.
    """
    if len(order_param_series) == 0:
        return np.nan

    # Ensure it's a numpy array for vectorized operations
    order_param_series = np.asarray(order_param_series)

    phi_squared = order_param_series**2
    phi_fourth = order_param_series**4

    mean_phi_squared = np.mean(phi_squared)
    mean_phi_fourth = np.mean(phi_fourth)

    if mean_phi_squared == 0:
        # If <phi^2> is 0, then phi must be 0 always, so <phi^4> is also 0.
        # The cumulant is conventionally 0 or 2/3 in some contexts for a Gaussian distribution around 0.
        # Returning NaN to indicate this specific condition for plotting.
        return np.nan

    denominator = 3.0 * (mean_phi_squared**2)
    if denominator == 0:
        # This case should ideally be covered by mean_phi_squared == 0,
        # but it's a safeguard against potential floating point issues
        # if mean_phi_squared is extremely small but not exactly zero.
        return np.nan

    return 1.0 - mean_phi_fourth / denominator


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    from interaction_functions import grid_neighbors, update_vicsek_numba, update_soc_numba

    # Parameters for the simple Vicsek model simulation
    T = 3000  # Number of time steps
    L = 25  # Box size
    rho = 1.0  # Density of particles
    N = int(rho * L ** 2)  # Number of particles
    r = 1.0  # Interaction radius
    noise = 0.5  # Noise level
    speed = 0.5  # Speed of particles
    dt = 1.0  # Time step
    epsilon = .3
    gamma = -.5

    cell_neighbors = grid_neighbors(L, r)

    # Create mock data for simple Vicsek model
    # Initialize array to store particle data: (time, [x,y,theta], particle)
    data = np.zeros((T, 3, N))

    # Initialize random positions and orientations
    data[0, 0, :] = np.random.uniform(0, L, N)  # x positions
    data[0, 1, :] = np.random.uniform(0, L, N)  # y positions
    data[0, 2, :] = np.random.uniform(0, 2 * np.pi, N)  # orientations

    # Simple Vicsek model simulation
    for t in range(1, T):
        # Update positions and orientations
        data[t, 0:2, :], data[t, 2, :] = update_soc_numba(data[t - 1], dt, r, noise, L, speed, epsilon, gamma,
                                                          cell_neighbors)

    # Create plots to visualize all the observables
    plt.figure(figsize=(16, 9))

    # 1. Plot order parameter over time
    plt.subplot(2, 3, 1)
    order_params = op(data[:, 2, :], axis=1)
    plt.plot(order_params)
    plt.title('Order Parameter Over Time')
    plt.xlabel('Time step')
    plt.ylabel('Order Parameter')

    # 2. Plot velocity correlation
    plt.subplot(2, 3, 2)
    rs = np.linspace(0, L / 2, 20)  # distance bins
    corr = vel_corr(data, rs, L, fluctuation_correlation=True)
    plt.plot(rs, corr)
    plt.title('Velocity Correlation')
    plt.xlabel('Distance r')
    plt.ylabel('Correlation')

    # 3. Plot autocorrelation
    plt.subplot(2, 3, 3)
    auto_corr = autocorrelation(order_params, max_lag=T // 2)
    plt.plot(range(len(auto_corr)), auto_corr)
    plt.title('Order Parameter Autocorrelation')
    plt.xlabel('Time lag')
    plt.ylabel('Autocorrelation')

    # estimate correlation time from exponential fit of autocorrelation
    f = lambda x, b: np.exp(-b * x)
    fitparams, covmatrix = curve_fit(f, range(len(auto_corr)), auto_corr, p0=(0.01,))
    tc = int(10 / fitparams[0])  # equilibration time
    plt.plot(range(len(auto_corr)), f(range(len(auto_corr)), *fitparams), 'r--', label='Fit')

    # 4. Plot return time distribution
    plt.subplot(2, 3, 4)
    threshold = np.mean(order_params[tc:]) - np.std(order_params[tc:], ddof=1)  # mean - std
    ret_times = return_time(order_params, T, threshold)
    if len(ret_times) > 0:
        ax, _ = plot_return_time_distribution(ret_times, plt.gca())
        plt.title('Return Time Distribution')
        plt.xlabel('Return Time')
        plt.ylabel('Probability Density')
    else:
        plt.text(0.5, 0.5, 'No return times found',
                 horizontalalignment='center', verticalalignment='center')
        plt.title('Return Time Distribution')

    # 5. Plot local order parameter distribution
    plt.subplot(2, 3, 5)
    local_op = local_order_over_time(data, L, r, cell_neighbors)
    plt.hist(local_op.flatten(), bins=30, density=True)
    plt.title('Local Order Parameter Distribution')
    plt.xlabel('Local Order Parameter')
    plt.ylabel('Frequency')

    # 6. Plot mean squared displacement
    plt.subplot(2, 3, 6)
    # Calculate MSD up to 1/4 of the total time steps
    max_lag = T // 2
    # First unwrap the trajectories
    unwrapped = unwrap_trajectory(data[:, :2, :], L)
    msd_values = mean_squared_displacement(unwrapped, max_lag=max_lag)
    plt.plot(range(len(msd_values)), msd_values)
    plt.title('Mean Squared Displacement')
    plt.xlabel('Time lag')
    plt.ylabel('MSD')
    plt.loglog()

    plt.tight_layout()
    plt.show()

    # Bonus: Visualize particle trajectories
    plt.figure(figsize=(8, 8))
    unwrapped = unwrap_trajectory(data[:, :2, :], L)
    for i in range(min(10, N)):  # Plot first 10 particles
        plt.plot(unwrapped[:, 0, i], unwrapped[:, 1, i], alpha=0.7)
    plt.title('Particle Trajectories (Unwrapped)')
    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.grid(True)
    plt.show()

    # Plot velocity autocorrelation
    plt.figure(figsize=(8, 6))
    velocity_autocorr = velocity_autocorrelation(data, max_lag=T // 2)
    plt.plot(range(len(velocity_autocorr)), velocity_autocorr)
    plt.title('Velocity Autocorrelation')
    plt.xlabel('Time lag')
    plt.ylabel('Autocorrelation')
    plt.grid(True)
    plt.show()

    # Plot ordered time distribution
    plt.figure(figsize=(8, 6))
    ordered_times = ordered_time(order_params, T, threshold)
    if len(ordered_times) > 0:
        ax, _ = plot_return_time_distribution(ordered_times, ax=plt.gca(), label="Ordered Time")
        plt.title('Ordered Time Distribution')
        plt.xlabel('Ordered Time')
        plt.ylabel('Probability Density')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No ordered times found',
                 horizontalalignment='center', verticalalignment='center')
        plt.title('Ordered Time Distribution')
    plt.show()


