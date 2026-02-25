import numpy as np
import matplotlib.pyplot as plt
import interaction_functions as ia
import analysis_functions as af
import string
from itertools import cycle

# Use the existing PRL style file
plt.style.use('prl_style.mplstyle')

# Additional parameters not covered by the style file
# plt.rcParams.update({
#     'text.usetex': True,
#     'text.latex.preamble': r'\usepackage{amsmath}'
# })


def label_axes(fig, labels=None, loc=None, xycoords='axes fraction', **kwargs):
    """
    Walks through axes and labels each.

    kwargs are collected and passed to `annotate`

    Parameters
    ----------
    fig : Figure
         Figure object to work on

    labels : iterable or None
        iterable of strings to use to label the axes.
        If None, lower case letters are used.

    loc : len=2 tuple of floats
        Where to put the label in axes-fraction units
    """
    if labels is None:
        labels = string.ascii_uppercase

    # re-use labels rather than stop labeling
    labels = cycle(labels)
    if loc is None:
        loc = (-0.1, 1.1)
    for ax, lab in zip(fig.axes, labels):
        ax.annotate(lab, xy=loc, ha='right', xycoords=xycoords, **kwargs)

def run_simulation(model_type, params, time_steps):
    """Run a simulation using either SOC or Vicsek model.
    
    Args:
        model_type (str): Either 'SOC' or 'Vicsek'
        params (dict): Dictionary of parameters
        time_steps (int): Number of time steps to simulate
        
    Returns:
        tuple: (positions, angles, order_parameters, average_angles)
    """
    # Extract parameters
    L = params['L']
    N = params['N']
    r = params['r']
    noise = params['noise']
    dt = params['dt']
    speed = params['speed']
    
    # Initialize arrays to store results
    arr = np.empty((time_steps, 3, N))
    order_params = np.zeros(time_steps)
    avg_angles = np.zeros(time_steps)
    
    # Initialize random positions and orientations
    xs = np.random.rand(N) * L
    ys = np.random.rand(N) * L
    ths = np.random.uniform(0, 2 * np.pi, size=N)
    
    # Store initial state
    arr[0] = xs, ys, ths
    order_params[0] = af.op(ths)
    avg_angles[0] = af.average_angle(ths)
    
    # Precompute grid neighbors
    cell_neighbors = ia.grid_neighbors(L, r)
    
    # Main simulation loop
    for t in range(1, time_steps):
        if model_type == 'SOC':
            pos, ths = ia.update_soc_numba(
                arr[t-1], dt, r, noise, L, speed, 
                params['epsilon'], params['gamma'], cell_neighbors
            )
        else:  # Vicsek model
            pos, ths = ia.update_vicsek_numba(
                arr[t-1], dt, r, noise, L, speed, cell_neighbors
            )
        
        arr[t] = pos[0], pos[1], ths
        order_params[t] = af.op(ths)
        avg_angles[t] = af.average_angle(ths)
    
    return arr, order_params, avg_angles

def main():
    # Define parameters for both models
    base_params = {
        'L': 32,           # System size
        'density': 1.0,    # Particle density
        'r': 1.0,          # Interaction radius
        'dt': 1.0,         # Time step
        'speed': 0.5,      # Particle speed
        'noise': 0.1       # Noise strength (scaled appropriately below)
    }
    
    # Calculate number of particles
    base_params['N'] = int(base_params['density'] * base_params['L']**2)
    
    # Scale noise properly for Vicsek model
    # Factor 2π/√12 to match uniform distribution variance
    base_params['noise'] *= 2 * np.pi / np.sqrt(12)
    
    # Additional parameters for SOC model
    soc_params = base_params.copy()
    soc_params.update({
        'gamma': -0.2,
        'epsilon': 0.6
    })
    
    # Number of time steps to simulate
    time_steps = 10000
    # color for the SOC model is red from tab10 palette
    soc_color = plt.get_cmap('tab10')(3)
    
    print(f"Starting simulation with {base_params['N']} particles for {time_steps} time steps")
    
    # Run simulations
    print("Running SOC model simulation...")
    soc_arr, soc_order_params, soc_avg_angles = run_simulation('SOC', soc_params, time_steps)
    
    print("Running Vicsek model simulation...")
    vicsek_arr, vicsek_order_params, vicsek_avg_angles = run_simulation('Vicsek', base_params, time_steps)
    
    # Use the figure width from prl_style.mplstyle
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(3.375, 1.7))
    
    # Time points for x-axis
    time = np.arange(time_steps)
    equilibrium_time = 2000  # Time after which system is considered in equilibrium
    
    # Panel a: Order parameters
    ax1.plot(time, soc_order_params, label=r'Minority interaction', color=soc_color)
    ax1.plot(time, vicsek_order_params, label=r'Vicsek Model', color='grey')
    
    # Threshold for avalanche detection - only show after equilibrium time
    threshold = np.mean(vicsek_order_params[equilibrium_time:]) - 3 * np.std(vicsek_order_params[equilibrium_time:])

    ax1.plot(np.arange(equilibrium_time, time_steps), 
             np.ones(time_steps-equilibrium_time) * threshold, 
             linestyle='--', label=r'$\phi_c$', color='k')
    
    # Add '(a)' label to upper left corner
    axis_label_position = (0.05, 0.9)
    ax1.text(*axis_label_position, '(a)', transform=ax1.transAxes)
    
    ax1.set_ylabel(r'Order parameter $\phi$')

    # Panel b: Cosine of average angle
    ax2.plot(time, np.cos(soc_avg_angles), label=r'Minority interaction', color=soc_color)
    ax2.plot(time, np.cos(vicsek_avg_angles), label=r'Vicsek Model', color='grey')
    
    # Add '(b)' label to upper left corner
    ax2.text(*axis_label_position, '(b)', transform=ax2.transAxes)
    
    ax2.set_ylabel(r'Orientation $\cos(\Theta)$')
    ax2.set_xlabel(r'Time $t$')
    ax1.set_xlabel(r'Time $t$')

    # Set y-limits for better visualization
    ax1.set_ylim(0, 1.1)
    ax2.set_ylim(-1, 1.2)
    ax2.set_yticks([-1, 0, 1])
    ax1.set_yticks([0, 1])
    ax1.legend(loc='best', frameon=True)

    
    # Format x-axis with fewer ticks
    ax2.set_xlim(0, time_steps)
    # plt.xticks(np.arange(0, time_steps+1, 2000))
    # set x tick labels
    ax2.set_xticks((0, 5000))

    
    # Add tight layout
    plt.tight_layout()
    
    # Save figure as SVG, EPS and PDF. make sure that text is not converted to paths
    plt.savefig('figure2_orderparam.svg', format='svg', bbox_inches='tight')
    # plt.savefig('figure2_orderparam2.eps', format='eps', bbox_inches='tight')
    # plt.savefig('figure2_orderparam2.pdf', format='pdf', bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    main()
