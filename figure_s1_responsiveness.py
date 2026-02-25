"""
Supplementary Figure S1: responsiveness of SOC and Vicsek models.
Runs a 3x3 grid with Vicsek in top-left and 8 SOC parameter combinations.
Uses dual y-axis plots to show both order parameter and orientation.
Styled to match PRL manuscript figures.
Saves figure to figures/figure_s1_responsiveness.pdf.
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import string
from itertools import cycle
from parameterscan_responsiveness import (
    initialize_memmap_observables,
    initialize_ordered_flock_with_defector,
    save_parameters
)
import interaction_functions as ia
import analysis_functions as af

# Use PRL style
plt.style.use('prl_style.mplstyle')


def label_axes(fig, labels=None, loc=None, xycoords='axes fraction', **kwargs):
    """
    Walks through axes and labels each with (a), (b), (c), etc.

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
        labels = string.ascii_lowercase

    # re-use labels rather than stop labeling
    labels = cycle(labels)
    if loc is None:
        loc = (-0.1, 1.05)
    for ax, lab in zip(fig.axes, labels):
        ax.annotate(f'({lab})', xy=loc, ha='right', xycoords=xycoords,
                    fontsize=9, **kwargs)


def save_snapshot(state, t, tau, L, run_dir, snapshot_id):
    """
    Save a visualization of the particle configuration.

    Parameters:
    - state: Current state array (3, N)
    - t: Current timestep
    - tau: Perturbation duration
    - L: System size
    - run_dir: Directory to save snapshot
    - snapshot_id: Unique identifier for the snapshot
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    xs, ys, ths = state
    N = len(xs)

    # Plot flock particles (all except last one)
    flock_xs = xs[:-1]
    flock_ys = ys[:-1]
    flock_ths = ths[:-1]

    # Plot defector (last particle)
    defector_x = xs[-1]
    defector_y = ys[-1]
    defector_th = ths[-1]

    # Draw flock particles
    ax.quiver(flock_xs, flock_ys, np.cos(flock_ths), np.sin(flock_ths),
              color='blue', alpha=0.6, scale=20, width=0.003, label='Flock')

    # Draw defector particle
    ax.quiver([defector_x], [defector_y], [np.cos(defector_th)], [np.sin(defector_th)],
              color='red', scale=20, width=0.006, label='Defector')

    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Add title with perturbation status
    if t < tau:
        status = "CONTROLLED"
    else:
        status = "RELEASED"
    ax.set_title(f't = {t} ({status})')

    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Save figure
    os.makedirs(run_dir, exist_ok=True)
    plt.savefig(os.path.join(run_dir, f'snapshot_{snapshot_id:03d}_t{t:04d}.png'), dpi=100, bbox_inches='tight')
    plt.close()


def run_single_simulation_with_snapshots(parameters, mmap, snapshot_dir, snapshot_interval=10):
    """
    Run a single simulation with snapshot saving.

    Parameters:
    - parameters: dict containing simulation parameters
    - mmap: Preallocated memmap array to store observables
    - snapshot_dir: Directory to save particle snapshots
    - snapshot_interval: Save snapshot every N timesteps
    """
    tmax = parameters['tmax']
    N = parameters['N']
    L = parameters['L']
    tau = parameters['tau']
    model = parameters['model']

    # Initialize ordered flock with defector (same as Fig 1)
    current_state, defector_idx = initialize_ordered_flock_with_defector(N, L)

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

    snapshot_counter = 0

    # Main simulation loop
    for t in range(tmax + 1):
        # Compute and store observables for current state
        angles = current_state[2]
        order_param = af.op(angles)
        avg_angle = af.average_angle(angles)

        mmap[t, 0] = t
        mmap[t, 1] = order_param
        mmap[t, 2] = np.cos(avg_angle)
        mmap[t, 3] = avg_angle

        # Save snapshot at specific intervals
        if t % snapshot_interval == 0 or t == tau or t == tau + 1:
            save_snapshot(current_state, t, tau, L, snapshot_dir, snapshot_counter)
            snapshot_counter += 1

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

    mmap.flush()


def test_grid_with_visualization():
    """
    Run 3x3 grid: Vicsek in top-left, 8 SOC parameter combinations in remaining positions.
    Create single grid with dual y-axis plots (order parameter + orientation).
    Styled to match PRL manuscript figures.
    """
    # Fixed parameters (same as Fig 1)
    base_params = {
        'L': 32,
        'N': 200,
        'r': 1.0,
        'dt': 1.0,
        'speed': 0.5,
        'noise': 0.1 * 2 * np.pi / np.sqrt(12),  # Same as Fig 1
        'tmax': 100,
        'tau': 5  # Adjusted so defector stays near/in flock
    }

    # Grid layout: 3x3 with Vicsek at [0,0] and 8 SOC cases
    # Using 8 out of 9 possible (eps, gamma) combinations
    grid_configs = [
        # Position [0,0]: Vicsek
        {'row': 0, 'col': 0, 'model': 'Vicsek', 'epsilon': None, 'gamma': None, 'label': 'Vicsek'},
        # Position [0,1]: SOC eps=0.3, gamma=-0.3
        {'row': 0, 'col': 1, 'model': 'SOC', 'epsilon': 0.3, 'gamma': -0.3, 'label': r'$\epsilon=0.3$, $\gamma=-0.3$'},
        # Position [0,2]: SOC eps=0.3, gamma=-0.5
        {'row': 0, 'col': 2, 'model': 'SOC', 'epsilon': 0.3, 'gamma': -0.5, 'label': r'$\epsilon=0.3$, $\gamma=-0.5$'},
        # Position [1,0]: SOC eps=0.3, gamma=-0.7
        {'row': 1, 'col': 0, 'model': 'SOC', 'epsilon': 0.3, 'gamma': -0.7, 'label': r'$\epsilon=0.3$, $\gamma=-0.7$'},
        # Position [1,1]: SOC eps=0.5, gamma=-0.3
        {'row': 1, 'col': 1, 'model': 'SOC', 'epsilon': 0.5, 'gamma': -0.3, 'label': r'$\epsilon=0.5$, $\gamma=-0.3$'},
        # Position [1,2]: SOC eps=0.5, gamma=-0.5
        {'row': 1, 'col': 2, 'model': 'SOC', 'epsilon': 0.5, 'gamma': -0.5, 'label': r'$\epsilon=0.5$, $\gamma=-0.5$'},
        # Position [2,0]: SOC eps=0.5, gamma=-0.7
        {'row': 2, 'col': 0, 'model': 'SOC', 'epsilon': 0.5, 'gamma': -0.7, 'label': r'$\epsilon=0.5$, $\gamma=-0.7$'},
        # Position [2,1]: SOC eps=0.7, gamma=-0.3
        {'row': 2, 'col': 1, 'model': 'SOC', 'epsilon': 0.7, 'gamma': -0.3, 'label': r'$\epsilon=0.7$, $\gamma=-0.3$'},
        # Position [2,2]: SOC eps=0.7, gamma=-0.5
        {'row': 2, 'col': 2, 'model': 'SOC', 'epsilon': 0.7, 'gamma': -0.5, 'label': r'$\epsilon=0.7$, $\gamma=-0.5$'},
    ]

    base_dir = 'test_responsiveness_runs'
    snapshots_base_dir = os.path.join(base_dir, 'snapshots')

    # Storage for results
    results = {}

    print("="*60)
    print("Running 3x3 grid: 1 Vicsek + 8 SOC parameter combinations")
    print(f"tau = {base_params['tau']}")
    print("="*60)

    for run_id, config in enumerate(grid_configs):
        model = config['model']
        epsilon = config['epsilon']
        gamma = config['gamma']
        label = config['label']

        print(f"\n[{run_id+1}/9] {label}")

        # Set up parameters
        params = base_params.copy()
        params['model'] = model
        if model == 'SOC':
            params['epsilon'] = epsilon
            params['gamma'] = gamma
            filename_suffix = f"eps{epsilon}_gam{gamma}"
        else:
            filename_suffix = "vicsek"

        # File paths
        data_file = os.path.join(base_dir, f'run_{run_id}_{filename_suffix}.dat')
        params_file = os.path.join(base_dir, f'run_{run_id}_{filename_suffix}_params.json')
        snapshot_dir = os.path.join(snapshots_base_dir, f'run_{run_id}_{filename_suffix}')

        # Initialize and run
        mmap = initialize_memmap_observables(data_file, params['tmax'], dtype=np.float32)
        save_parameters(params, params_file)

        print(f"  Running {model} model...")
        run_single_simulation_with_snapshots(params, mmap, snapshot_dir, snapshot_interval=10)

        # Store results
        results[run_id] = {
            'data_file': data_file,
            'params': params,
            'row': config['row'],
            'col': config['col'],
            'label': label,
            'model': model
        }

        del mmap

    print("\n" + "="*60)
    print("Creating PRL-style visualization with dual y-axes...")
    print("="*60)

    # Paul Tol's colorblind-friendly palette (different from red/grey used in Fig 1-3)
    color_phi = '#0077BB'      # Blue for order parameter
    color_theta = '#EE7733'    # Orange for orientation
    color_tau = '#009988'      # Teal for tau line (distinct from blue, orange, black, grey)
    # Special colors for Vicsek case (subplot a) to make it stand out
    color_phi_vicsek = 'black'
    color_theta_vicsek = 'grey'

    # Create single 3x3 grid plot with dual y-axes
    # Use sharex and sharey for common axis labels
    fig, axes = plt.subplots(3, 3, figsize=(7.0, 5.5), sharex=True)

    # Store twin axes for consistent y-limits
    twin_axes = []

    for run_id, result in results.items():
        row = result['row']
        col = result['col']
        label = result['label']
        model = result['model']

        ax1 = axes[row, col]
        data = np.memmap(result['data_file'], dtype=np.float32, mode='r', shape=(101, 4))

        time = data[:, 0]
        phi = data[:, 1]
        cos_theta = data[:, 2]

        # Choose colors based on model (Vicsek uses black/gray)
        if model == 'Vicsek':
            c_phi = color_phi_vicsek
            c_theta = color_theta_vicsek
        else:
            c_phi = color_phi
            c_theta = color_theta

        # Plot order parameter on left y-axis
        line_phi, = ax1.plot(time, phi, color=c_phi, linewidth=1.0,
                             label=r'Order parameter $\phi$')
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_yticks([0, 1])
        ax1.minorticks_off()
        ax1.grid(False)

        # Create second y-axis for cos(theta)
        ax2 = ax1.twinx()
        line_theta, = ax2.plot(time, cos_theta, color=c_theta, linewidth=1.0,
                               label=r'Orientation $\cos(\Theta)$')
        ax2.set_ylim(-1.1, 1.1)
        ax2.set_yticks([-1, 0, 1])
        ax2.minorticks_off()
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.2, linewidth=0.5)
        twin_axes.append(ax2)

        # Add vertical line at tau
        tau_line = ax1.axvline(x=base_params['tau'], color=color_tau, linestyle='--',
                               alpha=0.7, linewidth=1.0, zorder=10)

        # Set title (rename Vicsek for clarity)
        title = 'Vicsek model' if model == 'Vicsek' else label
        ax1.set_title(title, fontsize=8)

        # Only show y-axis labels and ticks on edge subplots
        if col == 0:
            ax1.set_ylabel(r'Order parameter $\phi$')
        else:
            ax1.tick_params(labelleft=False)
        if col == 2:
            ax2.set_ylabel(r'Orientation $\cos(\Theta)$')
        else:
            ax2.tick_params(labelright=False)
            ax2.spines['right'].set_visible(False)

        # Add legends: top-left includes tau line label, top-right shows observables only
        if row == 0 and col == 0:
            lines = [line_phi, line_theta, tau_line]
            labels_leg = [r'$\phi$', r'$\cos(\Theta)$', r'$t = \tau$']
            ax1.legend(lines, labels_leg, loc='lower right', frameon=True)
        elif row == 0 and col == 2:
            lines = [line_phi, line_theta]
            labels_leg = [r'$\phi$', r'$\cos(\Theta)$']
            ax1.legend(lines, labels_leg, loc='best', frameon=True)

        # Highlight the Vicsek subplot with a thick frame
        if model == 'Vicsek':
            for spine in ax1.spines.values():
                spine.set_linewidth(1.5)

    # Add common x-axis label using supxlabel
    fig.supxlabel(r'Time $t$')

    # Add subfigure labels (a), (b), (c), etc.
    # Only label the main axes (left y-axis), not the twin axes
    main_axes = [axes[i, j] for i in range(3) for j in range(3)]
    label_axes(fig, labels=string.ascii_lowercase[:9], loc=(-0.1, 1.05))

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, 'figure_s1_responsiveness.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_file}")

    # Also save as PDF for publication
    output_pdf = os.path.join(out_dir, 'figure_s1_responsiveness.pdf')
    plt.savefig(output_pdf, dpi=600, bbox_inches='tight')
    print(f"[OK] Saved: {output_pdf}")
    plt.close()

    print("\n" + "="*60)
    print("[SUCCESS] All simulations and visualizations completed!")
    print("="*60)
    print(f"\nResults saved to: {base_dir}")
    print(f"  - 1 combined grid plot (PNG + PDF)")
    print(f"  - 9 simulation data files (1 Vicsek + 8 SOC)")
    print(f"  - Particle snapshots in: {snapshots_base_dir}")
    print("="*60)


if __name__ == "__main__":
    test_grid_with_visualization()
