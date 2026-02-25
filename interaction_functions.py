from math import ceil
from math import sin, cos, atan2

import numpy as np
from numba import jit, prange

PI2 = 2 * np.pi


def grid_neighbors(L, r):
    n_cells = int(np.ceil(L / r))
    cell_neighbors = np.empty((n_cells * n_cells, 9), dtype=np.int64)
    for cell_y in range(n_cells):
        for cell_x in range(n_cells):
            cell_index = cell_y * n_cells + cell_x
            # Fill in the 9 neighbors
            count = 0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    adj_x = (cell_x + dx) % n_cells
                    adj_y = (cell_y + dy) % n_cells
                    adj_idx = adj_y * n_cells + adj_x
                    cell_neighbors[cell_index, count] = adj_idx
                    count += 1
    return cell_neighbors


@jit(nopython=True, nogil=True, fastmath=True)
def build_cell_list(positions, L, r=1.0):
    # positions has shape (N, 2)
    N = positions.shape[0]
    cell_size = r
    n_cells = int(ceil(L / cell_size))
    total_cells = n_cells * n_cells

    cell_indices = np.empty(N, np.uint32)
    cell_particle_counts = np.zeros(total_cells, np.uint32)

    for i in prange(N):
        cell_x = int(positions[i, 0] // cell_size)
        cell_y = int(positions[i, 1] // cell_size)
        cell_indices[i] = cell_y * n_cells + cell_x
        cell_particle_counts[cell_indices[i]] += 1

    # Choose a capacity that fits the densest cell.
    capacity = np.max(cell_particle_counts)

    # Phase 2: fill the cell list.
    cell_list = np.empty((total_cells, capacity), np.uint32)
    cur_counts = np.zeros(total_cells, np.uint32)
    for i in range(N):
        idx = cell_indices[i]
        pos = cur_counts[idx]
        cell_list[idx, pos] = i
        cur_counts[idx] += 1

    return cell_list, cell_particle_counts, cell_indices, n_cells


@jit(nopython=True, nogil=True, fastmath=True)
def find_neighbors(positions, L, cell_list, cell_particle_counts, cell_indices,
                   cell_neighbors, r=1.0, initial_max_neighbors=50):
    # positions has shape (N, 2)
    N = positions.shape[0]
    r2 = r * r
    max_neighbors = initial_max_neighbors
    overflow = True
    # Loop until every particle's neighbor list fits.
    while overflow:
        overflow = False
        neighbors = np.empty((N, max_neighbors), np.uint64)
        neighbor_counts = np.zeros(N, np.uint16)
        for i in prange(N):
            neighbors[i, 0] = i  # count yourself as neighbor
            cnt = 1
            x = positions[i, 0]
            y = positions[i, 1]
            cell_idx = cell_indices[i]
            # cell_neighbors[cell_idx] must be a 1D array of adjacent cell indices.
            num_adj = cell_neighbors[cell_idx].shape[0]
            for k in range(num_adj):
                adj = cell_neighbors[cell_idx][k]
                for j in range(cell_particle_counts[adj]):
                    p_idx = cell_list[adj, j]
                    if p_idx != i:  # don't count yourself twice
                        dx = positions[p_idx, 0] - x
                        dy = positions[p_idx, 1] - y
                        # Periodic boundary adjustment.
                        if dx > L/2:
                            dx -= L
                        elif dx < -L/2:
                            dx += L
                        if dy > L/2:
                            dy -= L
                        elif dy < -L/2:
                            dy += L
                        if dx*dx + dy*dy <= r2:
                            if cnt < max_neighbors:
                                neighbors[i, cnt] = p_idx
                                cnt += 1
                            else:
                                cnt += 1
            neighbor_counts[i] = cnt
        max_neighbor_count = np.max(neighbor_counts)
        if max_neighbor_count >= max_neighbors:
            overflow = True
            max_neighbors = max_neighbor_count + 1
    return neighbors, neighbor_counts


@jit(nopython=True, nogil=True, parallel=True, fastmath=True)
def update_vicsek_numba(state, dt, r, eta, L, speed, cell_neighbors):
    # state[:2] are positions (shape (2, N)) and state[2] are directions (shape (N,))
    positions = state[:2]
    directions = state[2]
    N = directions.shape[0]
    # Transpose to shape (N, 2)
    # positions_t = positions.T
    vx = np.cos(directions)
    vy = np.sin(directions)
    new_positions = np.empty_like(positions)
    new_directions = np.empty_like(directions)

    # Build the grid cell list. Initial allocation is 20 per cell.
    cell_list, cell_particle_counts, cell_indices, n_cells = build_cell_list(positions.T, L, r)
    # Find neighbors. Initial allocation is 50 neighbors.
    neighbors, neighbor_counts = find_neighbors(positions.T, L, cell_list, cell_particle_counts, cell_indices,
                                                cell_neighbors, r=r)
    noise = np.random.normal(0, eta, N).astype(directions.dtype)

    # Update each particle.
    for i in prange(N):
        sin_avg = 0.0
        cos_avg = 0.0
        cnt = neighbor_counts[i]
        for j in range(cnt):
            p_idx = neighbors[i, j]
            sin_avg += vy[p_idx]
            cos_avg += vx[p_idx]
        sin_avg /= cnt
        cos_avg /= cnt

        new_dir = (atan2(sin_avg, cos_avg) + noise[i]) % PI2
        new_directions[i] = new_dir
        new_positions[0, i] = (positions[0, i] + speed * dt * cos(new_dir)) % L
        new_positions[1, i] = (positions[1, i] + speed * dt * sin(new_dir)) % L

    return new_positions, new_directions


###############################################################################
# Optimized update functions for the different models.
###############################################################################

@jit(nopython=True, nogil=True, parallel=True, fastmath=True)
def update_soc_numba(state, dt, r, eta, L, speed, eps, gamma, cell_neighbors,
                         initial_max_particles=20, initial_max_neighbors=50):
    """
    Update positions and directions using the SOC model.
    positions has shape (2, N).
    """
    positions = state[:2]
    directions = state[2]
    N = len(directions)
    new_positions = np.empty_like(positions)
    new_directions = np.empty_like(directions)
    vx = np.cos(directions)
    vy = np.sin(directions)
    # Convert positions to (N,2)
    positions_T = positions.T
    cell_list, cell_particle_counts, cell_indices, n_cells = build_cell_list(positions_T, L, r=r)  # TO CHECK
    neighbors, neighbor_counts = find_neighbors(positions_T, L, cell_list, cell_particle_counts,
                                                 cell_indices, cell_neighbors, r=r)
    noise = np.random.normal(0, eta, N).astype(directions.dtype)

    for i in prange(N):
        x = positions[0, i]
        y = positions[1, i]
        alignment = True
        ct = 1.0
        Dcos = 0.0
        Dsin = 0.0
        nb_count = neighbor_counts[i]
        # Average neighborhood velocity.
        for j in range(nb_count):
            idx = neighbors[i, j]
            Dcos += vx[idx]
            Dsin += vy[idx]
        Dcos /= nb_count
        Dsin /= nb_count
        cc = Dcos * vx[i] + Dsin * vy[i]
        sin_avg = 0.0
        cos_avg = 0.0
        # Minority interaction check.
        if cc > eps:
            for j in range(nb_count):
                idx = neighbors[i, j]
                dotprod = Dcos * vx[idx] + Dsin * vy[idx]
                if dotprod < ct:
                    ct = dotprod
                    if ct < gamma:
                        alignment = False
                        sin_avg = vy[idx]
                        cos_avg = vx[idx]
        if alignment:
            sin_avg = Dsin
            cos_avg = Dcos
        new_directions[i] = (atan2(sin_avg, cos_avg) + noise[i]) % PI2
        new_positions[0, i] = (x + speed * dt * cos(new_directions[i])) % L
        new_positions[1, i] = (y + speed * dt * sin(new_directions[i])) % L
    return new_positions, new_directions


@jit(nopython=True, nogil=True, parallel=True, fastmath=True)
def update_soc_numba_async(state, dt, r, eta, L, speed, eps, gamma, cell_neighbors,
                           initial_max_particles=20, initial_max_neighbors=50):
    """
    Update positions and directions using the asynchronous SOC model.
    positions has shape (2, N).
    """
    positions = state[:2]
    directions = state[2]
    N = len(directions)
    vx = np.cos(directions)
    vy = np.sin(directions)
    new_directions = np.empty_like(directions)
    new_positions = np.empty_like(positions)
    positions_T = positions.T
    cell_list, cell_particle_counts, cell_indices, n_cells = build_cell_list(positions_T, L, r=r)
    neighbors, neighbor_counts = find_neighbors(positions_T, L, cell_list, cell_particle_counts, cell_indices,
                                                cell_neighbors, r=r)
    alignment = np.ones(N, np.bool_)
    noise = np.random.normal(0, eta, N).astype(directions.dtype)

    # First pass: decide new directions.
    for i in prange(N):
        ct = 1.0
        Dcos = 0.0
        Dsin = 0.0
        nb_count = neighbor_counts[i]
        for j in range(nb_count):
            idx = neighbors[i, j]
            Dcos += vx[idx]
            Dsin += vy[idx]
        Dcos /= nb_count
        Dsin /= nb_count
        cc = Dcos * vx[i] + Dsin * vy[i]
        sin_avg = 0.0
        cos_avg = 0.0
        if cc > eps:
            for j in range(nb_count):
                idx = neighbors[i, j]
                dotprod = Dcos * vx[idx] + Dsin * vy[idx]
                if dotprod < ct:
                    ct = dotprod
                    if ct < gamma:
                        alignment[i] = False
                        sin_avg = vy[idx]
                        cos_avg = vx[idx]
        if not alignment[i]:
            new_directions[i] = (atan2(sin_avg, cos_avg) + noise[i]) % PI2

    # Second pass: update velocities for nonaligned particles.
    for i in prange(N):
        if not alignment[i]:
            vx[i] = cos(new_directions[i])
            vy[i] = sin(new_directions[i])

    # Third pass: update positions (and compute new directions for aligned ones).
    for i in prange(N):
        x = positions[0, i]
        y = positions[1, i]
        if alignment[i]:
            sin_avg = 0.0
            cos_avg = 0.0
            nb_count = neighbor_counts[i]
            for j in range(nb_count):
                idx = neighbors[i, j]
                sin_avg += vy[idx]
                cos_avg += vx[idx]
            sin_avg /= nb_count
            cos_avg /= nb_count
            new_directions[i] = (atan2(sin_avg, cos_avg) + noise[i]) % PI2
        new_positions[0, i] = (x + speed * dt * cos(new_directions[i])) % L
        new_positions[1, i] = (y + speed * dt * sin(new_directions[i])) % L

    return new_positions, new_directions

@jit(nopython=True, nogil=True, parallel=True, fastmath=True)
def update_vicsek_selftuning_numba(state, dt, r, b, eta, L, speed, cell_neighbors,
                                   initial_max_particles=20, initial_max_neighbors=50):
    """
    Update positions and directions using the self-tuning Vicsek model.
    positions has shape (2, N).
    """
    positions = state[:2]
    directions = state[2]
    N = len(directions)
    vx = np.cos(directions)
    vy = np.sin(directions)
    new_positions = np.empty_like(positions)
    new_directions = np.empty_like(directions)
    positions_T = positions.T
    cell_list, cell_particle_counts, cell_indices, n_cells = build_cell_list(positions_T, L, r=r)
    neighbors, neighbor_counts = find_neighbors(positions_T, L, cell_list, cell_particle_counts, cell_indices,
                                                cell_neighbors, r=r)
    noise = np.random.normal(0, eta, N).astype(directions.dtype)

    for i in prange(N):
        cos_avg = 0.0
        sin_avg = 0.0
        x = positions[0, i]
        y = positions[1, i]
        nb_count = neighbor_counts[i]
        # Sum the neighbor velocities.
        for j in range(nb_count):
            idx = neighbors[i, j]
            cos_avg += vx[idx]
            sin_avg += vy[idx]
        cos_avg /= nb_count
        sin_avg /= nb_count
        local_order = cos_avg * cos_avg + sin_avg * sin_avg
        local_direction = atan2(sin_avg, cos_avg)
        new_directions[i] = (local_direction + b * local_order * noise[i]) % PI2
        new_positions[0, i] = (x + speed * dt * cos(new_directions[i])) % L
        new_positions[1, i] = (y + speed * dt * sin(new_directions[i])) % L

    return new_positions, new_directions