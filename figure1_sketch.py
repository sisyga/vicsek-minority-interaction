import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# --- Try to use the provided PRL style ---
STYLE_FILE = "prl_style.mplstyle"
try:
    if os.path.exists(STYLE_FILE):
        plt.style.use(STYLE_FILE)
        print(f"Using stylesheet: {STYLE_FILE}")
    else:
        print(f"'{STYLE_FILE}' not found. Using default Matplotlib styles.")
except Exception as e:
    print(f"Error applying style '{STYLE_FILE}': {e}. Using default Matplotlib styles.")


# Helper function to get vector from angle
def angle_to_vec(angle_deg):
    rad = np.deg2rad(angle_deg)
    return np.array([np.cos(rad), np.sin(rad)])

# Helper function to get angle from vector
def vec_to_angle_deg(vec):
    return np.rad2deg(np.arctan2(vec[1], vec[0]))

# Helper function to calculate average direction vector and angle
def average_direction_vector(vectors):
    if not vectors:
        return np.array([0,0])
    mean_vec = np.mean(vectors, axis=0)
    return mean_vec # Return the mean vector (not necessarily normalized)

# --- Figure Setup ---
fig, ax = plt.subplots(1, 1)

# Common parameters
interaction_radius = 1.0
focal_pos = np.array([0, 0]) # Focal particle at origin
arrow_scale = 0.5
noise_level_deg = 90 # Increased noise for neighbors

# Arrow style parameters
common_arrow_width = 0.008
common_headwidth = 5
common_headlength = 7
thicker_arrow_width = 0.01

# --- Particle Setup for Minority Interaction ---
# Neighbors (gray)
num_neighbors = 3
neighbor_base_angle = 30 # Original base angle for neighbors
neighbor_angles = np.random.normal(neighbor_base_angle, noise_level_deg, num_neighbors)
neighbor_vectors = [angle_to_vec(ang) for ang in neighbor_angles]

# Fixed, visually pleasing positions for neighbors to the left of the focal particle
neighbor_positions = []
if num_neighbors >= 1:
    neighbor_positions.append(focal_pos + np.array([-0.7 * interaction_radius, 0.3 * interaction_radius]))
if num_neighbors >= 2:
    neighbor_positions.append(focal_pos + np.array([-0.5 * interaction_radius, 0.6 * interaction_radius]))
if num_neighbors >= 3:
    neighbor_positions.append(focal_pos + np.array([-0.5 * interaction_radius, -0.6 * interaction_radius]))
if num_neighbors >= 4:
    neighbor_positions.append(focal_pos + np.array([-0.7 * interaction_radius, -0.3 * interaction_radius]))
neighbor_positions = neighbor_positions[:num_neighbors]

# Defector (red) - approaching from the right
defector_angle = 180 # Pointing left
defector_vec = angle_to_vec(defector_angle)
# Position defector to the right of focal_pos, within interaction_radius
defector_pos = focal_pos + np.array([interaction_radius * 0.8, np.random.uniform(-0.1, 0.1) * interaction_radius])

# Focal particle's initial orientation (must be reasonably aligned for condition 1)
focal_initial_angle = neighbor_base_angle + 8 # Slightly off from base, but still aligned
focal_initial_vec = angle_to_vec(focal_initial_angle)

# Calculate local average direction *including the defector and the focal particle itself*
# This average is the reference for epsilon and gamma conditions, and also the target for standard Vicsek
all_particle_vectors_for_avg = neighbor_vectors + [defector_vec] + [focal_initial_vec]
avg_local_vec_raw = average_direction_vector(all_particle_vectors_for_avg)
avg_local_angle = vec_to_angle_deg(avg_local_vec_raw)
# For plotting the average direction arrow, we use its normalized version if its magnitude is non-zero
avg_local_vec_normalized = avg_local_vec_raw / np.linalg.norm(avg_local_vec_raw) if np.linalg.norm(avg_local_vec_raw) > 1e-6 else np.array([1,0])


# Condition 1: Local alignment (focal particle vs local average)
# cos(alpha) > epsilon
dot_prod_alpha = np.dot(focal_initial_vec, avg_local_vec_normalized)
angle_alpha_rad = np.arccos(np.clip(dot_prod_alpha, -1.0, 1.0))
angle_alpha_deg = np.rad2deg(angle_alpha_rad)
# epsilon_thresh = 0.8 # Example

# Condition 2: Defector deviation (defector vs local average)
# cos(beta) < gamma_thresh
dot_prod_beta = np.dot(defector_vec, avg_local_vec_normalized)
angle_beta_rad = np.arccos(np.clip(dot_prod_beta, -1.0, 1.0))
angle_beta_deg = np.rad2deg(angle_beta_rad)
# gamma_thresh = -0.3 # Example

# Focal particle's new orientation (minority rule: follows defector)
focal_new_minority_vec = defector_vec # (Ignoring noise for clarity of diagram)

# Focal particle's new orientation (hypothetical standard Vicsek: aligns with local average)
focal_new_vicsek_vec = avg_local_vec_normalized


# --- Plotting ---
# Interaction radius circle
circle = patches.Circle(focal_pos, interaction_radius, edgecolor='dimgray', facecolor='none', linestyle='--', linewidth=1)
ax.add_patch(circle)

# Neighbors
for i, (pos, vec) in enumerate(zip(neighbor_positions, neighbor_vectors)):
    ax.quiver(pos[0], pos[1], vec[0], vec[1], angles='xy', scale_units='xy', scale=1/arrow_scale,
              color='gray', width=common_arrow_width, headwidth=common_headwidth, headlength=common_headlength,
              label='_nolegend_' if i > 0 else 'Neighbor')

# Defector
ax.quiver(defector_pos[0], defector_pos[1], defector_vec[0], defector_vec[1],
          angles='xy', scale_units='xy', scale=1/arrow_scale,
          color='red', width=common_arrow_width, headwidth=common_headwidth, headlength=common_headlength, label='Defector')

# Focal particle - initial
ax.quiver(focal_pos[0], focal_pos[1], focal_initial_vec[0], focal_initial_vec[1],
          angles='xy', scale_units='xy', scale=1/arrow_scale,
          color='k', width=common_arrow_width, headwidth=common_headwidth, headlength=common_headlength, label='Focal (initial)')

# Add label "i" to the focal particle
ax.text(focal_pos[0] + 0.05, focal_pos[1] + 0.05, r'$i$', color='black', ha='left', va='bottom')

# Average local orientation (target for standard Vicsek, reference for conditions)
# This arrow now also represents the hypothetical standard Vicsek outcome.
ax.quiver(focal_pos[0], focal_pos[1], focal_new_vicsek_vec[0], focal_new_vicsek_vec[1],
          angles='xy', scale_units='xy', scale=1/arrow_scale,
          color='green', width=thicker_arrow_width, headwidth=common_headwidth, headlength=common_headlength, label='Focal (standard Vicsek outcome)')
ax.text(focal_new_vicsek_vec[0]*arrow_scale*0.7, focal_new_vicsek_vec[1]*arrow_scale*0.7 - 0.2,
         r'$\langle \mathbf{v} \rangle_i$', color='green', ha='center', va='top')

# Focal particle - new (minority rule)
ax.quiver(focal_pos[0], focal_pos[1], focal_new_minority_vec[0], focal_new_minority_vec[1],
          angles='xy', scale_units='xy', scale=1/arrow_scale,
          color='red', width=thicker_arrow_width, headwidth=common_headwidth, headlength=common_headlength, label='Focal (minority rule outcome)')

# Arcs for conditions (alpha and beta)
# Arc for alpha (focal_initial vs avg_local)
arc_radius_alpha = 0.35 * arrow_scale
ang1_alpha_plot = min(focal_initial_angle, avg_local_angle)
ang2_alpha_plot = max(focal_initial_angle, avg_local_angle)
if ang2_alpha_plot - ang1_alpha_plot > 180: # Correct for wraparound
    arc_alpha = patches.Arc(focal_pos, 2*arc_radius_alpha, 2*arc_radius_alpha, angle=0,
                            theta1=ang2_alpha_plot, theta2=ang1_alpha_plot + 360, color='green')
else:
    arc_alpha = patches.Arc(focal_pos, 2*arc_radius_alpha, 2*arc_radius_alpha, angle=0,
                            theta1=ang1_alpha_plot, theta2=ang2_alpha_plot, color='green')
ax.add_patch(arc_alpha)
# Alpha label for arc
mid_angle_alpha_plot_rad = np.deg2rad((ang1_alpha_plot + ang2_alpha_plot) / 2)
if ang2_alpha_plot - ang1_alpha_plot > 180: mid_angle_alpha_plot_rad += np.pi
ax.text(focal_pos[0] + (arc_radius_alpha + 0.05) * np.cos(mid_angle_alpha_plot_rad),
        focal_pos[1] + (arc_radius_alpha + 0.05) * np.sin(mid_angle_alpha_plot_rad),
        r'$\alpha$', color='green', ha='center', va='center')

# Arc for beta (defector vs avg_local)
arc_radius_beta = 0.55 * arrow_scale
ang1_beta_plot = min(defector_angle, avg_local_angle)
ang2_beta_plot = max(defector_angle, avg_local_angle)
if ang2_beta_plot - ang1_beta_plot > 180: # Correct for wraparound
    arc_beta = patches.Arc(focal_pos, 2*arc_radius_beta, 2*arc_radius_beta, angle=0,
                           theta1=ang2_beta_plot, theta2=ang1_beta_plot + 360, color='red')
else:
    arc_beta = patches.Arc(focal_pos, 2*arc_radius_beta, 2*arc_radius_beta, angle=0,
                           theta1=ang1_beta_plot, theta2=ang2_beta_plot, color='red')
ax.add_patch(arc_beta)
# Beta label for arc
mid_angle_beta_plot_rad = np.deg2rad((ang1_beta_plot + ang2_beta_plot) / 2)
if ang2_beta_plot - ang1_beta_plot > 180: mid_angle_beta_plot_rad += np.pi
ax.text(focal_pos[0] + (arc_radius_beta + 0.05) * np.cos(mid_angle_beta_plot_rad),
        focal_pos[1] + (arc_radius_beta + 0.05) * np.sin(mid_angle_beta_plot_rad),
        r'$\beta$', color='red', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, pad=0.1, edgecolor='none'))


# Text for conditions - positioned for better readability
text_box_props = dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8, lw=0.5)

# Condition 1 Text
ax.annotate(r'Condition 1 (Alignment):' + '\n' + r'$\cos(\alpha) > \epsilon$',
            xy=(focal_pos[0] - interaction_radius * 0.8, focal_pos[1] + interaction_radius * 0.7),
            xytext=(focal_pos[0] - interaction_radius * 1.5, focal_pos[1] + interaction_radius * 1.1), # Adjusted for less clutter
            textcoords='data',
            color='green', ha='left', va='top',
            arrowprops=None, # Removed arrow
            bbox=text_box_props)

# Condition 2 Text
ax.annotate(r'Condition 2 (Deviation):' + '\n' + r'$\cos(\beta) < \gamma$',
            xy=(focal_pos[0] + interaction_radius * 0.8, focal_pos[1] - interaction_radius * 0.7),
            xytext=(focal_pos[0] + interaction_radius * 0.5, focal_pos[1] - interaction_radius * 1.4), # Adjusted for less clutter
            textcoords='data',
            color='red', ha='left', va='top', 
            arrowprops=None, # Removed arrow
            bbox=text_box_props)


# --- General Figure Settings ---
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-interaction_radius - 1.0, interaction_radius + 1.0)
ax.set_ylim(-interaction_radius - 1.0, interaction_radius + 1.0)

# Turn off axes and labels
ax.axis('off')

plt.tight_layout() # Adjusted rect for suptitle and legend removed
plt.savefig("figure1_minority_interaction.svg", dpi=300, bbox_inches='tight')
plt.show()

print(f"Angle alpha (local alignment, focal vs avg): {angle_alpha_deg:.1f} deg, cos(alpha) = {np.cos(np.deg2rad(angle_alpha_deg)):.2f}")
print(f"Angle beta (defector deviation, defector vs avg): {angle_beta_deg:.1f} deg, cos(beta) = {np.cos(np.deg2rad(angle_beta_deg)):.2f}")
print(f"Local average vector <v>_i angle: {avg_local_angle:.1f} deg")
print(f"Focal initial angle: {focal_initial_angle:.1f} deg")
print(f"Defector angle: {defector_angle:.1f} deg")

