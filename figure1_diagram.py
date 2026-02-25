import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.patheffects as path_effects

# Set the style sheet
plt.style.use('default')

def create_direction_arrows(positions, directions, colors, sizes=None):
    """Helper function to create direction arrows only"""
    if sizes is None:
        sizes = [1.0] * len(positions)
    
    arrows = []
    
    for pos, dir, color, size in zip(positions, directions, colors, sizes):
        # Create direction arrow
        arrow_length = 0.8*size
        end_point = (pos[0] + arrow_length * np.cos(dir), 
                     pos[1] + arrow_length * np.sin(dir))
        arrow = FancyArrowPatch(pos, end_point, color=color, arrowstyle='->', 
                               linewidth=2, mutation_scale=15, zorder=11)
        
        arrows.append(arrow)
        
    return arrows

# Create figure - single column width
fig, ax = plt.subplots(figsize=(5, 5))

# Common parameters
focal_color = 'black'  # Changed from red to black
neighbor_color = 'dimgray'  # Changed from skyblue to dark grey
defector_color = 'red'  # Changed from green to red

# Set plot limits
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')

# Reposition focal particle to be closest to defector
defector_pos = (-1.5, -1.5)  # Moved to a more opposite position from the majority
focal_pos = (-0.9, 0.0)  # Moved closer to defector
focal_dir = np.pi/4  # 45 degrees

# Create neighbors
neighbors_pos = [(1, 1), (-0.8, 1.2), (0.7, -1), (-1, -0.8)]
neighbors_dir = [np.pi/4, np.pi/4+0.2, np.pi/4-0.1, np.pi/4+0.15]
defector_dir = 5*np.pi/4  # Directly opposite to the majority direction (which is ~pi/4)

# Draw interaction radius
interaction_circle = Circle(focal_pos, radius=2, fill=False, linestyle='--', 
                           color='gray', alpha=0.6, zorder=1)
ax.add_patch(interaction_circle)

# Add radius line and label
radius_end = (focal_pos[0] + 2, focal_pos[1])
radius_line = plt.Line2D([focal_pos[0], radius_end[0]], [focal_pos[1], radius_end[1]], 
                         linestyle='--', color='gray', alpha=0.6)
ax.add_line(radius_line)
ax.text(focal_pos[0] + 1.0, focal_pos[1] + 0.1, r"$r$", fontsize=10)

# Create and add arrows only
focal_arrows = create_direction_arrows([focal_pos], [focal_dir], [focal_color], [1.2])
neighbor_arrows = create_direction_arrows(neighbors_pos, neighbors_dir, 
                                       [neighbor_color]*len(neighbors_pos))
defector_arrows = create_direction_arrows([defector_pos], [defector_dir], 
                                       [defector_color], [1.2])

for a in focal_arrows + neighbor_arrows + defector_arrows:
    ax.add_patch(a)

# Add new direction arrow (dashed) for focal particle showing the resulting change
new_dir = 5*np.pi/4  # Updated to match defector's direction
new_focal_arrow = FancyArrowPatch(focal_pos, 
                                (focal_pos[0] + np.cos(new_dir), 
                                 focal_pos[1] + np.sin(new_dir)), 
                                color=defector_color, arrowstyle='->', 
                                linestyle='--', linewidth=2, mutation_scale=15)
ax.add_patch(new_focal_arrow)

# Remove axes
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
# save figure with editable text in SVG format. text should be editable in Inkscape
plt.savefig('sketch.svg', format='svg', dpi=300, bbox_inches='tight')
plt.show()
