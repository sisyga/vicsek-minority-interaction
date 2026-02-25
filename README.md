# Minority-Triggered Reorientation in the Vicsek Model

Code accompanying the paper:

> Simon Syga, Chandraniva Guha Ray, Josué Manik Nava Sedeño, Fernando Peruani, Andreas Deutsch.
> *Minority-Triggered Reorientation Yields Macroscopic Cascades and Enhanced Responsiveness in a Vicsek Swarm.*
> [Preprint / DOI — to be added upon publication]

## Overview

This repository has all the simulation and analysis code for reproducing the figures in the paper. We study a variant of the Vicsek model where particles can align with a dissenting minority neighbor (one whose orientation diverges sufficiently from the local majority) rather than following the crowd. This minority-interaction rule drives the swarm to a self-organized critical state: rare fluctuations trigger system-spanning avalanches in collective order, and the swarm becomes noticeably more sensitive to external orientation signals.

The model is implemented in Python with Numba JIT compilation for speed. Large-scale parameter scans use memory-mapped arrays to keep memory usage manageable.

## Authors

Simon Syga¹, Chandraniva Guha Ray²³⁴⁵, Josué Manik Nava Sedeño⁶, Fernando Peruani⁷, Andreas Deutsch¹

¹ Information Services and High Performance Computing, Center for Interdisciplinary Digital Sciences, TUD Dresden University of Technology, Dresden, Germany
² Max Planck Institute for the Physics of Complex Systems, Dresden, Germany
³ Max Planck Institute of Molecular Cell Biology and Genetics, Dresden, Germany
⁴ Center for Systems Biology Dresden, Dresden, Germany
⁵ Department of Physical Sciences, Indian Institute of Science Education and Research Kolkata, Mohanpur, India
⁶ Departamento de Matemáticas, Facultad de Ciencias, Universidad Nacional Autónoma de México, Ciudad de México, México
⁷ Laboratoire de Physique Théorique et Modélisation, UMR 8089, CY Cergy Paris Université, Cergy-Pontoise, France

## Repository structure

```text
.
├── interaction_functions.py        # Numba-accelerated update rules (Vicsek + minority variant)
├── analysis_functions.py           # Order parameter, correlations, avalanche statistics, MSD
├── parameterscan.py                # Batch scan runner; writes memory-mapped simulation data
├── parameterscan_analysis.py       # Computes and caches observables for completed scan runs
├── parameterscan_responsiveness.py # Responsiveness analysis (supplementary Fig. S1)
├── plotting.py                     # Heatmap plotting utilities
│
├── figure1_sketch.py               # Fig. 1: Schematic of the minority interaction rule
├── figure1_diagram.py              # Fig. 1: Phase diagram
├── figure2_orderparam.py           # Fig. 2: Order parameter time series comparison
├── figure3_avalanches.py           # Fig. 3: Avalanche size and duration distributions
├── figure4_correlation_length.py   # Fig. 4: Correlation length scaling with system size
├── figure5_heatmap.py              # Fig. 5: Parameter space heatmaps
│
├── figure_s1_responsiveness.py     # Supp. Fig. S1: Responsiveness measurements
├── figure_s2_avalanchesL.py        # Supp. Fig. S2: Avalanche distributions by system size
├── figure_s3_correlation_length.py # Supp. Fig. S3: Correlation length (supplementary)
├── figure_s4_orderparameter.py     # Supp. Fig. S4: Order parameter dynamics (supplementary)
├── figure_s5_orderVSsize.py        # Supp. Fig. S5: Order parameter vs. system size scaling
│
├── figures/                        # Pre-generated output figures (PDF, EPS, PNG)
├── prl_style.mplstyle              # Matplotlib style matching PRL figure requirements
├── requirements.txt                # Python package dependencies
└── soc_avalanche_movie.mp4         # Supplementary movie: avalanche event in real space
```

## Requirements

- Python ≥ 3.10
- See `requirements.txt` for package versions

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The first run of any simulation script triggers Numba JIT compilation (roughly 30-60 s, one time only). Subsequent runs use the cached compiled code and start immediately.

## Reproducing the figures

### Using the pre-generated figures

All paper figures are already in `figures/` as PDF and EPS files. You can use them directly without running any simulations.

### Running the figure scripts

Most figure scripts load pre-computed scan data from disk. Before running them, open each script and set `BASE_DIR` to the directory containing your simulation data (lines marked `# <-- CHANGE THIS`).

**Figure 2 - Order parameter dynamics:**

```bash
python figure2_orderparam.py
```

Runs both the standard Vicsek model and the minority-interaction variant for a single parameter set and plots the order parameter trajectories.

**Figure 3 - Avalanche statistics:**

```bash
python figure3_avalanches.py
```

Loads observables from `BASE_DIR` and plots distributions of avalanche sizes and durations.

**Figure 4 - Correlation length:**

```bash
python figure4_correlation_length.py
```

Plots the spatial correlation length as a function of system size near the critical point.

**Figure 5 - Parameter space heatmaps:**

```bash
python figure5_heatmap.py
```

Heatmaps of mean order parameter and order parameter variance over the minority-interaction parameter space (ε, γ).

**Supplementary figures S1-S5:** Run the corresponding `figure_s*.py` script.

### Running parameter scans from scratch

To regenerate the full simulation dataset:

1. Set `BASE_DIR` in `parameterscan.py` to a directory with enough free disk space (scans produce several GB of `.dat` files).

2. Run the scan:

   ```bash
   python parameterscan.py
   ```

3. Compute observables for all completed runs:

   ```bash
   python parameterscan_analysis.py
   ```

   This writes `observables_{id}.npz` files alongside the raw data. The figure scripts pick these up automatically.

## Model

**Standard Vicsek model.** Particles move at constant speed in a 2D periodic domain. At each time step, particle *i* aligns its heading with the mean heading of all neighbors within radius *r*, plus Gaussian noise *η*.

**Minority-interaction variant.** The same alignment rule applies by default. However, if two conditions are simultaneously satisfied, particle *i* instead aligns with its most strongly deviating neighbor — the "defector," defined as the neighbor whose orientation has the smallest dot product with the local mean velocity:

1. **Local order condition:** the dot product of the local mean velocity and particle *i*'s own heading exceeds *ε*. This ensures the minority rule only fires when the neighborhood is already well-aligned — there must be a clear majority to deviate from.
2. **Defector condition:** the dot product of the local mean velocity and the defector's heading falls below *γ*. This ensures the defector is genuinely anti-aligned, not merely slightly off.

When both conditions hold, particle *i* adopts the defector's heading (plus noise) rather than the majority direction. Higher *ε* or lower (more negative) *γ* make the rule harder to trigger. Across a broad intermediate range of both parameters, the competition between majority alignment and minority defection drives the system to a self-organized critical state with scale-free avalanches and heightened collective responsiveness.

## Citation

If you use this code, please cite the paper:

```text
Syga, S., Guha Ray, C., Nava Sedeño, J. M., Peruani, F., & Deutsch, A.
Minority-Triggered Reorientation Yields Macroscopic Cascades and Enhanced Responsiveness
in a Vicsek Swarm. [Journal, Year. DOI: to be added]
```
