# ml-accelerated-md-gpcr

# Accelerating G Protein Coupled Receptor (GPCR) Molecular Dynamic Simulations Using Machine Learning

This repository contains code and data for evaluating how OCN backbone angle perturbations at switch residues affect predicted activity of the β₂-adrenergic receptor (β₂AR). The workflow combines structural sampling, CHARMM-GUI–based equilibration, and machine learning to compute per-residue sensitivity using predicted activity changes.

---

## Project Summary

For each switch residue:
- Sampled 10 conformations across the observed OCN angle range
- Replaced the corresponding residue in the β₂AR active-state reference structure
- Ran energy minimization using CHARMM-GUI and OpenMM (GPU-enabled)
- Predicted activity using a pretrained XGBoost model
- Quantified sensitivity using:  
**ΔA / Δθ** = change in predicted activity / change in OCN angle

---

## Directory Structure

```bash
data/                         # Input data files
├── 192_switch/              # Per-residue CSVs with OCN angle/frame info
├── pdb_beta2_inverse_ago/   # Raw trajectory PDBs
└── delta_angles_from_reference.csv

b2_active_crystal_reference.pdb  # Active-state template of β₂AR

model/
└── xgb_activity_regressor.pkl       # XGBoost model for activity prediction

scripts/
├── sample_and_save_frames.py         # Samples 10 frames per residue based on OCN angle bins
├── align_replace.py     # Aligns sampled frames and replaces residue in reference
├── run_equilibration.py     # Automates CHARMM-GUI upload and runs OpenMM minimization
└── pred_activity.ipynb   # Computes ΔA/Δθ using ML model and summarizes results

sampled_by_residue/          # Sampled frames per residue
Aligned/                     # Aligned versions of sampled PDBs
Replaced/                    # Structures with replaced residues
minimized_inactive/          # Equilibrated output PDBs
results/                     # Summary of final sensitivity values
```
## Environment & Installation

- Requires **Python 3.8+**
- **OpenMM must be GPU-enabled** (CUDA required for equilibration)

Install all core dependencies:

```bash
pip install -r requirements.txt
```

---
## Data Setup

To run the code, you’ll need to download the required PDB structure files.

[Download `pdb_beta2_inverse_ago.zip`](https://cmu.app.box.com/s/y0fhqh78s5j8iaqh2m0xtc7vzl0cn53j)

Once downloaded, extract the contents into the `data/` directory
## Pipeline Execution

Run the following steps **in order** from the project root:

### 1. Sample frames based on OCN angle bins

```bash
python scripts/sample_and_save_frames.py
```

### 2. Align sampled frames and replace target residue in reference structure

```bash
python scripts/align_replace.py
```

### 3. Submit structures to CHARMM-GUI and run OpenMM equilibration  
Requires a valid CHARMM-GUI account and a CUDA-enabled GPU

```bash
python scripts/run_equilibration.py
```

### 4. Predict activity and compute ΔA / Δθ  
Open and run all cells in:

```bash
scripts/pred_activity.ipynb
```
