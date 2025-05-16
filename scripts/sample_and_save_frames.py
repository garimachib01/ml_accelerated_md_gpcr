import os
import pandas as pd
import numpy as np
import mdtraj as md
from Bio.PDB import PDBParser
import shutil
import random

random.seed(42)

# === Paths ===
switch_csv_dir = "../data/192_switch"
pdb_folder = "../data/pdb_beta2_inverse_ago"
reference_pdb = "../b2_active_crystal_reference.pdb"
output_root = "../sampled_by_residue"

os.makedirs(output_root, exist_ok=True)

# === Helper: Map atom serials to residue index and name ===
def get_common_residue_index(pdb_path, atom_serials):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("ref", pdb_path)
    atom_to_residue_index = {}
    residues = list(structure.get_residues())
    for i, res in enumerate(residues):
        for atom in res:
            atom_to_residue_index[atom.serial_number] = (i, res.get_resname())
    mapped = [atom_to_residue_index.get(s + 1) for s in atom_serials]  # PDB atom serials are 1-indexed
    if None in mapped:
        raise ValueError("Some atom serials not found in the PDB.")
    residue_indices = [x[0] for x in mapped]
    residue_names = [x[1] for x in mapped]
    if all(i == residue_indices[0] for i in residue_indices):
        return residue_indices[0], residue_names[0]
    raise ValueError(f"Atom serials map to different residues: {residue_indices}")

# === Track processed residues ===
processed_residues = set()

# === Main loop: iterate over switch CSV files ===
for csv_file in sorted(os.listdir(switch_csv_dir)):
    if not csv_file.endswith(".csv"):
        continue

    df = pd.read_csv(os.path.join(switch_csv_dir, csv_file))
    
    try:
        atom_indices = eval(df.iloc[0][1])
        residue_index, residue_name = get_common_residue_index(reference_pdb, atom_indices)
    except Exception as e:
        print(f"Skipping {csv_file} due to error: {e}")
        continue

    if residue_index in processed_residues:
        print(f"Skipping {csv_file} (already processed residue {residue_index} - {residue_name})")
        continue
    processed_residues.add(residue_index)

    print(f"Using {csv_file} for residue {residue_index} ({residue_name})")

    # === Step 1: Extract (traj_file, frame_index, angle) entries ===
    all_entries = []
    for entry in df.iloc[:, 2]:
        if isinstance(entry, str):
            all_entries.extend(eval(entry))

    # === Step 2: Filter by available PDBs ===
    pdb_files = [f for f in os.listdir(pdb_folder) if f.endswith(".pdb")]
    pdb_ids = {f.split(".")[0][3:] for f in pdb_files}

    filtered_entries = []
    filtered_angles = []

    for traj in all_entries:
        traj_id = traj[0].split(".")[0][3:]
        if traj_id in pdb_ids:
            filtered_entries.append(traj)
            filtered_angles.append(traj[-1])

    if len(filtered_entries) < 10:
        print(f"Not enough valid entries for residue {residue_index}")
        continue

    # === Step 3: Bin angles and sample one per bin ===
    filtered_angles = np.array(filtered_angles)
    bin_edges = np.linspace(min(filtered_angles), max(filtered_angles), 11)

    sampled_entries = []
    for i in range(10):
        bin_start, bin_end = bin_edges[i], bin_edges[i + 1]
        bin_entries = [entry for entry in filtered_entries if bin_start <= entry[-1] < bin_end]
        if bin_entries:
            sampled_entries.append(random.choice(bin_entries))

    # === Step 4: Save sampled frames to residue-specific folder ===
    output_folder = os.path.join(output_root, f"residue_{residue_index}")
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    for traj_file, frame_index, angle in sampled_entries:
        pdb_name = f"trj{traj_file.split('.')[0][3:]}.pdb"
        pdb_path = os.path.join(pdb_folder, pdb_name)

        try:
            traj = md.load_pdb(pdb_path)
            frame = traj[frame_index]
            out_file = f"{pdb_name[:-4]}_frame{frame_index}.pdb"
            out_path = os.path.join(output_folder, out_file)
            frame.save_pdb(out_path)
            print(f"Saved {out_path}")
        except Exception as e:
            print(f"Error processing {pdb_name}: {e}")

print("Done — one switch file used per residue, 10 binned samples saved each.")
