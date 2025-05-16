import os
import warnings
from Bio import BiopythonWarning
from Bio.PDB import PDBParser, PDBIO, Superimposer

warnings.simplefilter('ignore', BiopythonWarning)

# === User Parameters ===
reference_pdb_path = "../b2_active_crystal_reference.pdb"
input_root = "../sampled_by_residue"
aligned_output_root = "../Aligned"
replaced_output_root = "../Replaced"

os.makedirs(aligned_output_root, exist_ok=True)
os.makedirs(replaced_output_root, exist_ok=True)

# === Helper: Align sample PDB to reference ===
def align_pdb_files(sample_pdb, reference_pdb, output_path):
    parser = PDBParser(QUIET=True)
    ref_structure = parser.get_structure('ref', reference_pdb)
    sample_structure = parser.get_structure('sample', sample_pdb)

    ref_atoms = list(ref_structure[0].get_atoms())
    sample_atoms = list(sample_structure[0].get_atoms())

    sup = Superimposer()
    sup.set_atoms(ref_atoms, sample_atoms)
    sup.apply(sample_atoms)

    io = PDBIO()
    io.set_structure(sample_structure)
    io.save(output_path)

# === Helper: Replace residue coordinates in reference with those from sample ===
def replace_residue(sample_pdb, reference_pdb, target_residue_index, output_path):
    parser = PDBParser(QUIET=True)
    sample_structure = parser.get_structure('sample', sample_pdb)
    ref_structure = parser.get_structure('ref', reference_pdb)

    sample_residue = list(sample_structure[0].get_residues())[target_residue_index]
    ref_residue = list(ref_structure[0].get_residues())[target_residue_index]

    sample_coords = {atom.id: atom.get_coord() for atom in sample_residue}

    print(f"Replacing residue at index {target_residue_index} in file {sample_pdb}:")
    print(f"Residue name in reference: {ref_residue.get_resname()} | Atom count: {len(ref_residue)}")
    print(f"Residue name in sample   : {sample_residue.get_resname()} | Atom count: {len(sample_residue)}")

    for atom in ref_residue:
        if atom.id in sample_coords:
            print(f"  Atom {atom.id}: before = {atom.get_coord()} → after = {sample_coords[atom.id]}")
            atom.set_coord(sample_coords[atom.id])

    io = PDBIO()
    io.set_structure(ref_structure)
    io.save(output_path)

# === Process each residue folder ===
for folder in sorted(os.listdir(input_root)):
    if not folder.startswith("residue_"):
        continue

    residue_index = int(folder.split("_")[-1])
    sample_folder = os.path.join(input_root, folder)
    aligned_output_folder = os.path.join(aligned_output_root, folder)
    replaced_output_folder = os.path.join(replaced_output_root, folder)

    os.makedirs(aligned_output_folder, exist_ok=True)
    os.makedirs(replaced_output_folder, exist_ok=True)

    all_files = [f for f in os.listdir(sample_folder) if f.endswith(".pdb")]

    for pdb_file in all_files:
        input_path = os.path.join(sample_folder, pdb_file)
        aligned_path = os.path.join(aligned_output_folder, f"aligned_{pdb_file}")
        replaced_path = os.path.join(replaced_output_folder, f"replaced_{pdb_file}")

        align_pdb_files(input_path, reference_pdb_path, aligned_path)
        replace_residue(aligned_path, reference_pdb_path, residue_index, replaced_path)

print("Done aligning and replacing for all residue folders!")