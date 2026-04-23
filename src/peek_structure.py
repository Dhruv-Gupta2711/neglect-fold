# Quick peek at a downloaded PDB structure file

import os

# Get the first PDB file we downloaded
structure_dir = "data/processed/structures/trypanosoma_cruzi"
pdb_files = os.listdir(structure_dir)
first_file = os.path.join(structure_dir, pdb_files[0])

print(f"Looking at: {pdb_files[0]}")
print("=" * 50)

with open(first_file) as f:
    lines = f.readlines()

print(f"Total lines in file: {len(lines)}")
print("\nFirst 15 lines:")
for line in lines[:15]:
    print(line.strip())