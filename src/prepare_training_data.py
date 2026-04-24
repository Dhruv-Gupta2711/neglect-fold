# =============================================================
# Neglect-Fold | Phase 3: Prepare Training Data
# =============================================================
# To train our GNN we need labeled examples:
# - Input: protein 3D structure
# - Label: which amino acids are in the binding pocket
#
# We get this from PDB entries where a protein is
# co-crystallized with a ligand (drug molecule).
# The amino acids touching the ligand = binding pocket.

import requests
import os
import numpy as np
from Bio import PDB
from Bio.PDB import NeighborSearch
import pandas as pd
import time

OUTPUT_DIR = "data/processed/training"

# These are PDB IDs of NTD proteins co-crystallized with ligands
# Each one is a protein structure with a drug already bound to it
# We know EXACTLY where the pocket is in these structures
NTD_PDB_ENTRIES = [
    # Trypanosoma cruzi proteins with bound ligands
    "1CL8",  # T. cruzi GAPDH with inhibitor
    "2OT6",  # T. cruzi TIM with ligand
    "3MFP",  # T. cruzi cruzain with inhibitor
    "1U0D",  # T. cruzi DHFR-TS
    "2H7S",  # T. cruzi trans-sialidase
    
    # Leishmania proteins with bound ligands  
    "2JK6",  # L. donovani DHFR with inhibitor
    "3RGF",  # Leishmania pteridine reductase
    "1E7W",  # Leishmania DHFR-TS
    "2BFA",  # Leishmania trypanothione reductase
    "3GRS",  # Leishmania GSH reductase
    
    # Schistosoma proteins with bound ligands
    "3QQ4",  # S. mansoni thioredoxin GSH reductase  
    "4CGO",  # S. mansoni kinase
    "2V6O",  # S. mansoni HDAC
]

def download_pdb(pdb_id):
    """
    Downloads a PDB file from the RCSB PDB database.
    These are experimentally solved structures with real
    drug molecules bound to the protein.
    """
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url, timeout=30)
    
    if response.status_code == 200:
        return response.text
    return None

def find_binding_pocket_residues(pdb_content, distance_threshold=4.5):
    """
    Finds which amino acids form the binding pocket.
    
    Method: any amino acid within 4.5 Angstroms of the
    bound ligand is considered part of the binding pocket.
    
    This is the ground truth label for our GNN.
    """
    parser = PDB.PDBParser(QUIET=True)
    
    import io
    structure = parser.get_structure(
        'protein', 
        io.StringIO(pdb_content)
    )
    
    # Separate protein residues from ligand atoms
    protein_atoms = []
    ligand_atoms = []
    protein_residues = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                res_id = residue.get_id()
                
                # HETATM records are ligands/drugs
                if res_id[0] != ' ' and res_id[0] != 'W':
                    # This is a ligand (not water, not standard AA)
                    for atom in residue:
                        ligand_atoms.append(atom)
                
                # Standard amino acids
                elif res_id[0] == ' ':
                    if 'CA' in residue:
                        protein_residues.append(residue)
                        for atom in residue:
                            protein_atoms.append(atom)
    
    if not ligand_atoms or not protein_residues:
        return None, None
    
    # Find protein atoms near ligand atoms
    ns = NeighborSearch(protein_atoms)
    
    pocket_residue_ids = set()
    
    for lig_atom in ligand_atoms:
        nearby_atoms = ns.search(
            lig_atom.get_vector().get_array(),
            distance_threshold,
            'R'  # Search at Residue level
        )
        for residue in nearby_atoms:
            pocket_residue_ids.add(residue.get_id())
    
    # Create labels: 1 if pocket residue, 0 if not
    residue_labels = {}
    for residue in protein_residues:
        res_id = residue.get_id()
        label = 1 if res_id in pocket_residue_ids else 0
        residue_labels[res_id] = label
    
    return protein_residues, residue_labels

def process_pdb_entry(pdb_id):
    """
    Downloads and processes one PDB entry.
    Returns labeled training data.
    """
    print(f"  Processing {pdb_id}...")
    
    # Download PDB
    pdb_content = download_pdb(pdb_id)
    if not pdb_content:
        print(f"  Could not download {pdb_id}")
        return None
    
    # Save raw PDB
    os.makedirs(f"{OUTPUT_DIR}/pdbs", exist_ok=True)
    pdb_path = f"{OUTPUT_DIR}/pdbs/{pdb_id}.pdb"
    with open(pdb_path, 'w') as f:
        f.write(pdb_content)
    
    # Find binding pocket residues
    residues, labels = find_binding_pocket_residues(pdb_content)
    
    if residues is None:
        print(f"  No ligand found in {pdb_id}")
        return None
    
    # Count pocket vs non-pocket residues
    pocket_count = sum(labels.values())
    total_count = len(labels)
    
    print(f"  Found {pocket_count} pocket residues out of {total_count} total")
    
    return {
        'pdb_id': pdb_id,
        'pdb_path': pdb_path,
        'total_residues': total_count,
        'pocket_residues': pocket_count,
        'labels': labels
    }

# ---- Main program ----
if __name__ == "__main__":
    print("=== Neglect-Fold: Prepare Training Data ===")
    print(f"Processing {len(NTD_PDB_ENTRIES)} NTD protein structures")
    print()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    results = []
    successful = 0
    
    for pdb_id in NTD_PDB_ENTRIES:
        result = process_pdb_entry(pdb_id)
        
        if result:
            results.append({
                'pdb_id': result['pdb_id'],
                'pdb_path': result['pdb_path'],
                'total_residues': result['total_residues'],
                'pocket_residues': result['pocket_residues'],
                'pocket_fraction': result['pocket_residues'] / result['total_residues']
            })
            successful += 1
        
        time.sleep(0.5)
    
    # Save summary
    df = pd.DataFrame(results)
    df.to_csv(f"{OUTPUT_DIR}/training_data_summary.csv", index=False)
    
    print(f"\n=== Summary ===")
    print(f"Successfully processed: {successful}/{len(NTD_PDB_ENTRIES)}")
    if len(df) > 0:
        print(f"Average pocket fraction: {df['pocket_fraction'].mean():.3f}")
        print(f"Total residues: {df['total_residues'].sum()}")
        print(f"Total pocket residues: {df['pocket_residues'].sum()}")
    print(f"\nTraining data saved to {OUTPUT_DIR}/")