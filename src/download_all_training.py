# =============================================================
# Neglect-Fold | Download All Training Structures
# =============================================================
# Downloads all 1,225 structures found by search_pdb.py
# Processes each one to find pocket residues
# Saves labeled training data for GNN training

import requests
import os
import time
import pandas as pd
from Bio import PDB
from Bio.PDB import NeighborSearch
import io
import warnings
warnings.filterwarnings('ignore')

SEARCH_RESULTS = "data/processed/pdb_search/pdb_search_results.csv"
OUTPUT_DIR = "data/processed/training_full"

def download_pdb(pdb_id):
    """Downloads a PDB file from RCSB."""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return response.text
    except:
        pass
    return None

def find_pocket_residues(pdb_content, distance_threshold=4.5):
    """Finds pocket residues near bound ligands."""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(
        'protein',
        io.StringIO(pdb_content)
    )
    
    protein_atoms = []
    ligand_atoms = []
    protein_residues = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                res_id = residue.get_id()
                if res_id[0] != ' ' and res_id[0] != 'W':
                    for atom in residue:
                        ligand_atoms.append(atom)
                elif res_id[0] == ' ':
                    if 'CA' in residue:
                        protein_residues.append(residue)
                        for atom in residue:
                            protein_atoms.append(atom)
    
    if not ligand_atoms or not protein_residues:
        return None, None
    
    ns = NeighborSearch(protein_atoms)
    pocket_residue_ids = set()
    
    for lig_atom in ligand_atoms:
        nearby = ns.search(
            lig_atom.get_vector().get_array(),
            distance_threshold,
            'R'
        )
        for residue in nearby:
            pocket_residue_ids.add(residue.get_id())
    
    residue_labels = {}
    for residue in protein_residues:
        res_id = residue.get_id()
        label = 1 if res_id in pocket_residue_ids else 0
        residue_labels[res_id] = label
    
    pocket_count = sum(residue_labels.values())
    total_count = len(residue_labels)
    
    if total_count == 0:
        return None, None
    
    pocket_fraction = pocket_count / total_count
    
    return residue_labels, pocket_fraction

if __name__ == "__main__":
    print("=== Neglect-Fold: Download All Training Structures ===")
    
    # Load search results
    df = pd.read_csv(SEARCH_RESULTS)
    print(f"Total structures to download: {len(df)}")
    print(f"Parasite: {len(df[df['label']==1])}")
    print(f"Human: {len(df[df['label']==0])}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/pdbs", exist_ok=True)
    
    results = []
    downloaded = 0
    failed = 0
    skipped = 0
    
    for i, row in df.iterrows():
        pdb_id = row['pdb_id']
        organism = row['organism']
        label = row['label']
        
        # Skip if already downloaded
        pdb_path = f"{OUTPUT_DIR}/pdbs/{pdb_id}.pdb"
        if os.path.exists(pdb_path):
            skipped += 1
            continue
        
        # Download
        pdb_content = download_pdb(pdb_id)
        
        if not pdb_content:
            failed += 1
            continue
        
        # Find pocket residues
        residue_labels, pocket_fraction = find_pocket_residues(
            pdb_content
        )
        
        if residue_labels is None:
            failed += 1
            continue
        
        # Filter by pocket fraction (5-30%)
        if pocket_fraction < 0.02 or pocket_fraction > 0.35:
            skipped += 1
            continue
        
        # Save PDB file
        with open(pdb_path, 'w') as f:
            f.write(pdb_content)
        
        pocket_count = sum(residue_labels.values())
        total_count = len(residue_labels)
        
        results.append({
            'pdb_id': pdb_id,
            'organism': organism,
            'label': label,
            'pdb_path': pdb_path,
            'total_residues': total_count,
            'pocket_residues': pocket_count,
            'pocket_fraction': round(pocket_fraction, 3)
        })
        
        downloaded += 1
        
        # Progress update every 50
        if downloaded % 50 == 0:
            print(f"Downloaded: {downloaded} | "
                  f"Failed: {failed} | "
                  f"Skipped: {skipped}")
            
            # Save progress
            pd.DataFrame(results).to_csv(
                f"{OUTPUT_DIR}/training_summary.csv",
                index=False
            )
        
        time.sleep(0.3)
    
    # Final save
    result_df = pd.DataFrame(results)
    result_df.to_csv(
        f"{OUTPUT_DIR}/training_summary.csv",
        index=False
    )
    
    print(f"\n{'='*50}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*50}")
    print(f"Successfully downloaded: {downloaded}")
    print(f"Failed: {failed}")
    print(f"Skipped (bad pocket fraction): {skipped}")
    print(f"Parasite: {len(result_df[result_df['label']==1])}")
    print(f"Human: {len(result_df[result_df['label']==0])}")
    print(f"\nSaved to {OUTPUT_DIR}/")