# =============================================================
# Neglect-Fold | Improved Training Data Collection
# =============================================================
# Downloads TWO types of training data:
#
# Type 1: Parasite proteins with pockets (GOOD TARGETS)
#   - T. cruzi, Leishmania, Schistosoma proteins
#   - Co-crystallized with drug-like ligands
#   - Label: 1 (these are what we want to find)
#
# Type 2: Human proteins with pockets (AVOID)
#   - Human proteins with similar binding pockets
#   - Label: 0 (these must not be targeted)
#
# This teaches the GNN to find parasite-specific pockets
# while automatically avoiding human-similar ones.

import requests
import os
import time
import json
import pandas as pd
from Bio import PDB
from Bio.PDB import NeighborSearch
import io
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = "data/processed/training_v2"

# ============================================================
# PART 1: PDB IDs for each category
# ============================================================

# Parasite proteins with confirmed drug binding pockets
# These are our POSITIVE examples (label = 1)
PARASITE_PDB_IDS = {
    "trypanosoma_cruzi": [
        "1CL8",  # GAPDH with inhibitor
        "3MFP",  # Cruzain with inhibitor
        "2H7S",  # Trans-sialidase
        "2OZ2",  # Pteridine reductase
        "1AON",  # GroEL chaperonin
        "3L4Q",  # CYP51 with inhibitor
        "3ITF",  # Trypanothione reductase
        "4KOY",  # Cruzain new inhibitor
        "2YIT",  # Glucokinase
        "3QP5",  # Trypanothione synthetase
    ],
    "leishmania": [
        "2JK6",  # DHFR with inhibitor
        "3RGF",  # Pteridine reductase
        "1E7W",  # DHFR-TS
        "2BFA",  # Trypanothione reductase
        "3GRS",  # GSH reductase
        "3KCU",  # L. major pteridine reductase
        "3MHR",  # Trypanothione synthetase
        "4OAE",  # CYP51
        "2YNF",  # MAPK with inhibitor
        "3RY1",  # N-myristoyltransferase
    ],
    "schistosoma": [
        "4CGO",  # Kinase
        "2V6O",  # HDAC
        "4MQT",  # Carbonic anhydrase
        "4CQF",  # HDAC8 with inhibitor
        "3ZMQ",  # GST with inhibitor
        "4FTP",  # Phosphoglycerate mutase
        "3O8O",  # Adenosine kinase
        "4CAD",  # GST
        "3ZYT",  # HDAC with inhibitor
        "2XZF",  # Thioredoxin glutathione reductase
    ]
}

# Human proteins with drug binding pockets
# These are our NEGATIVE examples (label = 0)
# We choose human proteins that are in similar 
# functional categories to parasite targets
HUMAN_PDB_IDS = [
    # Human kinases (similar to parasite kinases)
    "1ATP",  # Human PKA kinase
    "2ITY",  # Human CDK2 kinase
    "1FPU",  # Human EGFR kinase
    "3EKO",  # Human Aurora kinase
    "2F4J",  # Human p38 MAPK
    
    # Human reductases (similar to parasite reductases)
    "1DXO",  # Human DHFR
    "1GRA",  # Human glutathione reductase
    "3GRS",  # Human GSH reductase
    "1YBV",  # Human thioredoxin reductase
    "2QQ5",  # Human NADPH reductase
    
    # Human proteases (similar to cruzain)
    "1NPM",  # Human cathepsin B
    "1CS8",  # Human cathepsin L
    "2XU3",  # Human caspase-3
    "1RHB",  # Human renin
    "1TNI",  # Human trypsin
    
    # Human metabolic enzymes
    "1HRD",  # Human GAPDH
    "3TG0",  # Human CYP51
    "1E66",  # Human adenosine kinase
    "1NL5",  # Human carbonic anhydrase
    "2VHL",  # Human HDAC
]

# ============================================================
# PART 2: Download and process functions
# ============================================================

def download_pdb(pdb_id):
    """Downloads a PDB file from RCSB."""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url, timeout=30)
    if response.status_code == 200:
        return response.text
    return None

def find_pocket_residues(pdb_content, distance_threshold=4.5):
    """
    Finds pocket residues by detecting amino acids
    within 4.5 Angstroms of any bound ligand.
    """
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
    
    # Check pocket fraction
    pocket_count = sum(residue_labels.values())
    total_count = len(residue_labels)
    pocket_fraction = pocket_count / total_count
    
    return residue_labels, pocket_fraction

def process_pdb(pdb_id, category, label):
    """
    Downloads and processes one PDB structure.
    
    category: which organism/group
    label: 1 = parasite (good target), 0 = human (avoid)
    """
    print(f"  Processing {pdb_id} ({category})...")
    
    pdb_content = download_pdb(pdb_id)
    if not pdb_content:
        print(f"  Could not download {pdb_id}")
        return None
    
    # Save PDB file
    category_dir = os.path.join(OUTPUT_DIR, 'pdbs', category)
    os.makedirs(category_dir, exist_ok=True)
    pdb_path = os.path.join(category_dir, f"{pdb_id}.pdb")
    
    with open(pdb_path, 'w') as f:
        f.write(pdb_content)
    
    # Find pocket residues
    residue_labels, pocket_fraction = find_pocket_residues(
        pdb_content
    )
    
    if residue_labels is None:
        print(f"  No ligand found in {pdb_id}")
        return None
    
    pocket_count = sum(residue_labels.values())
    total_count = len(residue_labels)
    
    print(f"  Found {pocket_count}/{total_count} pocket residues "
          f"({pocket_fraction:.1%}) - Label: {label}")
    
    return {
        'pdb_id': pdb_id,
        'category': category,
        'label': label,
        'pdb_path': pdb_path,
        'total_residues': total_count,
        'pocket_residues': pocket_count,
        'pocket_fraction': pocket_fraction,
        'residue_labels': residue_labels
    }

# ============================================================
# PART 3: Main program
# ============================================================

if __name__ == "__main__":
    print("=== Neglect-Fold: Download Training Structures ===")
    print("Downloading parasite AND human proteins")
    print()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_results = []
    
    # Download parasite proteins (label = 1)
    print("=" * 50)
    print("PARASITE PROTEINS (Good targets - Label: 1)")
    print("=" * 50)
    
    for organism, pdb_ids in PARASITE_PDB_IDS.items():
        print(f"\n{organism.upper()}:")
        for pdb_id in pdb_ids:
            result = process_pdb(pdb_id, organism, label=1)
            if result:
                all_results.append(result)
            time.sleep(0.5)
    
    # Download human proteins (label = 0)
    print("\n" + "=" * 50)
    print("HUMAN PROTEINS (Avoid - Label: 0)")
    print("=" * 50)
    
    for pdb_id in HUMAN_PDB_IDS:
        result = process_pdb(pdb_id, 'human', label=0)
        if result:
            all_results.append(result)
        time.sleep(0.5)
    
    # Save summary
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            'pdb_id': r['pdb_id'],
            'category': r['category'],
            'label': r['label'],
            'pdb_path': r['pdb_path'],
            'total_residues': r['total_residues'],
            'pocket_residues': r['pocket_residues'],
            'pocket_fraction': round(r['pocket_fraction'], 3)
        })
    
    df = pd.DataFrame(summary_rows)
    df.to_csv(
        f"{OUTPUT_DIR}/training_summary.csv",
        index=False
    )
    
    # Print summary
    parasite = df[df['label'] == 1]
    human = df[df['label'] == 0]
    
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Parasite proteins: {len(parasite)}")
    print(f"Human proteins: {len(human)}")
    print(f"Total: {len(df)}")
    print(f"Average pocket fraction: {df['pocket_fraction'].mean():.1%}")
    print(f"\nSaved to {OUTPUT_DIR}/")
    print("\n=== Download Complete! ===")