# =============================================================
# Neglect-Fold | Phase 2: Download Structures from AlphaFold DB
# =============================================================
# Now that we know AlphaFold has ~98% of our proteins,
# we download their 3D structure files (PDB format).
# PDB files describe exact 3D coordinates of every atom
# in the protein.

import requests
import os
import time
import pandas as pd
from Bio import SeqIO

# ---- Settings ----
ORGANISMS = {
    "trypanosoma_cruzi": "data/processed/trypanosoma_cruzi_cleaned.fasta",
    "leishmania_donovani": "data/processed/leishmania_donovani_cleaned.fasta",
    "schistosoma_mansoni": "data/processed/schistosoma_mansoni_cleaned.fasta"
}

ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction"
OUTPUT_DIR = "data/processed/structures"

# Download 50 per organism for now (full download later on cloud)
PROTEINS_PER_ORGANISM = 50

def extract_uniprot_id(protein_id):
    """
    Extracts UniProt ID from FASTA header.
    Example: sp|Q2VLK6|MCA3_TRYCC -> Q2VLK6
    """
    parts = protein_id.split('|')
    if len(parts) >= 2:
        return parts[1]
    return protein_id

def download_structure(uniprot_id, organism_name):
    """
    Downloads the PDB structure file for one protein
    from AlphaFold DB.
    """
    # First get the metadata to find the PDB download URL
    url = f"{ALPHAFOLD_API}/{uniprot_id}"
    response = requests.get(url, timeout=10)
    
    if response.status_code != 200:
        return False, "Not in AlphaFold DB"
    
    data = response.json()
    
    if not data:
        return False, "Empty response"
    
    # Get the PDB file URL
    pdb_url = data[0].get('pdbUrl')
    confidence_score = data[0].get('confidenceScore', 0)
    
    if not pdb_url:
        return False, "No PDB URL"
    
    # Download the actual PDB file
    pdb_response = requests.get(pdb_url, timeout=30)
    
    if pdb_response.status_code != 200:
        return False, "Could not download PDB"
    
    # Save to file
    organism_dir = os.path.join(OUTPUT_DIR, organism_name)
    os.makedirs(organism_dir, exist_ok=True)
    
    output_path = os.path.join(organism_dir, f"{uniprot_id}.pdb")
    
    with open(output_path, 'w') as f:
        f.write(pdb_response.text)
    
    return True, confidence_score

def download_organism_structures(organism_name, fasta_path):
    """
    Downloads structures for one organism.
    """
    print(f"\n{'='*50}")
    print(f"Downloading structures for {organism_name}")
    print(f"{'='*50}")
    
    # Read proteins
    proteins = list(SeqIO.parse(fasta_path, "fasta"))
    sample = proteins[:PROTEINS_PER_ORGANISM]
    
    print(f"Downloading {len(sample)} structures...")
    
    results = []
    successful = 0
    failed = 0
    
    for i, protein in enumerate(sample):
        uniprot_id = extract_uniprot_id(protein.id)
        
        success, result = download_structure(uniprot_id, organism_name)
        
        if success:
            successful += 1
            results.append({
                "uniprot_id": uniprot_id,
                "organism": organism_name,
                "status": "downloaded",
                "confidence_score": result
            })
        else:
            failed += 1
            results.append({
                "uniprot_id": uniprot_id,
                "organism": organism_name,
                "status": f"failed: {result}",
                "confidence_score": None
            })
        
        # Progress update
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(sample)} "
                  f"(✓ {successful} downloaded, ✗ {failed} failed)")
        
        time.sleep(0.3)
    
    print(f"\nFinal: {successful} downloaded, {failed} failed")
    return results

# ---- Main program ----
if __name__ == "__main__":
    print("=== Neglect-Fold: Structure Downloader ===")
    print(f"Downloading {PROTEINS_PER_ORGANISM} structures per organism")
    print(f"Total: ~{PROTEINS_PER_ORGANISM * 3} structure files")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_results = []
    
    for organism_name, fasta_path in ORGANISMS.items():
        results = download_organism_structures(organism_name, fasta_path)
        all_results.extend(results)
    
    # Save download report
    df = pd.DataFrame(all_results)
    df.to_csv("data/processed/structure_download_report.csv", index=False)
    
    # Summary
    total_downloaded = len(df[df['status'] == 'downloaded'])
    avg_confidence = df[df['confidence_score'].notna()]['confidence_score'].mean()
    
    print(f"\n=== Download Complete ===")
    print(f"Total structures downloaded: {total_downloaded}")
    print(f"Average confidence score: {avg_confidence:.1f}")
    print(f"Structures saved in: {OUTPUT_DIR}/")
    print(f"Report saved to: data/processed/structure_download_report.csv")