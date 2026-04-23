# =============================================================
# Neglect-Fold | Phase 2: Check AlphaFold DB Coverage
# =============================================================
# AlphaFold DB has pre-computed structures for many proteins.
# We check which of our 58,265 proteins are already there.
# For those that are - we download directly (free, no GPU needed)
# For those that aren't - we note them for ESMFold later

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

def extract_uniprot_id(protein_id):
    """
    Extracts the UniProt ID from a FASTA header.
    Example: sp|Q2VLK6|MCA3_TRYCC -> Q2VLK6
    """
    parts = protein_id.split('|')
    if len(parts) >= 2:
        return parts[1]
    return protein_id

def check_alphafold_availability(uniprot_id):
    """
    Checks if AlphaFold DB has a structure for this protein.
    Returns True if available, False if not.
    """
    url = f"{ALPHAFOLD_API}/{uniprot_id}"
    response = requests.get(url, timeout=10)
    return response.status_code == 200

def check_organism_coverage(organism_name, fasta_path):
    """
    Checks AlphaFold coverage for one organism.
    Only checks first 100 proteins to save time.
    """
    print(f"\nChecking AlphaFold coverage for {organism_name}...")
    
    # Read proteins
    proteins = list(SeqIO.parse(fasta_path, "fasta"))
    
    # Only check first 100 to save time
    sample = proteins[:100]
    print(f"Checking first {len(sample)} proteins...")
    
    available = []
    not_available = []
    
    for i, protein in enumerate(sample):
        uniprot_id = extract_uniprot_id(protein.id)
        
        if check_alphafold_availability(uniprot_id):
            available.append(uniprot_id)
        else:
            not_available.append(uniprot_id)
        
        # Progress update every 10 proteins
        if (i + 1) % 10 == 0:
            print(f"  Checked {i+1}/{len(sample)} proteins...")
        
        # Be polite to the server
        time.sleep(0.2)
    
    coverage = len(available) / len(sample) * 100
    print(f"\nResults for {organism_name}:")
    print(f"  Available in AlphaFold DB: {len(available)}/100 ({coverage:.1f}%)")
    print(f"  Not available: {len(not_available)}/100")
    
    return available, not_available, coverage

# ---- Main program ----
if __name__ == "__main__":
    print("=== Neglect-Fold: AlphaFold Coverage Check ===")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    results = []
    
    for organism_name, fasta_path in ORGANISMS.items():
        available, not_available, coverage = check_organism_coverage(
            organism_name, fasta_path
        )
        
        results.append({
            "organism": organism_name,
            "available_in_alphafold": len(available),
            "not_available": len(not_available),
            "coverage_percent": coverage
        })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv("data/processed/alphafold_coverage.csv", index=False)
    
    print("\n=== Coverage Summary ===")
    print(df.to_string(index=False))
    print("\nSaved to data/processed/alphafold_coverage.csv")