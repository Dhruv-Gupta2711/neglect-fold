# =============================================================
# Neglect-Fold | Phase 1: Clean and Filter Proteomes
# =============================================================
# Raw data from UniProt contains proteins we don't want:
# - Too short (not real proteins, just fragments)
# - Too long (hard to process, often poorly annotated)
# - Duplicate sequences (would bias our models)
# - Non-standard amino acids (would confuse our models)
# This script filters them all out.

import os
import pandas as pd
from Bio import SeqIO
from collections import Counter

# ---- Settings ----
MIN_LENGTH = 50       # Shorter than this = probably a fragment
MAX_LENGTH = 2000     # Longer than this = too large for our models now
ORGANISMS = {
    "trypanosoma_cruzi": "data/raw/trypanosoma_cruzi.fasta",
    "leishmania_donovani": "data/raw/leishmania_donovani.fasta",
    "schistosoma_mansoni": "data/raw/schistosoma_mansoni.fasta"
}

# Standard amino acid letters - anything else is non-standard
STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

def has_nonstandard_amino_acids(sequence):
    """
    Returns True if the sequence contains unusual amino acid letters.
    X means unknown, B/Z/J are ambiguous - we want to remove these.
    """
    letters_in_sequence = set(sequence.upper())
    non_standard = letters_in_sequence - STANDARD_AMINO_ACIDS
    return len(non_standard) > 0

def clean_proteome(organism_name, filepath):
    """
    Cleans one organism's proteins and returns filtered results.
    """
    print(f"\n{'='*50}")
    print(f"Cleaning: {organism_name}")
    print(f"{'='*50}")
    
    # Read all proteins
    all_proteins = list(SeqIO.parse(filepath, "fasta"))
    total_original = len(all_proteins)
    print(f"Starting proteins: {total_original}")
    
    # ---- Filter 1: Remove too short ----
    after_min = [p for p in all_proteins if len(p.seq) >= MIN_LENGTH]
    removed_short = total_original - len(after_min)
    print(f"Removed (too short <{MIN_LENGTH} aa): {removed_short}")
    
    # ---- Filter 2: Remove too long ----
    after_max = [p for p in after_min if len(p.seq) <= MAX_LENGTH]
    removed_long = len(after_min) - len(after_max)
    print(f"Removed (too long >{MAX_LENGTH} aa): {removed_long}")
    
    # ---- Filter 3: Remove non-standard amino acids ----
    after_standard = [p for p in after_max 
                      if not has_nonstandard_amino_acids(str(p.seq))]
    removed_nonstandard = len(after_max) - len(after_standard)
    print(f"Removed (non-standard amino acids): {removed_nonstandard}")
    
    # ---- Filter 4: Remove duplicate sequences ----
    seen_sequences = set()
    unique_proteins = []
    
    for protein in after_standard:
        seq_str = str(protein.seq)
        if seq_str not in seen_sequences:
            seen_sequences.add(seq_str)
            unique_proteins.append(protein)
    
    removed_duplicates = len(after_standard) - len(unique_proteins)
    print(f"Removed (duplicate sequences): {removed_duplicates}")
    
    # ---- Final count ----
    final_count = len(unique_proteins)
    removed_total = total_original - final_count
    kept_percent = (final_count / total_original) * 100
    
    print(f"\nFinal proteins kept: {final_count}")
    print(f"Total removed: {removed_total} ({100-kept_percent:.1f}%)")
    print(f"Kept: {kept_percent:.1f}% of original")
    
    return unique_proteins

def save_cleaned_fasta(proteins, organism_name):
    """
    Saves the cleaned proteins to a new FASTA file.
    """
    output_path = f"data/processed/{organism_name}_cleaned.fasta"
    SeqIO.write(proteins, output_path, "fasta")
    print(f"Saved cleaned file: {output_path}")
    return output_path

def save_cleaning_report(reports):
    """
    Saves a CSV summary of what was removed and why.
    """
    df = pd.DataFrame(reports)
    df.to_csv("data/processed/cleaning_report.csv", index=False)
    print("\nCleaning report saved to data/processed/cleaning_report.csv")

# ---- Main program ----
if __name__ == "__main__":
    print("=== Neglect-Fold: Proteome Cleaner ===")
    
    # Make sure output folder exists
    os.makedirs("data/processed", exist_ok=True)
    
    reports = []
    
    for organism_name, filepath in ORGANISMS.items():
        # Clean the proteome
        clean_proteins = clean_proteome(organism_name, filepath)
        
        # Save cleaned file
        save_cleaned_fasta(clean_proteins, organism_name)
        
        # Record the report
        reports.append({
            "organism": organism_name,
            "original_count": len(list(SeqIO.parse(filepath, "fasta"))),
            "cleaned_count": len(clean_proteins),
        })
    
    # Save the cleaning report
    save_cleaning_report(reports)
    
    print("\n=== Cleaning Complete! ===")
    print("Cleaned files are in data/processed/")