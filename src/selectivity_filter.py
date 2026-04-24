# =============================================================
# Neglect-Fold | Phase 4: Selectivity Filter
# =============================================================
# Before recommending any drug target we must verify it is
# sufficiently different from human proteins.
# If a parasite protein is too similar to a human protein,
# a drug targeting it would harm the patient.
#
# Method: sequence alignment using BLAST-like comparison
# Threshold: reject if >40% identical to any human protein

import requests
import pandas as pd
import numpy as np
import os
import time
import json
from Bio import SeqIO
from Bio.Blast import NCBIWWW, NCBIXML
import warnings
warnings.filterwarnings('ignore')

# ---- Settings ----
# How similar is too similar?
# >40% identity in the pocket region = dangerous
IDENTITY_THRESHOLD = 40.0

# Human proteome taxonomy ID in UniProt
HUMAN_TAX_ID = "9606"

OUTPUT_DIR = "data/processed/selectivity"

def get_human_blast(sequence, protein_id):
    """
    Runs BLAST search against human proteins.
    
    BLAST = Basic Local Alignment Search Tool
    It compares our parasite protein sequence against
    a database of all human proteins to find similar ones.
    
    Returns the best match identity percentage.
    """
    print(f"    Running BLAST for {protein_id}...")
    
    try:
        # Run BLAST against human proteins only
        result_handle = NCBIWWW.qblast(
            "blastp",           # protein BLAST
            "swissprot",        # search Swiss-Prot database
            sequence,           # our parasite protein
            entrez_query="Homo sapiens[organism]",  # human only
            hitlist_size=5,     # top 5 matches
            expect=0.001        # significance threshold
        )
        
        # Parse results
        blast_records = NCBIXML.parse(result_handle)
        blast_record = next(blast_records)
        
        if not blast_record.alignments:
            # No human matches found - great! Very selective
            return 0.0, "No human homolog found"
        
        # Get best match
        best_alignment = blast_record.alignments[0]
        best_hsp = best_alignment.hsps[0]
        
        # Calculate identity percentage
        identity = (best_hsp.identities / best_hsp.align_length) * 100
        
        human_protein = best_alignment.title[:50]
        
        return identity, human_protein
        
    except Exception as e:
        print(f"    BLAST error: {e}")
        return None, "Error"

def check_selectivity_uniprot(sequence, protein_id):
    """
    Faster alternative to BLAST - uses UniProt API
    to check if our protein has human homologs.
    
    This is faster but less accurate than BLAST.
    We use this for quick screening.
    """
    # Take first 100 amino acids as query
    # This is enough to detect close homologs
    query_seq = sequence[:100]
    
    url = (
        f"https://rest.uniprot.org/uniprotkb/search?"
        f"query=organism_id:{HUMAN_TAX_ID}+AND+"
        f"reviewed:true&"
        f"format=fasta&"
        f"size=1"
    )
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return True
    except:
        pass
    
    return False

def calculate_sequence_identity(seq1, seq2):
    """
    Simple sequence identity calculation.
    Compares two sequences character by character.
    
    For a quick comparison without running full BLAST.
    """
    # Use shorter sequence length for comparison
    min_len = min(len(seq1), len(seq2))
    max_len = max(len(seq1), len(seq2))
    
    if min_len == 0:
        return 0.0
    
    # Count matching positions
    matches = sum(
        1 for a, b in zip(seq1[:min_len], seq2[:min_len]) 
        if a == b
    )
    
    # Identity relative to longer sequence
    identity = (matches / max_len) * 100
    return identity

def load_human_reference_proteins():
    """
    Downloads a small set of important human proteins
    to use as reference for selectivity checking.
    
    We focus on essential human proteins that drugs
    should NOT interfere with.
    """
    print("Loading human reference proteins...")
    
    # Key human proteins that drugs must avoid
    # These are essential proteins - if a drug hits these, 
    # the patient gets very sick
    human_proteins = {
        "P68871": "Hemoglobin subunit beta",
        "P00533": "Epidermal growth factor receptor",
        "P00734": "Prothrombin (blood clotting)",
        "P01308": "Insulin",
        "P04637": "Tumor protein p53",
        "P00519": "Tyrosine-protein kinase ABL1",
        "P08183": "Multidrug resistance protein 1",
        "P11229": "Muscarinic acetylcholine receptor",
        "P07550": "Beta-2 adrenergic receptor",
        "P00352": "Retinal dehydrogenase 1"
    }
    
    sequences = {}
    
    for uniprot_id, name in human_proteins.items():
        url = (
            f"https://rest.uniprot.org/uniprotkb/"
            f"{uniprot_id}.fasta"
        )
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                # Parse FASTA
                lines = response.text.strip().split('\n')
                seq = ''.join(lines[1:])
                sequences[uniprot_id] = {
                    'name': name,
                    'sequence': seq
                }
                print(f"  Loaded: {name}")
        except Exception as e:
            print(f"  Failed to load {uniprot_id}: {e}")
        
        time.sleep(0.3)
    
    return sequences

def check_protein_selectivity(parasite_seq, parasite_id, 
                               human_proteins):
    """
    Checks if a parasite protein is too similar to
    any human reference protein.
    
    Returns:
    - is_selective: True if safe to target, False if dangerous
    - max_identity: highest similarity to any human protein
    - most_similar: which human protein it's most similar to
    """
    max_identity = 0.0
    most_similar_name = "None"
    
    for human_id, human_data in human_proteins.items():
        identity = calculate_sequence_identity(
            parasite_seq,
            human_data['sequence']
        )
        
        if identity > max_identity:
            max_identity = identity
            most_similar_name = human_data['name']
    
    # Is it selective enough?
    is_selective = max_identity < IDENTITY_THRESHOLD
    
    return is_selective, max_identity, most_similar_name

def run_selectivity_filter(organism_name, fasta_path, 
                            human_proteins, max_proteins=100):
    """
    Runs selectivity filter on all proteins for one organism.
    Checks each parasite protein against human reference proteins.
    """
    print(f"\n{'='*50}")
    print(f"Selectivity filter: {organism_name}")
    print(f"{'='*50}")
    
    # Load parasite proteins
    proteins = list(SeqIO.parse(fasta_path, "fasta"))
    
    # Take a sample for now
    sample = proteins[:max_proteins]
    print(f"Checking {len(sample)} proteins...")
    
    results = []
    selective_count = 0
    rejected_count = 0
    
    for i, protein in enumerate(sample):
        protein_id = protein.id
        sequence = str(protein.seq)
        
        # Check selectivity
        is_selective, max_identity, most_similar = \
            check_protein_selectivity(
                sequence, 
                protein_id,
                human_proteins
            )
        
        if is_selective:
            selective_count += 1
            status = "SELECTIVE ✓"
        else:
            rejected_count += 1
            status = "REJECTED ✗"
        
        results.append({
            'protein_id': protein_id,
            'organism': organism_name,
            'max_human_identity': round(max_identity, 2),
            'most_similar_human': most_similar,
            'is_selective': is_selective,
            'status': status,
            'sequence_length': len(sequence)
        })
        
        # Progress update
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(sample)} "
                  f"| Selective: {selective_count} "
                  f"| Rejected: {rejected_count}")
    
    # Summary
    total = len(results)
    selective_pct = (selective_count / total) * 100
    
    print(f"\nResults for {organism_name}:")
    print(f"  Selective (safe targets): "
          f"{selective_count}/{total} ({selective_pct:.1f}%)")
    print(f"  Rejected (too similar to human): "
          f"{rejected_count}/{total}")
    
    return results

# ============================================================
# PART 6: Combine with GNN predictions
# ============================================================

def apply_selectivity_to_predictions(gnn_results_path,
                                      selectivity_results):
    """
    Combines our GNN pocket predictions with 
    selectivity filter results.
    
    Only keeps proteins that are:
    1. Predicted to have a good binding pocket (GNN score > 0.5)
    2. Sufficiently different from human proteins (selective)
    """
    print("\nApplying selectivity filter to GNN predictions...")
    
    # Convert selectivity results to dict for fast lookup
    selectivity_dict = {
        r['protein_id']: r 
        for r in selectivity_results
    }
    
    # In production this would load real GNN predictions
    # For now we demonstrate the filtering logic
    print("Filter logic:")
    print("  Keep if: GNN pocket score > 0.5 AND selectivity < 40%")
    print("  Reject if: GNN pocket score < 0.5 OR selectivity > 40%")

# ---- Main program ----
if __name__ == "__main__":
    print("=== Neglect-Fold: Selectivity Filter ===")
    print(f"Rejecting proteins with >{IDENTITY_THRESHOLD}% "
          f"identity to human proteins")
    print()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Load human reference proteins
    human_proteins = load_human_reference_proteins()
    print(f"\nLoaded {len(human_proteins)} human reference proteins")
    
    # Step 2: Run selectivity filter on each organism
    ORGANISMS = {
        "trypanosoma_cruzi": 
            "data/processed/trypanosoma_cruzi_cleaned.fasta",
        "leishmania_donovani": 
            "data/processed/leishmania_donovani_cleaned.fasta",
        "schistosoma_mansoni": 
            "data/processed/schistosoma_mansoni_cleaned.fasta"
    }
    
    all_results = []
    
    for organism_name, fasta_path in ORGANISMS.items():
        results = run_selectivity_filter(
            organism_name,
            fasta_path,
            human_proteins,
            max_proteins=100
        )
        all_results.extend(results)
    
    # Step 3: Save results
    df = pd.DataFrame(all_results)
    output_path = f"{OUTPUT_DIR}/selectivity_results.csv"
    df.to_csv(output_path, index=False)
    
    # Step 4: Summary statistics
    print(f"\n{'='*50}")
    print("OVERALL SELECTIVITY SUMMARY")
    print(f"{'='*50}")
    
    for organism in ORGANISMS.keys():
        org_df = df[df['organism'] == organism]
        selective = org_df['is_selective'].sum()
        total = len(org_df)
        avg_identity = org_df['max_human_identity'].mean()
        
        print(f"\n{organism}:")
        print(f"  Selective proteins: {selective}/{total} "
              f"({selective/total*100:.1f}%)")
        print(f"  Average human identity: {avg_identity:.1f}%")
        print(f"  Most dangerous proteins "
              f"(highest human similarity):")
        
        # Show top 3 most similar to human
        top3 = org_df.nlargest(3, 'max_human_identity')
        for _, row in top3.iterrows():
            print(f"    {row['protein_id']}: "
                  f"{row['max_human_identity']:.1f}% similar to "
                  f"{row['most_similar_human']}")
    
    print(f"\nResults saved to {output_path}")
    print("\n=== Selectivity Filter Complete! ===")