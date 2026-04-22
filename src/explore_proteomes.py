# =============================================================
# Neglect-Fold | Phase 1: Explore and Summarize Proteomes
# =============================================================
# Before doing any ML, we need to understand our data.
# This script reads the FASTA files and gives us statistics
# about the proteins we downloaded.

import os
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO

# ---- Our three organisms ----
ORGANISMS = {
    "trypanosoma_cruzi": "data/raw/trypanosoma_cruzi.fasta",
    "leishmania_donovani": "data/raw/leishmania_donovani.fasta",
    "schistosoma_mansoni": "data/raw/schistosoma_mansoni.fasta"
}

def parse_fasta(filepath):
    """
    Reads a FASTA file and returns a list of proteins.
    Each protein has an id, name, and sequence.
    """
    proteins = []
    
    for record in SeqIO.parse(filepath, "fasta"):
        proteins.append({
            "id": record.id,
            "name": record.description,
            "sequence": str(record.seq),
            "length": len(record.seq)
        })
    
    return proteins

def explore_organism(organism_name, filepath):
    """
    Explores the proteins for one organism and prints statistics.
    """
    print(f"\n{'='*50}")
    print(f"Organism: {organism_name}")
    print(f"{'='*50}")
    
    # Parse the FASTA file
    proteins = parse_fasta(filepath)
    
    # Convert to a pandas DataFrame
    # A DataFrame is like an Excel spreadsheet in Python
    df = pd.DataFrame(proteins)
    
    # Basic statistics
    print(f"Total proteins: {len(df)}")
    print(f"Shortest protein: {df['length'].min()} amino acids")
    print(f"Longest protein: {df['length'].max()} amino acids")
    print(f"Average length: {df['length'].mean():.1f} amino acids")
    
    # How many are very short? (less than 50 amino acids)
    very_short = len(df[df['length'] < 50])
    print(f"Very short proteins (<50 aa): {very_short}")
    
    # How many are very long? (more than 2000 amino acids)
    very_long = len(df[df['length'] > 2000])
    print(f"Very long proteins (>2000 aa): {very_long}")
    
    return df

def plot_length_distributions(dataframes):
    """
    Creates a plot showing protein length distributions
    for all three organisms.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Protein Length Distributions', fontsize=16)
    
    colors = ['#2196F3', '#4CAF50', '#FF5722']
    
    for i, (organism_name, df) in enumerate(dataframes.items()):
        # Only plot proteins under 2000 aa for readability
        lengths = df[df['length'] < 2000]['length']
        
        axes[i].hist(lengths, bins=50, color=colors[i], alpha=0.7)
        axes[i].set_title(organism_name.replace('_', ' ').title())
        axes[i].set_xlabel('Protein Length (amino acids)')
        axes[i].set_ylabel('Number of Proteins')
    
    plt.tight_layout()
    plt.savefig('results/figures/protein_length_distributions.png', dpi=150)
    print("\nPlot saved to results/figures/protein_length_distributions.png")
    plt.show()

# ---- Main program ----
if __name__ == "__main__":
    print("=== Neglect-Fold: Proteome Explorer ===")
    
    dataframes = {}
    
    for organism_name, filepath in ORGANISMS.items():
        df = explore_organism(organism_name, filepath)
        dataframes[organism_name] = df
    
    # Save summary to CSV
    print("\nSaving summary statistics...")
    summary_rows = []
    for organism_name, df in dataframes.items():
        summary_rows.append({
            "organism": organism_name,
            "total_proteins": len(df),
            "min_length": df['length'].min(),
            "max_length": df['length'].max(),
            "mean_length": round(df['length'].mean(), 1)
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv('data/processed/proteome_summary.csv', index=False)
    print("Summary saved to data/processed/proteome_summary.csv")
    
    # Plot length distributions
    plot_length_distributions(dataframes)
    
    print("\nExploration complete!")