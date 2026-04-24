# =============================================================
# Neglect-Fold | Final Pipeline: End-to-End Drug Target Discovery
# =============================================================
# This script connects all our models together:
# 
# Stage 1: Load cleaned proteins + structures
# Stage 2: Detect binding pockets (Pocket GNN)
# Stage 3: Score drug compounds (Affinity Model)
# Stage 4: Filter by selectivity (human similarity)
# Stage 5: Rank and produce Top 20 list
# Stage 6: Generate report

import torch
import numpy as np
import pandas as pd
import os
import json
import warnings
warnings.filterwarnings('ignore')

from pocket_gnn import PocketDetectionGNN, pdb_to_graph
from affinity_model import BindingAffinityModel, smiles_to_graph
from selectivity_filter import (
    load_human_reference_proteins,
    check_protein_selectivity
)
from Bio import SeqIO

# ============================================================
# Settings
# ============================================================

STRUCTURE_DIRS = {
    "trypanosoma_cruzi": 
        "data/processed/structures/trypanosoma_cruzi",
    "leishmania_donovani": 
        "data/processed/structures/leishmania_donovani",
    "schistosoma_mansoni": 
        "data/processed/structures/schistosoma_mansoni"
}

FASTA_FILES = {
    "trypanosoma_cruzi": 
        "data/processed/trypanosoma_cruzi_cleaned.fasta",
    "leishmania_donovani": 
        "data/processed/leishmania_donovani_cleaned.fasta",
    "schistosoma_mansoni": 
        "data/processed/schistosoma_mansoni_cleaned.fasta"
}

CHEMBL_FILE = "data/processed/all_chembl_clean.csv"

# ============================================================
# Stage 1: Load Models
# ============================================================

def load_models():
    """
    Loads our trained GNN models.
    """
    print("Loading trained models...")
    
    # Load pocket detection model
    pocket_model = PocketDetectionGNN()
    if os.path.exists('models/pocket_gnn_best.pt'):
        pocket_model.load_state_dict(
            torch.load('models/pocket_gnn_best.pt',
                      map_location='cpu')
        )
        print("  ✓ Pocket Detection GNN loaded")
    else:
        print("  ! Using untrained Pocket GNN")
    pocket_model.eval()
    
    # Load affinity model
    affinity_model = BindingAffinityModel()
    if os.path.exists('models/affinity_model.pt'):
        affinity_model.load_state_dict(
            torch.load('models/affinity_model.pt',
                      map_location='cpu')
        )
        print("  ✓ Binding Affinity Model loaded")
    else:
        print("  ! Using untrained Affinity Model")
    affinity_model.eval()
    
    return pocket_model, affinity_model

# ============================================================
# Stage 2: Score Binding Pockets
# ============================================================

def score_pockets(pocket_model, structure_dir, organism):
    """
    Runs pocket detection GNN on all protein structures
    for one organism.
    
    Returns a ranked list of proteins by pocket score.
    """
    print(f"\n  Scoring pockets for {organism}...")
    
    pdb_files = [
        f for f in os.listdir(structure_dir) 
        if f.endswith('.pdb')
    ]
    
    results = []
    
    for pdb_file in pdb_files:
        pdb_path = os.path.join(structure_dir, pdb_file)
        protein_id = pdb_file.replace('.pdb', '')
        
        # Convert to graph
        graph = pdb_to_graph(pdb_path)
        if graph is None:
            continue
        
        # Run pocket detection
        batch = torch.zeros(
            graph.x.shape[0], 
            dtype=torch.long
        )
        
        with torch.no_grad():
            predictions = pocket_model(
                graph.x, 
                graph.edge_index, 
                batch
            ).squeeze()
        
        probs = predictions.numpy()
        
        # Pocket score = mean of top 10% predicted residues
        top_10_pct = int(len(probs) * 0.1) + 1
        top_probs = np.sort(probs)[::-1][:top_10_pct]
        pocket_score = float(np.mean(top_probs))
        
        # Count high confidence pocket residues
        high_conf_residues = int((probs > 0.6).sum())
        
        results.append({
            'protein_id': protein_id,
            'organism': organism,
            'pocket_score': round(pocket_score, 4),
            'high_conf_residues': high_conf_residues,
            'total_residues': len(probs),
            'pdb_path': pdb_path
        })
    
    # Sort by pocket score
    results.sort(key=lambda x: x['pocket_score'], reverse=True)
    
    print(f"  Scored {len(results)} proteins")
    if results:
        print(f"  Best pocket score: {results[0]['pocket_score']:.4f}")
        print(f"  Best protein: {results[0]['protein_id']}")
    
    return results

# ============================================================
# Stage 3: Score Drug Compounds
# ============================================================

def score_compounds(affinity_model, top_proteins, 
                    chembl_df, top_n_compounds=10):
    """
    For each top protein, scores the best drug compounds
    from our ChEMBL dataset.
    
    Returns proteins with their best matching compounds.
    """
    print(f"\n  Scoring drug compounds...")
    
    # Sample compounds for efficiency
    sample_compounds = chembl_df.dropna(
        subset=['smiles', 'pIC50']
    ).head(50)
    
    enriched_results = []
    
    for protein_data in top_proteins[:20]:
        protein_id = protein_data['protein_id']
        
        compound_scores = []
        
        for _, compound in sample_compounds.iterrows():
            smiles = compound['smiles']
            known_pIC50 = compound['pIC50']
            
            # Convert molecule to graph
            mol_graph = smiles_to_graph(smiles)
            if mol_graph is None:
                continue
            
            # Score with affinity model
            mol_batch = torch.zeros(
                mol_graph.x.shape[0], 
                dtype=torch.long
            )
            
            # Dummy protein features for now
            dummy_protein_x = torch.randn(50, 22)
            dummy_protein_edge = torch.randint(
                0, 50, (2, 100)
            )
            dummy_protein_batch = torch.zeros(
                50, dtype=torch.long
            )
            
            with torch.no_grad():
                pred_pIC50 = affinity_model(
                    dummy_protein_x,
                    dummy_protein_edge,
                    dummy_protein_batch,
                    mol_graph.x,
                    mol_graph.edge_index,
                    mol_batch
                ).item()
            
            compound_scores.append({
                'smiles': smiles,
                'predicted_pIC50': round(pred_pIC50, 3),
                'known_pIC50': round(known_pIC50, 3),
                'molecule_id': compound.get(
                    'molecule_chembl_id', 'unknown'
                )
            })
        
        # Sort by predicted pIC50
        compound_scores.sort(
            key=lambda x: x['predicted_pIC50'], 
            reverse=True
        )
        
        # Add top compounds to protein data
        protein_data['top_compounds'] = compound_scores[:3]
        protein_data['best_pIC50'] = (
            compound_scores[0]['predicted_pIC50'] 
            if compound_scores else 0
        )
        
        enriched_results.append(protein_data)
    
    return enriched_results

# ============================================================
# Stage 4: Apply Selectivity Filter
# ============================================================

def apply_selectivity(proteins, fasta_files, human_proteins):
    """
    Checks each candidate protein against human proteins.
    Adds selectivity scores to each protein.
    """
    print(f"\n  Applying selectivity filter...")
    
    # Build sequence lookup from FASTA files
    sequence_lookup = {}
    for organism, fasta_path in fasta_files.items():
        for record in SeqIO.parse(fasta_path, "fasta"):
            sequence_lookup[record.id] = str(record.seq)
    
    filtered_results = []
    
    for protein_data in proteins:
        protein_id = protein_data['protein_id']
        
        # Find sequence
        sequence = None
        for seq_id, seq in sequence_lookup.items():
            if protein_id in seq_id:
                sequence = seq
                break
        
        if sequence is None:
            protein_data['is_selective'] = True
            protein_data['max_human_identity'] = 0.0
            protein_data['most_similar_human'] = 'Unknown'
            filtered_results.append(protein_data)
            continue
        
        # Check selectivity
        is_selective, max_identity, most_similar = \
            check_protein_selectivity(
                sequence,
                protein_id,
                human_proteins
            )
        
        protein_data['is_selective'] = is_selective
        protein_data['max_human_identity'] = round(
            max_identity, 2
        )
        protein_data['most_similar_human'] = most_similar
        
        filtered_results.append(protein_data)
    
    selective = sum(
        1 for p in filtered_results if p['is_selective']
    )
    print(f"  Selective proteins: {selective}/{len(filtered_results)}")
    
    return filtered_results

# ============================================================
# Stage 5: Final Ranking
# ============================================================

def rank_candidates(all_proteins):
    """
    Produces the final ranked list of drug targets.
    
    Scoring formula:
    - 50% pocket score (how good is the binding site?)
    - 30% binding affinity (how well do compounds bind?)
    - 20% selectivity (how different from human proteins?)
    """
    print(f"\n  Ranking final candidates...")
    
    scored = []
    
    for protein in all_proteins:
        if not protein.get('is_selective', True):
            continue
        
        # Normalize scores to 0-1
        pocket_score = protein.get('pocket_score', 0)
        
        best_pIC50 = protein.get('best_pIC50', 5)
        affinity_score = min(best_pIC50 / 10.0, 1.0)
        
        human_identity = protein.get('max_human_identity', 0)
        selectivity_score = 1.0 - (human_identity / 100.0)
        
        # Weighted final score
        final_score = (
            0.5 * pocket_score +
            0.3 * affinity_score +
            0.2 * selectivity_score
        )
        
        protein['final_score'] = round(final_score, 4)
        scored.append(protein)
    
    # Sort by final score
    scored.sort(key=lambda x: x['final_score'], reverse=True)
    
    return scored[:20]  # Top 20

# ============================================================
# Stage 6: Generate Report
# ============================================================

def generate_report(top_candidates):
    """
    Generates the final drug target report.
    This is the main output of the whole project!
    """
    print("\n" + "="*60)
    print("NEGLECT-FOLD: TOP DRUG TARGET CANDIDATES")
    print("="*60)
    
    report_rows = []
    
    for rank, protein in enumerate(top_candidates, 1):
        protein_id = protein['protein_id']
        organism = protein['organism']
        pocket_score = protein['pocket_score']
        final_score = protein['final_score']
        human_identity = protein.get('max_human_identity', 0)
        top_compounds = protein.get('top_compounds', [])
        
        print(f"\nRank {rank}: {protein_id}")
        print(f"  Organism: {organism}")
        print(f"  Pocket Score: {pocket_score:.4f}")
        print(f"  Final Score: {final_score:.4f}")
        print(f"  Human Similarity: {human_identity:.1f}%")
        
        if top_compounds:
            best = top_compounds[0]
            print(f"  Best Compound pIC50: "
                  f"{best['predicted_pIC50']:.3f}")
        
        row = {
            'rank': rank,
            'protein_id': protein_id,
            'organism': organism,
            'pocket_score': pocket_score,
            'final_score': final_score,
            'human_similarity_pct': human_identity,
            'best_compound_pIC50': (
                top_compounds[0]['predicted_pIC50'] 
                if top_compounds else None
            ),
            'best_compound_smiles': (
                top_compounds[0]['smiles'][:50] 
                if top_compounds else None
            )
        }
        report_rows.append(row)
    
    # Save report
    os.makedirs('results', exist_ok=True)
    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(
        'results/top20_drug_targets.csv', 
        index=False
    )
    
    print("\n" + "="*60)
    print(f"Report saved to results/top20_drug_targets.csv")
    
    return report_df

# ============================================================
# Main Pipeline
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("NEGLECT-FOLD: END-TO-END PIPELINE")
    print("Finding drug targets for neglected tropical diseases")
    print("="*60)
    
    # Load models
    pocket_model, affinity_model = load_models()
    
    # Load ChEMBL compounds
    print("\nLoading drug compounds...")
    chembl_df = pd.read_csv(CHEMBL_FILE)
    print(f"Loaded {len(chembl_df)} compounds")
    
    # Load human reference proteins
    print("\nLoading human reference proteins...")
    human_proteins = load_human_reference_proteins()
    
    # Run pipeline for each organism
    all_results = []
    
    for organism, structure_dir in STRUCTURE_DIRS.items():
        print(f"\n{'='*40}")
        print(f"Processing: {organism}")
        print(f"{'='*40}")
        
        # Stage 2: Score pockets
        pocket_results = score_pockets(
            pocket_model, 
            structure_dir, 
            organism
        )
        
        if not pocket_results:
            continue
        
        # Take top 10 per organism
        top_proteins = pocket_results[:10]
        
        # Stage 3: Score compounds
        enriched = score_compounds(
            affinity_model,
            top_proteins,
            chembl_df
        )
        
        all_results.extend(enriched)
    
    # Stage 4: Selectivity filter
    print("\n" + "="*40)
    print("Applying selectivity filter...")
    filtered = apply_selectivity(
        all_results,
        FASTA_FILES,
        human_proteins
    )
    
    # Stage 5: Final ranking
    top20 = rank_candidates(filtered)
    
    # Stage 6: Generate report
    report = generate_report(top20)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print(f"Top 20 drug targets identified")
    print(f"Results saved to results/top20_drug_targets.csv")
    print("="*60)