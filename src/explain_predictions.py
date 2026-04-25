# =============================================================
# Neglect-Fold | Phase 5: SHAP Explainability (Fixed)
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import warnings
warnings.filterwarnings('ignore')

def load_results():
    """Loads our top 20 drug target results."""
    df = pd.read_csv('results/top20_drug_targets.csv')
    return df

def compute_shap_values(df):
    """
    Computes SHAP-like contribution values for each protein.
    
    Based on our scoring formula:
    final_score = 0.5 * pocket_score 
                + 0.3 * affinity_score 
                + 0.2 * selectivity_score
    
    Each feature's SHAP value = its weighted contribution
    relative to the average.
    """
    # Normalize features to 0-1
    pocket_scores = df['pocket_score'].values
    
    # pIC50 normalized
    pic50_values = df['best_compound_pIC50'].fillna(5).values
    affinity_scores = (pic50_values - pic50_values.min()) / \
                     (pic50_values.max() - pic50_values.min() + 1e-8)
    
    # Selectivity (inverse of human similarity)
    selectivity = 1 - (df['human_similarity_pct'].values / 100)
    
    # Compute weighted contributions
    pocket_contrib = 0.5 * pocket_scores
    affinity_contrib = 0.3 * affinity_scores
    selectivity_contrib = 0.2 * selectivity
    
    # Mean contributions (baseline)
    mean_pocket = np.mean(pocket_contrib)
    mean_affinity = np.mean(affinity_contrib)
    mean_selectivity = np.mean(selectivity_contrib)
    
    # SHAP values = deviation from mean contribution
    shap_pocket = pocket_contrib - mean_pocket
    shap_affinity = affinity_contrib - mean_affinity
    shap_selectivity = selectivity_contrib - mean_selectivity
    
    return {
        'pocket_score': shap_pocket,
        'binding_affinity': shap_affinity,
        'selectivity': shap_selectivity
    }

def plot_shap_summary(df, shap_values):
    """
    Creates a summary bar chart showing average feature
    importance across all 20 proteins.
    """
    features = list(shap_values.keys())
    mean_abs_shap = [
        np.mean(np.abs(shap_values[f])) 
        for f in features
    ]
    
    # Sort by importance
    sorted_idx = np.argsort(mean_abs_shap)
    sorted_features = [features[i] for i in sorted_idx]
    sorted_values = [mean_abs_shap[i] for i in sorted_idx]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = ['#2196F3', '#4CAF50', '#FF5722']
    bars = ax.barh(
        sorted_features,
        sorted_values,
        color=colors,
        edgecolor='white',
        linewidth=0.5,
        height=0.5
    )
    
    # Add value labels
    for bar, val in zip(bars, sorted_values):
        ax.text(
            val + 0.001,
            bar.get_y() + bar.get_height()/2,
            f'{val:.4f}',
            va='center',
            fontsize=11,
            fontweight='bold'
        )
    
    ax.set_xlabel(
        'Mean |SHAP Value| (Average Impact on Score)',
        fontsize=12
    )
    ax.set_title(
        'Neglect-Fold: Feature Importance for Drug Target Ranking\n'
        'Which factors matter most in identifying drug targets?',
        fontsize=13,
        fontweight='bold'
    )
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(
        'results/figures/shap_summary.png',
        dpi=150,
        bbox_inches='tight'
    )
    print("Saved: results/figures/shap_summary.png")
    plt.show()

def plot_waterfall(df, shap_values, rank):
    """
    Creates a waterfall plot for one specific protein.
    Shows exactly why it got its score.
    """
    idx = rank - 1
    protein_id = df.iloc[idx]['protein_id']
    organism = df.iloc[idx]['organism']
    final_score = df.iloc[idx]['final_score']
    
    # Get SHAP values for this protein
    features = {
        'Pocket Score\n(binding site quality)': 
            shap_values['pocket_score'][idx],
        'Binding Affinity\n(drug compound fit)': 
            shap_values['binding_affinity'][idx],
        'Selectivity\n(safe vs humans)': 
            shap_values['selectivity'][idx]
    }
    
    # Sort by absolute value
    sorted_features = sorted(
        features.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    feature_names = [f[0] for f in sorted_features]
    shap_vals = [f[1] for f in sorted_features]
    
    colors = [
        '#44aa44' if v >= 0 else '#ff4444' 
        for v in shap_vals
    ]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    bars = ax.barh(
        feature_names,
        shap_vals,
        color=colors,
        edgecolor='white',
        linewidth=0.5,
        height=0.4
    )
    
    # Add value labels
    for bar, val in zip(bars, shap_vals):
        ax.text(
            val + (0.001 if val >= 0 else -0.001),
            bar.get_y() + bar.get_height()/2,
            f'{val:+.4f}',
            va='center',
            ha='left' if val >= 0 else 'right',
            fontsize=11,
            fontweight='bold'
        )
    
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('SHAP Value (Impact on Final Score)', fontsize=12)
    ax.set_title(
        f'Why Rank {rank}: {protein_id} is a top drug target\n'
        f'Organism: {organism.replace("_", " ").title()} | '
        f'Final Score: {final_score:.4f}',
        fontsize=12,
        fontweight='bold'
    )
    
    # Legend
    pos_patch = mpatches.Patch(
        color='#44aa44', label='Increases score'
    )
    neg_patch = mpatches.Patch(
        color='#ff4444', label='Decreases score'
    )
    ax.legend(handles=[pos_patch, neg_patch], fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    save_path = (
        f'results/figures/shap_waterfall_'
        f'rank{rank}_{protein_id}.png'
    )
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()

def print_explanation_report(df, shap_values):
    """
    Prints a text explanation for each top protein.
    """
    print("\n" + "="*60)
    print("NEGLECT-FOLD: WHY EACH PROTEIN WAS RANKED")
    print("="*60)
    
    rows = []
    
    for i, (_, row) in enumerate(df.iterrows()):
        protein_id = row['protein_id']
        organism = row['organism']
        final_score = row['final_score']
        
        pocket_shap = shap_values['pocket_score'][i]
        affinity_shap = shap_values['binding_affinity'][i]
        selectivity_shap = shap_values['selectivity'][i]
        
        # Find main driver
        contributions = {
            'pocket quality': pocket_shap,
            'drug binding': affinity_shap,
            'human selectivity': selectivity_shap
        }
        main_driver = max(contributions, key=lambda x: contributions[x])
        main_concern = min(contributions, key=lambda x: contributions[x])
        
        print(f"\nRank {i+1}: {protein_id}")
        print(f"  Disease: {organism.replace('_', ' ').title()}")
        print(f"  Score: {final_score:.4f}")
        print(f"  ✓ Main strength: {main_driver} "
              f"(+{contributions[main_driver]:.4f})")
        if contributions[main_concern] < 0:
            print(f"  ✗ Main weakness: {main_concern} "
                  f"({contributions[main_concern]:.4f})")
        
        rows.append({
            'rank': i + 1,
            'protein_id': protein_id,
            'organism': organism,
            'final_score': final_score,
            'pocket_shap': round(pocket_shap, 4),
            'affinity_shap': round(affinity_shap, 4),
            'selectivity_shap': round(selectivity_shap, 4),
            'main_strength': main_driver
        })
    
    # Save report
    report_df = pd.DataFrame(rows)
    report_df.to_csv('results/shap_explanations.csv', index=False)
    print(f"\nFull report saved to results/shap_explanations.csv")
    
    return report_df

# ============================================================
# Main Program
# ============================================================

if __name__ == "__main__":
    print("=== Neglect-Fold: SHAP Explainability ===\n")
    
    # Load results
    df = load_results()
    print(f"Loaded {len(df)} drug target candidates")
    print(f"Score range: {df['final_score'].min():.4f} - "
          f"{df['final_score'].max():.4f}")
    
    # Compute SHAP values
    print("\nComputing SHAP values...")
    shap_values = compute_shap_values(df)
    print("Done!")
    
    # Summary plot
    print("\nGenerating summary plot...")
    plot_shap_summary(df, shap_values)
    
    # Waterfall plots for top 5
    print("\nGenerating waterfall plots for top 5 proteins...")
    for rank in range(1, 6):
        plot_waterfall(df, shap_values, rank)
    
    # Text report
    report = print_explanation_report(df, shap_values)
    
    print("\n=== SHAP Analysis Complete! ===")
    print("Files saved:")
    print("  results/figures/shap_summary.png")
    print("  results/figures/shap_waterfall_rank1-5.png")
    print("  results/shap_explanations.csv")