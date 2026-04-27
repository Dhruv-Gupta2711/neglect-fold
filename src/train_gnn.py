# =============================================================
# Neglect-Fold | Phase 3: Train the Pocket Detection GNN
# =============================================================
# This script teaches our GNN to detect binding pockets.
# 
# The training loop:
# 1. Show the GNN a protein with known pocket labels
# 2. GNN makes predictions
# 3. Calculate how wrong it was (loss)
# 4. Adjust parameters to do better (backpropagation)
# 5. Repeat until accurate

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from Bio import PDB
import io
import warnings
warnings.filterwarnings('ignore')

# Import our GNN from the previous file
from pocket_gnn import PocketDetectionGNN, pdb_to_graph, AMINO_ACIDS

# ============================================================
# PART 1: Load Training Data
# ============================================================

def load_labeled_graph(pdb_path, labels_dict):
    """
    Converts a PDB file to a graph AND adds pocket labels.
    
    labels_dict: {residue_id: 0 or 1}
    1 = part of binding pocket
    0 = not part of binding pocket
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    
    residues = []
    residue_ids = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] != ' ':
                    continue
                if 'CA' not in residue:
                    continue
                
                ca = residue['CA']
                pos = ca.get_vector().get_array()
                res_name = residue.get_resname()
                aa_idx = AMINO_ACIDS.get(res_name, 20)
                plddt = ca.get_bfactor()
                
                residues.append({
                    'position': pos,
                    'aa_type': aa_idx,
                    'plddt': plddt
                })
                residue_ids.append(residue.get_id())
    
    if len(residues) < 10:
        return None
    
    # Build node features
    positions = []
    node_features = []
    pocket_labels = []
    
    for i, (res, res_id) in enumerate(zip(residues, residue_ids)):
        one_hot = [0] * 21
        one_hot[res['aa_type']] = 1
        plddt_norm = res['plddt'] / 100.0
        features = one_hot + [plddt_norm]
        node_features.append(features)
        positions.append(res['position'])
        
        # Get label for this residue
        label = labels_dict.get(res_id, 0)
        pocket_labels.append(label)
    
    # Build edges
    positions_arr = np.array(positions)
    edge_list = []
    
    for i in range(len(residues)):
        for j in range(i + 1, len(residues)):
            diff = positions_arr[i] - positions_arr[j]
            distance = np.sqrt(np.sum(diff ** 2))
            if distance <= 8.0:
                edge_list.append([i, j])
                edge_list.append([j, i])
    
    if not edge_list:
        return None
    
    from torch_geometric.data import Data
    
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    y = torch.tensor(pocket_labels, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, y=y)

def load_all_training_data():
    """
    Loads all 10 labeled protein structures as graphs.
    """
    print("Loading training data...")
    
    training_dir = "data/processed/training"
    summary_path = f"{training_dir}/training_data_summary.csv"
    
    if not os.path.exists(summary_path):
        print("Training data not found! Run prepare_training_data.py first.")
        return []
    
    df = pd.read_csv(summary_path)
    
    # We need to re-run pocket detection to get labels
    # Load labels from the PDB files
    from prepare_training_data import (
        find_binding_pocket_residues, 
        download_pdb
    )
    
    graphs = []
    
    for _, row in df.iterrows():
        pdb_id = row['pdb_id']
        pdb_path = row['pdb_path']
        
        if not os.path.exists(pdb_path):
            continue
        
        # Re-detect pocket to get labels
        with open(pdb_path) as f:
            pdb_content = f.read()
        
        residues, labels = find_binding_pocket_residues(pdb_content)
        
        if labels is None:
            continue
        
        # Convert to labeled graph
        graph = load_labeled_graph(pdb_path, labels)
        
        if graph is not None:
            graphs.append((pdb_id, graph))
            pocket_count = int(graph.y.sum().item())
            total = graph.y.shape[0]
            print(f"  Loaded {pdb_id}: {total} residues, {pocket_count} pocket")
    
    print(f"Total graphs loaded: {len(graphs)}")
    return graphs

# ============================================================
# PART 2: The Training Loop
# ============================================================

def calculate_metrics(predictions, labels):
    """
    Calculates how well our model is performing.
    
    Precision: of residues we predicted as pocket, 
               how many actually are?
    Recall: of actual pocket residues, 
            how many did we find?
    F1: harmonic mean of precision and recall
    """
    pred_binary = (predictions > 0.5).float()
    
    tp = ((pred_binary == 1) & (labels == 1)).sum().item()
    fp = ((pred_binary == 1) & (labels == 0)).sum().item()
    fn = ((pred_binary == 0) & (labels == 1)).sum().item()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return precision, recall, f1

def train_model(graphs, num_epochs=50):
    """
    The main training loop.
    
    For each epoch:
    - Go through all proteins
    - Make predictions
    - Calculate loss
    - Backpropagate
    - Track progress
    """
    print("\n=== Starting Training ===")
    print(f"Training on {len(graphs)} proteins")
    print(f"Epochs: {num_epochs}")
    print()
    
    # Create model
    model = PocketDetectionGNN()
    
    # Optimizer - AdamW adjusts parameters during training
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=0.001,           # Learning rate - how big each step is
        weight_decay=0.01   # Prevents overfitting
    )
    
    # Loss function - Binary Cross Entropy
    # Measures how wrong our predictions are
    # pos_weight handles imbalanced data (few pocket vs many non-pocket)
    
    criterion = nn.BCELoss()
    
    # Learning rate scheduler - reduces lr when progress stalls
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=20, 
        gamma=0.5
    )
    
    # Track metrics over time
    history = {
        'epoch': [],
        'loss': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    best_f1 = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        
        epoch_losses = []
        epoch_preds = []
        epoch_labels = []
        
        # Shuffle training data each epoch
        import random
        random.shuffle(graphs)
        
        for pdb_id, graph in graphs:
            # Zero gradients from previous step
            optimizer.zero_grad()
            
            # Create batch tensor
            batch = torch.zeros(
                graph.x.shape[0], 
                dtype=torch.long
            )
            
            # Forward pass - make predictions
            predictions = model(
                graph.x, 
                graph.edge_index, 
                batch
            ).squeeze()
            
            # Get labels
            labels = graph.y
            
            # Calculate loss
            loss = criterion(predictions, labels)
            
            # Backward pass - calculate gradients
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=1.0
            )
            
            # Update parameters
            optimizer.step()
            
            epoch_losses.append(loss.item())
            epoch_preds.append(predictions.detach())
            epoch_labels.append(labels)
        
        # Update learning rate
        scheduler.step()
        
        # Calculate epoch metrics
        avg_loss = np.mean(epoch_losses)
        all_preds = torch.cat(epoch_preds)
        all_labels = torch.cat(epoch_labels)
        precision, recall, f1 = calculate_metrics(all_preds, all_labels)
        
        # Save history
        history['epoch'].append(epoch + 1)
        history['loss'].append(avg_loss)
        history['precision'].append(precision)
        history['recall'].append(recall)
        history['f1'].append(f1)
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            best_model_state = model.state_dict().copy()
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Precision: {precision:.3f} | "
                  f"Recall: {recall:.3f} | "
                  f"F1: {f1:.3f}")
    
    print(f"\nBest F1 score: {best_f1:.3f}")
    
    return model, history, best_model_state

def plot_training_history(history):
    """
    Plots the training progress over epochs.
    Shows how loss decreases and metrics improve.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Neglect-Fold GNN Training Progress', fontsize=14)
    
    # Plot loss
    axes[0].plot(history['epoch'], history['loss'], 
                 'b-', linewidth=2, label='Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Over Training')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot metrics
    axes[1].plot(history['epoch'], history['precision'], 
                 'g-', linewidth=2, label='Precision')
    axes[1].plot(history['epoch'], history['recall'], 
                 'r-', linewidth=2, label='Recall')
    axes[1].plot(history['epoch'], history['f1'], 
                 'b-', linewidth=2, label='F1 Score')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Metrics Over Training')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('results/figures/training_progress.png', dpi=150)
    print("Training plot saved to results/figures/training_progress.png")
    plt.show()

# ============================================================
# PART 3: Save and Test the Model
# ============================================================

def save_model(model, best_state, history):
    """
    Saves the trained model for later use.
    """
    os.makedirs('models', exist_ok=True)
    
    # Save best model
    torch.save(best_state, 'models/pocket_gnn_best.pt')
    print("Best model saved to models/pocket_gnn_best.pt")
    
    # Save training history
    df = pd.DataFrame(history)
    df.to_csv('models/training_history.csv', index=False)
    print("Training history saved to models/training_history.csv")

# ---- Main program ----
if __name__ == "__main__":
    print("=== Neglect-Fold: Train Pocket Detection GNN ===")
    print("Architecture: Local GNN + Global GNN + Fusion")
    print()
    
    # Load training data
    graphs = load_all_training_data()
    
    if len(graphs) < 2:
        print("Not enough training data!")
        exit()
    
    # Train the model
    model, history, best_state = train_model(graphs, num_epochs=50)
    
    # Save model and results
    save_model(model, best_state, history)
    
    # Plot training progress
    plot_training_history(history)
    
    print("\n=== Training Complete! ===")
    print(f"Final F1 Score: {history['f1'][-1]:.3f}")
    print(f"Best F1 Score: {max(history['f1']):.3f}")
    print("\nNext step: evaluate on test proteins")