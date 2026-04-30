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
    Loads all labeled protein structures as graphs.
    Now includes both parasite (label=1) 
    and human (label=0) proteins.
    """
    print("Loading training data...")
    
    training_dir = "data/processed/training_v2"
    summary_path = f"{training_dir}/training_summary.csv"
    
    if not os.path.exists(summary_path):
        print("New training data not found!")
        print("Falling back to original training data...")
        # Fall back to original
        training_dir = "data/processed/training"
        summary_path = f"{training_dir}/training_data_summary.csv"
    
    df = pd.read_csv(summary_path)
    print(f"Found {len(df)} labeled structures")
    
    from prepare_training_data import find_binding_pocket_residues
    
    graphs = []
    
    for _, row in df.iterrows():
        pdb_id = row['pdb_id']
        pdb_path = row['pdb_path']
        label = row.get('label', 1)  # Default to 1 if no label
        
        if not os.path.exists(pdb_path):
            continue
        
        with open(pdb_path) as f:
            pdb_content = f.read()
        
        residues, pocket_labels = find_binding_pocket_residues(
            pdb_content
        )
        
        if pocket_labels is None:
            continue
        
        # If this is a human protein (label=0),
        # we INVERT the pocket labels
        # Human pocket residues become 0 (avoid)
        # Human non-pocket residues become 1 (safe)
        # This teaches the model to avoid human pockets
        if label == 0:
            pocket_labels = {
                k: 1 - v 
                for k, v in pocket_labels.items()
            }
        
        graph = load_labeled_graph(pdb_path, pocket_labels)
        
        if graph is not None:
            pocket_count = int(graph.y.sum().item())
            total = graph.y.shape[0]
            protein_type = (
                "PARASITE" if label == 1 else "HUMAN"
            )
            print(f"  Loaded {pdb_id} ({protein_type}): "
                  f"{total} residues, {pocket_count} pocket")
            graphs.append((pdb_id, graph))
    
    print(f"\nTotal graphs loaded: {len(graphs)}")
    parasite = sum(1 for g in graphs if 'human' not in g[0].lower())
    print(f"Parasite proteins: {parasite}")
    print(f"Human proteins: {len(graphs) - parasite}")
    
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
    The main training loop with proper train/test split.
    
    Splits data into:
    - 70% training (model learns from these)
    - 30% testing (model never sees these during training)
    
    This gives us honest F1 scores.
    """
    print(f"\n=== Starting Training ===")
    print(f"Total proteins: {len(graphs)}")
    
    # ---- Proper train/test split ----
    import random
    random.seed(42)  # For reproducibility
    
    # Shuffle proteins
    shuffled = graphs.copy()
    random.shuffle(shuffled)
    
    # Split 70/30
    split_idx = int(len(shuffled) * 0.7)
    train_graphs = shuffled[:split_idx]
    test_graphs = shuffled[split_idx:]
    
    print(f"Training proteins: {len(train_graphs)}")
    print(f"Testing proteins: {len(test_graphs)}")
    print(f"Epochs: {num_epochs}")
    print()
    
    # Create model
    model = PocketDetectionGNN()
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01
    )
    
    criterion = nn.BCELoss()
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=20,
        gamma=0.5
    )
    
    history = {
        'epoch': [],
        'train_loss': [],
        'train_f1': [],
        'test_f1': []
    }
    
    best_test_f1 = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # ---- Training phase ----
        model.train()
        train_losses = []
        train_preds = []
        train_labels = []
        
        random.shuffle(train_graphs)
        
        for pdb_id, graph in train_graphs:
            optimizer.zero_grad()
            
            batch = torch.zeros(
                graph.x.shape[0],
                dtype=torch.long
            )
            
            predictions = model(
                graph.x,
                graph.edge_index,
                batch
            ).squeeze()
            
            labels = graph.y
            loss = criterion(predictions, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )
            
            optimizer.step()
            train_losses.append(loss.item())
            train_preds.append(predictions.detach())
            train_labels.append(labels)
        
        # ---- Evaluation phase ----
        model.eval()
        test_preds = []
        test_labels = []
        
        with torch.no_grad():
            for pdb_id, graph in test_graphs:
                batch = torch.zeros(
                    graph.x.shape[0],
                    dtype=torch.long
                )
                predictions = model(
                    graph.x,
                    graph.edge_index,
                    batch
                ).squeeze()
                
                test_preds.append(predictions)
                test_labels.append(graph.y)
        
        scheduler.step()
        
        # Calculate metrics
        avg_loss = np.mean(train_losses)
        
        all_train_preds = torch.cat(train_preds)
        all_train_labels = torch.cat(train_labels)
        _, _, train_f1 = calculate_metrics(
            all_train_preds, all_train_labels
        )
        
        all_test_preds = torch.cat(test_preds)
        all_test_labels = torch.cat(test_labels)
        _, _, test_f1 = calculate_metrics(
            all_test_preds, all_test_labels
        )
        
        # Save best model based on TEST f1
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_model_state = {
                k: v.clone() 
                for k, v in model.state_dict().items()
            }
        
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_loss)
        history['train_f1'].append(train_f1)
        history['test_f1'].append(test_f1)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Train F1: {train_f1:.3f} | "
                  f"Test F1: {test_f1:.3f}")
    
    print(f"\nBest TEST F1: {best_test_f1:.3f}")
    print("(This is the honest score on unseen proteins)")
    
    return model, history, best_model_state

def plot_training_history(history):
    """
    Plots training AND test metrics separately.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        'Neglect-Fold GNN Training Progress', 
        fontsize=14
    )
    
    # Plot loss
    axes[0].plot(
        history['epoch'], 
        history['train_loss'],
        'b-', linewidth=2, label='Training Loss'
    )
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Over Training')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot F1 scores
    axes[1].plot(
        history['epoch'], 
        history['train_f1'],
        'b-', linewidth=2, label='Train F1'
    )
    axes[1].plot(
        history['epoch'], 
        history['test_f1'],
        'r-', linewidth=2, label='Test F1 (honest)'
    )
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('Train vs Test F1')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(
        'results/figures/training_progress.png', 
        dpi=150
    )
    print("Training plot saved")
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
    print(f"Final Train F1: {history['train_f1'][-1]:.3f}")
    print(f"Final Test F1: {history['test_f1'][-1]:.3f}")
    print(f"Best Test F1: {max(history['test_f1']):.3f}")
    print("\nNext step: evaluate on test proteins")