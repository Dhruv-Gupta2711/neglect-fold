# =============================================================
# Neglect-Fold | Pocket Detection GNN V2
# =============================================================
# Improved dual-output architecture that simultaneously:
# 1. Detects binding pockets (pocket_head)
# 2. Distinguishes parasite vs human pockets (selectivity_head)
#
# This replaces the separate selectivity filter with
# a learned distinction built into the model itself.
#
# Training data:
# - Parasite proteins: pocket_label + selectivity_label=1
# - Human proteins: pocket_label + selectivity_label=0
#
# Final score = pocket_score × selectivity_score
# Only proteins with BOTH a good pocket AND
# parasite-specific features score highly.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from Bio import PDB
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PART 1: Amino acid encoding (same as v1)
# ============================================================

AMINO_ACIDS = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
    'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
    'UNK': 20
}

# ============================================================
# PART 2: PDB to graph conversion (same as v1)
# ============================================================

def pdb_to_graph(pdb_path, distance_threshold=8.0):
    """
    Converts a PDB file into a graph for the GNN.
    Same as v1 — no changes needed here.
    """
    parser = PDB.PDBParser(QUIET=True)
    
    try:
        structure = parser.get_structure('protein', pdb_path)
    except Exception as e:
        return None
    
    residues = []
    
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
    
    if len(residues) < 10:
        return None
    
    num_residues = len(residues)
    node_features = []
    positions = []
    
    for res in residues:
        one_hot = [0] * 21
        one_hot[res['aa_type']] = 1
        plddt_normalized = res['plddt'] / 100.0
        features = one_hot + [plddt_normalized]
        node_features.append(features)
        positions.append(res['position'])
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    edge_list = []
    
    for i in range(num_residues):
        for j in range(i + 1, num_residues):
            diff = np.array(positions[i]) - np.array(positions[j])
            distance = np.sqrt(np.sum(diff ** 2))
            if distance <= distance_threshold:
                edge_list.append([i, j])
                edge_list.append([j, i])
    
    if not edge_list:
        return None
    
    edge_index = torch.tensor(
        edge_list, dtype=torch.long
    ).t().contiguous()
    
    return Data(x=x, edge_index=edge_index)

# ============================================================
# PART 3: Shared GNN backbone
# ============================================================

class LocalGNN(nn.Module):
    """
    Looks at local neighborhood of each residue.
    Same as v1.
    """
    def __init__(self, input_dim=22, hidden_dim=64, output_dim=32):
        super(LocalGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        return x

class GlobalGNN(nn.Module):
    """
    Looks at the whole protein structure.
    Same as v1.
    """
    def __init__(self, input_dim=22, hidden_dim=64, output_dim=32):
        super(GlobalGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.global_pool_linear = nn.Linear(output_dim * 2, output_dim)
    
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        global_mean = global_mean_pool(x, batch)
        global_max = global_max_pool(x, batch)
        global_repr = torch.cat([global_mean, global_max], dim=1)
        global_repr = self.global_pool_linear(global_repr)
        return x, global_repr

# ============================================================
# PART 4: Improved dual-output GNN
# ============================================================

class ImprovedPocketGNN(nn.Module):
    """
    Dual-output GNN with two prediction heads:
    
    Head 1 — Pocket Detection:
    "Is this residue part of a binding pocket?"
    Output: probability per residue (0-1)
    
    Head 2 — Selectivity:
    "Is this protein parasite-specific?"
    Output: single probability for whole protein (0-1)
    1 = parasite protein (good target)
    0 = human protein (avoid)
    
    Final drug target score:
    score = pocket_prob × selectivity_prob
    
    Only proteins with BOTH:
    - Good binding pocket
    - Parasite-specific features
    will score highly.
    """
    def __init__(self):
        super(ImprovedPocketGNN, self).__init__()
        
        # Shared backbone — learns general protein features
        self.local_gnn = LocalGNN(
            input_dim=22,
            hidden_dim=64,
            output_dim=32
        )
        self.global_gnn = GlobalGNN(
            input_dim=22,
            hidden_dim=64,
            output_dim=32
        )
        
        # Head 1: Pocket detection (per residue)
        # Takes local + global features
        self.pocket_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Head 2: Selectivity (per protein)
        # Takes only global features
        # Asks: "is this whole protein parasite-specific?"
        self.selectivity_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, batch):
        """
        Forward pass through both heads.
        
        Returns:
        - pocket_prob: per-residue pocket probability [N, 1]
        - selectivity_prob: per-protein selectivity [B, 1]
          where B = batch size (number of proteins)
        """
        # Shared feature extraction
        local_features = self.local_gnn(x, edge_index)
        node_global, global_repr = self.global_gnn(
            x, edge_index, batch
        )
        
        # Expand global to match residue count
        global_expanded = global_repr[batch]
        
        # Combine local + global for pocket detection
        combined = torch.cat(
            [local_features, global_expanded],
            dim=1
        )
        
        # Head 1: pocket probability per residue
        pocket_prob = self.pocket_head(combined)
        
        # Head 2: selectivity for whole protein
        # Uses global representation only
        selectivity_prob = self.selectivity_head(global_repr)
        
        return pocket_prob, selectivity_prob

# ============================================================
# PART 5: Load labeled training data
# ============================================================

def load_labeled_graph(pdb_path, pocket_labels):
    """
    Converts PDB to graph with pocket labels.
    Same logic as train_gnn.py.
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
    
    positions = []
    node_features = []
    pocket_label_list = []
    
    for res, res_id in zip(residues, residue_ids):
        one_hot = [0] * 21
        one_hot[res['aa_type']] = 1
        plddt_norm = res['plddt'] / 100.0
        features = one_hot + [plddt_norm]
        node_features.append(features)
        positions.append(res['position'])
        label = pocket_labels.get(res_id, 0)
        pocket_label_list.append(label)
    
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
    
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(
        edge_list, dtype=torch.long
    ).t().contiguous()
    y = torch.tensor(pocket_label_list, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, y=y)

def load_all_training_data_v2():
    """
    Loads training data with BOTH:
    - Pocket labels (per residue)
    - Selectivity labels (per protein: 1=parasite, 0=human)
    """
    print("Loading training data for V2 model...")
    
    training_dir = "data/processed/training_full"
    summary_path = f"{training_dir}/training_summary.csv"
    
    if not os.path.exists(summary_path):
        print("Full training data not found!")
        print("Run download_all_training.py first")
        return []
    
    df = pd.read_csv(summary_path)
    print(f"Found {len(df)} labeled structures")
    
    import sys
    sys.path.insert(0, 'src')
    from prepare_training_data import find_binding_pocket_residues
    
    graphs = []
    
    for _, row in df.iterrows():
        pdb_id = row['pdb_id']
        pdb_path = row['pdb_path']
        selectivity_label = row.get('label', 1)
        
        if not os.path.exists(pdb_path):
            continue
        
        with open(pdb_path) as f:
            pdb_content = f.read()
        
        residues, pocket_labels = find_binding_pocket_residues(
            pdb_content
        )
        
        if pocket_labels is None:
            continue
        
        graph = load_labeled_graph(pdb_path, pocket_labels)
        
        if graph is not None:
            graphs.append({
                'pdb_id': pdb_id,
                'graph': graph,
                'selectivity_label': float(selectivity_label)
            })
    
    print(f"Total graphs loaded: {len(graphs)}")
    parasite = sum(
        1 for g in graphs if g['selectivity_label'] == 1
    )
    human = sum(
        1 for g in graphs if g['selectivity_label'] == 0
    )
    print(f"Parasite: {parasite}, Human: {human}")
    
    return graphs

# ============================================================
# PART 6: Training with dual loss
# ============================================================

def calculate_metrics(predictions, labels):
    """Calculates precision, recall, F1."""
    pred_binary = (predictions > 0.5).float()
    tp = ((pred_binary == 1) & (labels == 1)).sum().item()
    fp = ((pred_binary == 1) & (labels == 0)).sum().item()
    fn = ((pred_binary == 0) & (labels == 1)).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1

def train_model_v2(graphs, num_epochs=100):
    """
    Trains the dual-output GNN with two loss functions:
    
    Loss 1 (pocket): BCELoss on per-residue pocket labels
    Loss 2 (selectivity): BCELoss on per-protein organism label
    
    Total loss = 0.6 * pocket_loss + 0.4 * selectivity_loss
    
    Why 0.6/0.4?
    Pocket detection is our primary task (higher weight)
    Selectivity is secondary but important (lower weight)
    """
    print(f"\n=== Training Improved Dual-Output GNN ===")
    print(f"Total proteins: {len(graphs)}")
    
    # Train/test split
    import random
    random.seed(42)
    shuffled = graphs.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * 0.7)
    train_graphs = shuffled[:split_idx]
    test_graphs = shuffled[split_idx:]
    
    print(f"Training: {len(train_graphs)}")
    print(f"Testing: {len(test_graphs)}")
    
    model = ImprovedPocketGNN()
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01
    )
    
    # Two loss functions
    pocket_criterion = nn.BCELoss()
    selectivity_criterion = nn.BCELoss()
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=30, gamma=0.5
    )
    
    history = {
        'epoch': [],
        'train_loss': [],
        'train_pocket_f1': [],
        'test_pocket_f1': [],
        'train_selectivity_acc': [],
        'test_selectivity_acc': []
    }
    
    best_test_f1 = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        
        train_losses = []
        train_pocket_preds = []
        train_pocket_labels = []
        train_sel_preds = []
        train_sel_labels = []
        
        random.shuffle(train_graphs)
        
        for item in train_graphs:
            optimizer.zero_grad()
            
            graph = item['graph']
            sel_label = item['selectivity_label']
            
            batch = torch.zeros(
                graph.x.shape[0],
                dtype=torch.long
            )
            
            # Forward pass — get both outputs
            pocket_prob, sel_prob = model(
                graph.x,
                graph.edge_index,
                batch
            )
            
            pocket_prob = pocket_prob.squeeze()
            sel_prob = sel_prob.squeeze()
            
            # Loss 1: pocket detection
            pocket_loss = pocket_criterion(
                pocket_prob,
                graph.y
            )
            
            # Loss 2: selectivity
            sel_target = torch.tensor(
                [sel_label],
                dtype=torch.float
            )
            sel_loss = selectivity_criterion(
                sel_prob.unsqueeze(0),
                sel_target
            )
            
            # Combined loss
            total_loss = (
                0.6 * pocket_loss +
                0.4 * sel_loss
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )
            optimizer.step()
            
            train_losses.append(total_loss.item())
            train_pocket_preds.append(pocket_prob.detach())
            train_pocket_labels.append(graph.y)
            train_sel_preds.append(sel_prob.detach())
            train_sel_labels.append(sel_label)
        
        # Evaluation
        model.eval()
        test_pocket_preds = []
        test_pocket_labels = []
        test_sel_preds = []
        test_sel_labels = []
        
        with torch.no_grad():
            for item in test_graphs:
                graph = item['graph']
                sel_label = item['selectivity_label']
                
                batch = torch.zeros(
                    graph.x.shape[0],
                    dtype=torch.long
                )
                
                pocket_prob, sel_prob = model(
                    graph.x,
                    graph.edge_index,
                    batch
                )
                
                test_pocket_preds.append(
                    pocket_prob.squeeze()
                )
                test_pocket_labels.append(graph.y)
                test_sel_preds.append(sel_prob.item())
                test_sel_labels.append(sel_label)
        
        scheduler.step()
        
        # Calculate metrics
        avg_loss = np.mean(train_losses)
        
        all_train_preds = torch.cat(train_pocket_preds)
        all_train_labels = torch.cat(train_pocket_labels)
        _, _, train_f1 = calculate_metrics(
            all_train_preds, all_train_labels
        )
        
        all_test_preds = torch.cat(test_pocket_preds)
        all_test_labels = torch.cat(test_pocket_labels)
        _, _, test_f1 = calculate_metrics(
            all_test_preds, all_test_labels
        )
        
        # Selectivity accuracy
        train_sel_acc = np.mean([
            1 if (p > 0.5) == (l == 1) else 0
            for p, l in zip(train_sel_preds, train_sel_labels)
        ])
        test_sel_acc = np.mean([
            1 if (p > 0.5) == (l == 1) else 0
            for p, l in zip(test_sel_preds, test_sel_labels)
        ])
        
        # Save best model
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_model_state = {
                k: v.clone()
                for k, v in model.state_dict().items()
            }
        
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_loss)
        history['train_pocket_f1'].append(train_f1)
        history['test_pocket_f1'].append(test_f1)
        history['train_selectivity_acc'].append(train_sel_acc)
        history['test_selectivity_acc'].append(test_sel_acc)
        
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Pocket F1: {test_f1:.3f} | "
                f"Selectivity Acc: {test_sel_acc:.3f}"
            )
    
    print(f"\nBest Test Pocket F1: {best_test_f1:.3f}")
    return model, history, best_model_state

# ============================================================
# PART 7: Save and plot
# ============================================================

def plot_training_v2(history):
    """Plots both pocket F1 and selectivity accuracy."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        'Neglect-Fold GNN V2 Training Progress',
        fontsize=14
    )
    
    # Loss
    axes[0].plot(
        history['epoch'],
        history['train_loss'],
        'b-', linewidth=2
    )
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].grid(True, alpha=0.3)
    
    # Pocket F1
    axes[1].plot(
        history['epoch'],
        history['train_pocket_f1'],
        'b-', linewidth=2, label='Train'
    )
    axes[1].plot(
        history['epoch'],
        history['test_pocket_f1'],
        'r-', linewidth=2, label='Test'
    )
    axes[1].set_title('Pocket Detection F1')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3)
    
    # Selectivity accuracy
    axes[2].plot(
        history['epoch'],
        history['train_selectivity_acc'],
        'b-', linewidth=2, label='Train'
    )
    axes[2].plot(
        history['epoch'],
        history['test_selectivity_acc'],
        'r-', linewidth=2, label='Test'
    )
    axes[2].set_title('Selectivity Accuracy\n(Parasite vs Human)')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()
    axes[2].set_ylim([0, 1])
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(
        'results/figures/training_progress_v2.png',
        dpi=150
    )
    print("Plot saved!")
    plt.show()

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=== Neglect-Fold: Improved GNN V2 ===")
    print("Dual-output: Pocket Detection + Selectivity")
    print()
    
    # Count parameters
    model = ImprovedPocketGNN()
    total_params = sum(
        p.numel() for p in model.parameters()
    )
    print(f"Total parameters: {total_params:,}")
    
    # Load data
    graphs = load_all_training_data_v2()
    
    if len(graphs) < 10:
        print("Not enough training data!")
        exit()
    
    # Train
    model, history, best_state = train_model_v2(
        graphs, num_epochs=100
    )
    
    # Save
    os.makedirs('models', exist_ok=True)
    torch.save(best_state, 'models/pocket_gnn_v2_best.pt')
    print("Model saved to models/pocket_gnn_v2_best.pt")
    
    # Save history
    pd.DataFrame(history).to_csv(
        'models/training_history_v2.csv',
        index=False
    )
    
    # Plot
    plot_training_v2(history)
    
    print("\n=== Training Complete! ===")
    print(f"Best Test Pocket F1: {max(history['test_pocket_f1']):.3f}")
    print(f"Best Selectivity Acc: {max(history['test_selectivity_acc']):.3f}")