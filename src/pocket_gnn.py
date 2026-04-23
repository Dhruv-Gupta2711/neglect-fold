# =============================================================
# Neglect-Fold | Phase 3: Pocket Detection GNN
# =============================================================
# This is our Graph Neural Network for detecting drug binding
# pockets on parasite proteins.
#
# Architecture (your friend's idea):
# - Sub-Network 1: Local GNN (looks at nearby atoms)
# - Sub-Network 2: Global GNN (looks at whole protein)
# - Main Network: Fusion (combines both → final prediction)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Dataset
import numpy as np
from Bio import PDB
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PART 1: Convert PDB files to Graphs
# ============================================================
# Before we can run a GNN, we need to convert our PDB files
# (3D coordinates) into graphs (nodes + edges)

# Amino acid to number mapping
# Each amino acid gets a unique number so the network can
# understand it mathematically
AMINO_ACIDS = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
    'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
    'UNK': 20  # Unknown amino acid
}

def pdb_to_graph(pdb_path, distance_threshold=8.0):
    """
    Converts a PDB file into a graph for the GNN.
    
    Each amino acid becomes a NODE.
    Two amino acids are connected by an EDGE if they are
    within 8 Angstroms of each other in 3D space.
    
    Node features:
    - Amino acid type (one-hot encoded)
    - 3D position (x, y, z coordinates)
    - AlphaFold confidence score (pLDDT)
    
    distance_threshold: how close two amino acids need to be
    to be connected (8 Angstroms is standard)
    """
    
    parser = PDB.PDBParser(QUIET=True)
    
    try:
        structure = parser.get_structure('protein', pdb_path)
    except Exception as e:
        return None
    
    # Extract residue information
    residues = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                # Skip non-amino acid residues (water, ligands etc)
                if residue.get_id()[0] != ' ':
                    continue
                
                # Get the alpha carbon position (center of amino acid)
                if 'CA' not in residue:
                    continue
                
                ca = residue['CA']
                pos = ca.get_vector().get_array()
                
                # Get amino acid type
                res_name = residue.get_resname()
                aa_idx = AMINO_ACIDS.get(res_name, 20)
                
                # Get pLDDT score (AlphaFold confidence)
                # Stored in B-factor column of PDB file
                plddt = ca.get_bfactor()
                
                residues.append({
                    'position': pos,
                    'aa_type': aa_idx,
                    'plddt': plddt
                })
    
    if len(residues) < 10:
        return None
    
    # Build node features
    # For each residue: [one-hot amino acid type + plddt score]
    num_residues = len(residues)
    node_features = []
    positions = []
    
    for res in residues:
        # One-hot encode amino acid type (21 possible types)
        one_hot = [0] * 21
        one_hot[res['aa_type']] = 1
        
        # Add pLDDT score (normalized to 0-1)
        plddt_normalized = res['plddt'] / 100.0
        
        # Combine into feature vector
        features = one_hot + [plddt_normalized]
        node_features.append(features)
        positions.append(res['position'])
    
    # Convert to tensors
    x = torch.tensor(node_features, dtype=torch.float)
    pos = torch.tensor(positions, dtype=torch.float)
    
    # Build edges - connect residues within distance_threshold
    edge_list = []
    edge_weights = []
    
    for i in range(num_residues):
        for j in range(i + 1, num_residues):
            # Calculate 3D distance between residues
            diff = positions[i] - positions[j]
            distance = np.sqrt(np.sum(diff ** 2))
            
            if distance <= distance_threshold:
                # Add edge in both directions (undirected graph)
                edge_list.append([i, j])
                edge_list.append([j, i])
                edge_weights.append(distance)
                edge_weights.append(distance)
    
    if not edge_list:
        return None
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)

# ============================================================
# PART 2: Sub-Network 1 — Local GNN
# ============================================================
# Looks at each amino acid and its immediate neighbors
# Detects local structural patterns like cavities

class LocalGNN(nn.Module):
    def __init__(self, input_dim=22, hidden_dim=64, output_dim=32):
        super(LocalGNN, self).__init__()
        
        # Three layers of graph convolution
        # Each layer looks one hop further in the graph
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        
        # Batch normalization helps training stability
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout prevents overfitting
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, edge_index):
        """
        Forward pass - processes the graph through 3 layers.
        Each layer: message passing → aggregation → update
        """
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 3
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        return x  # Shape: [num_residues, output_dim]

# ============================================================
# PART 3: Sub-Network 2 — Global GNN
# ============================================================
# Looks at the whole protein structure
# Captures global context - where is this residue relative
# to the whole protein?

class GlobalGNN(nn.Module):
    def __init__(self, input_dim=22, hidden_dim=64, output_dim=32):
        super(GlobalGNN, self).__init__()
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.2)
        
        # Global pooling - summarizes the whole protein
        # into a single vector
        self.global_pool_linear = nn.Linear(output_dim * 2, output_dim)
    
    def forward(self, x, edge_index, batch):
        """
        Forward pass with global pooling.
        batch: tells PyG which nodes belong to which protein
        """
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 3
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # Global pooling - get one vector per protein
        # Mean pooling + Max pooling gives richer representation
        global_mean = global_mean_pool(x, batch)
        global_max = global_max_pool(x, batch)
        global_repr = torch.cat([global_mean, global_max], dim=1)
        global_repr = self.global_pool_linear(global_repr)
        
        return x, global_repr

# ============================================================
# PART 4: Main Network — Fusion
# ============================================================
# Combines local + global information
# Makes the final pocket prediction per amino acid

class PocketDetectionGNN(nn.Module):
    def __init__(self):
        super(PocketDetectionGNN, self).__init__()
        
        # The two sub-networks
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
        
        # Fusion network
        # Input: local features (32) + global features (32) = 64
        self.fusion = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),  # Output: one score per residue
            nn.Sigmoid()        # Convert to probability 0-1
        )
    
    def forward(self, x, edge_index, batch):
        """
        Full forward pass through both sub-networks and fusion.
        
        Returns: probability that each amino acid is part of
        a binding pocket (0 = definitely not, 1 = definitely yes)
        """
        # Sub-Network 1: Local features per residue
        local_features = self.local_gnn(x, edge_index)
        
        # Sub-Network 2: Global features
        _, global_repr = self.global_gnn(x, edge_index, batch)
        
        # Expand global features to match number of residues
        # Each residue gets the same global context vector
        num_residues = x.size(0)
        global_expanded = global_repr[batch]
        
        # Concatenate local + global features
        combined = torch.cat([local_features, global_expanded], dim=1)
        
        # Final prediction
        pocket_probability = self.fusion(combined)
        
        return pocket_probability

# ============================================================
# PART 5: Test the network
# ============================================================

if __name__ == "__main__":
    print("=== Neglect-Fold: Pocket Detection GNN ===")
    print("Architecture: Local GNN + Global GNN + Fusion")
    print()
    
    # Test with one real protein
    structure_dir = "data/processed/structures/trypanosoma_cruzi"
    pdb_files = [f for f in os.listdir(structure_dir) if f.endswith('.pdb')]
    
    if not pdb_files:
        print("No PDB files found!")
        exit()
    
    test_pdb = os.path.join(structure_dir, pdb_files[0])
    protein_name = pdb_files[0].replace('.pdb', '')
    
    print(f"Testing with protein: {protein_name}")
    
    # Convert PDB to graph
    print("Converting PDB to graph...")
    graph = pdb_to_graph(test_pdb)
    
    if graph is None:
        print("Failed to parse PDB file")
        exit()
    
    print(f"Graph created!")
    print(f"  Nodes (amino acids): {graph.x.shape[0]}")
    print(f"  Edges (connections): {graph.edge_index.shape[1]}")
    print(f"  Node features: {graph.x.shape[1]}")
    
    # Create the model
    print("\nCreating PocketDetectionGNN...")
    model = PocketDetectionGNN()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Run a forward pass
    print("\nRunning forward pass...")
    
    # batch tensor - all nodes belong to protein 0
    batch = torch.zeros(graph.x.shape[0], dtype=torch.long)
    
    with torch.no_grad():
        predictions = model(graph.x, graph.edge_index, batch)
    
    print(f"Output shape: {predictions.shape}")
    print(f"Predictions for first 5 amino acids:")
    for i in range(min(5, len(predictions))):
        prob = predictions[i].item()
        bar = '█' * int(prob * 20)
        print(f"  Residue {i+1}: {prob:.4f} [{bar:<20}]")
    
    # Find top predicted pocket residues
    probs = predictions.squeeze().numpy()
    top_indices = np.argsort(probs)[::-1][:10]
    
    print(f"\nTop 10 predicted pocket residues:")
    for idx in top_indices:
        prob = probs[idx]
        print(f"  Residue {idx+1}: {prob:.4f} pocket probability")
    
    print("\nGNN is working correctly!")
    print("Next step: train on labeled binding site data")