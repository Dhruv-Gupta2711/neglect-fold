# =============================================================
# Neglect-Fold | Phase 4: Binding Affinity Model
# =============================================================
# This model predicts how strongly a drug compound binds
# to a protein pocket.
#
# Architecture:
# - Sub-Network 1: Protein GNN (processes pocket features)
# - Sub-Network 2: Molecule GNN (processes drug compound)
# - Main Network: Combines both → predicts pIC50 score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PART 1: Convert Drug Molecules to Graphs
# ============================================================
# Just like we converted proteins to graphs,
# we convert drug molecules to graphs.
# Atoms = nodes, Bonds = edges

# Atom types we'll recognize
ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'P', 'I', 'OTHER']

# Bond types
BOND_TYPES = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3
}

def atom_to_features(atom):
    """
    Converts one atom into a feature vector.
    Features describe the atom's chemical properties.
    """
    # Atom type (one-hot)
    atom_type = atom.GetSymbol()
    atom_one_hot = [0] * len(ATOM_TYPES)
    if atom_type in ATOM_TYPES:
        atom_one_hot[ATOM_TYPES.index(atom_type)] = 1
    else:
        atom_one_hot[-1] = 1  # OTHER
    
    # Additional chemical features
    features = atom_one_hot + [
        atom.GetDegree() / 10.0,           # Number of bonds
        atom.GetFormalCharge() / 5.0,       # Electric charge
        atom.GetTotalNumHs() / 8.0,              # Hydrogen count
        float(atom.GetIsAromatic()),         # In aromatic ring?
        atom.GetMass() / 100.0,             # Atomic mass
    ]
    
    return features

def smiles_to_graph(smiles):
    """
    Converts a SMILES string into a graph for the GNN.
    
    Example: CC(=O)Oc1ccccc1C(=O)O (aspirin)
    → Graph with atoms as nodes, bonds as edges
    """
    # Parse SMILES
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return None
    
    # Add hydrogens for more complete representation
    mol = Chem.AddHs(mol)
    
    if mol.GetNumAtoms() == 0:
        return None
    
    # Build node features (one per atom)
    atom_features = []
    for atom in mol.GetAtoms():
        features = atom_to_features(atom)
        atom_features.append(features)
    
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # Build edges (one per bond, both directions)
    edge_list = []
    edge_features = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = BOND_TYPES.get(bond.GetBondType(), 0)
        
        edge_list.append([i, j])
        edge_list.append([j, i])
        edge_features.append([bond_type])
        edge_features.append([bond_type])
    
    if not edge_list:
        return None
    
    edge_index = torch.tensor(
        edge_list, dtype=torch.long
    ).t().contiguous()
    
    return Data(x=x, edge_index=edge_index)

# ============================================================
# PART 2: Sub-Network 1 — Protein Pocket GNN
# ============================================================

class ProteinPocketGNN(nn.Module):
    """
    Processes protein pocket features.
    Same architecture as our pocket detection GNN
    but focused on generating a pocket representation.
    """
    def __init__(self, input_dim=22, hidden_dim=64, output_dim=128):
        super(ProteinPocketGNN, self).__init__()
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, edge_index, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        
        # Global pooling — one vector per protein
        x = global_mean_pool(x, batch)
        return x

# ============================================================
# PART 3: Sub-Network 2 — Molecule GNN
# ============================================================

class MoleculeGNN(nn.Module):
    """
    Processes drug molecule features.
    Takes a molecular graph and produces a
    fixed-size representation of the molecule.
    """
    def __init__(self, input_dim=15, hidden_dim=64, output_dim=128):
        super(MoleculeGNN, self).__init__()
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, edge_index, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        
        # Global pooling — one vector per molecule
        x = global_mean_pool(x, batch)
        return x

# ============================================================
# PART 4: Main Network — Binding Affinity Predictor
# ============================================================

class BindingAffinityModel(nn.Module):
    """
    Combines protein pocket + drug molecule representations
    to predict binding affinity (pIC50).
    
    Higher pIC50 = stronger binding = better drug candidate
    """
    def __init__(self):
        super(BindingAffinityModel, self).__init__()
        
        # Sub-networks
        self.protein_gnn = ProteinPocketGNN(
            input_dim=22,
            hidden_dim=64,
            output_dim=128
        )
        
        self.molecule_gnn = MoleculeGNN(
            input_dim=15,
            hidden_dim=64,
            output_dim=128
        )
        
        # Fusion network
        # Input: protein (128) + molecule (128) = 256
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Output: pIC50 value
        )
    
    def forward(self, protein_x, protein_edge, protein_batch,
                mol_x, mol_edge, mol_batch):
        """
        Forward pass through both sub-networks and fusion.
        """
        # Process protein pocket
        protein_repr = self.protein_gnn(
            protein_x, protein_edge, protein_batch
        )
        
        # Process drug molecule
        mol_repr = self.molecule_gnn(
            mol_x, mol_edge, mol_batch
        )
        
        # Combine both representations
        combined = torch.cat([protein_repr, mol_repr], dim=1)
        
        # Predict pIC50
        pIC50 = self.fusion(combined)
        
        return pIC50

# ============================================================
# PART 5: Prepare Training Data
# ============================================================

def prepare_affinity_data(chembl_path, max_samples=500):
    """
    Loads ChEMBL data and converts molecules to graphs.
    Returns paired (molecule_graph, pIC50) training examples.
    """
    print(f"Loading ChEMBL data from {chembl_path}...")
    
    df = pd.read_csv(chembl_path)
    
    # Take a sample for training
    df = df.dropna(subset=['smiles', 'pIC50'])
    df = df.head(max_samples)
    
    print(f"Processing {len(df)} compounds...")
    
    valid_data = []
    failed = 0
    
    for _, row in df.iterrows():
        smiles = row['smiles']
        pIC50 = row['pIC50']
        
        # Convert SMILES to graph
        mol_graph = smiles_to_graph(smiles)
        
        if mol_graph is None:
            failed += 1
            continue
        
        valid_data.append({
            'mol_graph': mol_graph,
            'pIC50': float(pIC50),
            'smiles': smiles
        })
    
    print(f"Valid molecules: {len(valid_data)}")
    print(f"Failed to parse: {failed}")
    
    return valid_data

# ============================================================
# PART 6: Training Loop
# ============================================================

def train_affinity_model(train_data, num_epochs=50):
    """
    Trains the binding affinity model.
    Uses mean squared error loss — we're predicting
    a continuous number (pIC50) not a binary label.
    """
    print(f"\n=== Training Binding Affinity Model ===")
    print(f"Training samples: {len(train_data)}")
    print(f"Epochs: {num_epochs}")
    
    model = BindingAffinityModel()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01
    )
    
    # MSE loss for regression (predicting a number)
    criterion = nn.MSELoss()
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=20,
        gamma=0.5
    )
    
    history = {
        'epoch': [],
        'loss': [],
        'rmse': []
    }
    
    # Create a simple protein graph for testing
    # In production we'd use real pocket graphs
    # For now use a placeholder protein
    dummy_protein_x = torch.randn(50, 22)
    dummy_protein_edge = torch.randint(0, 50, (2, 100))
    dummy_protein_batch = torch.zeros(50, dtype=torch.long)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        import random
        random.shuffle(train_data)
        
        for item in train_data:
            optimizer.zero_grad()
            
            mol_graph = item['mol_graph']
            target_pIC50 = torch.tensor(
                [[item['pIC50']]], 
                dtype=torch.float
            )
            
            mol_batch = torch.zeros(
                mol_graph.x.shape[0],
                dtype=torch.long
            )
            
            # Forward pass
            pred_pIC50 = model(
                dummy_protein_x,
                dummy_protein_edge,
                dummy_protein_batch,
                mol_graph.x,
                mol_graph.edge_index,
                mol_batch
            )
            
            # Calculate loss
            loss = criterion(pred_pIC50, target_pIC50)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )
            optimizer.step()
            epoch_losses.append(loss.item())
        
        scheduler.step()
        
        avg_loss = np.mean(epoch_losses)
        rmse = np.sqrt(avg_loss)
        
        history['epoch'].append(epoch + 1)
        history['loss'].append(avg_loss)
        history['rmse'].append(rmse)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"RMSE: {rmse:.4f}")
    
    return model, history

# ============================================================
# PART 7: Test the Model
# ============================================================

if __name__ == "__main__":
    print("=== Neglect-Fold: Binding Affinity Model ===")
    print("Architecture: Protein GNN + Molecule GNN + Fusion")
    print()
    
    # Count parameters
    model = BindingAffinityModel()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test with a real molecule from ChEMBL
    test_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # Testosterone
        "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",  # Pyrene
    ]
    
    print("\nTesting molecule → graph conversion:")
    for smiles in test_smiles:
        graph = smiles_to_graph(smiles)
        if graph:
            print(f"  SMILES: {smiles[:30]}...")
            print(f"  Atoms: {graph.x.shape[0]}, "
                  f"Bonds: {graph.edge_index.shape[1]//2}")
    
    # Load ChEMBL training data
    chembl_path = "data/processed/trypanosoma_cruzi_chembl_clean.csv"
    
    if os.path.exists(chembl_path):
        train_data = prepare_affinity_data(
            chembl_path, 
            max_samples=200
        )
        
        if len(train_data) > 10:
            # Train the model
            model, history = train_affinity_model(
                train_data,
                num_epochs=50
            )
            
            # Save model
            os.makedirs('models', exist_ok=True)
            torch.save(
                model.state_dict(),
                'models/affinity_model.pt'
            )
            print("\nModel saved to models/affinity_model.pt")
            
            # Final RMSE
            final_rmse = history['rmse'][-1]
            print(f"Final RMSE: {final_rmse:.4f}")
            print("\nWhat RMSE means:")
            print(f"Our predictions are off by ~{final_rmse:.2f} "
                  f"pIC50 units on average")
            print("Target: RMSE < 1.0 (good), < 0.5 (excellent)")
    
    print("\n=== Affinity Model Complete! ===")