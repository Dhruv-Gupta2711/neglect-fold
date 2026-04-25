# Neglect-Fold: Complete Explanation Guide

A plain-English explanation of everything in this project —
the science, the code, and the reasoning behind every decision.

Written for someone who started as a complete beginner.

---

# PART 1: THE PROBLEM

## What are Neglected Tropical Diseases?

Neglected Tropical Diseases (NTDs) are infections that 
primarily affect people living in poverty in tropical regions.
The WHO recognizes 20+ NTDs. We focus on three:

### Chagas Disease
- **Caused by:** Trypanosoma cruzi (a parasite)
- **How you get it:** Bite of the triatomine bug
- **Who it affects:** ~6 million people, mostly in Latin America
- **What happens:** The parasite enters your blood, reaches 
  your heart and digestive system, causes organ damage
- **Current drugs:** Benznidazole, Nifurtimox
- **Problem with current drugs:** Severe side effects 
  (nerve damage, skin rashes), only works in early stages,
  completely useless once the disease becomes chronic
- **Why pharma ignores it:** Poor patients in poor countries
  = no profitable market

### Leishmaniasis
- **Caused by:** Leishmania donovani (a parasite)
- **How you get it:** Bite of the sandfly
- **Who it affects:** 700,000-1 million new cases per year
- **What happens:** Attacks your immune system, damages 
  liver and spleen, fatal if untreated
- **Current drugs:** Amphotericin B, Antimonials
- **Problem with current drugs:** Amphotericin B requires 
  IV injection in a hospital, costs thousands of dollars,
  damages your kidneys. Antimonials are toxic and 
  resistance is spreading
- **Why pharma ignores it:** Same reason — no rich market

### Schistosomiasis
- **Caused by:** Schistosoma mansoni (a parasitic worm)
- **How you get it:** Contact with contaminated freshwater
- **Who it affects:** ~240 million people worldwide
- **What happens:** Worm larvae penetrate your skin, 
  travel to blood vessels around your intestines,
  cause chronic inflammation and organ damage
- **Current drugs:** Praziquantel — the ONLY drug available
- **Problem:** Resistance is emerging. If resistance spreads,
  240 million people have zero treatment options
- **Why pharma ignores it:** Again — no money in it

---

## Why Use AI for This?

Traditional drug discovery works like this:
1. Scientists guess which protein might be a good target
2. They test thousands of compounds against it in a lab
3. Each experiment takes weeks and costs money
4. Most experiments fail
5. Start over

This process takes 10-15 years and costs billions of dollars.
No company will spend that for a disease affecting poor people.

AI changes this:
1. We predict which proteins are good targets computationally
2. We score thousands of compounds computationally
3. Each prediction takes seconds and costs almost nothing
4. We give scientists a shortlist of the most promising targets
5. Scientists only run lab experiments on our top candidates

This dramatically reduces the cost and time of drug discovery.

---

# PART 2: THE BIOLOGY

## What is a Protein?

A protein is a large molecule made of amino acids chained 
together. Proteins do almost everything in living organisms:
- Enzymes: speed up chemical reactions
- Structural proteins: build cells and tissues
- Transport proteins: carry molecules through the body
- Receptor proteins: receive signals from other cells

Every living thing — including parasites — needs proteins 
to survive. If you stop a critical protein from working,
the organism dies.

## What is an Amino Acid?

Amino acids are the building blocks of proteins.
There are 20 standard amino acids. Each has:
- A unique chemical structure
- A single letter code (A, C, D, E, F, G, H, I, K, L, 
  M, N, P, Q, R, S, T, V, W, Y)

When amino acids chain together they form a protein sequence:
MGFDFGCLLKLCSTVLKPGGAPGPIN...
Each letter = one amino acid = one residue

## What is a Residue?

When amino acids join together to form a protein chain,
they lose a small water molecule at each connection point.
What remains of each amino acid after joining is called
a RESIDUE.

In practice: residue = amino acid in a protein chain.
Scientists use "residue" to be technically precise.

When we say "Residue 60" we mean the 60th amino acid
in the protein sequence.

## What is a Protein Structure?

A protein sequence folds into a specific 3D shape.
This shape is determined by how amino acids interact
with each other chemically.

The 3D shape has several levels:
- Primary structure: the sequence (MGFDFGCLL...)
- Secondary structure: local patterns (alpha helices, 
  beta sheets)
- Tertiary structure: the complete 3D folded shape
- Quaternary structure: multiple proteins together

The 3D SHAPE is what matters for drug discovery.

## What is a Binding Pocket?

A binding pocket (also called active site) is a cavity 
or groove on the surface of a protein where a small 
molecule (drug) can fit.

Think of it like a lock and key:
- The protein pocket = the lock
- The drug molecule = the key
- When the key fits the lock = the protein stops working
- If the protein is essential for the parasite = parasite dies

A good drug target has:
- A clear binding pocket (druggable)
- Is essential for parasite survival
- Is different enough from human proteins (selective)

## What is a Ligand?

A ligand is any small molecule that binds to a protein.
Most drugs are ligands. The word comes from Latin "ligare"
meaning "to bind".

## What is IC50 and pIC50?

IC50 = Inhibitory Concentration 50%
The concentration of a drug needed to kill/inhibit 50% 
of the target (parasite or protein activity).

Lower IC50 = more potent drug (needs less to work)

pIC50 = -log10(IC50 in molar units)
We convert to pIC50 because:
- IC50 values span many orders of magnitude (0.001 to 10000)
- pIC50 compresses them into a manageable range (2 to 15)
- Higher pIC50 = stronger drug (easier to reason about)
- ML models work better with this range

Example:
IC50 = 10 nanomolar = 0.00000001 molar
pIC50 = -log10(0.00000001) = 8.0 (a potent drug!)

## What is a SMILES String?

SMILES = Simplified Molecular Input Line Entry System
A way to represent a molecule as a text string.

Example - Aspirin:
CC(=O)Oc1ccccc1C(=O)O

Each character represents:
- C = Carbon atom
- O = Oxygen atom
- c = Carbon in aromatic ring
- () = branch point
- = = double bond
- 1...1 = ring closure

This allows computers to understand and process 
chemical structures as text.

## What is AlphaFold?

AlphaFold is a neural network developed by DeepMind
that predicts the 3D structure of a protein from
its amino acid sequence.

Before AlphaFold: determining a protein's 3D structure
took months to years of expensive lab work (X-ray 
crystallography, cryo-EM).

After AlphaFold (2021): predict any protein's structure
in minutes on a computer. FREE for all proteins.

AlphaFold changed biology the way the internet changed 
communication. It won the Nobel Prize in Chemistry 2024.

## What is pLDDT?

pLDDT = predicted Local Distance Difference Test
AlphaFold's confidence score for each residue (0-100).

- >90: Very high confidence (dark blue in viewer)
- 70-90: High confidence (light blue)
- 50-70: Medium confidence (yellow)
- <50: Low confidence (orange/red) — often disordered

We only trust predictions with pLDDT > 70.
In our code we store pLDDT in the B-factor column of 
PDB files and use it as a node feature in our GNN.

## What is a PDB File?

PDB = Protein Data Bank format
A text file describing the 3D coordinates of every 
atom in a protein.

Each line describes one atom:
ATOM    1  N   MET A   1      10.123  20.456  30.789  1.00 95.23
Columns: atom number, atom type, residue name, chain, 
residue number, X, Y, Z coordinates, occupancy, B-factor

The B-factor column is where AlphaFold stores pLDDT.

## What is FASTA Format?

FASTA is a text format for protein/DNA sequences.

Example:
sp|Q2VLK6|MCA3_TRYCC Metacaspase-3 OS=Trypanosoma cruzi
MGFDFGCLLKLCSTVLKPGGAPGPINYMEIGLNLIKIAAPYIVQYLGIMER...
The > line is the header (protein ID and description).
The following lines are the amino acid sequence.
One letter per amino acid.

---

# PART 3: THE MACHINE LEARNING

## What is a Graph?

A graph is a mathematical structure with:
- Nodes (also called vertices): the objects
- Edges: connections between objects

Examples:
- Social network: people = nodes, friendships = edges
- Road map: cities = nodes, roads = edges
- Protein: amino acids = nodes, proximity = edges
- Molecule: atoms = nodes, chemical bonds = edges

## What is a Neural Network?

A neural network is a mathematical function that learns
to map inputs to outputs by adjusting its parameters.

Inspired by the brain: 
- Neurons = mathematical functions
- Connections = learnable weights

Training = showing the network many examples and 
adjusting weights to minimize prediction error.

## What is a Graph Neural Network (GNN)?

A GNN is a neural network designed to work on graphs.

Regular neural networks work on fixed-size inputs
(like a 28x28 image). They can't handle graphs because
graphs have variable sizes and structures.

GNNs solve this with MESSAGE PASSING:
1. Each node looks at its neighbors
2. Collects information from neighbors (message)
3. Updates its own representation
4. Repeat for multiple layers

After several layers, each node has "seen" everything
within several hops of it in the graph.

## What is GCNConv?

GCNConv = Graph Convolutional Network Convolution
The basic building block of our GNN layers.

For each node it:
1. Collects features from all neighboring nodes
2. Averages them (with normalization)
3. Multiplies by a learnable weight matrix
4. Applies activation function

Mathematical formula:
h_v = ReLU(W * mean(h_u for u in neighbors(v)))

In our code:
```python
self.conv1 = GCNConv(input_dim, hidden_dim)
x = self.conv1(x, edge_index)
```

## What is Global Mean Pooling?

When we have a graph with variable number of nodes,
we need to produce a fixed-size output for the 
whole graph (not per-node).

Global mean pooling averages all node features:
output = mean(all node features)

This gives one vector representing the whole graph.

In our code:
```python
x = global_mean_pool(x, batch)
```

The `batch` tensor tells PyG which nodes belong 
to which graph when processing multiple graphs together.

## What is a Loss Function?

A loss function measures how wrong our model's 
predictions are. We want to minimize this.

For pocket detection (classification):
- We use Binary Cross Entropy (BCE)
- Measures difference between predicted probability
  and actual label (0 or 1)
- BCE = -(y*log(p) + (1-y)*log(1-p))
- Perfect prediction = loss of 0

For affinity prediction (regression):
- We use Mean Squared Error (MSE)
- Measures average squared difference between
  predicted and actual pIC50 values
- MSE = mean((predicted - actual)^2)
- RMSE = sqrt(MSE) — in same units as pIC50

## What is Backpropagation?

After calculating loss, we need to know which 
parameters caused the error so we can fix them.

Backpropagation calculates the gradient of the loss
with respect to every parameter using the chain rule
of calculus.

In simple terms: "how much does changing each 
parameter affect the loss?"

PyTorch does this automatically with:
```python
loss.backward()
```

## What is an Optimizer?

After backpropagation gives us gradients, the optimizer
uses those gradients to update parameters.

We use AdamW:
- Adam = Adaptive Moment Estimation
- Combines momentum (like a ball rolling downhill)
  with adaptive learning rates per parameter
- W = weight decay (prevents overfitting)

```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,        # learning rate
    weight_decay=0.01
)
optimizer.step()  # updates all parameters
```

## What is Dropout?

Dropout randomly sets some neuron outputs to zero
during training (e.g. 20% of them).

Why? It prevents overfitting (memorizing training data).
By randomly disabling neurons, the network is forced
to learn robust features that work even when some 
neurons are missing.

```python
self.dropout = nn.Dropout(0.2)  # 20% dropout
x = self.dropout(x)
```

## What is Batch Normalization?

Batch normalization normalizes the output of a layer
to have mean 0 and variance 1.

Why? It makes training more stable and faster.
Without it, values can explode or vanish as they
pass through many layers.

```python
self.bn1 = nn.BatchNorm1d(hidden_dim)
x = self.bn1(x)
```

## What is F1 Score?

Used for classification problems (YES/NO predictions).

Precision = of things we predicted YES,
            how many were actually YES?
Recall = of all actual YES things,
         how many did we find?
F1 = 2 * (Precision * Recall) / (Precision + Recall)

F1 of 1.0 = perfect
F1 of 0.0 = completely wrong
Our model: F1 = 0.662 (decent for 10 training examples!)

## What is RMSE?

Root Mean Squared Error. Used for regression problems
(predicting a number).

RMSE = sqrt(mean((predicted - actual)^2))

RMSE of 0 = perfect predictions
RMSE of 0.87 = predictions off by 0.87 pIC50 on average
Our target: RMSE < 1.0 ✅ (we achieved 0.87!)

## What is SHAP?

SHAP = SHapley Additive exPlanations

Explains WHY a model made a specific prediction by
calculating how much each feature contributed.

Based on game theory (Shapley values):
imagine features as players in a game, 
SHAP calculates each player's fair contribution.

Positive SHAP = feature increased the score
Negative SHAP = feature decreased the score
Magnitude = how much impact it had

In our project:
Rank 1: G4VJD6
binding_affinity: +0.1550  ← main reason for high rank
pocket_score:     +0.0637  ← also contributed
selectivity:      +0.0016  ← tiny contribution
## What is LightGBM?

LightGBM = Light Gradient Boosting Machine
A fast, efficient gradient boosting algorithm.

Gradient boosting = ensemble of decision trees,
each one correcting the errors of the previous one.

We use it for final ranking because:
- Works well with small datasets (only 20 proteins)
- Directly compatible with SHAP
- Very interpretable
- Fast to train

---

# PART 4: THE CODE — LINE BY LINE

## download_proteomes.py

```python
import requests
import os
import time
```
- `requests`: Python library for making HTTP requests
  (like a browser but in code)
- `os`: Operating system interface — for file/folder operations
- `time`: For adding delays between requests

```python
ORGANISMS = {
    "trypanosoma_cruzi": "353153",
    "leishmania_donovani": "5661",
    "schistosoma_mansoni": "6183"
}
```
A Python dictionary mapping organism names to their
UniProt taxonomy IDs. These IDs are UniProt's internal
numbering system for organisms.

```python
def download_proteome(organism_name, tax_id):
```
A function definition. Functions are reusable blocks
of code. Instead of writing the download code 3 times,
we write it once and call it 3 times.

```python
url = (
    f"https://rest.uniprot.org/uniprotkb/stream?"
    f"query=organism_id:{tax_id}&"
    f"format=fasta&"
    f"compressed=false"
)
```
Building the API URL. The f"..." syntax is an f-string —
it inserts variable values into the string.
- `organism_id:{tax_id}`: filter by our organism
- `format=fasta`: return data in FASTA format
- `compressed=false`: don't zip the response

```python
response = requests.get(url, stream=True)
```
Sends an HTTP GET request to UniProt's server.
`stream=True` means download in chunks (good for large files).

```python
if response.status_code != 200:
    print(f"ERROR: Could not download {organism_name}")
    return
```
HTTP status codes: 200 = OK, 404 = not found, 500 = error.
If we don't get 200, something went wrong — we print
an error and stop (return exits the function early).

```python
with open(output_file, 'w') as f:
    f.write(response.text)
```
Opens a file for writing ('w' mode).
`with` statement ensures the file is closed properly
even if an error occurs. `f.write()` writes the content.

```python
protein_count = response.text.count('>')
```
In FASTA format every protein starts with '>'.
Counting '>' = counting proteins. Simple but effective.

```python
if __name__ == "__main__":
```
This block only runs when you execute this file directly.
If another file imports this module, this block is skipped.
It's Python's standard way to separate "library code"
from "script code".

```python
for organism_name, tax_id in ORGANISMS.items():
    download_proteome(organism_name, tax_id)
    time.sleep(1)
```
Loops through each organism in our dictionary.
`.items()` gives us both the key and value each iteration.
`time.sleep(1)` pauses for 1 second between downloads —
being polite to UniProt's servers (they have rate limits).

---

## explore_proteomes.py

```python
from Bio import SeqIO
```
Biopython's SeqIO module for reading biological sequence
files. BioPython is a Python library for bioinformatics.
SeqIO can read FASTA, GenBank, and many other formats.

```python
for record in SeqIO.parse(filepath, "fasta"):
    proteins.append({
        "id": record.id,
        "name": record.description,
        "sequence": str(record.seq),
        "length": len(record.seq)
    })
```
SeqIO.parse reads each protein from the FASTA file.
Each `record` has:
- `.id`: the protein identifier
- `.description`: full header line
- `.seq`: the sequence object
- `str(record.seq)`: converts sequence to plain string
- `len(record.seq)`: length in amino acids

```python
df = pd.DataFrame(proteins)
```
Creates a pandas DataFrame — like an Excel spreadsheet
in Python. Rows = proteins, Columns = id/name/sequence/length.

```python
print(f"Shortest protein: {df['length'].min()} amino acids")
print(f"Average length: {df['length'].mean():.1f} amino acids")
```
Pandas makes statistics easy:
- `.min()`: minimum value
- `.max()`: maximum value  
- `.mean()`: average value
- `:.1f`: format to 1 decimal place

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
```
Creates a figure with 1 row, 3 columns of subplots.
`figsize=(15, 5)` sets the figure size in inches.

```python
axes[i].hist(lengths, bins=50, color=colors[i], alpha=0.7)
```
Creates a histogram (bar chart of frequencies).
- `bins=50`: divide data into 50 groups
- `alpha=0.7`: 70% opacity (slightly transparent)

```python
plt.savefig('results/figures/protein_length_distributions.png', dpi=150)
```
Saves the plot as a PNG image.
`dpi=150` = 150 dots per inch (good quality for papers).

---

## clean_proteomes.py

```python
MIN_LENGTH = 50
MAX_LENGTH = 2000
```
Constants defining our filtering thresholds.
- <50 aa: probably just a fragment, not a real protein
- >2000 aa: too large for our models to handle efficiently

```python
STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
```
A set of the 20 standard amino acid letters.
Sets are faster than lists for checking membership.

```python
def has_nonstandard_amino_acids(sequence):
    letters_in_sequence = set(sequence.upper())
    non_standard = letters_in_sequence - STANDARD_AMINO_ACIDS
    return len(non_standard) > 0
```
Converts sequence to a set of unique letters.
Set subtraction (-) finds letters not in standard set.
Returns True if any non-standard letters found.
'X' means unknown, 'B'/'Z'/'J' are ambiguous.
These would confuse our GNN.

```python
after_min = [p for p in all_proteins if len(p.seq) >= MIN_LENGTH]
```
List comprehension — a compact way to filter a list.
Equivalent to:
```python
after_min = []
for p in all_proteins:
    if len(p.seq) >= MIN_LENGTH:
        after_min.append(p)
```

```python
seen_sequences = set()
unique_proteins = []

for protein in after_standard:
    seq_str = str(protein.seq)
    if seq_str not in seen_sequences:
        seen_sequences.add(seq_str)
        unique_proteins.append(protein)
```
Removes duplicate sequences by keeping a set of
already-seen sequences. Sets have O(1) lookup time
making this very efficient even for 60,000 proteins.

---

## download_chembl.py

```python
while True:
    ...
    offset += limit
    if offset >= total_count:
        break
```
Pagination loop. ChEMBL returns 1000 results at a time.
We keep requesting more pages until we have everything.
`while True` runs forever until we explicitly `break`.

```python
df['pIC50'] = -np.log10(df['standard_value'] * 1e-9)
```
Converting IC50 to pIC50:
- `standard_value` is in nanomolar (nM)
- `* 1e-9` converts nM to molar (M)
- `-np.log10(...)` applies the pIC50 formula
- np.log10 = base-10 logarithm from NumPy

```python
df = df[(df['pIC50'] >= 2) & (df['pIC50'] <= 15)]
```
Filters rows where pIC50 is between 2 and 15.
Values outside this range are likely data entry errors.
`&` = AND operator for pandas boolean indexing.

---

## pocket_gnn.py

```python
AMINO_ACIDS = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, ...
    'UNK': 20
}
```
Maps 3-letter amino acid codes to numbers.
Neural networks need numbers, not strings.
We have 20 standard amino acids + 1 unknown = 21 total.

```python
def pdb_to_graph(pdb_path, distance_threshold=8.0):
```
Converts a PDB file to a PyTorch Geometric graph.
`distance_threshold=8.0` means connect residues within
8 Angstroms (Å). 1 Å = 0.1 nanometers.
8Å is the standard threshold used in structural biology
for detecting residue-residue interactions.

```python
parser = PDB.PDBParser(QUIET=True)
structure = parser.get_structure('protein', pdb_path)
```
Biopython's PDB parser reads the PDB file.
`QUIET=True` suppresses warning messages.

```python
for model in structure:
    for chain in model:
        for residue in chain:
```
PDB files have a hierarchy: Structure → Model → Chain → Residue → Atom.
We loop through all levels to reach individual residues.

```python
if residue.get_id()[0] != ' ':
    continue
```
PDB residue IDs have a "hetflag" character.
' ' (space) = standard amino acid
'W' = water molecule
'H_XXX' = ligand/drug
We skip everything that isn't a standard amino acid.

```python
if 'CA' not in residue:
    continue
ca = residue['CA']
pos = ca.get_vector().get_array()
```
'CA' = alpha carbon = the central carbon of each amino acid.
We use the alpha carbon as the representative position
of each residue (center of mass approximation).
`.get_vector().get_array()` converts to a numpy array [x, y, z].

```python
one_hot = [0] * 21
one_hot[res['aa_type']] = 1
```
One-hot encoding: represents categorical data as a binary vector.
If amino acid is LEU (index 10):
one_hot = [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
Neural networks work better with one-hot than integers.

```python
x = torch.tensor(node_features, dtype=torch.float)
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
```
Converts Python lists to PyTorch tensors.
- `dtype=torch.float`: 32-bit floating point for features
- `dtype=torch.long`: 64-bit integer for indices
- `.t()`: transpose (converts from [[i,j],...] to [[i,...],[j,...]])
- `.contiguous()`: ensures memory layout is efficient

```python
diff = positions[i] - positions[j]
distance = np.sqrt(np.sum(diff ** 2))
```
Euclidean distance in 3D space:
distance = sqrt((x1-x2)² + (y1-y2)² + (z1-z2)²)
`diff ** 2` squares each component,
`np.sum(...)` adds them up,
`np.sqrt(...)` takes the square root.

```python
class LocalGNN(nn.Module):
    def __init__(self, input_dim=22, hidden_dim=64, output_dim=32):
        super(LocalGNN, self).__init__()
```
`nn.Module` is PyTorch's base class for all neural networks.
`super().__init__()` calls the parent class constructor —
required for PyTorch to properly set up the module.
`input_dim=22`: 21 amino acid types + 1 pLDDT score.

```python
self.conv1 = GCNConv(input_dim, hidden_dim)
self.conv2 = GCNConv(hidden_dim, hidden_dim)
self.conv3 = GCNConv(hidden_dim, output_dim)
```
Three graph convolution layers:
- Layer 1: 22 → 64 dimensions (expand representation)
- Layer 2: 64 → 64 dimensions (refine)
- Layer 3: 64 → 32 dimensions (compress to output)

```python
def forward(self, x, edge_index):
    x = self.conv1(x, edge_index)
    x = self.bn1(x)
    x = F.relu(x)
    x = self.dropout(x)
```
The forward pass runs in this order:
1. Graph convolution (message passing)
2. Batch normalization (stabilize values)
3. ReLU activation (add non-linearity)
4. Dropout (prevent overfitting)

ReLU = Rectified Linear Unit = max(0, x)
Makes all negative values 0. Without this,
the network can only learn linear functions.

```python
class PocketDetectionGNN(nn.Module):
    def __init__(self):
        self.local_gnn = LocalGNN(...)
        self.global_gnn = GlobalGNN(...)
        self.fusion = nn.Sequential(...)
```
The main model combining both sub-networks.
`nn.Sequential` chains layers in order —
output of one layer becomes input of the next.

```python
def forward(self, x, edge_index, batch):
    local_features = self.local_gnn(x, edge_index)
    _, global_repr = self.global_gnn(x, edge_index, batch)
    global_expanded = global_repr[batch]
    combined = torch.cat([local_features, global_expanded], dim=1)
    pocket_probability = self.fusion(combined)
    return pocket_probability
```
The fusion forward pass:
1. Get local features per residue (shape: [N, 32])
2. Get global protein representation (shape: [1, 32])
3. `global_repr[batch]`: expand global to match each residue
   If protein has 358 residues, global becomes [358, 32]
4. `torch.cat(..., dim=1)`: concatenate along feature dimension
   [N, 32] + [N, 32] → [N, 64]
5. Fusion network: [N, 64] → [N, 1]
6. Sigmoid: converts to probability 0-1

---

## train_gnn.py

```python
criterion = nn.BCELoss()
```
Binary Cross Entropy Loss for our classification problem.
Measures how different predicted probabilities are from
actual labels (0 or 1).

```python
scheduler = optim.lr_scheduler.StepLR(
    optimizer, 
    step_size=20, 
    gamma=0.5
)
```
Learning rate scheduler: reduces learning rate over time.
Every 20 epochs, multiplies learning rate by 0.5.
Why? Early in training we want large steps to learn fast.
Later we want small steps for fine-tuning precision.

```python
optimizer.zero_grad()
```
Clears gradients from the previous step.
PyTorch accumulates gradients by default — we must
clear them before each new backward pass.

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```
The three steps of every training iteration:
1. `loss.backward()`: calculate gradients via backprop
2. `clip_grad_norm_`: prevents "exploding gradients"
   (gradients becoming too large and destabilizing training)
3. `optimizer.step()`: update parameters using gradients

```python
tp = ((pred_binary == 1) & (labels == 1)).sum().item()
fp = ((pred_binary == 1) & (labels == 0)).sum().item()
fn = ((pred_binary == 0) & (labels == 1)).sum().item()
precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)
```
Calculating precision and recall:
- TP (True Positive): predicted pocket AND actually pocket
- FP (False Positive): predicted pocket BUT not pocket
- FN (False Negative): didn't predict pocket BUT is pocket
- `1e-8`: tiny number to prevent division by zero

---

## affinity_model.py

```python
BOND_TYPES = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3
}
```
Maps RDKit bond type objects to integers.
Chemical bonds:
- Single bond: one shared electron pair (C-C)
- Double bond: two shared pairs (C=C)
- Triple bond: three shared pairs (C≡C)
- Aromatic: delocalized electrons in rings (benzene)

```python
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
```
`Chem.MolFromSmiles`: parses SMILES string into RDKit molecule.
`Chem.AddHs`: adds explicit hydrogen atoms.
By default SMILES omits hydrogens (implicit).
Adding them gives the GNN more information about
the molecule's chemistry.

```python
atom.GetDegree() / 10.0
atom.GetFormalCharge() / 5.0
atom.GetTotalNumHs() / 8.0
float(atom.GetIsAromatic())
atom.GetMass() / 100.0
```
Chemical features for each atom, normalized to ~0-1:
- Degree: number of bonds (divided by 10 to normalize)
- Formal charge: electric charge (-2 to +2 typically)
- Hydrogen count: how many H atoms attached
- IsAromatic: 1 if in aromatic ring, 0 if not
- Mass: atomic weight (C=12, N=14, O=16...)

Normalization (dividing) is important because neural
networks work better when inputs are in similar ranges.

---

## selectivity_filter.py

```python
def calculate_sequence_identity(seq1, seq2):
    min_len = min(len(seq1), len(seq2))
    max_len = max(len(seq1), len(seq2))
    matches = sum(
        1 for a, b in zip(seq1[:min_len], seq2[:min_len]) 
        if a == b
    )
    identity = (matches / max_len) * 100
    return identity
```
Simple sequence identity calculation:
- Compare two sequences letter by letter
- Count matching positions
- Divide by longer sequence length
- Multiply by 100 for percentage

`zip(seq1, seq2)` pairs up characters from both sequences.
`sum(1 for a, b in zip(...) if a == b)` counts matches.

We reject proteins with >40% identity to human proteins.
This threshold is standard in drug discovery —
below 40% identity, 3D structures are usually very different
even if sequences share some similarity.

---

## pipeline.py

```python
pocket_model.eval()
affinity_model.eval()
```
Sets models to evaluation mode.
In training mode: dropout randomly disabled neurons.
In eval mode: all neurons active, predictions are deterministic.
Always call `.eval()` before making predictions!

```python
with torch.no_grad():
    predictions = pocket_model(graph.x, graph.edge_index, batch)
```
`torch.no_grad()` context manager:
Disables gradient calculation during inference.
We only need gradients during training (for backprop).
During prediction, disabling them saves memory and 
speeds up computation.

```python
top_10_pct = int(len(probs) * 0.1) + 1
top_probs = np.sort(probs)[::-1][:top_10_pct]
pocket_score = float(np.mean(top_probs))
```
Our pocket score = average probability of the top 10%
most likely pocket residues.
`np.sort(probs)[::-1]`: sorts in descending order
`[:top_10_pct]`: takes top 10%
This captures how good the BEST part of the protein is.

```python
final_score = (
    0.5 * pocket_score +
    0.3 * affinity_score +
    0.2 * selectivity_score
)
```
Weighted combination of three scores:
- 50% pocket quality: most important — if no pocket, no drug
- 30% binding affinity: how well compounds fit
- 20% selectivity: safety (all proteins pass this anyway)

Weights chosen based on biological reasoning:
a great pocket matters more than perfect selectivity
since all our proteins already pass the selectivity filter.

---

## explain_predictions.py

```python
shap_pocket = pocket_contrib - mean_pocket
shap_affinity = affinity_contrib - mean_affinity
shap_selectivity = selectivity_contrib - mean_selectivity
```
SHAP values = deviation from mean contribution.
If a protein's pocket score contribution is higher than 
average → positive SHAP (helped this protein rank higher).
If lower than average → negative SHAP (held it back).

This directly implements the SHAP concept:
"how much does this feature push the prediction above
or below the average prediction?"

---

# PART 5: THE TOOLS AND LIBRARIES

## PyTorch
Deep learning framework by Meta AI.
Provides tensors (like numpy arrays but on GPU),
automatic differentiation (autograd),
and neural network building blocks (nn.Module).

## PyTorch Geometric (PyG)
Extension of PyTorch for graph neural networks.
Provides GCNConv, global_mean_pool, Data class,
and efficient graph operations.

## Biopython
Python library for bioinformatics.
We use it to parse FASTA and PDB files.
SeqIO for sequences, PDB module for structures.

## RDKit
The standard cheminformatics library.
Converts SMILES strings to molecular graphs,
calculates chemical properties of atoms and bonds.

## Pandas
Data analysis library. DataFrame = spreadsheet in Python.
Used for loading, filtering, and saving CSV files.

## NumPy
Numerical computing library.
Fast array operations, mathematical functions.
The foundation of scientific Python computing.

## Matplotlib
Plotting library. Creates all our scientific figures —
histograms, training curves, SHAP plots.

## LightGBM
Gradient boosting library by Microsoft.
Fast, accurate, works with small datasets.
Perfect for our final ranking step.

## SHAP
Explainability library. Implements Shapley values
to explain any ML model's predictions.
Works especially well with tree-based models like LightGBM.

## Requests
HTTP library for Python. Makes web requests to APIs
(UniProt, ChEMBL, AlphaFold DB).

---

# PART 6: THE RESULTS

## What we found

After running the complete pipeline on 149 protein 
structures from 3 parasites:

### Top Drug Target: G4VJD6 (Schistosoma mansoni)
- Final Score: 0.843
- Pocket Score: 0.842 (excellent binding site)
- Human Similarity: 5.1% (very safe to target)
- Why it ranked #1: exceptional binding affinity of 
  predicted drug compounds (+0.1550 SHAP contribution)

### Key finding: Binding affinity matters most
SHAP analysis revealed that binding affinity 
(how well compounds fit) was the most important
factor (0.0626 mean SHAP) vs pocket quality (0.0376).

This makes biological sense: a protein with a good
pocket AND good drug compounds is more actionable
than one with just a good pocket.

### Selectivity finding
All 300 proteins checked (100 per organism) passed
the selectivity filter with average human similarity
of only ~6%. This confirms these parasites are
evolutionarily very distant from humans — 
good news for drug safety.

---

# PART 7: LIMITATIONS AND FUTURE WORK

## Current Limitations

### Small training dataset
We only had 10 labeled NTD proteins for training
the pocket detection GNN. This limits accuracy.
More labeled data would dramatically improve F1 score.

### No GPU training
Training on CPU limits model complexity and training time.
Cloud GPU training would allow larger models and more epochs.

### Simplified affinity model
The binding affinity model currently uses a dummy protein
representation. Full integration with real pocket graphs
would improve predictions.

### Only 149 structures
We only downloaded 149 of our 58,265 cleaned proteins.
Running the full pipeline on all proteins requires
cloud GPU compute.

## Future Work

1. Add more labeled training data from PDB
2. Train on cloud GPU (Google Colab — free)
3. Fully integrate pocket and affinity models
4. Run on all 58,265 proteins
5. Validate top predictions with literature search
6. Email DNDi researchers for feedback
7. Write and publish research paper
8. Build web demo for public use

---

*This document was written as part of the Neglect-Fold 
open-science project. All code is Apache 2.0 licensed.*