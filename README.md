# 🧬 Neglect-Fold

> A deep learning pipeline for druggable target discovery in neglected tropical diseases

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

**Target diseases:** Chagas · Leishmaniasis · Schistosomiasis

Neglected tropical diseases affect over **1 billion people** worldwide yet receive less than 1% of pharmaceutical R&D spending. Neglect-Fold uses modern deep learning to identify druggable protein targets and propose candidate compounds for three NTDs — building an end-to-end pipeline from raw genome data to ranked drug target predictions.

---

## 🔬 What this pipeline does
58,265 parasite proteins
↓
Pocket Detection GNN (F1: 0.662)
↓
Binding Affinity Model (RMSE: 0.87)
↓
Selectivity Filter vs human proteins
↓
Top 20 ranked drug targets with explanations
---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Proteins analyzed | 58,265 |
| Drug compounds | 33,006 |
| 3D structures | 149 |
| Pocket detection F1 | 0.662 |
| Binding affinity RMSE | 0.87 |
| Human selectivity | 100% pass rate |
| Top drug targets identified | 20 |

### Top 5 Drug Target Candidates

| Rank | Protein | Disease | Score |
|------|---------|---------|-------|
| 1 | G4VJD6 | Schistosomiasis | 0.843 |
| 2 | Q4CQJ5 | Chagas | 0.817 |
| 3 | P15964 | Schistosomiasis | 0.811 |
| 4 | Q4CMY9 | Chagas | 0.788 |
| 5 | G4VB10 | Schistosomiasis | 0.778 |

---

## 🏗️ Architecture

Neglect-Fold uses a dual-network Graph Neural Network architecture:

### Pocket Detection GNN
Protein 3D Structure
↓
┌─────────────────┐   ┌─────────────────┐
│   Local GNN     │   │   Global GNN    │
│ (neighborhood)  │   │ (whole protein) │
└────────┬────────┘   └────────┬────────┘
└──────────┬──────────┘
↓
Fusion Network
↓
Pocket probability per residue
### Binding Affinity Model
Protein Pocket + Drug Molecule
↓
┌─────────────┐   ┌─────────────┐
│ Protein GNN │   │ Molecule GNN│
└──────┬──────┘   └──────┬──────┘
└────────┬─────────┘
↓
Fusion Network
↓
Predicted pIC50
---

## 📁 Project Structure
neglect-fold/
├── data/
│   ├── raw/                    # Downloaded proteomes + ChEMBL
│   └── processed/              # Cleaned data + structures
├── models/
│   ├── pocket_gnn_best.pt      # Trained pocket detection GNN
│   └── affinity_model.pt       # Trained binding affinity model
├── results/
│   ├── top20_drug_targets.csv  # Final ranked predictions
│   ├── shap_explanations.csv   # Per-protein explanations
│   └── figures/                # All plots and visualizations
└── src/
├── download_proteomes.py   # Data collection
├── clean_proteomes.py      # Data preprocessing
├── download_chembl.py      # Drug compound collection
├── download_structures.py  # 3D structure collection
├── pocket_gnn.py           # Pocket detection model
├── affinity_model.py       # Binding affinity model
├── selectivity_filter.py   # Human similarity filter
├── train_gnn.py            # Model training
├── explain_predictions.py  # SHAP explainability
└── pipeline.py             # End-to-end pipeline
---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Dhruv-Gupta2711/neglect-fold.git
cd neglect-fold
```

### 2. Set up environment
```bash
conda create -n neglectfold python=3.11 -y
conda activate neglectfold
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric numpy pandas matplotlib biopython rdkit lightgbm shap
```

### 3. Download data
```bash
python src/download_proteomes.py
python src/download_chembl.py
python src/download_structures.py
```

### 4. Train models
```bash
python src/prepare_training_data.py
python src/train_gnn.py
python src/affinity_model.py
```

### 5. Run full pipeline
```bash
python src/pipeline.py
python src/explain_predictions.py
```

---

## 📈 Data Sources

| Dataset | Description | Size |
|---------|-------------|------|
| UniProt | Parasite protein sequences | 60,896 proteins |
| AlphaFold DB | Predicted 3D structures | 149 structures |
| ChEMBL | Bioactivity data | 33,006 compounds |
| PDB | Experimental structures for training | 10 structures |

---

## 🎯 Target Diseases

### Chagas Disease
- Caused by *Trypanosoma cruzi*
- ~6 million people affected
- Current drugs have severe side effects

### Leishmaniasis  
- Caused by *Leishmania donovani*
- 700,000–1 million new cases/year
- Treatment is toxic and expensive

### Schistosomiasis
- Caused by *Schistosoma mansoni*
- ~240 million people infected
- Only one drug available, resistance emerging

---

## 🔍 How it works

1. **Data Collection** — Downloads all proteins for 3 parasites from UniProt, drug compounds from ChEMBL, and 3D structures from AlphaFold DB

2. **Pocket Detection** — A dual-network GNN predicts which amino acids form drug binding pockets on each protein

3. **Binding Affinity** — A second GNN scores how well known drug compounds fit into predicted pockets

4. **Selectivity Filter** — Rejects any protein too similar to human proteins (would cause side effects)

5. **Ranking + Explanation** — LightGBM + SHAP produces an explainable ranked list of the top 20 drug targets

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE)

---

## 🙏 Acknowledgements

- [AlphaFold](https://alphafold.ebi.ac.uk/) by DeepMind for protein structure predictions
- [ChEMBL](https://www.ebi.ac.uk/chembl/) by EMBL-EBI for bioactivity data
- [UniProt](https://www.uniprot.org/) for protein sequence data
- [DNDi](https://dndi.org/) for inspiration and open science in NTD research

---

*Built as an open-science contribution to neglected tropical disease research.*