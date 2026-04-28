# Molecular Glue Classifier

This document serves as the main entry point and guide for the **Classifier** portion of the Molecular Glues project. The classifier determines whether a given molecule (SMILES string) is a likely molecular glue out of an embedded graph structure.

---

## 🏗️ Architecture at a Glance

- **Model:** Graph Transformer with node, edge, and optional global context pooling (`model/classifier.py`).
- **Layers:** Uses `GraphAttentionLayer` from the  architecture but introduces `ClassifierTransformerBlock`.
- **Pre-Processing:** Standard Graph encoding with fixed `drop_edge_rate` to prevent GNN topological overfitting.
- **Config:** Fully controlled via `config_classifier.py`.

---

## 🚀 Usage Guide

### 1. Data Preparation
To start training, you must construct the classifier dataset. Make sure you have positive components (glues) and negative counterparts (non-glues, ideally matched on molecular weight via stratification, or sampled as hard negatives).

```bash
# Create proper scaffold splits to prevent data leakage
python create_classifier_data.py
```
This generates `data/classifier_train.csv`, `data/classifier_val.csv`, and `data/classifier_test.csv` containing SMILES and a binary `label`.

*(Note: Data loaders are defined in `data/classifier_dataset.py`)*

### 2. Single-Run Training
To train the classifier once using the current config:

```bash
python train_classifier.py --epochs 50 --lr 1e-4 --batch_size 64
```

*(You can also use multi-GPU via `torchrun`). Check `config_classifier.py` for all hyperparameter defaults.*
Checkpoints fall into the `checkpoints/classifier/` directory out of the box.

### 3. Multi-Seed Training (For robust metrics)
Neural network initializations vary randomly. To prove robust classification performance, run a multi-seed experiment:

```bash
python run_multi_seed.py --num_seeds 5 --epochs 30
```
This script trains the model independently across random seeds, evaluates them on the test set, and reports averaged baseline statistics (AUC, MCC, F1, Accuracy, SPCC, PCC, Precision, Recall).

### 4. Evaluation
Evaluate the final trained model on the unseen test set:

```bash
python evaluate_classifier.py
```
This script generates overall model statistics and creates figures natively:
- `roc_curve.png` (ROC-AUC performance)
- `confusion_matrix.png` (Confusion Matrix Heatmap)
- `evaluation_metrics.csv` (Spreadsheet of Accuracy, F1, MCC, SPCC, etc.)
- `error_cases.csv` (Lists all false positives / false negatives)

### 5. Deep Diagnostics
If your model is achieving 99% accuracy off the bat, it's likely learning shortcuts (like predicting on molecular weight or sequence length rather than molecular bonds). 

Run the comprehensive diagnostics script to uncover data leakage and plot performance distributions for features:

```bash
python scripts/diagnose_classifier.py --checkpoint checkpoints/classifier/best_classifier.pt
```
*Outputs into `diagnostics/`.*

### 6. Interpretability (Attribution Visualizations)
To understand **why** the model predicted a molecule to be a glue or non-glue, generate an RDKit attribution map:

```bash
python interpret_classifier.py
```
The script runs gradients backward into input atom features and scales them, highlighting the specific atoms (green = important in prediction) that contributed to the model's forward pass!

---

## 🛠 File Hierarchy Map

| File/Folder | Purpose |
| ----------- | ------- |
| **`config_classifier.py`** | Central configuration file for classifier training parameters. |
| **`model/classifier.py`** | PyTorch model (`MolecularGlueClassifier`) and Graph Transformer layers. |
| **`data/classifier_dataset.py`** | Dataset class for extracting features from SMILES specifically for classification. |
| **`train_classifier.py`** | Main entry script to train the classifier locally or via DDP. |
| **`train/classifier_trainer.py`** | PyTorch training loop, batching, and loss computation object. |
| **`evaluate_classifier.py`** | Tests `best_classifier.pt` and dumps metrics + plots. |
| **`run_multi_seed.py`** | Trains N classifier models consecutively to report mean and standard deviation. |
| **`interpret_classifier.py`** | Script creating heatmap visual attributions mapping model predictions back onto specific RDKit atoms. |
| **`scripts/diagnose_classifier.py`** | Data validation framework to check for trivial chemical shortcuts and leakage. |
