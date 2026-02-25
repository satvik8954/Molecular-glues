#!/bin/bash
# ==========================================================================
# Molecular Glue Classifier Pipeline
# ==========================================================================
# Full pipeline: data preparation → training → evaluation
#
# Usage:
#   chmod +x run_classifier_pipeline.sh
#   ./run_classifier_pipeline.sh              # Run full pipeline
#   ./run_classifier_pipeline.sh --skip-data   # Skip data prep (if CSV exists)
#   ./run_classifier_pipeline.sh --epochs 10   # Quick test run
# ==========================================================================

set -e  # Exit on any error

# ---- Configuration ----
EPOCHS=50
BATCH_SIZE=32
LR=1e-4
DATA_PATH="data/classifier_dataset.csv"
CHECKPOINT_DIR="checkpoints/classifier"
SKIP_DATA=false

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-data)    SKIP_DATA=true; shift ;;
        --epochs)       EPOCHS="$2"; shift 2 ;;
        --batch-size)   BATCH_SIZE="$2"; shift 2 ;;
        --lr)           LR="$2"; shift 2 ;;
        --data-path)    DATA_PATH="$2"; shift 2 ;;
        *)              echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "========================================================================"
echo "  MOLECULAR GLUE CLASSIFIER PIPELINE"
echo "========================================================================"
echo ""
echo "  Settings:"
echo "    Epochs:     $EPOCHS"
echo "    Batch size: $BATCH_SIZE"
echo "    LR:         $LR"
echo "    Data path:  $DATA_PATH"
echo "    Skip data:  $SKIP_DATA"
echo ""

# ---- Step 0: Verify environment ----
echo "========================================================================"
echo "  STEP 0: Verifying environment"
echo "========================================================================"
echo ""

python -c "
import torch
print(f'  PyTorch:         {torch.__version__}')
print(f'  CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:             {torch.cuda.get_device_name(0)}')
" || { echo "ERROR: PyTorch not found"; exit 1; }

python -c "import torch_geometric; print(f'  PyG:             {torch_geometric.__version__}')" \
    || { echo "ERROR: torch_geometric not found. Run: pip install torch-geometric"; exit 1; }

python -c "import rdkit; print(f'  RDKit:           {rdkit.__version__}')" \
    || { echo "ERROR: rdkit not found. Run: pip install rdkit"; exit 1; }

python -c "import sklearn; print(f'  scikit-learn:    {sklearn.__version__}')" \
    || { echo "ERROR: scikit-learn not found. Run: pip install scikit-learn"; exit 1; }

echo ""
echo "  ✓ All dependencies OK"
echo ""

# ---- Step 1: Verify imports ----
echo "========================================================================"
echo "  STEP 1: Verifying classifier imports"
echo "========================================================================"
echo ""

python -c "
from model.classifier import MolecularGlueClassifier
from data.classifier_dataset import MolecularGlueClassifierDataset
from train.classifier_trainer import ClassifierTrainer
from config_classifier import ClassifierConfig
print('  ✓ All classifier imports OK')

model = MolecularGlueClassifier()
params = sum(p.numel() for p in model.parameters())
print(f'  ✓ Model created: {params:,} parameters')
" || { echo "ERROR: Import verification failed"; exit 1; }

echo ""

# ---- Step 2: Prepare data ----
echo "========================================================================"
echo "  STEP 2: Preparing dataset"
echo "========================================================================"
echo ""

if [ "$SKIP_DATA" = true ] && [ -f "$DATA_PATH" ]; then
    echo "  Skipping data preparation (--skip-data flag set)"
    echo "  Using existing: $DATA_PATH"
    python -c "
import pandas as pd
df = pd.read_csv('$DATA_PATH')
print(f'  Dataset: {len(df)} molecules')
print(f'    Glues:     {(df[\"label\"]==1).sum()}')
print(f'    Non-glues: {(df[\"label\"]==0).sum()}')
print(f'    Balance:   {df[\"label\"].mean():.2%} positive')
"
else
    if [ -f "$DATA_PATH" ]; then
        echo "  Dataset already exists: $DATA_PATH"
        python -c "
import pandas as pd
df = pd.read_csv('$DATA_PATH')
print(f'  Dataset: {len(df)} molecules')
print(f'    Glues:     {(df[\"label\"]==1).sum()}')
print(f'    Non-glues: {(df[\"label\"]==0).sum()}')
print(f'    Balance:   {df[\"label\"].mean():.2%} positive')
"
    else
        echo "  No dataset found at $DATA_PATH"
        echo "  Running data collection script..."
        echo ""

        # Try the data prep script
        if [ -f "data_prep/collect_classifier_data.py" ]; then
            python data_prep/collect_classifier_data.py || {
                echo ""
                echo "  WARNING: Data collection script failed."
                echo "  Creating a minimal demo dataset instead..."
                echo ""
                python -c "
import pandas as pd
from rdkit import Chem

# Load existing glue chemotypes
glues = pd.read_csv('data/glue_chemotypes.csv')
glues['label'] = 1
glues = glues[['SMILES', 'label']]

# Load ChEMBL druglike as non-glues
chembl = pd.read_csv('data/chembl_druglike.csv')
chembl['label'] = 0
chembl = chembl[['SMILES', 'label']]

# Balance: take same number of non-glues as glues
n_glues = len(glues)
non_glues = chembl.sample(n=n_glues, random_state=42)

# Combine
dataset = pd.concat([glues, non_glues], ignore_index=True)
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

# Validate SMILES
valid = dataset[dataset['SMILES'].apply(lambda s: Chem.MolFromSmiles(s) is not None)]
valid.to_csv('$DATA_PATH', index=False)

print(f'  ✓ Dataset created: {len(valid)} molecules')
print(f'    Glues:     {(valid[\"label\"]==1).sum()}')
print(f'    Non-glues: {(valid[\"label\"]==0).sum()}')
"
            }
        else
            echo "  No data prep script found. Creating from existing CSV files..."
            python -c "
import pandas as pd
from rdkit import Chem

# Load existing glue chemotypes
glues = pd.read_csv('data/glue_chemotypes.csv')
glues['label'] = 1
glues = glues[['SMILES', 'label']]

# Load ChEMBL druglike as non-glues
chembl = pd.read_csv('data/chembl_druglike.csv')
chembl['label'] = 0
chembl = chembl[['SMILES', 'label']]

# Balance: take same number of non-glues as glues
n_glues = len(glues)
non_glues = chembl.sample(n=n_glues, random_state=42)

# Combine
dataset = pd.concat([glues, non_glues], ignore_index=True)
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

# Validate SMILES
valid = dataset[dataset['SMILES'].apply(lambda s: Chem.MolFromSmiles(s) is not None)]
valid.to_csv('$DATA_PATH', index=False)

print(f'  ✓ Dataset created: {len(valid)} molecules')
print(f'    Glues:     {(valid[\"label\"]==1).sum()}')
print(f'    Non-glues: {(valid[\"label\"]==0).sum()}')
"
        fi
    fi
fi

echo ""

# ---- Step 3: Test forward pass ----
echo "========================================================================"
echo "  STEP 3: Testing model forward pass"
echo "========================================================================"
echo ""

python -c "
import torch
from data.molecular_graph import smiles_to_graph
from model.classifier import MolecularGlueClassifier

model = MolecularGlueClassifier()
model.eval()

test_molecules = {
    'Benzene':     'c1ccccc1',
    'Ethanol':     'CCO',
    'Aspirin':     'CC(=O)Oc1ccccc1C(=O)O',
}

print('  Forward pass test:')
for name, smiles in test_molecules.items():
    graph = smiles_to_graph(smiles)
    if graph is not None:
        batch = torch.zeros(graph.num_nodes, dtype=torch.long)
        with torch.no_grad():
            logits = model(graph.x, graph.edge_index, graph.edge_attr, batch)
            prob = torch.sigmoid(logits).item()
        print(f'    {name:15s}  prob={prob:.4f}  (untrained, expect ~0.5)')
    else:
        print(f'    {name:15s}  FAILED to convert')

print()
print('  ✓ Forward pass OK')
" || { echo "ERROR: Forward pass test failed"; exit 1; }

echo ""

# ---- Step 4: Test data loading ----
echo "========================================================================"
echo "  STEP 4: Testing data loading"
echo "========================================================================"
echo ""

python -c "
from data.classifier_dataset import MolecularGlueClassifierDataset

ds = MolecularGlueClassifierDataset('$DATA_PATH', split='train', val_split=0.2)
sample = ds[0]
print(f'  Sample: {sample.num_nodes} nodes, {sample.edge_index.shape[1]} edges, label={sample.y.item():.0f}')
print(f'  Node features shape: {sample.x.shape}')
print(f'  Edge features shape: {sample.edge_attr.shape}')
print()
print('  ✓ Data loading OK')
" || { echo "ERROR: Data loading test failed"; exit 1; }

echo ""

# ---- Step 5: Train ----
echo "========================================================================"
echo "  STEP 5: Training classifier ($EPOCHS epochs)"
echo "========================================================================"
echo ""

mkdir -p "$CHECKPOINT_DIR"

python train_classifier.py \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --data_path "$DATA_PATH"

echo ""

# ---- Step 6: Evaluate ----
echo "========================================================================"
echo "  STEP 6: Evaluating trained model"
echo "========================================================================"
echo ""

if [ -f "$CHECKPOINT_DIR/best_classifier.pt" ]; then
    python -c "
import torch
from data.molecular_graph import smiles_to_graph
from model.classifier import MolecularGlueClassifier
from config_classifier import ClassifierConfig

# Load best model
config = ClassifierConfig()
model = MolecularGlueClassifier(
    hidden_dim=config.hidden_dim,
    num_layers=config.num_layers,
    num_heads=config.num_heads,
    dropout=config.dropout
)
checkpoint = torch.load('$CHECKPOINT_DIR/best_classifier.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

best_acc = checkpoint.get('best_val_acc', 'N/A')
epoch = checkpoint.get('epoch', 'N/A')
print(f'  Best model from epoch {epoch}')
print(f'  Best validation accuracy: {best_acc}')
print()

# Test on known molecules
test_molecules = {
    'Lenalidomide (glue)':    'O=C1CCC(=O)N1c1cccc2[nH]ccc12',
    'Thalidomide (glue)':     'O=C1CCC(=O)N1C1CCC(=O)NC1=O',
    'Pomalidomide (glue)':    'O=C1CCC(=O)N1c1cccc2c(N)nccc12',
    'Aspirin (non-glue)':     'CC(=O)Oc1ccccc1C(=O)O',
    'Paracetamol (non-glue)': 'CC(=O)Nc1ccc(O)cc1',
    'Caffeine (non-glue)':    'Cn1c(=O)c2c(ncn2C)n(C)c1=O',
    'Ibuprofen (non-glue)':   'CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O',
}

print('  Predictions on test molecules:')
print('  ' + '-' * 60)
for name, smiles in test_molecules.items():
    graph = smiles_to_graph(smiles)
    if graph is not None:
        batch = torch.zeros(graph.num_nodes, dtype=torch.long)
        with torch.no_grad():
            logits = model(graph.x, graph.edge_index, graph.edge_attr, batch)
            prob = torch.sigmoid(logits).item()
        label = 'GLUE' if prob > 0.5 else 'non-glue'
        marker = '✓' if ('glue' in name.lower() and prob > 0.5) or ('non-glue' in name.lower() and prob <= 0.5) else '✗'
        print(f'    {marker} {name:30s}  prob={prob:.4f}  [{label}]')
    else:
        print(f'    ? {name:30s}  FAILED to convert')

print()
" || echo "  WARNING: Evaluation script had issues"
else
    echo "  No checkpoint found at $CHECKPOINT_DIR/best_classifier.pt"
    echo "  Training may not have saved a checkpoint."
fi

echo ""

# ---- Summary ----
echo "========================================================================"
echo "  PIPELINE COMPLETE"
echo "========================================================================"
echo ""
echo "  Outputs:"
echo "    Checkpoint:  $CHECKPOINT_DIR/best_classifier.pt"
echo "    Dataset:     $DATA_PATH"
echo ""
echo "  Next steps:"
echo "    1. Evaluate:  python evaluate_classifier.py"
echo "    2. Interpret: python interpret_classifier.py"
echo "    3. Retrain:   ./run_classifier_pipeline.sh --skip-data --epochs 100"
echo ""
echo "========================================================================"
