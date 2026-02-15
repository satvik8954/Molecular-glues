#!/bin/bash
# run_full_pipeline.sh
# Complete end-to-end workflow for molecular glue generation
#
# Usage:
#   bash scripts/run_full_pipeline.sh
#   bash scripts/run_full_pipeline.sh --skip_download   # skip data collection
#   bash scripts/run_full_pipeline.sh --quick            # quick test run

set -e  # Exit on error

# ── Configuration ──────────────────────────────────────────────────
TARGET_SIZE=${TARGET_SIZE:-50000}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_EPOCHS=${NUM_EPOCHS:-100}
NUM_GENERATE=${NUM_GENERATE:-5000}
TOP_N=${TOP_N:-100}
CHECKPOINT=${CHECKPOINT:-checkpoints/best_model.pt}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/generation_1}

SKIP_DOWNLOAD=false
QUICK=false

for arg in "$@"; do
    case $arg in
        --skip_download) SKIP_DOWNLOAD=true ;;
        --quick)
            QUICK=true
            TARGET_SIZE=200
            NUM_EPOCHS=5
            NUM_GENERATE=100
            BATCH_SIZE=16
            TOP_N=20
            ;;
    esac
done

echo "========================================="
echo "MOLECULAR GLUE DIFFUSION PIPELINE"
echo "========================================="
echo "Target download size: $TARGET_SIZE"
echo "Batch size:           $BATCH_SIZE"
echo "Epochs:               $NUM_EPOCHS"
echo "Generate:             $NUM_GENERATE molecules"
echo "Top N:                $TOP_N"
echo "Output:               $OUTPUT_DIR"
echo "========================================="

START_TIME=$(date +%s)

# ── Step 1: Data Collection ────────────────────────────────────────
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo ""
    echo "STEP 1: Collecting training data ..."
    python scripts/collect_data.py \
        --target_size "$TARGET_SIZE" \
        --output data/raw/chembl_druglike.csv \
        --force

    if [ ! -f data/raw/chembl_druglike.csv ]; then
        echo "ERROR: Data collection failed."
        exit 1
    fi
    echo "✓ Data collection complete."
else
    echo ""
    echo "STEP 1: Skipping data collection (--skip_download)"
    if [ ! -f data/raw/chembl_druglike.csv ]; then
        echo "ERROR: data/raw/chembl_druglike.csv not found."
        echo "Run without --skip_download first."
        exit 1
    fi
fi

# ── Step 2: Preprocess Data ────────────────────────────────────────
echo ""
echo "STEP 2: Preprocessing data ..."
python scripts/preprocess_data.py \
    --chembl data/raw/chembl_druglike.csv \
    --glues data/glue_chemotypes.csv \
    --output data/processed/training_data.csv \
    --glue_fraction 0.15 \
    --force

if [ ! -f data/processed/training_data.csv ]; then
    echo "ERROR: Preprocessing failed."
    exit 1
fi
echo "✓ Preprocessing complete."

# ── Step 3: Train Model ───────────────────────────────────────────
echo ""
echo "STEP 3: Training model ..."
TRAIN_ARGS="--data_path data/processed/training_data.csv --batch_size $BATCH_SIZE --epochs $NUM_EPOCHS"

if [ "$QUICK" = true ]; then
    TRAIN_ARGS="$TRAIN_ARGS --quick_test"
fi

# Resume from checkpoint if it exists
if [ -f "$CHECKPOINT" ]; then
    echo "  Resuming from existing checkpoint: $CHECKPOINT"
    TRAIN_ARGS="$TRAIN_ARGS --checkpoint $CHECKPOINT"
fi

python train_model.py $TRAIN_ARGS

echo "✓ Training complete."

# ── Step 4: Validate Checkpoint ────────────────────────────────────
echo ""
echo "STEP 4: Validating checkpoint ..."
if [ ! -f "$CHECKPOINT" ]; then
    # Try to find any checkpoint
    LATEST=$(ls -t checkpoints/*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        CHECKPOINT="$LATEST"
        echo "  Using latest checkpoint: $CHECKPOINT"
    else
        echo "ERROR: No checkpoint found in checkpoints/"
        exit 1
    fi
fi
echo "✓ Checkpoint validated: $CHECKPOINT"

# ── Step 5: Generate Molecules ─────────────────────────────────────
echo ""
echo "STEP 5: Generating molecules ..."
mkdir -p "$OUTPUT_DIR"

python generate_molecules.py \
    --checkpoint "$CHECKPOINT" \
    --num_molecules "$NUM_GENERATE" \
    --diverse \
    --evaluate \
    --training_data data/processed/training_data.csv \
    --known_glues data/glue_chemotypes.csv \
    --output_dir "$OUTPUT_DIR"

echo "✓ Generation complete."

# ── Step 6: Filter and Rank ────────────────────────────────────────
echo ""
echo "STEP 6: Filtering and ranking candidates ..."

# Find the generated CSV (name may vary)
GEN_CSV=$(ls "$OUTPUT_DIR"/*.csv 2>/dev/null | head -1)
if [ -z "$GEN_CSV" ]; then
    GEN_CSV="$OUTPUT_DIR/generated_molecules.csv"
fi

if [ -f "$GEN_CSV" ]; then
    python scripts/filter_molecules.py \
        --input "$GEN_CSV" \
        --output "$OUTPUT_DIR/top_candidates.csv" \
        --top_n "$TOP_N"
    echo "✓ Filtering complete."
else
    echo "⚠ Generated CSV not found. Skipping filtering."
fi

# ── Step 7: Visualise ──────────────────────────────────────────────
echo ""
echo "STEP 7: Generating visualisation report ..."

if [ -f "$OUTPUT_DIR/top_candidates.csv" ]; then
    python -c "
import sys; sys.path.insert(0, '.')
from eval.visualize import generate_full_report
import pandas as pd
gen = pd.read_csv('$OUTPUT_DIR/top_candidates.csv')['smiles'].dropna().tolist()
train = pd.read_csv('data/processed/training_data.csv')['smiles'].dropna().tolist()
glues = pd.read_csv('data/glue_chemotypes.csv')['smiles'].dropna().tolist()
generate_full_report(gen, train, '$OUTPUT_DIR/report', glues)
"
    echo "✓ Visualisation complete."
else
    echo "⚠ Candidates CSV not found. Skipping visualisation."
fi

# ── Done ───────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "========================================="
echo "PIPELINE COMPLETE!"
echo "========================================="
echo "Total time:  ${MINUTES}m ${SECONDS}s"
echo "Outputs:     $OUTPUT_DIR/"
echo ""
echo "Key files:"
echo "  Training data:    data/processed/training_data.csv"
echo "  Checkpoint:       $CHECKPOINT"
if [ -f "$OUTPUT_DIR/top_candidates.csv" ]; then
    N_CAND=$(wc -l < "$OUTPUT_DIR/top_candidates.csv")
    echo "  Top candidates:   $OUTPUT_DIR/top_candidates.csv ($((N_CAND - 1)) molecules)"
fi
if [ -d "$OUTPUT_DIR/report" ]; then
    echo "  Report:           $OUTPUT_DIR/report/"
fi
echo "========================================="
