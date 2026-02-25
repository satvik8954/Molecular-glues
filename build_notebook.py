#!/usr/bin/env python3
"""
Build script that assembles all source modules into a single Jupyter notebook.
Run: python build_notebook.py
Output: molecular_glues_pipeline.ipynb
"""
import json, textwrap, os

# ── helpers ────────────────────────────────────────────────────────
def md_cell(source: str):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(source)}

def code_cell(source: str, eid=None):
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(source),
    }
    if eid:
        cell["id"] = eid
    return cell

def _lines(text: str):
    """Split text into a list of lines each ending with \\n (except last)."""
    text = textwrap.dedent(text).strip()
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]

def read_src(relpath: str) -> str:
    """Read a source file from the project, strip sys.path hacks and rewrite imports."""
    import re

    base = os.path.dirname(os.path.abspath(__file__))
    full = os.path.join(base, relpath)
    with open(full, "r", encoding="utf-8") as f:
        text = f.read()

    # Normalise line endings to \n for consistent regex matching
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # ── Remove ALL sys.path / __file__ / _project_root injection blocks ──
    # Broad pattern: catches "import sys\nimport os" ... "_project_root = ..." ... "sys.path.insert(...)"
    # regardless of single vs double os.path.dirname, optional comments, blank lines between.
    text = re.sub(
        r'^import sys\n'
        r'import os\n'
        r'(?:\n|#[^\n]*\n)*'                            # optional blanks / comments
        r'_project_root\s*=\s*[^\n]+__file__[^\n]*\n'    # _project_root = ...(__file__)...
        r'if _project_root not in sys\.path:\n'
        r'    sys\.path\.insert\(0,\s*_project_root\)\n',
        '',
        text,
        flags=re.MULTILINE,
    )

    # Safety net: remove any surviving individual lines that use __file__ or _project_root
    # (some files have slightly different formatting)
    text = re.sub(r'^_project_root\s*=\s*[^\n]*__file__[^\n]*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^if _project_root not in sys\.path:\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*sys\.path\.insert\(0,\s*_project_root\)\s*$', '', text, flags=re.MULTILINE)
    # Remove standalone "import sys" and "import os" that were ONLY used for the path hack
    # (keep them if they appear inside functions or with other uses — but strip leading pairs)
    text = re.sub(r'^import sys\nimport os\n\n(?=\n|#|\n#)', '', text, flags=re.MULTILINE)

    # Mapping: module prefix → replacement comment
    _import_rewrites = {
        r'config':                   '# (config already defined above)',
        r'model\.noise_scheduler':   '# (NoiseScheduler defined above)',
        r'model\.graph_transformer': '# (GraphTransformer defined above)',
        r'model\.diffusion':         '# (MolecularDiffusion defined above)',
        r'model\.fragment_vocab':    '# (FragmentVocab defined above)',
        r'data\.molecular_graph':    '# (molecular_graph functions defined above)',
        r'data\.dataset':            '# (dataset functions defined above)',
        r'utils\.chemistry':         '# (chemistry utils defined above)',
        r'utils\.filters':           '# (filter functions defined above)',
        r'utils\.glue_scoring':      '# (glue scoring defined above)',
        r'generate\.sampler':        '# (MoleculeSampler defined above)',
        r'generate\.postprocess':    '# (postprocess functions defined above)',
        r'eval\.metrics':            '# (metrics functions defined above)',
        r'eval\.visualize':          '# (visualization functions defined above)',
        r'eval\.analyze':            '# (analyze functions defined above)',
        r'train\.trainer':           '# (Trainer defined above)',
    }

    for mod, comment in _import_rewrites.items():
        # 1) Multi-line: from mod import (\n  ...\n)
        text = re.sub(
            r'^from\s+' + mod + r'\s+import\s*\(.*?\)',
            comment,
            text,
            flags=re.MULTILINE | re.DOTALL,
        )
        # 2) Single-line: from mod import X, Y, Z
        text = re.sub(
            r'^from\s+' + mod + r'\s+import\s+[^\n]+$',
            comment,
            text,
            flags=re.MULTILINE,
        )

    # Clean up multiple consecutive blank lines left by removals
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

# ── cells ──────────────────────────────────────────────────────────
cells = []

# ═══════════════════════════════════════════════════════════════════
# 1. Title & Intro
# ═══════════════════════════════════════════════════════════════════
cells.append(md_cell("""
# 🧬 Molecular Glues Diffusion Model – Full Pipeline Notebook

This self-contained notebook implements the **complete pipeline** for training and using a
graph-based discrete diffusion model to generate novel molecular glue candidates.

**Pipeline stages:**
1. **Setup** – install dependencies, mount Google Drive, configure device
2. **Configuration** – all hyperparameters in one place
3. **Utilities** – chemistry helpers, molecular filters, glue scoring
4. **Data** – graph representations, dataset loading, preprocessing
5. **Model** – noise scheduler, graph transformer, diffusion model
6. **Training** – trainer with EMA, LR scheduling, checkpointing
7. **Generation** – sampler, post-processing, diverse sampling
8. **Evaluation** – metrics, visualizations, full report
9. **Run** – end-to-end training → generation → analysis

> **Platforms:** Google Colab (with Drive mount) · DGX · any CUDA machine

---
"""))

# ═══════════════════════════════════════════════════════════════════
# 2. Setup & Installation
# ═══════════════════════════════════════════════════════════════════
cells.append(md_cell("## 1 · Environment Setup"))

cells.append(code_cell("""\
# ── Install dependencies ──────────────────────────────────────────
# Run this cell once per runtime to install all required packages.
# On Colab the base torch is pre-installed; we add the extras.

!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q torch-geometric
!pip install -q torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
!pip install -q rdkit
!pip install -q scipy pandas seaborn tqdm matplotlib numpy

print("✅ All dependencies installed.")
"""))

cells.append(md_cell("### 1.1 · Google Drive Mount (optional)\nMount your Google Drive to persist checkpoints and data across sessions."))

cells.append(code_cell("""\
# ── Google Drive mount ────────────────────────────────────────────
# Uncomment the block below if running on Google Colab and you want to
# save / load checkpoints and data from your Drive.

# from google.colab import drive
# drive.mount('/content/drive')

# Set these paths to your Drive folders:
DRIVE_DATA_DIR   = "/content/drive/MyDrive/molecular_glues/data"
DRIVE_CKPT_DIR   = "/content/drive/MyDrive/molecular_glues/checkpoints"
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/molecular_glues/outputs"

# Local fallbacks (used when Drive is not mounted)
import os
DATA_DIR   = DRIVE_DATA_DIR   if os.path.isdir(DRIVE_DATA_DIR)   else "./data"
CKPT_DIR   = DRIVE_CKPT_DIR   if os.path.isdir(DRIVE_CKPT_DIR)   else "./checkpoints"
OUTPUT_DIR = DRIVE_OUTPUT_DIR  if os.path.isdir(DRIVE_OUTPUT_DIR)  else "./outputs"

for d in [DATA_DIR, CKPT_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"Data dir:       {DATA_DIR}")
print(f"Checkpoint dir: {CKPT_DIR}")
print(f"Output dir:     {OUTPUT_DIR}")
"""))

cells.append(md_cell("### 1.2 · Device Configuration"))

cells.append(code_cell("""\
# ── Device & seed ─────────────────────────────────────────────────
import torch
import numpy as np
import random

SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
"""))

# ═══════════════════════════════════════════════════════════════════
# 3. Configuration
# ═══════════════════════════════════════════════════════════════════
cells.append(md_cell("## 2 · Configuration\nAll hyperparameters, vocabulary mappings, and dataclass-based config objects."))

cells.append(code_cell(read_src("config.py")))

# ═══════════════════════════════════════════════════════════════════
# 4. Utilities – Chemistry
# ═══════════════════════════════════════════════════════════════════
cells.append(md_cell("## 3 · Utility Modules\n### 3.1 · Chemistry Utilities\nCore RDKit helpers for validation, property calculation, fingerprints, and scaffolds."))

cells.append(code_cell(read_src("utils/chemistry.py")))

# 4b. Filters
cells.append(md_cell("### 3.2 · Molecular Filters\nDrug-likeness, glue-likeness, PAINS, SA score, and reactive-group filters."))

cells.append(code_cell(read_src("utils/filters.py")))

# 4c. Glue Scoring
cells.append(md_cell("### 3.3 · Glue-Likeness Scoring\nScore molecules 0–100 for molecular-glue similarity based on warhead motifs, scaffolds, and physicochemical properties."))

cells.append(code_cell(read_src("utils/glue_scoring.py")))

# ═══════════════════════════════════════════════════════════════════
# 5. Data – Molecular Graph
# ═══════════════════════════════════════════════════════════════════
cells.append(md_cell("## 4 · Data Modules\n### 4.1 · Molecular Graph Representation\nConvert SMILES ↔ PyTorch Geometric `Data` objects with full node & edge features."))

cells.append(code_cell(read_src("data/molecular_graph.py")))

# 5b. Dataset
cells.append(md_cell("### 4.2 · Dataset & Data Loading\nPyTorch Geometric `InMemoryDataset` that loads molecules from CSV/SMILES files and converts them to graphs."))

cells.append(code_cell(read_src("data/dataset.py")))

# ═══════════════════════════════════════════════════════════════════
# 6. Data – Download & Preprocess Scripts (inlined as functions)
# ═══════════════════════════════════════════════════════════════════
cells.append(md_cell("### 4.3 · Data Download & Preprocessing\nFunctions to download training molecules from ChEMBL and preprocess/merge datasets."))

# Read and convert the scripts to function form (remove if __name__ blocks)
download_src = read_src("scripts/download_training_data.py")
preprocess_src = read_src("scripts/preprocess_data.py")

# Remove argparse and __main__ blocks
import re
download_src = re.sub(r'if __name__.*', '', download_src, flags=re.DOTALL).strip()
preprocess_src = re.sub(r'if __name__.*', '', preprocess_src, flags=re.DOTALL).strip()

cells.append(code_cell(download_src + "\n\n" + preprocess_src))

# ═══════════════════════════════════════════════════════════════════
# 7. Model – Noise Scheduler
# ═══════════════════════════════════════════════════════════════════
cells.append(md_cell("## 5 · Model Components\n### 5.1 · Noise Scheduler\nDiscrete diffusion noise scheduler with forward (noise-adding) and reverse (posterior) processes."))

cells.append(code_cell(read_src("model/noise_scheduler.py")))

# 7b. Fragment Vocab
cells.append(md_cell("### 5.2 · Fragment Vocabulary\nBRICS-based fragment vocabulary for scoring generated molecules."))

cells.append(code_cell(read_src("model/fragment_vocab.py")))

# 7c. Graph Transformer
cells.append(md_cell("""\
### 5.3 · Graph Transformer
Multi-scale graph transformer with:
- Graph attention with edge feature propagation
- Ring-centric attention
- Global graph pooling with gating
- FiLM property conditioning with classifier-free guidance
- Gradient checkpointing support
"""))

cells.append(code_cell(read_src("model/graph_transformer.py")))

# 7d. Diffusion Model
cells.append(md_cell("### 5.4 · Molecular Diffusion Model\nComplete diffusion model combining the noise scheduler and graph transformer for training and sampling."))

cells.append(code_cell(read_src("model/diffusion.py")))

# ═══════════════════════════════════════════════════════════════════
# 8. Training
# ═══════════════════════════════════════════════════════════════════
cells.append(md_cell("## 6 · Training\n### 6.1 · Trainer\nTraining loop with EMA, cosine-annealing LR schedule, warmup, checkpointing, and validation."))

cells.append(code_cell(read_src("train/trainer.py")))

# ═══════════════════════════════════════════════════════════════════
# 9. Generation – Sampler
# ═══════════════════════════════════════════════════════════════════
cells.append(md_cell("## 7 · Generation\n### 7.1 · Molecule Sampler\nHigh-level sampler supporting basic, guided, diverse, and rejection sampling."))

cells.append(code_cell(read_src("generate/sampler.py")))

# 9b. Post-processing
cells.append(md_cell("### 7.2 · Post-Processing\nConvert generated graphs to SMILES, filter, deduplicate, and save."))

cells.append(code_cell(read_src("generate/postprocess.py")))

# 9c. Diverse Sampler
cells.append(md_cell("### 7.3 · Diverse Sampler\nExtended sampler with variable sizes, temperatures, guidance scales, and scaffold-diversity tracking."))

cells.append(code_cell(read_src("generate/diverse_sampler.py")))

# ═══════════════════════════════════════════════════════════════════
# 10. Evaluation
# ═══════════════════════════════════════════════════════════════════
cells.append(md_cell("## 8 · Evaluation\n### 8.1 · Metrics\nValidity, uniqueness, novelty, diversity, scaffold diversity, QED, Wasserstein distance, and glue similarity."))

cells.append(code_cell(read_src("eval/metrics.py")))

# 10b. Visualization
cells.append(md_cell("### 8.2 · Visualization\nProperty distributions, molecule grids, scatter matrices, scaffold charts, correlation heatmaps, and full reports."))

cells.append(code_cell(read_src("eval/visualize.py")))

# 10c. Analyze – strip argparse and __main__ block before inlining
cells.append(md_cell("### 8.3 · Analysis\nHigh-level analysis utilities for computing properties and generating summary reports."))

analyze_src = read_src("eval/analyze.py")
import re as _re
# Remove argparse import and parse_args function
analyze_src = _re.sub(r'^import argparse\n', '', analyze_src, flags=_re.MULTILINE)
analyze_src = _re.sub(r'^def parse_args\(\):.*?(?=\ndef )', '', analyze_src, flags=_re.MULTILINE | _re.DOTALL)
# Remove if __name__ block
analyze_src = _re.sub(r'if __name__.*', '', analyze_src, flags=_re.DOTALL).strip()
cells.append(code_cell(analyze_src))

# ═══════════════════════════════════════════════════════════════════
# 11. End-to-end pipeline
# ═══════════════════════════════════════════════════════════════════
cells.append(md_cell("""\
---
## 9 · Run the Pipeline
The cells below wire everything together for a complete train → generate → evaluate workflow.
Adjust the configuration variables at the top of each cell to suit your hardware and data.
"""))

# 11a. Data preparation
cells.append(md_cell("### 9.1 · Prepare Training Data"))

cells.append(code_cell("""\
# ── Download 50k drug-like molecules from ChEMBL ─────────────────
# This cell downloads ~50,000 drug-like molecules from ChEMBL to use
# as training data alongside the glue chemotypes.
# ⚡ Takes ~5-15 minutes depending on network speed.
# If ChEMBL is unavailable, falls back to existing glue_chemotypes.csv.

!pip install -q chembl_webresource_client

import os
import pandas as pd

CHEMBL_CSV = os.path.join(DATA_DIR, "chembl_druglike.csv")
GLUE_CSV   = os.path.join(DATA_DIR, "glue_chemotypes.csv")
TRAINING_DATA_PATH = os.path.join(DATA_DIR, "training_data.csv")

# --- Download from ChEMBL ---
if os.path.exists(CHEMBL_CSV):
    print(f"✅ ChEMBL data already exists at {CHEMBL_CSV}, skipping download.")
    df_chembl = pd.read_csv(CHEMBL_CSV)
    print(f"   {len(df_chembl)} molecules loaded.")
else:
    print("📥 Downloading ~50,000 drug-like molecules from ChEMBL...")
    chembl_smiles = download_chembl_druglike(n_molecules=50_000)

    if chembl_smiles:
        # Validate with RDKit
        chembl_smiles = validate_smiles_list(chembl_smiles)
        df_chembl = pd.DataFrame({"smiles": chembl_smiles})
        os.makedirs(os.path.dirname(CHEMBL_CSV), exist_ok=True)
        df_chembl.to_csv(CHEMBL_CSV, index=False)
        print(f"✅ Saved {len(df_chembl)} validated molecules to {CHEMBL_CSV}")
    else:
        print("⚠️  ChEMBL download failed. Will use existing data only.")
        df_chembl = pd.DataFrame(columns=["smiles"])

# --- Merge with glue chemotypes ---
if os.path.exists(GLUE_CSV):
    df_glues = pd.read_csv(GLUE_CSV)
    print(f"🧬 Glue chemotypes: {len(df_glues)} molecules")
else:
    df_glues = pd.DataFrame(columns=["smiles"])
    print("ℹ️  No glue_chemotypes.csv found – using ChEMBL data only.")

# Combine & deduplicate
all_smiles = list(set(
    df_chembl["smiles"].dropna().tolist() +
    df_glues["smiles"].dropna().tolist()
))
df_train = pd.DataFrame({"smiles": all_smiles})
os.makedirs(os.path.dirname(TRAINING_DATA_PATH) or ".", exist_ok=True)
df_train.to_csv(TRAINING_DATA_PATH, index=False)

print(f"\\n✅ Training data ready: {len(df_train)} unique molecules")
print(f"   Saved to: {TRAINING_DATA_PATH}")
"""))

# 11b. Build config & dataset
cells.append(md_cell("### 9.2 · Configure & Load Dataset"))

cells.append(code_cell("""\
# ── Build configuration ───────────────────────────────────────────
# Adjust these hyperparameters as needed for your hardware.

config = Config(
    model=ModelConfig(
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
    ),
    diffusion=DiffusionConfig(
        num_timesteps=500,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule="cosine",
    ),
    loss=LossConfig(
        lambda_valency=0.1,
        lambda_property=0.05,
        lambda_fragment=0.01,
    ),
    guidance=GuidanceConfig(
        guidance_scale=2.0,
        condition_dropout=0.1,
    ),
    training=TrainingConfig(
        learning_rate=1e-4,
        batch_size=32,
        epochs=100,
        warmup_steps=1000,
        ema_decay=0.999,
        gradient_clip=1.0,
        save_frequency=10,
        val_frequency=5,
    ),
    generation=GenerationConfig(
        num_molecules=100,
        batch_size=10,
        temperature=0.8,
        min_atoms=15,
        max_atoms=35,
    ),
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=SEED,
)

print("Configuration created:")
print(f"  Model: {config.model.hidden_dim}d, {config.model.num_layers} layers, {config.model.num_heads} heads")
print(f"  Diffusion: {config.diffusion.num_timesteps} timesteps, {config.diffusion.beta_schedule} schedule")
print(f"  Training: lr={config.training.learning_rate}, batch={config.training.batch_size}, epochs={config.training.epochs}")
print(f"  Device: {config.device}")
"""))

cells.append(code_cell("""\
# ── Load dataset ──────────────────────────────────────────────────
from torch_geometric.loader import DataLoader

# Load molecules from CSV
dataset = MolecularGlueDataset(
    data_path=TRAINING_DATA_PATH,
    filter_drug_like=True,
    filter_glue_like=False,  # Keep broader chemical space for training
    max_atoms=50,
)

print(f"Dataset size: {len(dataset)} molecules")

# Train/validation split
train_dataset, val_dataset = create_train_val_split(dataset, val_fraction=0.1, seed=SEED)
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# Inspect first batch
batch = next(iter(train_loader))
print(f"\\nSample batch:")
print(f"  Nodes: {batch.x.shape}")
print(f"  Edges: {batch.edge_index.shape}")
print(f"  Edge attrs: {batch.edge_attr.shape}")
print(f"  Num graphs: {batch.num_graphs}")
"""))

# 11c. Training
cells.append(md_cell("### 9.3 · Train the Model"))

cells.append(code_cell("""\
# ── Initialise model & trainer ────────────────────────────────────
model = MolecularDiffusion(config)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

# Trainer requires train_loader (and optionally val_loader) at init time
trainer = Trainer(model, config, train_loader=train_loader, val_loader=val_loader)

# Optional: resume from checkpoint
resume_path = os.path.join(CKPT_DIR, "latest_checkpoint.pt")
if os.path.exists(resume_path):
    trainer.load_checkpoint(resume_path)
    print(f"Resumed from checkpoint")
else:
    print("Starting fresh training")
"""))

cells.append(code_cell("""\
# ── Training loop ─────────────────────────────────────────────────
# Adjust epochs for a full run. Use fewer for testing.
NUM_EPOCHS = config.training.epochs  # or set to e.g. 5 for a quick test

history = trainer.train(num_epochs=NUM_EPOCHS)

print("\\n✅ Training complete!")
"""))

# 11d. Generation
cells.append(md_cell("### 9.4 · Generate Molecules"))

cells.append(code_cell("""\
# ── Generate molecules ────────────────────────────────────────────
model.eval()

sampler = MoleculeSampler(model, config)

# Basic generation
print("Generating molecules...")
graphs = sampler.sample(
    num_molecules=config.generation.num_molecules,
    num_atoms=20,
    temperature=config.generation.temperature,
)

# Post-process: convert graphs → SMILES
raw_smiles = []
for g in graphs:
    smi = postprocess_molecule(g)
    if smi:
        raw_smiles.append(smi)

print(f"\\nRaw valid molecules: {len(raw_smiles)}/{len(graphs)}")
print(f"Validity rate: {len(raw_smiles)/max(1,len(graphs))*100:.1f}%")
"""))

cells.append(code_cell("""\
# ── Filter & deduplicate ─────────────────────────────────────────
filtered = filter_generated(
    raw_smiles,
    drug_like=True,
    glue_like=True,
    pains=True,
    sa_accessible=True,
    no_reactive=True,
)
print(f"After filtering: {len(filtered)} molecules")

# Load training SMILES for novelty check
train_smiles = dataset.smiles_list if hasattr(dataset, 'smiles_list') else []

unique = deduplicate(filtered, training_smiles=train_smiles if train_smiles else None)
print(f"After deduplication: {len(unique)} novel molecules")

# Save results
output_csv = os.path.join(OUTPUT_DIR, "generated_molecules.csv")
save_molecules(unique, output_csv, include_properties=True)
"""))

# 11e. Guided generation
cells.append(md_cell("### 9.5 · Guided Generation (Glue-like Targets)"))

cells.append(code_cell("""\
# ── Guided generation with glue-like targets ──────────────────────
glue_targets = MoleculeSampler.get_glue_targets()
print("Target properties for glue-like molecules:")
for k, v in glue_targets.items():
    print(f"  {k}: {v:.3f}")

guided_graphs = sampler.guided_sample(
    num_molecules=50,
    num_atoms=22,
    temperature=0.8,
    target_properties=glue_targets,
    guidance_scale=2.0,
)

guided_smiles = [postprocess_molecule(g) for g in guided_graphs]
guided_smiles = [s for s in guided_smiles if s is not None]

guided_filtered = filter_generated(guided_smiles, drug_like=True, glue_like=True)
print(f"\\nGuided generation: {len(guided_filtered)} glue-like molecules from {len(guided_graphs)} attempts")
"""))

# 11f. Evaluation
cells.append(md_cell("### 9.6 · Evaluate Generated Molecules"))

cells.append(code_cell("""\
# ── Compute metrics ───────────────────────────────────────────────
all_generated = unique  # Use the deduplicated set

metrics = compute_all_metrics(
    generated=all_generated,
    training_set=train_smiles if train_smiles else None,
    known_glues=None,  # Add known glue SMILES if available
)

print_metrics_report(metrics)
"""))

cells.append(code_cell("""\
# ── Glue-likeness scoring ────────────────────────────────────────
if all_generated:
    ranked = rank_molecules_by_glue_score(all_generated[:50])
    print("\\nTop 10 molecules by glue-likeness score:")
    print("-" * 60)
    for smi, score in ranked[:10]:
        print(f"  Score: {score:3d}  |  {smi}")
"""))

# 11g. Visualization
cells.append(md_cell("### 9.7 · Visualize Results"))

cells.append(code_cell("""\
# ── Inline plotting setup ────────────────────────────────────────
%matplotlib inline
import matplotlib
matplotlib.rcParams['figure.dpi'] = 100
import matplotlib.pyplot as plt
"""))

cells.append(code_cell("""\
# ── Generate full visualisation report ────────────────────────────
if all_generated and train_smiles:
    generate_full_report(
        gen_smiles=all_generated,
        train_smiles=train_smiles,
        output_dir=os.path.join(OUTPUT_DIR, "report"),
    )
else:
    print("Need both generated and training SMILES for full report.")

# ── Show molecule grid inline ─────────────────────────────────────
if all_generated:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    from IPython.display import display

    mols = []
    legends = []
    for smi in all_generated[:20]:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            AllChem.Compute2DCoords(mol)
            qed = calculate_qed(mol)
            mols.append(mol)
            legends.append(f"QED={qed:.2f}")

    if mols:
        img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(300, 300), legends=legends)
        display(img)
"""))

# ═══════════════════════════════════════════════════════════════════
# 12. Appendix
# ═══════════════════════════════════════════════════════════════════
cells.append(md_cell("""\
---
## 10 · Appendix

### Saving & Loading Checkpoints
```python
# Save
trainer.save_checkpoint("my_model.pt")

# Load
trainer.load_checkpoint("my_model.pt")
```

### Exporting to SDF
```python
save_molecules_sdf(all_generated, os.path.join(OUTPUT_DIR, "molecules.sdf"))
```

### Rejection Sampling
```python
valid_smiles, stats = sampler.generate_with_rejection(
    n_molecules=100,
    max_attempts_per_molecule=10,
    num_atoms=22,
    temperature=0.8,
    target_properties=glue_targets,
    guidance_scale=2.0,
)
```
"""))

# ═══════════════════════════════════════════════════════════════════
# Assemble notebook
# ═══════════════════════════════════════════════════════════════════
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        },
        "accelerator": "GPU",
        "colab": {
            "provenance": [],
            "gpuType": "T4"
        }
    },
    "cells": cells,
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "molecular_glues_pipeline.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"✅ Notebook written to {out_path}")
print(f"   Cells: {len(cells)} ({sum(1 for c in cells if c['cell_type']=='code')} code, "
      f"{sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
