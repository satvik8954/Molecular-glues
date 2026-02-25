"""Quick validation of the generated notebook."""
import json, os

path = "molecular_glues_pipeline.ipynb"
nb = json.load(open(path, encoding="utf-8"))

print(f"File size: {os.path.getsize(path) / 1024:.0f} KB")
print(f"nbformat: {nb['nbformat']}.{nb['nbformat_minor']}")
print(f"Kernel: {nb['metadata']['kernelspec']['display_name']}")
print(f"Total cells: {len(nb['cells'])}")

code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
md_cells = [c for c in nb['cells'] if c['cell_type'] == 'markdown']
print(f"  Code cells: {len(code_cells)}")
print(f"  Markdown cells: {len(md_cells)}")

total_code_lines = sum(len(c['source']) for c in code_cells)
print(f"  Total code lines: {total_code_lines}")

# Check all cells have required keys
for i, c in enumerate(nb['cells']):
    assert 'cell_type' in c, f"Cell {i} missing cell_type"
    assert 'source' in c, f"Cell {i} missing source"
    assert 'metadata' in c, f"Cell {i} missing metadata"
    if c['cell_type'] == 'code':
        assert 'outputs' in c, f"Cell {i} missing outputs"
        assert 'execution_count' in c, f"Cell {i} missing execution_count"

print("\n✅ All cells have valid structure")

# Print markdown headers for overview
print("\n--- Notebook Outline ---")
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'markdown':
        first_line = ''.join(c['source']).split('\n')[0].strip()
        if first_line.startswith('#'):
            n_lines = len(c['source'])
            print(f"  Cell [{i:2d}] {first_line}")

# Check for key classes/functions
full_source = '\n'.join(''.join(c['source']) for c in code_cells)
key_items = [
    'class Config', 'class ModelConfig', 'class MolecularGlueDataset',
    'class NoiseScheduler', 'class GraphTransformer', 'class MolecularDiffusion',
    'class Trainer', 'class MoleculeSampler', 'class DiverseMoleculeSampler',
    'def smiles_to_graph', 'def graph_to_smiles', 'def is_valid_molecule',
    'def compute_all_metrics', 'def generate_full_report',
    'def postprocess_molecule', 'def filter_generated',
    'def compute_glue_likeness_score', 'def rank_molecules_by_glue_score',
]

print("\n--- Key Definitions Check ---")
missing = []
for item in key_items:
    if item in full_source:
        print(f"  ✅ {item}")
    else:
        print(f"  ❌ {item} NOT FOUND")
        missing.append(item)

if missing:
    print(f"\n⚠️  {len(missing)} key items missing!")
else:
    print(f"\n✅ All {len(key_items)} key definitions found")
