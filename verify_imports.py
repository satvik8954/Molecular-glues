"""Quick verification of all module imports and basic functionality."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=== Config ===")
from config import Config, LossConfig, GuidanceConfig, HYBRIDIZATIONS, PROPERTY_NAMES, HYBRID_TO_IDX
c = Config()
print(f"  Hybridizations: {HYBRIDIZATIONS}")
print(f"  Properties: {PROPERTY_NAMES}")
print(f"  Loss: valency={c.loss.lambda_valency}, property={c.loss.lambda_property}")
print(f"  EMA: {c.training.ema_decay}")
print(f"  Guidance: scale={c.guidance.guidance_scale}, dropout={c.guidance.condition_dropout}")
print("  OK")

print("\n=== Chemistry Utils ===")
from utils.chemistry import is_valid_molecule, get_molecular_properties, calculate_qed, get_murcko_scaffold
props = get_molecular_properties('CC(=O)Oc1ccccc1C(=O)O')
print(f"  Aspirin: MW={props['molecular_weight']:.1f}, QED={props['qed']:.3f}, Fsp3={props['fraction_sp3']:.3f}")
s = get_murcko_scaffold('CC(=O)Oc1ccccc1C(=O)O')
print(f"  Scaffold: {s}")
print("  OK")

print("\n=== Filters ===")
from utils.filters import has_reactive_groups, is_glue_target_range, passes_all_filters, is_glue_like
print(f"  Aldehyde reactive: {has_reactive_groups('CC=O')}")
print(f"  Benzene reactive: {has_reactive_groups('c1ccccc1')}")
print(f"  Phthalimide glue-like: {is_glue_like('O=C1NC(=O)c2ccccc12')}")
print("  OK")

print("\n=== Fragment Vocab ===")
from model.fragment_vocab import FragmentVocab
fv = FragmentVocab.from_smiles_list(['CCO', 'c1ccccc1', 'CC(=O)O'])
print(f"  {fv}")
score = fv.score_molecule('CCO')
print(f"  Score for CCO: {score:.2f}")
print("  OK")

print("\n=== All basic imports verified! ===")
