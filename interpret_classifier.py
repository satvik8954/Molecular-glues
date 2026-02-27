# interpret_classifier.py

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw

from data.molecular_graph import smiles_to_graph
from model.classifier import MolecularGlueClassifier


def explain_prediction(model, smiles, device='cuda'):
    """
    Explain why model classified molecule as glue/non-glue.
    Uses gradient × input on logits (pre-sigmoid) for sharp attributions.

    Args:
        model: Trained MolecularGlueClassifier (already on device)
        smiles: SMILES string to explain
        device: 'cuda' or 'cpu'

    Returns:
        img: PIL image with atom highlighting
        atom_importance: numpy array of per-atom importance scores [0, 1]
    """
    # Convert SMILES to graph
    graph = smiles_to_graph(smiles)
    if graph is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")

    graph = graph.to(device)
    batch_idx = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)

    model.eval()

    # Enable gradients on node features
    x = graph.x.clone().requires_grad_(True)

    # Forward pass — use logits (NOT sigmoid) to avoid vanishing gradients
    logits = model(x, graph.edge_index, graph.edge_attr, batch_idx).squeeze(-1)
    prob = torch.sigmoid(logits).item()

    # Backward pass on logits
    logits.backward()

    # Gradient × input — captures which features in which atoms matter
    attr = (x.grad * x).detach()

    # Sum across features, take absolute value (both + and - contributions matter)
    atom_importance = attr.abs().sum(dim=-1).cpu().numpy()

    # Normalize to [0, 1]
    imp_range = atom_importance.max() - atom_importance.min()
    if imp_range > 1e-8:
        atom_importance = (atom_importance - atom_importance.min()) / imp_range
    else:
        atom_importance[:] = 0.5

    # Visualize with RDKit — green=important, red=unimportant
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(
        mol,
        highlightAtoms=list(range(mol.GetNumAtoms())),
        highlightAtomColors={i: (1 - v, v, 0) for i, v in enumerate(atom_importance)},
        size=(400, 400)
    )

    print(f"Prediction: {'GLUE' if prob > 0.5 else 'non-glue'} (prob={prob:.4f})")
    print(f"Top-3 important atoms: {np.argsort(atom_importance)[-3:][::-1]}")

    return img, atom_importance


# Usage:
# from model.classifier import MolecularGlueClassifier
# checkpoint = torch.load('checkpoints/classifier/best_classifier.pt')
# model = MolecularGlueClassifier(**checkpoint['config'])
# model.load_state_dict(checkpoint['model_state_dict'])
# model.to('cuda').eval()
#
# img, importance = explain_prediction(model, "CC(=O)Nc1ccc(O)cc1")
# img.save('interpretation.png')