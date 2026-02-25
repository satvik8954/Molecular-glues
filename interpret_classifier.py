# interpret_classifier.py

import torch
from captum.attr import IntegratedGradients
from rdkit import Chem
from rdkit.Chem import Draw

def explain_prediction(model, smiles, converter):
    """
    Explain why model classified molecule as glue/non-glue.
    Uses integrated gradients to highlight important atoms.
    """
    # Convert to graph
    graph = converter.smiles_to_graph(smiles)
    graph = graph.to('cuda')
    
    # Model in eval mode
    model.eval()
    
    # Integrated Gradients
    ig = IntegratedGradients(model)
    
    # Compute attributions
    attributions = ig.attribute(
        graph.x.unsqueeze(0),
        target=0,
        additional_forward_args=(graph.edge_index, graph.edge_attr, graph.batch)
    )
    
    # Sum attributions across features for each atom
    atom_importance = attributions.sum(dim=-1).squeeze().cpu().numpy()
    
    # Visualize
    mol = Chem.MolFromSmiles(smiles)
    
    # Normalize importance to [0, 1]
    atom_importance = (atom_importance - atom_importance.min()) / (atom_importance.max() - atom_importance.min())
    
    # Draw with highlighting
    img = Draw.MolToImage(mol, highlightAtoms=range(mol.GetNumAtoms()),
                         highlightAtomColors={i: (1-v, v, 0) for i, v in enumerate(atom_importance)})
    
    return img, atom_importance


# Usage:
# model = load_model('checkpoints/best_classifier.pt')
# img, importance = explain_prediction(model, "CC(=O)Nc1ccc(O)cc1", converter)
# img.save('interpretation.png')