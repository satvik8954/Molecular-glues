# evaluate_classifier.py

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, matthews_corrcoef
from scipy.stats import spearmanr, pearsonr
from torch_geometric.loader import DataLoader
from model.classifier import MolecularGlueClassifier
from data.classifier_dataset import MolecularGlueClassifierDataset

def evaluate_model(model_path, test_dataset):
    """
    Comprehensive evaluation of trained classifier.
    """
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})
    # Handle both old (no drop_edge_rate) and new checkpoint formats
    model = MolecularGlueClassifier(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    # Predictions
    all_preds = []
    all_probs = []
    all_labels = []
    all_smiles = []
    
    test_loader = DataLoader(test_dataset, batch_size=32, 
                             collate_fn=test_dataset.collate_fn)
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(probs > 0.5)
            all_labels.extend(batch.y.cpu().numpy())
            all_smiles.extend(batch.smiles)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Confusion matrix saved")
    
    # 2. ROC Curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    print("✓ ROC curve saved")
    
    # 3. Classification Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                                target_names=['Non-Glue', 'Glue']))
    
    # 4. MCC, SPCC, PCC
    import numpy as np
    labels_arr = np.array(all_labels)
    probs_arr = np.array(all_probs)
    preds_arr = np.array(all_preds).astype(float)
    
    mcc = matthews_corrcoef(labels_arr, preds_arr)
    spcc, spcc_pval = spearmanr(labels_arr, probs_arr)
    pcc, pcc_pval = pearsonr(labels_arr, probs_arr)
    
    print(f"Correlation Metrics:")
    print(f"  MCC  (Matthews Correlation Coefficient): {mcc:.4f}")
    print(f"  SPCC (Spearman Rank Correlation):        {spcc:.4f}  (p={spcc_pval:.2e})")
    print(f"  PCC  (Pearson Correlation):              {pcc:.4f}  (p={pcc_pval:.2e})")
    
    # 5. Error Analysis
    errors = pd.DataFrame({
        'SMILES': all_smiles,
        'True': all_labels,
        'Pred': all_preds,
        'Prob': all_probs
    })
    errors['Error'] = errors['True'] != errors['Pred']
    
    print(f"\nError Analysis:")
    print(f"  False Positives: {((errors['True']==0) & (errors['Pred']==1)).sum()}")
    print(f"  False Negatives: {((errors['True']==1) & (errors['Pred']==0)).sum()}")
    
    # Save error cases
    error_cases = errors[errors['Error']]
    error_cases.to_csv('error_cases.csv', index=False)
    print(f"✓ Error cases saved ({len(error_cases)} total)")
    
    return errors


if __name__ == '__main__':
    # Load test dataset (scaffold-split)
    test_dataset = MolecularGlueClassifierDataset(
        csv_file='data/classifier_test.csv',
        split_name='test'
    )
    
    # Evaluate
    results = evaluate_model('checkpoints/classifier/best_classifier.pt', test_dataset)