from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import torch

def calculate_deepfake_metrics(y_true, y_pred_probs):
    """
    Calculates the research metrics reported for the Audio Deepfake Detection project.
    Target: AUC 0.8293, F1 0.7221, Accuracy 74.74%.
    """
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred_probs > 0.5).astype(int)
    
    metrics = {
        "AUC": roc_auc_score(y_true, y_pred_probs),
        "F1-Score": f1_score(y_true, y_pred_binary),
        "Accuracy": accuracy_score(y_true, y_pred_binary)
    }
    
    return metrics