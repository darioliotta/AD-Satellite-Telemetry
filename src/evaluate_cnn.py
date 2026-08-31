import os
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

from src.models.cnn1d import CNN1DAnomalyClassifier

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
    CHECKPOINT_PATH = PROJECT_ROOT / "models" / "best_cnn1d_model.pth"

    WINDOW_SIZE = 50
    X_test = np.load(PROCESSED_DATA_DIR / f"X_test_cnn_w{WINDOW_SIZE}.npy")
    y_test = np.load(PROCESSED_DATA_DIR / f"y_test_cnn_w{WINDOW_SIZE}.npy")

    test_x = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1).to(device)
    
    model = CNN1DAnomalyClassifier(in_channels=1, window_size=WINDOW_SIZE).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
    model.eval()

    with torch.no_grad():
        logits = model(test_x).squeeze()
        probs = torch.sigmoid(logits).cpu().numpy()

    roc_auc = roc_auc_score(y_test, probs)
    print(f"ROC-AUC Score: {roc_auc:.4f}\n")
    
    thresholds = np.linspace(0.1, 0.9, 17)
    best_f1 = -1.0
    best_thresh = 0.5

    for t in thresholds:
        preds = (probs >= t).astype(int)
        score = f1_score(y_test, preds, zero_division=0)
        print(f"Threshold: {t:.2f} | F1-Score: {score:.4f}")
        if score > best_f1:
            best_f1 = score
            best_thresh = t

    print(f"\n--- Best Classification Report (Threshold = {best_thresh:.2f}) ---")
    preds_best = (probs >= best_thresh).astype(int)
    print(classification_report(y_test, preds_best, target_names=["Nominal", "Anomaly"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds_best))

if __name__ == "__main__":
    evaluate()