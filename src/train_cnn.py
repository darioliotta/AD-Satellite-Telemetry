import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.models.cnn1d import CNN1DAnomalyClassifier

def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
    MODEL_DIR = PROJECT_ROOT / "models"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    CHECKPOINT_PATH = MODEL_DIR / "best_cnn1d_model.pth"

    WINDOW_SIZE = 50
    BATCH_SIZE = 256
    ADDITIONAL_EPOCHS = 20
    LR = 0.0005
    IN_CHANNELS = 1

    print("Dataset loading...")
    X_train = np.load(PROCESSED_DATA_DIR / f"X_train_cnn_w{WINDOW_SIZE}.npy")
    y_train = np.load(PROCESSED_DATA_DIR / f"y_train_cnn_w{WINDOW_SIZE}.npy")
    X_val   = np.load(PROCESSED_DATA_DIR / f"X_val_cnn_w{WINDOW_SIZE}.npy")
    y_val   = np.load(PROCESSED_DATA_DIR / f"y_val_cnn_w{WINDOW_SIZE}.npy")

    train_x = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
    train_y = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    val_x   = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1)
    val_y   = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    class_counts = np.bincount(y_train.astype(int))
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train.astype(int)]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_dataset = TensorDataset(train_x, train_y)
    val_dataset   = TensorDataset(val_x, val_y)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = CNN1DAnomalyClassifier(in_channels=IN_CHANNELS, window_size=WINDOW_SIZE).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    start_epoch = 1
    best_val_loss = float("inf")

    if CHECKPOINT_PATH.exists():
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = LR
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            start_epoch = checkpoint.get("epoch", 0) + 1
            print(f"Checkpoint found. Starting from epoch {start_epoch} (Best Val Losss: {best_val_loss:.6f})")
        else:
            model.load_state_dict(checkpoint)
            print("Only model parameters loaded.")
    else:
        print("No checkpoint found. Starting from scratch.")

    end_epoch = start_epoch + ADDITIONAL_EPOCHS - 1
    print(f"\nInizio ciclo di training (Epoche da {start_epoch} a {end_epoch})...")

    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct_train += (preds == batch_y).sum().item()
            total_train += batch_y.size(0)

        train_loss /= len(train_loader)
        train_acc = (correct_train / total_train) * 100.0

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                
                val_loss += loss.item()
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct_val += (preds == batch_y).sum().item()
                total_val += batch_y.size(0)
                
        val_loss /= len(val_loader)
        val_acc = (correct_val / total_val) * 100.0

        print(f"Epoca {epoch:02d}/{end_epoch} | Train Loss: {train_loss:.4f} (Acc: {train_acc:.2f}%) | Val Loss: {val_loss:.4f} (Acc: {val_acc:.2f}%)")

        if val_loss < best_val_loss:
            print(f"  --> Miglioramento ({best_val_loss:.6f} -> {val_loss:.6f}). Salvataggio checkpoint...")
            best_val_loss = val_loss
            
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "window_size": WINDOW_SIZE,
                "in_channels": IN_CHANNELS
            }
            torch.save(checkpoint_data, CHECKPOINT_PATH)

    print(f"\nTraining completato. Miglior Validation Loss assoluta: {best_val_loss:.6f}")

if __name__ == "__main__":
    run_training()