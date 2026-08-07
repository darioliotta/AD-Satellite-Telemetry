import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.models.tranad import TranAD

def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
    MODEL_DIR = PROJECT_ROOT / "models"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    CHECKPOINT_PATH = MODEL_DIR / "best_tranad_model.pth"

    WINDOW_SIZE = 50
    BATCH_SIZE = 256
    EPOCHS = 5
    LR = 0.001

    print("Dataset loading...")
    X_train = np.load(PROCESSED_DATA_DIR / f"X_train_w{WINDOW_SIZE}.npy")
    X_val   = np.load(PROCESSED_DATA_DIR / f"X_val_w{WINDOW_SIZE}.npy")

    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    val_tensor   = torch.tensor(X_val, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(val_tensor), batch_size=BATCH_SIZE, shuffle=False)

    feats = 1
    model = TranAD(feats=feats, window_size=WINDOW_SIZE).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    l1_loss = nn.L1Loss(reduction='mean')

    if CHECKPOINT_PATH.exists():
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        print(f"Checkpoint found. Best Validation Loss: {best_val_loss:.6f}")
    else:
        best_val_loss = float('inf')
        print("No checkpoint found.")

    print("\nStarting Training Loop...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            x = batch[0].permute(1, 0, 2).to(device)
            optimizer.zero_grad()
            
            x1, x2 = model(x, x)
            loss1 = l1_loss(x1, x)
            loss2 = l1_loss(x2, x)
            loss = (1 / epoch) * loss1 + (1 - 1 / epoch) * loss2
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].permute(1, 0, 2).to(device)
                x1, x2 = model(x, x)
                loss1 = l1_loss(x1, x)
                loss2 = l1_loss(x2, x)
                val_loss += ((1 / epoch) * loss1 + (1 - 1 / epoch) * loss2).item()
                
        val_loss /= len(val_loader)

        print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            print(f"  --> Upgrade ({best_val_loss:.6f} -> {val_loss:.6f}). Checkpoint saving...")
            best_val_loss = val_loss
            
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "window_size": WINDOW_SIZE,
                "feats": feats
            }
            torch.save(checkpoint_data, CHECKPOINT_PATH)

    print(f"\nTraining completed. Best Validation Loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    run_training()