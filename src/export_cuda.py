import os
from pathlib import Path
import numpy as np
import torch

from src.models.cnn1d import CNN1DAnomalyClassifier

def export_all_for_cuda():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
    CHECKPOINT_PATH = PROJECT_ROOT / "models" / "best_cnn1d_model.pth"
    
    CUDA_DIR = PROJECT_ROOT / "cuda_engine"
    WEIGHTS_DIR = CUDA_DIR / "weights"
    DATA_DIR = CUDA_DIR / "data"
    
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    WINDOW_SIZE = 50

    model = CNN1DAnomalyClassifier(in_channels=1, window_size=WINDOW_SIZE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
    model.eval()

    def fuse_and_save_conv_bn(conv, bn, name):
        W = conv.weight.detach().numpy()
        b = conv.bias.detach().numpy() if conv.bias is not None else np.zeros(conv.out_channels, dtype=np.float32)
        
        gamma = bn.weight.detach().numpy()
        beta  = bn.bias.detach().numpy()
        mean  = bn.running_mean.detach().numpy()
        var   = bn.running_var.detach().numpy()
        eps   = bn.eps

        scale = gamma / np.sqrt(var + eps)
        W_fused = W * scale[:, None, None]
        b_fused = (b - mean) * scale + beta

        W_fused.astype(np.float32).tofile(WEIGHTS_DIR / f"{name}_weight.bin")
        b_fused.astype(np.float32).tofile(WEIGHTS_DIR / f"{name}_bias.bin")
        print(f"Exported {name:5s} -> W: {W_fused.shape}, b: {b_fused.shape}")

    fuse_and_save_conv_bn(model.conv1, model.bn1, "conv1")
    fuse_and_save_conv_bn(model.conv2, model.bn2, "conv2")
    fuse_and_save_conv_bn(model.conv3, model.bn3, "conv3")

    fc1_w = model.fc1.weight.detach().numpy().astype(np.float32)
    fc1_b = model.fc1.bias.detach().numpy().astype(np.float32)
    fc1_w.tofile(WEIGHTS_DIR / "fc1_weight.bin")
    fc1_b.tofile(WEIGHTS_DIR / "fc1_bias.bin")
    print(f"Exported fc1   -> W: {fc1_w.shape}, b: {fc1_b.shape}")

    fc2_w = model.fc2.weight.detach().numpy().astype(np.float32)
    fc2_b = model.fc2.bias.detach().numpy().astype(np.float32)
    fc2_w.tofile(WEIGHTS_DIR / "fc2_weight.bin")
    fc2_b.tofile(WEIGHTS_DIR / "fc2_bias.bin")
    print(f"Exported fc2   -> W: {fc2_w.shape}, b: {fc2_b.shape}")

    X_test = np.load(PROCESSED_DATA_DIR / f"X_test_cnn_w{WINDOW_SIZE}.npy")
    y_test = np.load(PROCESSED_DATA_DIR / f"y_test_cnn_w{WINDOW_SIZE}.npy")

    X_test_cuda = np.ascontiguousarray(X_test.transpose(0, 2, 1), dtype=np.float32)
    y_test_cuda = np.ascontiguousarray(y_test, dtype=np.float32)

    with torch.no_grad():
        test_tensor = torch.tensor(X_test_cuda, dtype=torch.float32)
        logits_ref = model(test_tensor).squeeze().cpu().numpy().astype(np.float32)
        probs_ref = 1.0 / (1.0 + np.exp(-logits_ref))

    X_test_cuda.tofile(DATA_DIR / "test_inputs.bin")
    y_test_cuda.tofile(DATA_DIR / "test_labels.bin")
    logits_ref.tofile(DATA_DIR / "pytorch_reference_logits.bin")
    probs_ref.tofile(DATA_DIR / "pytorch_reference_probs.bin")

    meta_info = f"{X_test_cuda.shape[0]} {X_test_cuda.shape[1]} {X_test_cuda.shape[2]}\n"
    with open(DATA_DIR / "meta.txt", "w") as f:
        f.write(meta_info)

    print(f"Test Input: {X_test_cuda.shape} ({X_test_cuda.nbytes / (1024*1024):.2f} MB)")
    print(f"Export completed in: {CUDA_DIR.resolve()}")

if __name__ == "__main__":
    export_all_for_cuda()