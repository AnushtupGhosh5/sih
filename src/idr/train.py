"""
Train the AI Speed & Vibration Filter on the synchronised IO-VNBD dataset.

Split is by drive (no leakage). Features are standardised using statistics from
the training split only. Best model (by validation RMSE) and the scaler are
saved to artifacts/.
"""
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from . import config as C
from . import dataset as D
from .model import SpeedCNN, count_params


def set_seed(seed=C.SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def frames_to_windows(frames, stride):
    """Compute features per drive then window; concatenate across drives."""
    Xs, ys = [], []
    for df in frames:
        feats = D.compute_features(df)
        sp = df["speed_ms"].to_numpy(np.float32)
        X, y = D.make_windows(feats, sp, window=C.WINDOW, stride=stride)
        if len(X):
            Xs.append(X)
            ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)


def main():
    set_seed()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}")

    train_frames, val_frames, test_frames = D.build_split()
    print(f"Drives -> train:{len(train_frames)} val:{len(val_frames)} test:{list(test_frames)}")

    Xtr, ytr = frames_to_windows(train_frames, C.STRIDE_TRAIN)
    Xva, yva = frames_to_windows(val_frames, C.WINDOW)  # non-overlapping for val
    print(f"Train windows: {Xtr.shape}  Val windows: {Xva.shape}")

    # ---- Standardise features (fit on train only) ----
    mean = Xtr.reshape(-1, C.NUM_FEATURES).mean(0)
    std = Xtr.reshape(-1, C.NUM_FEATURES).std(0) + 1e-6
    Xtr = (Xtr - mean) / std
    Xva = (Xva - mean) / std

    tr = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                    batch_size=C.BATCH_SIZE, shuffle=True, drop_last=True)
    va = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)),
                    batch_size=C.BATCH_SIZE, shuffle=False)

    model = SpeedCNN(in_ch=C.NUM_FEATURES).to(dev)
    print(f"Model params: {count_params(model):,}")
    opt = torch.optim.Adam(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    loss_fn = nn.SmoothL1Loss()  # robust to GPS speed outliers

    best_rmse, best_state, patience, bad = float("inf"), None, 8, 0
    for ep in range(1, C.EPOCHS + 1):
        model.train()
        t0 = time.time()
        for xb, yb in tr:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        # ---- validate ----
        model.eval()
        se, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in va:
                pred = model(xb.to(dev)).cpu().numpy()
                se += np.sum((pred - yb.numpy()) ** 2)
                n += len(yb)
        rmse = np.sqrt(se / max(n, 1))
        sched.step(rmse)
        print(f"epoch {ep:2d} | val RMSE {rmse:.3f} m/s ({rmse*3.6:.2f} km/h) | {time.time()-t0:.1f}s")

        if rmse < best_rmse - 1e-4:
            best_rmse, best_state, bad = rmse, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stop at epoch {ep}")
                break

    model.load_state_dict(best_state)
    ckpt = os.path.join(C.ART_DIR, "speed_cnn.pt")
    torch.save({"state_dict": best_state, "mean": mean, "std": std,
                "window": C.WINDOW, "num_features": C.NUM_FEATURES}, ckpt)
    with open(os.path.join(C.ART_DIR, "train_meta.json"), "w") as f:
        json.dump({"best_val_rmse_ms": float(best_rmse),
                   "best_val_rmse_kmh": float(best_rmse * 3.6),
                   "params": count_params(model),
                   "train_windows": int(len(Xtr)),
                   "test_drives": list(test_frames)}, f, indent=2)
    print(f"\nBest val RMSE: {best_rmse:.3f} m/s ({best_rmse*3.6:.2f} km/h)")
    print(f"Saved model -> {ckpt}")


if __name__ == "__main__":
    main()
