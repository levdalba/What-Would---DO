#!/usr/bin/env python3
"""
Pre-train a tiny Magnus-style move-prediction transformer,
log everything to MLflow, and keep full artefacts.

Optimised for Apple Silicon (M-series) + PyTorch ≥ 2.3.
"""

import os, time, platform, subprocess, json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import chess
import mlflow
import mlflow.pytorch
import torch.multiprocessing as mp
from mlflow.models.signature import infer_signature


# --------------------------------------------------------------------------- #
# 1.  FEN / move helpers
# --------------------------------------------------------------------------- #

PIECE_TO_ID = {
    '.': 0,
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
}


def fen_to_tokens(fen: str) -> torch.Tensor:
    """Board → 64 piece-ids + side-to-move token."""
    board = chess.Board(fen)
    toks = [(PIECE_TO_ID[board.piece_at(sq).symbol()]
             if board.piece_at(sq) else 0) for sq in chess.SQUARES]
    toks.append(13 if board.turn == chess.WHITE else 14)
    return torch.tensor(toks, dtype=torch.int8)      # int8 → compact


def encode_move(uci: str) -> int | None:
    try:
        mv = chess.Move.from_uci(uci)
        return mv.from_square * 64 + mv.to_square     # 0…4095
    except ValueError:
        return None


def process_data(path: str) -> list[tuple[str, str]]:
    """
    Each line:  <board> <side> <castling> <en-passant> <uci_move>
    → returns list[(full-fen-6, move)]  without legality checks.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    samples: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"[skip L{ln}] need 5 fields, got {len(parts)}")
                continue
            fen6 = " ".join(parts[:4]) + " 0 1"
            move = parts[4]
            try:
                chess.Board(fen6)
                chess.Move.from_uci(move)
            except ValueError as e:
                print(f"[skip L{ln}] bad FEN/move – {e}")
                continue
            samples.append((fen6, move))
    print(f"Loaded {len(samples)} samples from {path}")
    return samples


# --------------------------------------------------------------------------- #
# 2.  Dataset + model
# --------------------------------------------------------------------------- #

class MagnusDataset(Dataset):
    def __init__(self, pairs):             # [(fen, move)]
        xs = [fen_to_tokens(f) for f, _ in pairs]
        ys = [encode_move(m)  for _, m in pairs]
        self.x = torch.stack(xs)           # (N, 65)
        self.y = torch.tensor(ys, dtype=torch.int16)

    def __len__(self):  return len(self.x)

    def __getitem__(self, idx):  # returns two tensors
        return self.x[idx].long(), self.y[idx].long()


class MagnusTransformer(nn.Module):
    def __init__(self,
                 vocab_size: int = 15,
                 seq_len: int = 65,
                 num_moves: int = 4096,
                 d_model: int = 256,
                 nhead: int = 16,   
                 num_layers: int = 6,
                 dim_feedforward: int = 8192,
                 dropout: float = 0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos   = nn.Parameter(torch.randn(seq_len, d_model))
        enc = nn.TransformerEncoderLayer(d_model, nhead,
                                         dim_feedforward, dropout)
        self.tr = nn.TransformerEncoder(enc, num_layers)
        self.fc = nn.Linear(d_model * seq_len, num_moves)

    def forward(self, x):                  # (B, 65)
        x = self.embed(x) + self.pos       # (B, 65, d)
        x = self.tr(x.permute(1, 0, 2))    # (65, B, d)
        x = x.permute(1, 0, 2).reshape(x.shape[1], -1)
        return self.fc(x)                  # (B, 4096)


# --------------------------------------------------------------------------- #
# 3.  Training / logging
# --------------------------------------------------------------------------- #

def main() -> None:
    t0 = time.time()

    # ---------------- data ---------------- #
    data_path = "data_processing/v2/training_data/general/data.txt"
    pairs     = process_data(data_path)
    ds        = MagnusDataset(pairs)
    dl        = DataLoader(ds,
                           batch_size=32,
                           shuffle=True,
                           num_workers=0,           # faster on macOS
                           pin_memory=False)

    # ---------------- device -------------- #
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cpu")
    print("device:", device)

    # ---------------- model --------------- #
    model = MagnusTransformer().to(device)
    # model = torch.compile(model, backend="inductor")   # PyTorch 2.3+

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # ---------------- MLflow run ---------- #
    mlflow.set_experiment("MagnusTransformer")

    with mlflow.start_run(run_name="pretrain-m2-max"):

        # ---- run parameters / tags ---- #
        mlflow.log_param("dataset_size", len(ds))
        mlflow.log_params({
            "batch": 32, "lr": 1e-4, "epochs": 1,
            "d_model": 256, "layers": 6, "ffn": 8192
        })
        mlflow.set_tags({
            "device": device.type,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "os": platform.platform(),
            "git": subprocess.getoutput("git rev-parse --short HEAD") or "N/A"
        })

        # ---- training loop ------------- #
        for epoch in range(1):
            model.train(); tot = 0
            for xb, yb in tqdm(dl, desc=f"epoch {epoch+1}"):
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                with torch.amp.autocast(device_type=device.type,
                                        dtype=torch.float16):
                    out  = model(xb)
                    loss = loss_fn(out, yb)
                loss.backward(); opt.step()
                tot += loss.item()

            avg = tot / len(dl)
            mlflow.log_metric("avg_loss", avg, step=epoch)
            mlflow.pytorch.log_model(model, f"ckpt_epoch_{epoch+1}")

        # ---- final artefacts ----------- #
        weights_path = "data_processing/v2/models/magnus_transformer_pretrained.pth"
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        torch.save(model.state_dict(), weights_path)
        mlflow.log_artifact(weights_path, artifact_path="weights")

        # Prepare input/output examples for model signature
        x_example = ds[0][0].unsqueeze(0).to(device)         # 1 sample, shape [1, 65]
        y_example = model(x_example).detach().cpu().numpy()  # shape [1, 4096]
        signature = infer_signature(x_example.cpu().numpy(), y_example)

        # Log full model with signature + register
        mlflow.pytorch.log_model(
            model,
            artifact_path="final_model",
            registered_model_name="MagnusTransformer",
            signature=signature
        )

        # Time
        mlflow.log_metric("runtime_sec", time.time() - t0)

    print("done; artefacts logged.")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)   # macOS default but explicit
    main()
