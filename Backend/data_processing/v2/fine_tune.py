import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import chess
import os
import time
import platform
import subprocess
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature

# --------------------- Constants ---------------------
PRETRAINED_PATH = "data_processing/v2/models/magnus_transformer_pretrained.pth"
FINETUNED_PATH = "data_processing/v2/models/magnus_transformer_finetuned.pth"
DATA_PATH = "data_processing/v2/training_data/carlson/carlsen_fen_moves.txt"

# PIECE_TO_ID mapping
PIECE_TO_ID = {
    '.': 0, 'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
}

def fen_to_tokens(fen):
    board = chess.Board(fen)
    tokens = [(PIECE_TO_ID[piece.symbol()] if piece else 0)
              for square in chess.SQUARES
              if (piece := board.piece_at(square))]
    tokens += [0] * (64 - len(tokens))  # pad just in case
    tokens.append(13 if board.turn == chess.WHITE else 14)
    return torch.tensor(tokens, dtype=torch.long)

def encode_move(move_str):
    try:
        move = chess.Move.from_uci(move_str)
        return move.from_square * 64 + move.to_square
    except ValueError:
        return None

def process_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    samples = []
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
                samples.append((fen6, move))
            except ValueError as e:
                print(f"[skip L{ln}] bad FEN/move – {e}")
    print(f"Loaded {len(samples)} samples from {path}")
    return samples

class MagnusTransformerDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.samples = data
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        fen, move = self.samples[idx]
        x = fen_to_tokens(fen)
        y = encode_move(move)
        return x, torch.tensor(y)

class MagnusTransformer(nn.Module):
    def __init__(self, vocab_size=15, seq_len=65, num_moves=4096,
                 d_model=256, nhead=16, num_layers=6,
                 dim_feedforward=8192, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.tr = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model * seq_len, num_moves)
    def forward(self, x):
        x = self.embed(x.long()) + self.pos.unsqueeze(0)
        x = self.tr(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)

# ------------------ Fine-tuning Script ------------------
if __name__ == "__main__":
    t0 = time.time()
    carlsen_data = process_data(DATA_PATH)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = MagnusTransformer().to(device)
    model.load_state_dict(torch.load(PRETRAINED_PATH))

    train_data = MagnusTransformerDataset(carlsen_data)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    mlflow.set_experiment("MagnusTransformer-Finetune")
    with mlflow.start_run(run_name="FineTune-Carlsen-Games"):
        # Log static metadata
        mlflow.log_params({
            "dataset_size": len(train_data),
            "batch_size": 32,
            "lr": 1e-4,
            "epochs": 1,
            "device": device.type,
            "d_model": 256,
            "nhead": 16,
            "layers": 6,
            "dim_feedforward": 8192,
        })
        mlflow.set_tags({
            "host": platform.node(),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "os": platform.platform(),
            "git": subprocess.getoutput("git rev-parse --short HEAD") or "N/A"
        })

        for epoch in range(1):
            model.train()
            total_loss = 0
            step = 0

            loop = tqdm(train_loader, desc=f"Fine-tuning Epoch {epoch + 1}")
            for x, y in loop:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                    out = model(x)
                    loss = criterion(out, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                step += 1
                total_loss += loss.item()
                loop.set_postfix(step=step, loss=loss.item())

            avg_loss = total_loss / step
            mlflow.log_metric("avg_epoch_loss", avg_loss, step=epoch)

        # Save weights and model
        os.makedirs(os.path.dirname(FINETUNED_PATH), exist_ok=True)
        torch.save(model.state_dict(), FINETUNED_PATH)
        mlflow.log_artifact(FINETUNED_PATH, artifact_path="weights")

        # Signature
        example_x = train_data[0][0].unsqueeze(0).to(device)
        example_y = model(example_x).detach().cpu().numpy()
        signature = infer_signature(example_x.cpu().numpy(), example_y)

        mlflow.pytorch.log_model(
            model,
            artifact_path="final_model",
            registered_model_name="MagnusTransformer",
            signature=signature
        )

        mlflow.log_metric("runtime_sec", time.time() - t0)

    print("✅ Fine-tuning complete. Model logged to MLflow.")
