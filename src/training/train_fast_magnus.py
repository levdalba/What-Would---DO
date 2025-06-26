#!/usr/bin/env python3
"""
Fast Magnus Training - Optimized for Speed and Performance
GPU-optimized version with better architecture and faster training
"""

import sys
import time
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import numpy as np
import warnings

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# MLOps imports
try:
    import mlflow
    import mlflow.pytorch
    
    # Add model manager from new location
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    from src.mlops.model_manager import EnhancedMagnusModelManager, integrate_enhanced_training
    
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Chess processing
try:
    import chess
    import chess.pgn
    CHESS_AVAILABLE = True
except ImportError:
    CHESS_AVAILABLE = False
    print("⚠️ python-chess not available")


class FastMagnusDataset(Dataset):
    """Optimized dataset for fast GPU training"""

    def __init__(self, positions, features, magnus_moves, evaluations):
        # Pre-convert everything to tensors for faster loading
        self.positions = torch.FloatTensor(positions)

        # Handle features properly
        if len(features) > 0 and not isinstance(features[0], dict):
            self.features = torch.FloatTensor(features)
        else:
            # Create dummy features if none available
            self.features = torch.zeros(len(positions), 1)

        self.evaluations = torch.FloatTensor(evaluations)

        # Build vocabulary and convert moves
        unique_moves = list(set(magnus_moves))
        self.move_to_idx = {move: idx for idx, move in enumerate(unique_moves)}
        self.idx_to_move = {idx: move for move, idx in self.move_to_idx.items()}
        self.vocab_size = len(unique_moves)

        # Convert moves to indices
        self.magnus_moves = torch.LongTensor(
            [self.move_to_idx[move] for move in magnus_moves]
        )

        print(f"📚 Fast Dataset: {len(self)} samples, {self.vocab_size} moves")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return (
            self.positions[idx],
            self.features[idx],
            self.magnus_moves[idx],
            self.evaluations[idx],
        )


class FastMagnusModel(nn.Module):
    """Fast, GPU-optimized Magnus model with attention"""

    def __init__(self, vocab_size: int, feature_dim: int):
        super().__init__()

        # Efficient board encoder with residual connections
        self.board_encoder = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
        )

        # Fast feature encoder
        if feature_dim > 1:
            self.feature_encoder = nn.Sequential(
                nn.Linear(feature_dim, 32), nn.ReLU(inplace=True)
            )
            combined_dim = 128 + 32
        else:
            self.feature_encoder = None
            combined_dim = 128

        # Efficient attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=combined_dim, num_heads=4, dropout=0.1, batch_first=True
        )

        # Fast move predictor with residual connection
        self.move_predictor = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, vocab_size),
        )

        # Simple evaluation head
        self.eval_head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, position, features):
        # Encode inputs
        board_enc = self.board_encoder(position)

        if self.feature_encoder is not None:
            feat_enc = self.feature_encoder(features)
            combined = torch.cat([board_enc, feat_enc], dim=1)
        else:
            combined = board_enc

        # Apply attention (reshape for attention)
        combined_seq = combined.unsqueeze(1)  # Add sequence dimension
        attn_out, _ = self.attention(combined_seq, combined_seq, combined_seq)
        attn_out = attn_out.squeeze(1)  # Remove sequence dimension

        # Residual connection
        combined = combined + attn_out

        # Predictions
        move_logits = self.move_predictor(combined)
        eval_pred = self.eval_head(combined)

        return move_logits, eval_pred


class WeightedFocalLoss(nn.Module):
    """Fast focal loss implementation"""

    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def train_fast_magnus():
    """Fast training with GPU optimization"""

    print("🚀 Starting Fast Magnus Training...")

    # Device setup with optimization
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🎮 Device: mps (M3 Pro optimized)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🎮 Device: cuda")
    else:
        device = torch.device("cpu")
        print("🎮 Device: cpu")

    # Optimized training parameters
    config = {
        "learning_rate": 0.003,  # Higher LR for faster convergence
        "batch_size": 512,  # Larger batch for GPU efficiency
        "num_epochs": 30,  # Fewer epochs, better convergence
        "min_move_count": 20,  # Filter rare moves
        "weight_decay": 1e-4,
        "gradient_clip": 1.0,
    }

    # Load and filter data efficiently
    print("📂 Loading training data...")
    with open("magnus_extracted_positions_m3_pro.pkl", "rb") as f:
        data = pickle.load(f)

    positions = data["positions"]
    features = data.get("features", [])
    sf_moves = data["stockfish_moves"]
    magnus_moves = data["magnus_moves"]
    evaluations = data["evaluations"]

    # Fast filtering
    from collections import Counter

    move_counts = Counter(magnus_moves)
    valid_moves = {
        move for move, count in move_counts.items() if count >= config["min_move_count"]
    }

    # Filter data
    filtered_data = []
    for pos, feat, sf_move, mag_move, eval_val in zip(
        positions, features, sf_moves, magnus_moves, evaluations
    ):
        if mag_move in valid_moves:
            # Handle features properly - use dummy if dict or empty
            if isinstance(feat, dict) or not feat:
                feat_array = [0.0]  # Dummy feature
            else:
                feat_array = feat if isinstance(feat, list) else [feat]
            filtered_data.append((pos, feat_array, mag_move, eval_val))

    print(f"📊 Fast filtering: {len(magnus_moves)} → {len(filtered_data)} samples")
    print(f"📊 Vocabulary: {len(move_counts)} → {len(valid_moves)} moves")

    # Unpack filtered data
    positions, features, magnus_moves, evaluations = zip(*filtered_data)

    # Train/val/test split
    train_data, test_data = train_test_split(
        list(zip(positions, features, magnus_moves, evaluations)),
        test_size=0.2,
        random_state=42,
    )
    train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)

    # Create datasets
    def unpack_data(data_list):
        return list(zip(*data_list))

    train_dataset = FastMagnusDataset(*unpack_data(train_data))
    val_dataset = FastMagnusDataset(*unpack_data(val_data))
    test_dataset = FastMagnusDataset(*unpack_data(test_data))

    # Share vocabulary
    val_dataset.move_to_idx = train_dataset.move_to_idx
    val_dataset.vocab_size = train_dataset.vocab_size
    test_dataset.move_to_idx = train_dataset.move_to_idx
    test_dataset.vocab_size = train_dataset.vocab_size

    print(f"📚 Fast Dataset:")
    print(f"   Train: {len(train_dataset):,}")
    print(f"   Validation: {len(val_dataset):,}")
    print(f"   Test: {len(test_dataset):,}")
    print(f"   Vocabulary: {train_dataset.vocab_size:,} moves")

    # Fast data loaders with optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,  # Use 0 for MPS
        pin_memory=False,  # Don't use pin_memory for MPS
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0
    )

    # Create optimized model
    model = FastMagnusModel(
        train_dataset.vocab_size, train_dataset.features.shape[1]
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Fast Model: {total_params:,} parameters")

    # Optimized optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["learning_rate"],
        epochs=config["num_epochs"],
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )

    # Fast loss functions
    move_criterion = WeightedFocalLoss(alpha=1.0, gamma=2.0)
    eval_criterion = nn.MSELoss()

    # MLflow setup
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("magnus_fast_training")
        with mlflow.start_run():
            mlflow.log_params(config)
            mlflow.log_param("model_params", total_params)
            mlflow.log_param("device", str(device))

            # Fast training loop
            start_time = time.time()
            best_val_acc = 0

            for epoch in range(config["num_epochs"]):
                # Training
                model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0

                pbar = tqdm(
                    train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}"
                )
                for batch in pbar:
                    positions, features, magnus_moves, evaluations = [
                        x.to(device, non_blocking=True) for x in batch
                    ]

                    optimizer.zero_grad()

                    move_logits, eval_preds = model(positions, features)

                    move_loss = move_criterion(move_logits, magnus_moves)
                    eval_loss = eval_criterion(eval_preds.squeeze(), evaluations)
                    total_loss = move_loss + 0.1 * eval_loss  # Weight eval loss less

                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["gradient_clip"]
                    )
                    optimizer.step()
                    scheduler.step()

                    train_loss += total_loss.item()
                    train_correct += (
                        (torch.argmax(move_logits, dim=1) == magnus_moves).sum().item()
                    )
                    train_total += magnus_moves.size(0)

                    pbar.set_postfix(
                        {
                            "Loss": f"{total_loss.item():.4f}",
                            "Acc": f"{train_correct/train_total:.4f}",
                            "LR": f"{scheduler.get_last_lr()[0]:.6f}",
                        }
                    )

                # Fast validation
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch in val_loader:
                        positions, features, magnus_moves, evaluations = [
                            x.to(device, non_blocking=True) for x in batch
                        ]

                        move_logits, eval_preds = model(positions, features)

                        move_loss = move_criterion(move_logits, magnus_moves)
                        eval_loss = eval_criterion(eval_preds.squeeze(), evaluations)
                        total_loss = move_loss + 0.1 * eval_loss

                        val_loss += total_loss.item()
                        val_correct += (
                            (torch.argmax(move_logits, dim=1) == magnus_moves)
                            .sum()
                            .item()
                        )
                        val_total += magnus_moves.size(0)

                # Metrics
                train_acc = train_correct / train_total
                val_acc = val_correct / val_total

                if val_acc > best_val_acc:
                    best_val_acc = val_acc

                print(
                    f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
                )

                # Log to MLflow
                mlflow.log_metrics(
                    {
                        "train_accuracy": train_acc,
                        "val_accuracy": val_acc,
                        "train_loss": train_loss / len(train_loader),
                        "val_loss": val_loss / len(val_loader),
                    },
                    step=epoch,
                )

            # Final test evaluation
            model.eval()
            test_correct = 0
            test_total = 0
            top3_correct = 0
            top5_correct = 0

            with torch.no_grad():
                for batch in test_loader:
                    positions, features, magnus_moves, evaluations = [
                        x.to(device, non_blocking=True) for x in batch
                    ]

                    move_logits, _ = model(positions, features)

                    # Top-1 accuracy
                    test_correct += (
                        (torch.argmax(move_logits, dim=1) == magnus_moves).sum().item()
                    )

                    # Top-3 accuracy
                    _, top3_pred = torch.topk(move_logits, 3, dim=1)
                    top3_correct += (
                        (top3_pred == magnus_moves.unsqueeze(1)).any(dim=1).sum().item()
                    )

                    # Top-5 accuracy
                    _, top5_pred = torch.topk(move_logits, 5, dim=1)
                    top5_correct += (
                        (top5_pred == magnus_moves.unsqueeze(1)).any(dim=1).sum().item()
                    )

                    test_total += magnus_moves.size(0)

            # Final results
            test_acc = test_correct / test_total
            top3_acc = top3_correct / test_total
            top5_acc = top5_correct / test_total
            training_time = (time.time() - start_time) / 60

            print(f"\n🎯 Fast Training Results:")
            print(f"   Test Accuracy (Top-1): {test_acc:.4f}")
            print(f"   Test Accuracy (Top-3): {top3_acc:.4f}")
            print(f"   Test Accuracy (Top-5): {top5_acc:.4f}")
            print(f"   Training Time: {training_time:.2f} minutes")

            # Log final results
            mlflow.log_metrics(
                {
                    "final_test_accuracy": test_acc,
                    "final_top3_accuracy": top3_acc,
                    "final_top5_accuracy": top5_acc,
                    "training_time_minutes": training_time,
                    "best_val_accuracy": best_val_acc,
                }
            )

            # Save model with comprehensive management
            model_manager = EnhancedMagnusModelManager()

            final_metrics = {
                "test_accuracy": test_acc,
                "test_top3_accuracy": top3_acc,
                "test_top5_accuracy": top5_acc,
                "training_time_minutes": training_time,
                "best_val_accuracy": best_val_acc,
            }

            config_info = {
                "learning_rate": config["learning_rate"],
                "batch_size": config["batch_size"],
                "num_epochs": config["num_epochs"],
                "train_size": len(train_dataset),
                "val_size": len(val_dataset),
                "test_size": len(test_dataset),
                "vocab_size": train_dataset.vocab_size,
                "data_source": "magnus_extracted_positions_m3_pro.pkl",
                "device": str(device),
                "model_architecture": "FastMagnusModel",
                "optimization": "gpu_optimized",
            }

            model_version = integrate_enhanced_training(
                model_manager=model_manager,
                model=model,
                model_name="fast_magnus",
                experiment_name="magnus_fast_training",
                final_metrics=final_metrics,
                config=config_info,
            )

            print(f"💾 Model saved with ID: {model_version.model_id}")

            # Also save to MLflow
            mlflow.pytorch.log_model(model, "fast_magnus_model")

    else:
        print("❌ MLflow not available - running without tracking")
        # Run training without MLflow...

    return {
        "test_accuracy": test_acc,
        "top3_accuracy": top3_acc,
        "top5_accuracy": top5_acc,
        "training_time": training_time,
    }


if __name__ == "__main__":
    results = train_fast_magnus()
    print(f"\n✅ Fast training completed!")
    print(f"🎯 Final accuracy: {results['test_accuracy']:.4f}")
    print(f"🥇 Top-3 accuracy: {results['top3_accuracy']:.4f}")
    print(f"🏆 Top-5 accuracy: {results['top5_accuracy']:.4f}")
    print(f"⚡ Training time: {results['training_time']:.2f} minutes")
