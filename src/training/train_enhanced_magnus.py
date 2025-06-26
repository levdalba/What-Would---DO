#!/usr/bin/env python3
"""
Enhanced Magnus Carlsen Training with Improved Architecture and Training Techniques

This script implements several improvements to increase prediction accuracy:
1. Move filtering to reduce vocabulary size
2. Improved neural architecture with attention
3. Better training techniques (learning rate scheduling, focal loss)
4. Data augmentation and regularization
"""

import sys
import time
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import uuid
from tqdm import tqdm
import numpy as np
import warnings
import math

warnings.filterwarnings("ignore")

# Add the project directory to path
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from collections import Counter

# MLOps imports
try:
    import mlflow
    import mlflow.pytorch
    from mlflow.tracking import MlflowClient

    # Add model manager
    from mlops_model_manager import MagnusModelManager, integrate_with_training

    MLFLOW_AVAILABLE = True
    print("✅ MLflow available")
except ImportError as e:
    MLFLOW_AVAILABLE = False
    print(f"⚠️  MLflow not installed: {e}")

from stockfish_magnus_trainer import StockfishConfig


class FocalLoss(nn.Module):
    """Focal Loss to handle class imbalance"""

    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction="none")(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class PositionalEncoding(nn.Module):
    """Positional encoding for chess board positions"""

    def __init__(self, d_model, max_len=64):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1), :].unsqueeze(0)


class ImprovedMagnusModel(nn.Module):
    """Enhanced neural network with attention mechanism and better architecture"""

    def __init__(self, vocab_size: int, feature_dim: int = 1, hidden_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim

        # Enhanced board encoder with residual connections
        self.board_encoder = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )

        # Multi-head attention for position understanding
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // 2, num_heads=8, dropout=0.1, batch_first=True
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim // 2)

        # Feature encoder (minimal since we have limited features)
        if feature_dim > 1:
            self.feature_encoder = nn.Sequential(
                nn.Linear(feature_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 16),
            )
            self.use_features = True
            combined_dim = (hidden_dim // 2) + 16
        else:
            self.feature_encoder = None
            self.use_features = False
            combined_dim = hidden_dim // 2

        # Enhanced move prediction head with residual connections
        self.move_predictor = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, vocab_size),
        )

        # Evaluation head
        self.eval_adjustment = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, position, features):
        # Encode board position
        board_encoding = self.board_encoder(position)  # [batch, hidden_dim//2]

        # Reshape for attention (treat as sequence of 1)
        board_seq = board_encoding.unsqueeze(1)  # [batch, 1, hidden_dim//2]

        # Apply positional encoding and attention
        board_seq = self.pos_encoding(board_seq)
        attn_out, _ = self.attention(board_seq, board_seq, board_seq)
        board_encoding = attn_out.squeeze(1)  # [batch, hidden_dim//2]

        # Encode features if available
        if self.use_features and self.feature_encoder is not None:
            feature_encoding = self.feature_encoder(features)
            combined = torch.cat([board_encoding, feature_encoding], dim=1)
        else:
            combined = board_encoding

        # Predict move and evaluation
        move_logits = self.move_predictor(combined)
        eval_adjustment = self.eval_adjustment(combined)

        return move_logits, eval_adjustment


class FilteredMagnusDataset(Dataset):
    """Dataset with move filtering to reduce vocabulary size"""

    def __init__(
        self,
        positions,
        features,
        stockfish_moves,
        magnus_moves,
        evaluations,
        min_move_count=10,
    ):
        self.positions = positions
        self.features = (
            features if len(features) > 0 else [np.array([0.0]) for _ in positions]
        )
        self.stockfish_moves = stockfish_moves
        self.magnus_moves = magnus_moves
        self.evaluations = evaluations

        # Filter moves by frequency
        move_counts = Counter(magnus_moves)
        valid_moves = {
            move for move, count in move_counts.items() if count >= min_move_count
        }

        print(
            f"📊 Filtering moves: {len(move_counts)} → {len(valid_moves)} (min_count={min_move_count})"
        )

        # Filter dataset
        filtered_indices = [
            i for i, move in enumerate(magnus_moves) if move in valid_moves
        ]

        self.positions = [positions[i] for i in filtered_indices]
        self.features = [self.features[i] for i in filtered_indices]
        self.stockfish_moves = [stockfish_moves[i] for i in filtered_indices]
        self.magnus_moves = [magnus_moves[i] for i in filtered_indices]
        self.evaluations = [evaluations[i] for i in filtered_indices]

        # Create vocabulary mapping
        unique_moves = sorted(list(valid_moves))
        self.move_to_idx = {move: idx for idx, move in enumerate(unique_moves)}
        self.idx_to_move = {idx: move for move, idx in self.move_to_idx.items()}
        self.vocab_size = len(unique_moves)

        print(
            f"📚 Filtered dataset: {len(self.positions):,} samples, {self.vocab_size} moves"
        )

        # Compute feature array
        if len(self.features) > 0 and isinstance(self.features[0], np.ndarray):
            self.feature_array = np.array(self.features)
        else:
            self.feature_array = np.zeros((len(self.positions), 1))

        self.feature_names = [
            f"feature_{i}" for i in range(self.feature_array.shape[1])
        ]

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        # Convert to tensors
        position = torch.FloatTensor(self.positions[idx])
        features = torch.FloatTensor(self.feature_array[idx])

        # Convert move to index
        magnus_move_idx = self.move_to_idx.get(self.magnus_moves[idx], 0)

        # Convert evaluation
        try:
            eval_value = float(self.evaluations[idx])
            eval_value = max(-9000, min(9000, eval_value))  # Clamp
        except (ValueError, TypeError):
            eval_value = 0.0

        evaluation = torch.FloatTensor([eval_value])

        return position, features, torch.LongTensor([magnus_move_idx]), evaluation


class EnhancedMagnusTrainer:
    """Enhanced trainer with improved techniques"""

    def __init__(self, experiment_name: str = "magnus_enhanced"):
        self.experiment_name = experiment_name
        self.enable_mlflow = MLFLOW_AVAILABLE

        if self.enable_mlflow:
            self.setup_mlflow()

    def setup_mlflow(self):
        """Setup MLflow tracking"""
        try:
            mlflow.set_tracking_uri("./mlruns")

            try:
                mlflow.create_experiment(self.experiment_name)
            except:
                pass

            mlflow.set_experiment(self.experiment_name)
            print(f"🔬 MLflow experiment: {self.experiment_name}")
        except Exception as e:
            print(f"⚠️  MLflow setup error: {e}")
            self.enable_mlflow = False

    def train_enhanced_model(
        self,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        num_epochs: int = 50,
        min_move_count: int = 10,
        hidden_dim: int = 512,
        use_focal_loss: bool = True,
        use_scheduler: bool = True,
    ):
        """Train enhanced Magnus model"""

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"🎮 Device: {device}")

        # Start MLflow run
        if self.enable_mlflow:
            mlflow.start_run()

        try:
            # Load data
            print("📂 Loading training data...")
            with open("magnus_extracted_positions_m3_pro.pkl", "rb") as f:
                data = pickle.load(f)

            positions = data["positions"]
            features = data["features"]
            sf_moves = data["stockfish_moves"]
            magnus_moves = data["magnus_moves"]
            evaluations = data["evaluations"]

            # Create filtered dataset
            combined_data = list(
                zip(positions, features, sf_moves, magnus_moves, evaluations)
            )
            train_data, test_data = train_test_split(
                combined_data, test_size=0.2, random_state=42
            )
            train_data, val_data = train_test_split(
                train_data, test_size=0.25, random_state=42
            )

            def unpack_data(split_data):
                return list(zip(*split_data))

            train_dataset = FilteredMagnusDataset(
                *unpack_data(train_data), min_move_count=min_move_count
            )
            val_dataset = FilteredMagnusDataset(
                *unpack_data(val_data), min_move_count=min_move_count
            )
            test_dataset = FilteredMagnusDataset(
                *unpack_data(test_data), min_move_count=min_move_count
            )

            # Share vocabulary
            val_dataset.move_to_idx = train_dataset.move_to_idx
            val_dataset.idx_to_move = train_dataset.idx_to_move
            val_dataset.vocab_size = train_dataset.vocab_size

            test_dataset.move_to_idx = train_dataset.move_to_idx
            test_dataset.idx_to_move = train_dataset.idx_to_move
            test_dataset.vocab_size = train_dataset.vocab_size

            # Create data loaders
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
            )
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
            )

            print(f"📚 Enhanced Dataset:")
            print(f"   Train: {len(train_dataset):,}")
            print(f"   Validation: {len(val_dataset):,}")
            print(f"   Test: {len(test_dataset):,}")
            print(
                f"   Vocabulary: {train_dataset.vocab_size:,} moves (filtered from original)"
            )

            # Create enhanced model
            model = ImprovedMagnusModel(
                vocab_size=train_dataset.vocab_size,
                feature_dim=train_dataset.feature_array.shape[1],
                hidden_dim=hidden_dim,
            ).to(device)

            total_params = sum(p.numel() for p in model.parameters())
            print(f"🧠 Enhanced Model: {total_params:,} parameters")

            # Enhanced loss functions
            if use_focal_loss:
                move_criterion = FocalLoss(alpha=1, gamma=2)
                print("🎯 Using Focal Loss for move prediction")
            else:
                move_criterion = nn.CrossEntropyLoss()

            eval_criterion = nn.MSELoss()

            # Enhanced optimizer
            optimizer = optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=0.01
            )

            # Learning rate scheduler
            if use_scheduler:
                scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=learning_rate * 3,
                    epochs=num_epochs,
                    steps_per_epoch=len(train_loader),
                    pct_start=0.1,
                )
                print("📈 Using OneCycle learning rate scheduler")

            # Log hyperparameters
            if self.enable_mlflow:
                mlflow.log_params(
                    {
                        "learning_rate": learning_rate,
                        "batch_size": batch_size,
                        "num_epochs": num_epochs,
                        "min_move_count": min_move_count,
                        "hidden_dim": hidden_dim,
                        "use_focal_loss": use_focal_loss,
                        "use_scheduler": use_scheduler,
                        "model_type": "enhanced",
                        "vocab_size": train_dataset.vocab_size,
                        "total_params": total_params,
                        "device": str(device),
                    }
                )

            # Training loop
            history = {
                "train_loss": [],
                "val_loss": [],
                "train_accuracy": [],
                "val_accuracy": [],
            }
            best_val_accuracy = 0.0
            start_time = time.time()

            for epoch in range(num_epochs):
                # Training phase
                model.train()
                train_losses = []
                train_correct = 0
                train_total = 0

                train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

                for batch in train_pbar:
                    positions, features, magnus_moves, evaluations = [
                        x.to(device) for x in batch
                    ]

                    optimizer.zero_grad()

                    move_logits, eval_preds = model(positions, features)

                    magnus_moves = magnus_moves.squeeze(-1)
                    evaluations = evaluations.squeeze(-1)

                    move_loss = move_criterion(move_logits, magnus_moves)
                    eval_loss = eval_criterion(eval_preds.squeeze(), evaluations)
                    total_loss = move_loss + 0.1 * eval_loss  # Weight eval loss lower

                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0
                    )  # Gradient clipping
                    optimizer.step()

                    if use_scheduler:
                        scheduler.step()

                    train_losses.append(total_loss.item())
                    train_correct += (
                        (torch.argmax(move_logits, dim=1) == magnus_moves).sum().item()
                    )
                    train_total += magnus_moves.size(0)

                    current_lr = optimizer.param_groups[0]["lr"]
                    train_pbar.set_postfix(
                        {
                            "Loss": f"{total_loss.item():.4f}",
                            "Acc": f"{train_correct/train_total:.3f}",
                            "LR": f"{current_lr:.6f}",
                        }
                    )

                # Validation phase
                model.eval()
                val_losses = []
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch in val_loader:
                        positions, features, magnus_moves, evaluations = [
                            x.to(device) for x in batch
                        ]

                        move_logits, eval_preds = model(positions, features)

                        magnus_moves = magnus_moves.squeeze(-1)
                        evaluations = evaluations.squeeze(-1)

                        move_loss = move_criterion(move_logits, magnus_moves)
                        eval_loss = eval_criterion(eval_preds.squeeze(), evaluations)
                        total_loss = move_loss + 0.1 * eval_loss

                        val_losses.append(total_loss.item())
                        val_correct += (
                            (torch.argmax(move_logits, dim=1) == magnus_moves)
                            .sum()
                            .item()
                        )
                        val_total += magnus_moves.size(0)

                # Calculate metrics
                train_loss = np.mean(train_losses)
                val_loss = np.mean(val_losses)
                train_accuracy = train_correct / train_total
                val_accuracy = val_correct / val_total

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["train_accuracy"].append(train_accuracy)
                history["val_accuracy"].append(val_accuracy)

                # Track best model
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    if self.enable_mlflow:
                        mlflow.log_metric(
                            "best_val_accuracy", best_val_accuracy, step=epoch
                        )

                # Log metrics
                if self.enable_mlflow:
                    mlflow.log_metrics(
                        {
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "train_accuracy": train_accuracy,
                            "val_accuracy": val_accuracy,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        },
                        step=epoch,
                    )

                print(
                    f"Epoch {epoch+1}: Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, Val Loss: {val_loss:.4f}"
                )

            # Final test evaluation
            model.eval()
            test_correct = 0
            test_total = 0
            test_top3_correct = 0
            test_top5_correct = 0

            with torch.no_grad():
                for batch in test_loader:
                    positions, features, magnus_moves, evaluations = [
                        x.to(device) for x in batch
                    ]

                    move_logits, _ = model(positions, features)
                    magnus_moves = magnus_moves.squeeze(-1)

                    # Top-1 accuracy
                    predicted = torch.argmax(move_logits, dim=1)
                    test_correct += (predicted == magnus_moves).sum().item()
                    test_total += magnus_moves.size(0)

                    # Top-3 and Top-5 accuracy
                    _, top_k = torch.topk(move_logits, k=5, dim=1)
                    for i in range(magnus_moves.size(0)):
                        if magnus_moves[i] in top_k[i][:3]:
                            test_top3_correct += 1
                        if magnus_moves[i] in top_k[i]:
                            test_top5_correct += 1

            test_accuracy = test_correct / test_total
            test_top3_accuracy = test_top3_correct / test_total
            test_top5_accuracy = test_top5_correct / test_total
            training_time = (time.time() - start_time) / 60

            print(f"\n🎯 Final Results:")
            print(f"   Test Accuracy (Top-1): {test_accuracy:.4f}")
            print(f"   Test Accuracy (Top-3): {test_top3_accuracy:.4f}")
            print(f"   Test Accuracy (Top-5): {test_top5_accuracy:.4f}")
            print(f"   Training Time: {training_time:.2f} minutes")

            # Log final metrics
            if self.enable_mlflow:
                mlflow.log_metrics(
                    {
                        "final_test_accuracy": test_accuracy,
                        "test_top3_accuracy": test_top3_accuracy,
                        "test_top5_accuracy": test_top5_accuracy,
                        "training_time_minutes": training_time,
                    }
                )

                # Save model with comprehensive management
                model_manager = MagnusModelManager()

                final_metrics = {
                    "test_accuracy": test_accuracy,
                    "test_top3_accuracy": test_top3_accuracy,
                    "test_top5_accuracy": test_top5_accuracy,
                    "training_time_minutes": training_time,
                    "best_val_accuracy": best_val_accuracy,
                }

                config_info = {
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "min_move_count": min_move_count,
                    "hidden_dim": hidden_dim,
                    "use_focal_loss": use_focal_loss,
                    "use_scheduler": use_scheduler,
                    "train_size": len(train_dataset),
                    "val_size": len(val_dataset),
                    "test_size": len(test_dataset),
                    "vocab_size": train_dataset.vocab_size,
                    "data_source": "magnus_extracted_positions_m3_pro.pkl",
                    "device": str(device),
                    "model_architecture": "EnhancedMagnusModel",
                }

                model_id = integrate_with_training(
                    model_manager=model_manager,
                    model=model,
                    model_name="enhanced_magnus",
                    experiment_name=self.experiment_name,
                    final_metrics=final_metrics,
                    config=config_info,
                )

                print(f"💾 Model saved with ID: {model_id}")

                # Also save to MLflow
                mlflow.pytorch.log_model(model, "enhanced_magnus_model")

            return {
                "test_accuracy": test_accuracy,
                "test_top3_accuracy": test_top3_accuracy,
                "test_top5_accuracy": test_top5_accuracy,
                "training_time": training_time,
                "history": history,
            }

        finally:
            if self.enable_mlflow:
                mlflow.end_run()


if __name__ == "__main__":
    trainer = EnhancedMagnusTrainer("magnus_enhanced_experiments")

    print("🚀 Starting Enhanced Magnus Training...")

    # Test improved configuration
    result = trainer.train_enhanced_model(
        learning_rate=0.001,
        batch_size=128,
        num_epochs=40,
        min_move_count=15,  # Filter rare moves more aggressively
        hidden_dim=256,
        use_focal_loss=True,
        use_scheduler=True,
    )

    if result:
        print(f"\n✅ Training completed!")
        print(f"🎯 Final accuracy: {result['test_accuracy']:.4f}")
        print(f"🥇 Top-3 accuracy: {result['test_top3_accuracy']:.4f}")
        print(f"🏆 Top-5 accuracy: {result['test_top5_accuracy']:.4f}")
