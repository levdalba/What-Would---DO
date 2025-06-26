#!/usr/bin/env python3
"""
Advanced Magnus Model - Targeting 15%+ accuracy
Features: Attention mechanism, advanced features, ensemble techniques
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import time
from collections import Counter

# Add project path
sys.path.append(str(Path(__file__).parent))


class AdvancedChessFeatureExtractor:
    """Extract advanced chess features for better move prediction"""

    def __init__(self):
        self.piece_values = {
            "p": 1,
            "n": 3,
            "b": 3,
            "r": 5,
            "q": 9,
            "k": 0,
            "P": 1,
            "N": 3,
            "B": 3,
            "R": 5,
            "Q": 9,
            "K": 0,
        }

    def extract_features(self, position_data):
        """Extract comprehensive position features"""
        features = []

        # Basic piece counts and material balance
        white_material = sum(
            self.piece_values.get(p, 0) for p in str(position_data) if p.isupper()
        )
        black_material = sum(
            self.piece_values.get(p, 0) for p in str(position_data) if p.islower()
        )
        material_balance = white_material - black_material

        # Feature vector
        features.extend(
            [
                white_material / 39.0,  # Normalized material (max = Q+2R+2B+2N+8P)
                black_material / 39.0,
                material_balance / 39.0,
                abs(material_balance) / 39.0,  # Material imbalance magnitude
            ]
        )

        # Game phase estimation (opening/middlegame/endgame)
        total_material = white_material + black_material
        game_phase = total_material / 78.0  # 0 = endgame, 1 = opening
        features.extend(
            [
                game_phase,
                1 - game_phase,  # Endgame indicator
                min(game_phase * 2, 1),  # Opening indicator
                max(0, min((game_phase - 0.3) * 2, 1)),  # Middlegame indicator
            ]
        )

        return np.array(features, dtype=np.float32)


class MultiHeadAttention(nn.Module):
    """Multi-head attention for position encoding"""

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.size(0)

        # Linear transformations
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)

        # Concatenate heads
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )
        output = self.W_o(context)

        return output.mean(dim=1)  # Global average pooling


class AdvancedMagnusModel(nn.Module):
    """Advanced model with attention and sophisticated features"""

    def __init__(self, vocab_size, feature_dim=8):
        super().__init__()
        self.vocab_size = vocab_size

        # Advanced board encoder with residual connections
        self.board_encoder = nn.Sequential(
            nn.Linear(768, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        # Attention mechanism for board understanding
        self.board_attention = MultiHeadAttention(256, 8)

        # Advanced feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Combined feature processing
        combined_dim = 256 + 32
        self.feature_combiner = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Move prediction with multiple paths
        self.move_predictor = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, vocab_size),
        )

        # Evaluation head
        self.eval_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, position, features):
        # Process board position
        board_enc = self.board_encoder(position)

        # Apply attention (reshape for attention if needed)
        if len(board_enc.shape) == 2:
            board_enc_reshaped = board_enc.unsqueeze(1)  # Add sequence dimension
            board_att = self.board_attention(board_enc_reshaped)
        else:
            board_att = self.board_attention(board_enc)

        # Process additional features
        feature_enc = self.feature_encoder(features)

        # Combine features
        combined = torch.cat([board_att, feature_enc], dim=1)
        combined = self.feature_combiner(combined)

        # Predictions
        move_logits = self.move_predictor(combined)
        eval_pred = self.eval_predictor(combined)

        return move_logits, eval_pred


class AdvancedMagnusDataset(Dataset):
    """Advanced dataset with sophisticated feature extraction"""

    def __init__(
        self, positions, magnus_moves, evaluations, feature_extractor, min_move_count=20
    ):
        self.positions = positions
        self.magnus_moves = magnus_moves
        self.evaluations = evaluations
        self.feature_extractor = feature_extractor

        # Filter moves and create vocabulary
        move_counts = Counter(magnus_moves)
        self.valid_moves = {
            move for move, count in move_counts.items() if count >= min_move_count
        }

        # Filter data
        valid_indices = [
            i for i, move in enumerate(magnus_moves) if move in self.valid_moves
        ]
        self.positions = [positions[i] for i in valid_indices]
        self.magnus_moves = [magnus_moves[i] for i in valid_indices]
        self.evaluations = [evaluations[i] for i in valid_indices]

        # Create vocabulary
        unique_moves = sorted(self.valid_moves)
        self.move_to_idx = {move: idx for idx, move in enumerate(unique_moves)}
        self.idx_to_move = {idx: move for move, idx in self.move_to_idx.items()}
        self.vocab_size = len(unique_moves)

        print(
            f"📊 Advanced filtering: {len(move_counts)} → {len(self.valid_moves)} moves (min_count={min_move_count})"
        )
        print(
            f"📚 Advanced dataset: {len(self.positions):,} samples, {self.vocab_size} moves"
        )

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        # Position
        position = torch.FloatTensor(self.positions[idx])

        # Extract advanced features
        features = self.feature_extractor.extract_features(self.positions[idx])
        features = torch.FloatTensor(features)

        # Move index
        move_idx = self.move_to_idx.get(self.magnus_moves[idx], 0)

        # Evaluation
        try:
            eval_value = float(self.evaluations[idx])
            eval_value = max(-9999, min(9999, eval_value))  # Clamp
        except:
            eval_value = 0.0

        evaluation = torch.FloatTensor([eval_value / 1000.0])  # Normalize

        return position, features, torch.LongTensor([move_idx]), evaluation


def weighted_focal_loss(outputs, targets, alpha=0.25, gamma=2.0, class_weights=None):
    """Advanced focal loss with class weighting"""
    ce_loss = F.cross_entropy(outputs, targets, weight=class_weights, reduction="none")
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()


def train_advanced_magnus():
    """Train the advanced Magnus model"""

    print("🚀 Starting Advanced Magnus Training...")

    # Setup MLflow
    mlflow.set_tracking_uri("./mlruns")
    try:
        mlflow.create_experiment("magnus_advanced_experiments")
    except:
        pass
    mlflow.set_experiment("magnus_advanced_experiments")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🎮 Device: {device}")

    # Load data
    print("📂 Loading training data...")
    with open("magnus_extracted_positions_m3_pro.pkl", "rb") as f:
        data = pickle.load(f)

    positions = data["positions"]
    magnus_moves = data["magnus_moves"]
    evaluations = data["evaluations"]

    # Create feature extractor
    feature_extractor = AdvancedChessFeatureExtractor()

    # Split data first, then create datasets
    combined_data = list(zip(positions, magnus_moves, evaluations))
    train_data, test_data = train_test_split(
        combined_data, test_size=0.18, random_state=42
    )
    train_data, val_data = train_test_split(train_data, test_size=0.22, random_state=42)

    # Unpack data
    train_pos, train_moves, train_evals = zip(*train_data)
    val_pos, val_moves, val_evals = zip(*val_data)
    test_pos, test_moves, test_evals = zip(*test_data)

    # Create datasets
    train_dataset = AdvancedMagnusDataset(
        train_pos, train_moves, train_evals, feature_extractor, min_move_count=25
    )

    # Share vocabulary with validation and test sets
    val_dataset = AdvancedMagnusDataset(
        val_pos, val_moves, val_evals, feature_extractor, min_move_count=1
    )
    val_dataset.move_to_idx = train_dataset.move_to_idx
    val_dataset.idx_to_move = train_dataset.idx_to_move
    val_dataset.vocab_size = train_dataset.vocab_size
    val_dataset.valid_moves = train_dataset.valid_moves

    test_dataset = AdvancedMagnusDataset(
        test_pos, test_moves, test_evals, feature_extractor, min_move_count=1
    )
    test_dataset.move_to_idx = train_dataset.move_to_idx
    test_dataset.idx_to_move = train_dataset.idx_to_move
    test_dataset.vocab_size = train_dataset.vocab_size
    test_dataset.valid_moves = train_dataset.valid_moves

    print(f"📚 Advanced Dataset:")
    print(f"   Train: {len(train_dataset):,}")
    print(f"   Validation: {len(val_dataset):,}")
    print(f"   Test: {len(test_dataset):,}")
    print(f"   Vocabulary: {train_dataset.vocab_size:,} moves")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    # Create model
    model = AdvancedMagnusModel(train_dataset.vocab_size, feature_dim=8).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Advanced Model: {total_params:,} parameters")

    # Calculate class weights for imbalanced dataset
    move_counts = Counter(train_dataset.magnus_moves)
    max_count = max(move_counts.values())
    class_weights = torch.FloatTensor(
        [
            max_count / move_counts.get(train_dataset.idx_to_move[i], 1)
            for i in range(train_dataset.vocab_size)
        ]
    ).to(device)

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.005,
        epochs=50,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
    )

    # Loss functions
    eval_criterion = nn.MSELoss()

    print("🎯 Using Advanced Weighted Focal Loss")
    print("📈 Using OneCycle scheduler with cosine annealing")

    # Training loop
    best_val_accuracy = 0
    start_time = time.time()

    with mlflow.start_run():
        mlflow.log_params(
            {
                "model_type": "AdvancedMagnusModel",
                "vocab_size": train_dataset.vocab_size,
                "total_params": total_params,
                "batch_size": 128,
                "learning_rate": 0.002,
                "num_epochs": 50,
                "min_move_count": 25,
                "device": str(device),
            }
        )

        for epoch in range(50):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/50")
            for batch in pbar:
                positions, features, magnus_moves, evaluations = [
                    x.to(device) for x in batch
                ]

                optimizer.zero_grad()

                move_logits, eval_preds = model(positions, features)
                magnus_moves = magnus_moves.squeeze(-1)
                evaluations = evaluations.squeeze(-1)

                # Advanced loss calculation
                move_loss = weighted_focal_loss(
                    move_logits, magnus_moves, class_weights=class_weights
                )
                eval_loss = eval_criterion(eval_preds.squeeze(), evaluations)
                total_loss = move_loss + 0.1 * eval_loss  # Weight eval loss less

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )  # Gradient clipping
                optimizer.step()
                scheduler.step()

                train_loss += total_loss.item()
                train_correct += (
                    (torch.argmax(move_logits, dim=1) == magnus_moves).sum().item()
                )
                train_total += magnus_moves.size(0)

                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix(
                    {
                        "Loss": f"{total_loss.item():.4f}",
                        "Acc": f"{train_correct/train_total:.3f}",
                        "LR": f"{current_lr:.6f}",
                    }
                )

            # Validation
            model.eval()
            val_loss = 0
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

                    move_loss = weighted_focal_loss(
                        move_logits, magnus_moves, class_weights=class_weights
                    )
                    eval_loss = eval_criterion(eval_preds.squeeze(), evaluations)
                    total_loss = move_loss + 0.1 * eval_loss

                    val_loss += total_loss.item()
                    val_correct += (
                        (torch.argmax(move_logits, dim=1) == magnus_moves).sum().item()
                    )
                    val_total += magnus_moves.size(0)

            # Calculate metrics
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            print(
                f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {avg_val_loss:.4f}"
            )

            # Log metrics
            mlflow.log_metrics(
                {
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                },
                step=epoch,
            )

            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save(model.state_dict(), "best_advanced_magnus_model.pth")

        # Final testing
        model.load_state_dict(torch.load("best_advanced_magnus_model.pth"))
        model.eval()

        test_correct_top1 = 0
        test_correct_top3 = 0
        test_correct_top5 = 0
        test_total = 0

        with torch.no_grad():
            for batch in test_loader:
                positions, features, magnus_moves, evaluations = [
                    x.to(device) for x in batch
                ]

                move_logits, _ = model(positions, features)
                magnus_moves = magnus_moves.squeeze(-1)

                # Top-k accuracies
                _, top5_pred = torch.topk(move_logits, 5, dim=1)
                _, top3_pred = torch.topk(move_logits, 3, dim=1)
                _, top1_pred = torch.topk(move_logits, 1, dim=1)

                test_correct_top1 += (top1_pred.squeeze() == magnus_moves).sum().item()
                test_correct_top3 += sum(
                    (magnus_moves[i] in top3_pred[i]) for i in range(len(magnus_moves))
                )
                test_correct_top5 += sum(
                    (magnus_moves[i] in top5_pred[i]) for i in range(len(magnus_moves))
                )
                test_total += magnus_moves.size(0)

        # Final metrics
        test_acc_top1 = test_correct_top1 / test_total
        test_acc_top3 = test_correct_top3 / test_total
        test_acc_top5 = test_correct_top5 / test_total
        training_time = (time.time() - start_time) / 60

        print(f"\n🎯 Advanced Results:")
        print(f"   Test Accuracy (Top-1): {test_acc_top1:.4f}")
        print(f"   Test Accuracy (Top-3): {test_acc_top3:.4f}")
        print(f"   Test Accuracy (Top-5): {test_acc_top5:.4f}")
        print(f"   Training Time: {training_time:.2f} minutes")

        # Log final metrics
        mlflow.log_metrics(
            {
                "final_test_accuracy_top1": test_acc_top1,
                "final_test_accuracy_top3": test_acc_top3,
                "final_test_accuracy_top5": test_acc_top5,
                "training_time_minutes": training_time,
                "best_val_accuracy": best_val_accuracy,
            }
        )

        # Save model
        mlflow.pytorch.log_model(model, "advanced_magnus_model")

        return {
            "test_accuracy_top1": test_acc_top1,
            "test_accuracy_top3": test_acc_top3,
            "test_accuracy_top5": test_acc_top5,
            "training_time": training_time,
        }


if __name__ == "__main__":
    train_advanced_magnus()
