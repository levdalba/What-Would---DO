"""
Improved Magnus Carlsen Style Chess Training System

This module implements a practical approach to train a chess move prediction model
that mimics Magnus Carlsen's playing style using supervised learning on his games.

Unlike LC0 which requires massive self-play training, this approach uses:
1. Supervised learning on Magnus's actual moves
2. Position evaluation and move ranking
3. Style-aware feature extraction
"""

import chess
import chess.pgn
import chess.engine
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for Magnus Carlsen style training"""

    data_dir: str = "data_processing"
    pgn_file: str = "carlsen-games.pgn"
    model_save_dir: str = "models/magnus_style_model"
    batch_size: int = 512
    learning_rate: float = 0.001
    num_epochs: int = 100
    validation_split: float = 0.2
    test_split: float = 0.1
    early_stopping_patience: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_games: Optional[int] = None  # Set to limit games for testing

    # Model architecture
    board_embedding_dim: int = 512
    hidden_dim: int = 1024
    num_layers: int = 4
    dropout: float = 0.2

    # Feature extraction
    use_game_phase: bool = True
    use_time_pressure: bool = True
    use_position_evaluation: bool = True
    use_piece_activity: bool = True


class ChessPositionEncoder:
    """Encodes chess positions into numerical features"""

    def __init__(self):
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0,
        }

    def encode_board(self, board: chess.Board) -> np.ndarray:
        """
        Encode chess board into a numerical representation
        Returns: 8x8x12 array (12 piece types x colors)
        """
        encoding = np.zeros((8, 8, 12), dtype=np.float32)

        piece_map = {
            (chess.PAWN, chess.WHITE): 0,
            (chess.KNIGHT, chess.WHITE): 1,
            (chess.BISHOP, chess.WHITE): 2,
            (chess.ROOK, chess.WHITE): 3,
            (chess.QUEEN, chess.WHITE): 4,
            (chess.KING, chess.WHITE): 5,
            (chess.PAWN, chess.BLACK): 6,
            (chess.KNIGHT, chess.BLACK): 7,
            (chess.BISHOP, chess.BLACK): 8,
            (chess.ROOK, chess.BLACK): 9,
            (chess.QUEEN, chess.BLACK): 10,
            (chess.KING, chess.BLACK): 11,
        }

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row, col = divmod(square, 8)
                piece_idx = piece_map[(piece.piece_type, piece.color)]
                encoding[row, col, piece_idx] = 1.0

        return encoding

    def extract_position_features(self, board: chess.Board) -> np.ndarray:
        """Extract additional position features"""
        features = []

        # Basic game state
        features.append(float(board.turn))  # White to move = 1, Black = 0
        features.append(len(board.move_stack))  # Move number

        # Castling rights
        features.extend(
            [
                board.has_kingside_castling_rights(chess.WHITE),
                board.has_queenside_castling_rights(chess.WHITE),
                board.has_kingside_castling_rights(chess.BLACK),
                board.has_queenside_castling_rights(chess.BLACK),
            ]
        )

        # En passant
        features.append(board.ep_square is not None)

        # Material count
        white_material = sum(
            len(board.pieces(piece_type, chess.WHITE)) * value
            for piece_type, value in self.piece_values.items()
        )
        black_material = sum(
            len(board.pieces(piece_type, chess.BLACK)) * value
            for piece_type, value in self.piece_values.items()
        )

        features.extend(
            [white_material, black_material, white_material - black_material]
        )

        # Piece mobility (approximate)
        legal_moves = len(list(board.legal_moves))
        features.append(legal_moves)

        # King safety (simplified)
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)

        white_king_attackers = len(board.attackers(chess.BLACK, white_king_square))
        black_king_attackers = len(board.attackers(chess.WHITE, black_king_square))

        features.extend([white_king_attackers, black_king_attackers])

        return np.array(features, dtype=np.float32)


class ChessMoveDataset(Dataset):
    """Dataset for Magnus Carlsen chess moves"""

    def __init__(
        self,
        positions: List[np.ndarray],
        features: List[np.ndarray],
        moves: List[str],
        outcomes: List[float],
    ):
        self.positions = positions
        self.features = features
        self.moves = moves
        self.outcomes = outcomes

        # Create move vocabulary
        self.move_to_idx = {move: idx for idx, move in enumerate(sorted(set(moves)))}
        self.idx_to_move = {idx: move for move, idx in self.move_to_idx.items()}
        self.vocab_size = len(self.move_to_idx)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        position = torch.FloatTensor(self.positions[idx])
        features = torch.FloatTensor(self.features[idx])
        move_idx = self.move_to_idx[self.moves[idx]]
        outcome = torch.FloatTensor([self.outcomes[idx]])

        return {
            "position": position,
            "features": features,
            "move": torch.LongTensor([move_idx]),
            "outcome": outcome,
        }


class MagnusStyleModel(nn.Module):
    """Neural network to predict Magnus Carlsen's moves"""

    def __init__(self, config: TrainingConfig, vocab_size: int, feature_dim: int):
        super().__init__()
        self.config = config

        # Board encoder
        self.board_conv = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 16, config.board_embedding_dim),
        )

        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
        )

        # Combined encoder
        combined_dim = config.board_embedding_dim + config.hidden_dim // 4

        self.move_predictor = nn.Sequential(
            nn.Linear(combined_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, vocab_size),
        )

        # Outcome predictor (for auxiliary loss)
        self.outcome_predictor = nn.Sequential(
            nn.Linear(combined_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, position, features):
        # Encode board
        board_embedding = self.board_conv(position)

        # Encode features
        feature_embedding = self.feature_encoder(features)

        # Combine embeddings
        combined = torch.cat([board_embedding, feature_embedding], dim=1)

        # Predict move and outcome
        move_logits = self.move_predictor(combined)
        outcome_pred = self.outcome_predictor(combined)

        return move_logits, outcome_pred


class MagnusTrainer:
    """Trainer for Magnus Carlsen style model"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.encoder = ChessPositionEncoder()
        self.device = torch.device(config.device)

        # Create directories
        Path(config.model_save_dir).mkdir(parents=True, exist_ok=True)

    def extract_games_data(self) -> Tuple[List, List, List, List]:
        """Extract training data from Magnus Carlsen PGN file"""
        pgn_path = Path(self.config.data_dir) / self.config.pgn_file

        positions = []
        features = []
        moves = []
        outcomes = []

        logger.info(f"Loading games from {pgn_path}")

        with open(pgn_path, "r") as pgn_file:
            game_count = 0

            while True:
                if self.config.max_games and game_count >= self.config.max_games:
                    break

                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                # Only include games where Magnus played
                headers = game.headers
                if "Carlsen" not in headers.get(
                    "White", ""
                ) and "Carlsen" not in headers.get("Black", ""):
                    continue

                # Determine Magnus's color and game outcome
                magnus_color = (
                    chess.WHITE
                    if "Carlsen" in headers.get("White", "")
                    else chess.BLACK
                )
                result = headers.get("Result", "*")

                # Convert result to outcome from Magnus's perspective
                if result == "1-0":
                    outcome = 1.0 if magnus_color == chess.WHITE else 0.0
                elif result == "0-1":
                    outcome = 1.0 if magnus_color == chess.BLACK else 0.0
                else:
                    outcome = 0.5  # Draw

                # Process moves
                board = game.board()
                move_number = 0

                for move in game.mainline_moves():
                    # Only include Magnus's moves
                    if board.turn == magnus_color:
                        # Encode position
                        position_encoding = self.encoder.encode_board(board)
                        position_features = self.encoder.extract_position_features(
                            board
                        )

                        positions.append(position_encoding)
                        features.append(position_features)
                        moves.append(move.uci())
                        outcomes.append(outcome)

                    board.push(move)
                    move_number += 1

                game_count += 1
                if game_count % 100 == 0:
                    logger.info(
                        f"Processed {game_count} games, {len(positions)} positions"
                    )

        logger.info(f"Extracted {len(positions)} positions from {game_count} games")
        return positions, features, moves, outcomes

    def create_datasets(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test datasets"""
        positions, features, moves, outcomes = self.extract_games_data()

        # Split data
        # First split: separate test set
        (
            pos_train_val,
            pos_test,
            feat_train_val,
            feat_test,
            moves_train_val,
            moves_test,
            out_train_val,
            out_test,
        ) = train_test_split(
            positions,
            features,
            moves,
            outcomes,
            test_size=self.config.test_split,
            random_state=42,
            stratify=outcomes,
        )

        # Second split: separate train and validation
        (
            pos_train,
            pos_val,
            feat_train,
            feat_val,
            moves_train,
            moves_val,
            out_train,
            out_val,
        ) = train_test_split(
            pos_train_val,
            feat_train_val,
            moves_train_val,
            out_train_val,
            test_size=self.config.validation_split / (1 - self.config.test_split),
            random_state=42,
            stratify=out_train_val,
        )

        # Create datasets
        train_dataset = ChessMoveDataset(pos_train, feat_train, moves_train, out_train)
        val_dataset = ChessMoveDataset(pos_val, feat_val, moves_val, out_val)
        test_dataset = ChessMoveDataset(pos_test, feat_test, moves_test, out_test)

        # Update vocabularies to match
        val_dataset.move_to_idx = train_dataset.move_to_idx
        val_dataset.idx_to_move = train_dataset.idx_to_move
        val_dataset.vocab_size = train_dataset.vocab_size

        test_dataset.move_to_idx = train_dataset.move_to_idx
        test_dataset.idx_to_move = train_dataset.idx_to_move
        test_dataset.vocab_size = train_dataset.vocab_size

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=4
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
        )

        # Save dataset info
        dataset_info = {
            "vocab_size": train_dataset.vocab_size,
            "move_to_idx": train_dataset.move_to_idx,
            "feature_dim": len(features[0]),
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
        }

        with open(Path(self.config.model_save_dir) / "dataset_info.json", "w") as f:
            json.dump(dataset_info, f, indent=2)

        logger.info(
            f"Dataset created - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
        )
        logger.info(f"Vocabulary size: {train_dataset.vocab_size}")

        return train_loader, val_loader, test_loader

    def train(self):
        """Train the Magnus style model"""
        # Create datasets
        train_loader, val_loader, test_loader = self.create_datasets()

        # Load dataset info
        with open(Path(self.config.model_save_dir) / "dataset_info.json", "r") as f:
            dataset_info = json.load(f)

        # Create model
        model = MagnusStyleModel(
            self.config, dataset_info["vocab_size"], dataset_info["feature_dim"]
        ).to(self.device)

        # Loss functions and optimizer
        move_criterion = nn.CrossEntropyLoss()
        outcome_criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_move_acc": [],
            "val_move_acc": [],
            "train_top3_acc": [],
            "val_top3_acc": [],
        }

        best_val_loss = float("inf")
        patience_counter = 0

        logger.info("Starting training...")

        for epoch in range(self.config.num_epochs):
            # Training phase
            model.train()
            train_losses = []
            train_move_correct = 0
            train_top3_correct = 0
            train_total = 0

            for batch in tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}"
            ):
                optimizer.zero_grad()

                positions = batch["position"].to(self.device)
                features = batch["features"].to(self.device)
                moves = batch["move"].squeeze().to(self.device)
                outcomes = batch["outcome"].squeeze().to(self.device)

                # Forward pass
                move_logits, outcome_pred = model(positions, features)

                # Compute losses
                move_loss = move_criterion(move_logits, moves)
                outcome_loss = outcome_criterion(outcome_pred.squeeze(), outcomes)
                total_loss = move_loss + 0.1 * outcome_loss  # Weighted combination

                # Backward pass
                total_loss.backward()
                optimizer.step()

                # Metrics
                train_losses.append(total_loss.item())

                _, predicted = torch.max(move_logits, 1)
                train_move_correct += (predicted == moves).sum().item()

                # Top-3 accuracy
                _, top3_pred = torch.topk(move_logits, 3, dim=1)
                train_top3_correct += sum(
                    moves[i] in top3_pred[i] for i in range(len(moves))
                )

                train_total += moves.size(0)

            # Validation phase
            model.eval()
            val_losses = []
            val_move_correct = 0
            val_top3_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    positions = batch["position"].to(self.device)
                    features = batch["features"].to(self.device)
                    moves = batch["move"].squeeze().to(self.device)
                    outcomes = batch["outcome"].squeeze().to(self.device)

                    move_logits, outcome_pred = model(positions, features)

                    move_loss = move_criterion(move_logits, moves)
                    outcome_loss = outcome_criterion(outcome_pred.squeeze(), outcomes)
                    total_loss = move_loss + 0.1 * outcome_loss

                    val_losses.append(total_loss.item())

                    _, predicted = torch.max(move_logits, 1)
                    val_move_correct += (predicted == moves).sum().item()

                    _, top3_pred = torch.topk(move_logits, 3, dim=1)
                    val_top3_correct += sum(
                        moves[i] in top3_pred[i] for i in range(len(moves))
                    )

                    val_total += moves.size(0)

            # Calculate metrics
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            train_move_acc = train_move_correct / train_total
            val_move_acc = val_move_correct / val_total
            train_top3_acc = train_top3_correct / train_total
            val_top3_acc = val_top3_correct / val_total

            # Update history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_move_acc"].append(train_move_acc)
            history["val_move_acc"].append(val_move_acc)
            history["train_top3_acc"].append(train_top3_acc)
            history["val_top3_acc"].append(val_top3_acc)

            # Logging
            logger.info(
                f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
            logger.info(f"Train Acc: {train_move_acc:.4f}, Val Acc: {val_move_acc:.4f}")
            logger.info(
                f"Train Top-3: {train_top3_acc:.4f}, Val Top-3: {val_top3_acc:.4f}"
            )

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping and model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Save best model
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "config": self.config,
                    },
                    Path(self.config.model_save_dir) / "best_model.pth",
                )

                logger.info("Saved new best model!")
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info("Early stopping triggered!")
                    break

        # Save training history
        with open(Path(self.config.model_save_dir) / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Final evaluation on test set
        self.evaluate_model(model, test_loader, dataset_info)

        return model, history

    def evaluate_model(self, model, test_loader, dataset_info):
        """Evaluate model on test set"""
        model.eval()
        test_move_correct = 0
        test_top3_correct = 0
        test_top5_correct = 0
        test_total = 0

        with torch.no_grad():
            for batch in test_loader:
                positions = batch["position"].to(self.device)
                features = batch["features"].to(self.device)
                moves = batch["move"].squeeze().to(self.device)

                move_logits, _ = model(positions, features)

                _, predicted = torch.max(move_logits, 1)
                test_move_correct += (predicted == moves).sum().item()

                _, top3_pred = torch.topk(move_logits, 3, dim=1)
                test_top3_correct += sum(
                    moves[i] in top3_pred[i] for i in range(len(moves))
                )

                _, top5_pred = torch.topk(move_logits, 5, dim=1)
                test_top5_correct += sum(
                    moves[i] in top5_pred[i] for i in range(len(moves))
                )

                test_total += moves.size(0)

        test_acc = test_move_correct / test_total
        test_top3_acc = test_top3_correct / test_total
        test_top5_acc = test_top5_correct / test_total

        logger.info(f"Test Results:")
        logger.info(f"Accuracy: {test_acc:.4f}")
        logger.info(f"Top-3 Accuracy: {test_top3_acc:.4f}")
        logger.info(f"Top-5 Accuracy: {test_top5_acc:.4f}")

        # Save test results
        test_results = {
            "accuracy": test_acc,
            "top3_accuracy": test_top3_acc,
            "top5_accuracy": test_top5_acc,
            "total_samples": test_total,
        }

        with open(Path(self.config.model_save_dir) / "test_results.json", "w") as f:
            json.dump(test_results, f, indent=2)


class MagnusStylePredictor:
    """Inference class for Magnus Carlsen style predictions"""

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = ChessPositionEncoder()

        # Load dataset info
        with open(self.model_dir / "dataset_info.json", "r") as f:
            self.dataset_info = json.load(f)

        # Load model
        checkpoint = torch.load(
            self.model_dir / "best_model.pth", map_location=self.device
        )
        self.config = checkpoint["config"]

        self.model = MagnusStyleModel(
            self.config,
            self.dataset_info["vocab_size"],
            self.dataset_info["feature_dim"],
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Load move vocabulary
        self.idx_to_move = {
            int(k): v for k, v in self.dataset_info["move_to_idx"].items()
        }

    def predict_move(
        self, board: chess.Board, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Predict Magnus's likely moves for a position"""
        # Encode position
        position_encoding = self.encoder.encode_board(board)
        position_features = self.encoder.extract_position_features(board)

        # Convert to tensors
        position_tensor = (
            torch.FloatTensor(position_encoding).unsqueeze(0).to(self.device)
        )
        features_tensor = (
            torch.FloatTensor(position_features).unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            move_logits, outcome_pred = self.model(position_tensor, features_tensor)
            probabilities = torch.softmax(move_logits, dim=1)

            # Get top-k moves
            top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)

            predictions = []
            for i in range(top_k):
                move_idx = top_indices[0][i].item()
                prob = top_probs[0][i].item()
                move_uci = self.idx_to_move.get(move_idx, "unknown")

                # Validate move is legal
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in board.legal_moves:
                        predictions.append((move_uci, prob))
                except:
                    continue

            return predictions


def main():
    """Main training function"""
    config = TrainingConfig()

    # For testing, limit to 1000 games
    config.max_games = 1000
    config.num_epochs = 50

    trainer = MagnusTrainer(config)

    try:
        model, history = trainer.train()
        logger.info("Training completed successfully!")

        # Plot training curves
        plot_training_curves(history, config.model_save_dir)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def plot_training_curves(history: Dict, save_dir: str):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history["train_loss"], label="Train Loss")
    axes[0, 0].plot(history["val_loss"], label="Validation Loss")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    # Move Accuracy
    axes[0, 1].plot(history["train_move_acc"], label="Train Accuracy")
    axes[0, 1].plot(history["val_move_acc"], label="Validation Accuracy")
    axes[0, 1].set_title("Move Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()

    # Top-3 Accuracy
    axes[1, 0].plot(history["train_top3_acc"], label="Train Top-3")
    axes[1, 0].plot(history["val_top3_acc"], label="Validation Top-3")
    axes[1, 0].set_title("Top-3 Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()

    # Combined plot
    axes[1, 1].plot(history["val_move_acc"], label="Top-1 Acc")
    axes[1, 1].plot(history["val_top3_acc"], label="Top-3 Acc")
    axes[1, 1].set_title("Validation Accuracy Comparison")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(Path(save_dir) / "training_curves.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
