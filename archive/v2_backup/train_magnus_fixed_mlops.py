#!/usr/bin/env python3
"""
Fixed Magnus Carlsen Training with MLOps - Compatible Dataset Format

This script fixes the dataset compatibility issues and provides working MLOps.
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

# MLOps imports
try:
    import mlflow
    import mlflow.pytorch
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
    print("✅ MLflow available")
except ImportError as e:
    MLFLOW_AVAILABLE = False
    print(f"⚠️  MLflow not installed: {e}")

from stockfish_magnus_trainer import (
    StockfishConfig,
    MagnusStyleModel,
)


class MagnusDatasetFixed(Dataset):
    """Fixed Magnus dataset that returns tensors directly"""

    def __init__(self, positions, features, stockfish_moves, magnus_moves, evaluations):
        self.positions = positions
        self.features = features
        self.stockfish_moves = stockfish_moves
        self.magnus_moves = magnus_moves
        self.evaluations = evaluations

        # Create move vocabulary
        all_moves = set()
        for move in stockfish_moves + magnus_moves:
            if move and isinstance(move, str):
                all_moves.add(move)

        self.move_to_idx = {move: idx for idx, move in enumerate(sorted(all_moves))}
        self.idx_to_move = {idx: move for move, idx in self.move_to_idx.items()}
        self.vocab_size = len(self.move_to_idx)

        # Convert features to array
        if features and isinstance(features[0], dict):
            self.feature_names = list(features[0].keys())
            self.feature_array = self._features_to_array()
        else:
            self.feature_names = [
                f"feature_{i}" for i in range(len(features[0]) if features else 0)
            ]
            self.feature_array = np.array(features) if features else np.array([])

        print(f"📊 Dataset created:")
        print(f"   Samples: {len(self.positions)}")
        print(f"   Vocabulary: {self.vocab_size} moves")
        print(f"   Features: {len(self.feature_names)} per position")

    def _features_to_array(self) -> np.ndarray:
        """Convert feature dictionaries to numpy array"""
        if not self.features:
            return np.array([])

        feature_matrix = np.zeros((len(self.features), len(self.feature_names)))
        for i, feat_dict in enumerate(self.features):
            for j, feat_name in enumerate(self.feature_names):
                feature_matrix[i, j] = float(feat_dict.get(feat_name, 0))

        return feature_matrix

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        # Convert position to tensor
        if isinstance(self.positions[idx], np.ndarray):
            position = torch.FloatTensor(self.positions[idx])
        else:
            position = torch.FloatTensor(self.positions[idx])

        # Convert features to tensor
        if len(self.feature_array) > 0:
            features = torch.FloatTensor(self.feature_array[idx])
        else:
            features = torch.zeros(1)

        # Convert moves to indices
        magnus_move_idx = self.move_to_idx.get(self.magnus_moves[idx], 0)

        # Convert evaluation
        try:
            eval_value = float(self.evaluations[idx])
            if eval_value > 9000:  # Cap extreme values
                eval_value = 9000
            elif eval_value < -9000:
                eval_value = -9000
        except (ValueError, TypeError):
            eval_value = 0.0

        evaluation = torch.FloatTensor([eval_value])

        # Return as tuple: (position, features, move, evaluation)
        return position, features, torch.LongTensor([magnus_move_idx]), evaluation


class MagnusMLOpsFixed:
    """Fixed Magnus MLOps trainer with working dataset"""

    def __init__(self, experiment_name: str = "magnus_chess_fixed"):
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
                pass  # Experiment exists

            mlflow.set_experiment(self.experiment_name)
            print(f"🔬 MLflow experiment: {self.experiment_name}")

        except Exception as e:
            print(f"⚠️  MLflow setup failed: {e}")
            self.enable_mlflow = False

    def load_extracted_positions(self):
        """Load pre-extracted positions"""
        data_path = Path("magnus_extracted_positions_m3_pro.pkl")

        if not data_path.exists():
            raise FileNotFoundError(
                f"❌ Pre-extracted positions not found: {data_path}"
            )

        print(f"📂 Loading pre-extracted positions...")
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        print(f"✅ Loaded {len(data['positions']):,} positions")
        return data

    def train_model(
        self, learning_rate=0.001, batch_size=512, num_epochs=20, run_name=None
    ):
        """Train Magnus model with MLflow tracking"""

        if run_name is None:
            run_name = f"magnus_fixed_{learning_rate}_{batch_size}_{datetime.now().strftime('%H%M%S')}"

        print(f"\n🚀 Training Magnus Model: {run_name}")
        print(f"   Learning Rate: {learning_rate}")
        print(f"   Batch Size: {batch_size}")
        print(f"   Epochs: {num_epochs}")

        # Load data
        extracted_data = self.load_extracted_positions()
        positions = extracted_data["positions"]
        features = extracted_data["features"]
        sf_moves = extracted_data["stockfish_moves"]
        magnus_moves = extracted_data["magnus_moves"]
        evaluations = extracted_data["evaluations"]

        # Start MLflow run
        if self.enable_mlflow:
            run = mlflow.start_run(run_name=run_name)
            print(f"🔬 MLflow run ID: {run.info.run_id}")

        try:
            # Device setup
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            print(f"🎮 Device: {device}")

            # Data preparation
            combined_data = list(
                zip(positions, features, sf_moves, magnus_moves, evaluations)
            )

            # Split data
            train_data, test_data = train_test_split(
                combined_data, test_size=0.2, random_state=42
            )
            train_data, val_data = train_test_split(
                train_data, test_size=0.25, random_state=42
            )  # 0.25 * 0.8 = 0.2

            def unpack_data(split_data):
                return list(zip(*split_data))

            # Create datasets
            train_dataset = MagnusDatasetFixed(*unpack_data(train_data))
            val_dataset = MagnusDatasetFixed(*unpack_data(val_data))
            test_dataset = MagnusDatasetFixed(*unpack_data(test_data))

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

            print(f"📚 Dataset:")
            print(f"   Train: {len(train_dataset):,}")
            print(f"   Validation: {len(val_dataset):,}")
            print(f"   Test: {len(test_dataset):,}")
            print(f"   Vocabulary: {train_dataset.vocab_size:,} moves")

            # Create model
            config = StockfishConfig()
            config.learning_rate = learning_rate
            config.batch_size = batch_size
            config.num_epochs = num_epochs

            model = MagnusStyleModel(
                config, train_dataset.vocab_size, len(train_dataset.feature_names)
            ).to(device)

            total_params = sum(p.numel() for p in model.parameters())
            print(f"🧠 Model: {total_params:,} parameters")

            # Log parameters to MLflow
            if self.enable_mlflow:
                mlflow.log_params(
                    {
                        "learning_rate": learning_rate,
                        "batch_size": batch_size,
                        "num_epochs": num_epochs,
                        "device": str(device),
                        "total_params": total_params,
                        "train_size": len(train_dataset),
                        "val_size": len(val_dataset),
                        "test_size": len(test_dataset),
                        "vocab_size": train_dataset.vocab_size,
                        "hardware": "M3 Pro",
                    }
                )

            # Training setup
            move_criterion = nn.CrossEntropyLoss()
            eval_criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # Training loop
            history = {
                "train_loss": [],
                "val_loss": [],
                "train_accuracy": [],
                "val_accuracy": [],
            }

            best_val_accuracy = 0

            print(f"\n🔥 Starting training...")
            training_start_time = time.time()

            for epoch in range(num_epochs):
                # Training phase
                model.train()
                train_losses = []
                train_correct = 0
                train_total = 0

                train_pbar = tqdm(
                    train_loader, desc=f"Epoch {epoch+1:2d}/{num_epochs}", leave=False
                )

                for batch in train_pbar:
                    positions, features, magnus_moves, evaluations = [
                        x.to(device) for x in batch
                    ]

                    optimizer.zero_grad()
                    move_logits, eval_preds = model(positions, features)

                    # Flatten the targets
                    magnus_moves = magnus_moves.squeeze(-1)
                    evaluations = evaluations.squeeze(-1)

                    move_loss = move_criterion(move_logits, magnus_moves)
                    eval_loss = eval_criterion(eval_preds.squeeze(), evaluations)
                    total_loss = move_loss + eval_loss

                    total_loss.backward()
                    optimizer.step()

                    train_losses.append(total_loss.item())
                    train_correct += (
                        (torch.argmax(move_logits, dim=1) == magnus_moves).sum().item()
                    )
                    train_total += magnus_moves.size(0)

                    train_pbar.set_postfix(
                        {
                            "Loss": f"{total_loss.item():.4f}",
                            "Acc": f"{train_correct/train_total:.3f}",
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
                        total_loss = move_loss + eval_loss

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

                # Log to MLflow
                if self.enable_mlflow:
                    mlflow.log_metrics(
                        {
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "train_accuracy": train_accuracy,
                            "val_accuracy": val_accuracy,
                        },
                        step=epoch,
                    )

                print(
                    f"Epoch {epoch+1:2d}/{num_epochs} | "
                    f"Train: {train_accuracy:.3f} | "
                    f"Val: {val_accuracy:.3f} | "
                    f"Loss: {val_loss:.4f}"
                )

            # Final test evaluation
            model.eval()
            test_correct = 0
            test_total = 0

            with torch.no_grad():
                for batch in test_loader:
                    positions, features, magnus_moves, evaluations = [
                        x.to(device) for x in batch
                    ]

                    move_logits, eval_preds = model(positions, features)
                    magnus_moves = magnus_moves.squeeze(-1)

                    test_correct += (
                        (torch.argmax(move_logits, dim=1) == magnus_moves).sum().item()
                    )
                    test_total += magnus_moves.size(0)

            test_accuracy = test_correct / test_total
            total_training_time = time.time() - training_start_time

            print(f"\n✅ Training completed!")
            print(f"   Training time: {total_training_time/60:.1f} minutes")
            print(f"   Best validation accuracy: {best_val_accuracy:.4f}")
            print(f"   Final test accuracy: {test_accuracy:.4f}")

            # Log final metrics
            if self.enable_mlflow:
                mlflow.log_metrics(
                    {
                        "final_test_accuracy": test_accuracy,
                        "best_val_accuracy": best_val_accuracy,
                        "training_time_minutes": total_training_time / 60,
                    }
                )

                # Save model
                model_path = f"models/magnus_mlflow_{run.info.run_id}"
                Path(model_path).mkdir(parents=True, exist_ok=True)

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "vocab_size": train_dataset.vocab_size,
                        "feature_names": train_dataset.feature_names,
                        "move_to_idx": train_dataset.move_to_idx,
                        "idx_to_move": train_dataset.idx_to_move,
                    },
                    f"{model_path}/model.pth",
                )

                # Log model to MLflow
                mlflow.pytorch.log_model(model, "model")

                print(f"💾 Model saved and logged to MLflow")
                print(f"🔬 Run ID: {run.info.run_id}")

            return {
                "test_accuracy": test_accuracy,
                "best_val_accuracy": best_val_accuracy,
                "training_time": total_training_time,
                "history": history,
                "run_id": run.info.run_id if self.enable_mlflow else None,
            }

        finally:
            if self.enable_mlflow:
                mlflow.end_run()


def main():
    """Run multiple training experiments with different hyperparameters"""

    print("🎯 Magnus Carlsen Fixed MLOps Training")
    print("=" * 50)

    trainer = MagnusMLOpsFixed("magnus_chess_fixed_runs")

    # Define experiments
    experiments = [
        {"lr": 0.001, "batch": 512, "epochs": 15, "name": "baseline"},
        {"lr": 0.003, "batch": 512, "epochs": 15, "name": "high_lr"},
        {"lr": 0.0005, "batch": 1024, "epochs": 15, "name": "large_batch"},
        {"lr": 0.0001, "batch": 256, "epochs": 20, "name": "conservative"},
    ]

    results = []

    for i, exp in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"Experiment {i+1}/{len(experiments)}: {exp['name']}")
        print(f"{'='*60}")

        try:
            result = trainer.train_model(
                learning_rate=exp["lr"],
                batch_size=exp["batch"],
                num_epochs=exp["epochs"],
                run_name=exp["name"],
            )
            result["experiment"] = exp
            results.append(result)

            print(f"✅ {exp['name']} completed!")
            print(f"   Test accuracy: {result['test_accuracy']:.4f}")

        except Exception as e:
            print(f"❌ {exp['name']} failed: {e}")
            results.append({"experiment": exp, "error": str(e)})

    # Summary
    print(f"\n📊 Experiment Summary")
    print(f"{'='*50}")

    successful_results = [r for r in results if "error" not in r]
    if successful_results:
        best_result = max(successful_results, key=lambda x: x["test_accuracy"])

        print(f"🏆 Best performing experiment: {best_result['experiment']['name']}")
        print(f"   Test accuracy: {best_result['test_accuracy']:.4f}")
        print(f"   Training time: {best_result['training_time']/60:.1f} minutes")

        if best_result.get("run_id"):
            print(f"   MLflow run ID: {best_result['run_id']}")

        print(f"\n📈 All results:")
        for result in successful_results:
            exp_name = result["experiment"]["name"]
            accuracy = result["test_accuracy"]
            time_min = result["training_time"] / 60
            print(f"   {exp_name:12s}: {accuracy:.4f} ({time_min:.1f}m)")

    print(f"\n🔬 View results in MLflow UI:")
    print(f"   cd {Path.cwd()}")
    print(f"   mlflow ui --host 127.0.0.1 --port 5000")
    print(f"   Open: http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
