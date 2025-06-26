#!/usr/bin/env python3
"""
Magnus Carlsen Training with MLflow Integration (Simplified)

This script provides MLOps capabilities without overwhelming complexity.
"""

import sys
import time
import json
import pickle
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Add the project directory to path
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# MLflow integration
try:
    import mlflow
    import mlflow.pytorch

    MLFLOW_AVAILABLE = True
    print("✅ MLflow available")
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️  MLflow not installed. Run: pip install mlflow")

from stockfish_magnus_trainer import (
    StockfishConfig,
    MagnusDataset,
    MagnusStyleModel,
    plot_training_curves,
)


class MLflowMagnusTrainer:
    """Magnus training with MLflow tracking"""

    def __init__(self, experiment_name="magnus_chess_m3_pro"):
        self.experiment_name = experiment_name
        self.enable_mlflow = MLFLOW_AVAILABLE

        if self.enable_mlflow:
            self.setup_mlflow()

    def setup_mlflow(self):
        """Setup MLflow experiment tracking"""
        try:
            # Set tracking URI to local directory
            mlflow.set_tracking_uri("./mlruns")

            # Create or get experiment
            try:
                mlflow.create_experiment(self.experiment_name)
            except:
                pass  # Experiment already exists

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
        self, learning_rate=0.0001, batch_size=512, num_epochs=30, run_name=None
    ):
        """
        Train Magnus model with MLflow tracking

        Args:
            learning_rate: Learning rate for training
            batch_size: Batch size (512 works well for M3 Pro)
            num_epochs: Number of training epochs
            run_name: Custom name for this training run
        """

        if run_name is None:
            run_name = f"m3_pro_{learning_rate}_{batch_size}_{datetime.now().strftime('%H%M%S')}"

        print(f"🚀 Training Magnus Model: {run_name}")
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
            # Training configuration
            config = StockfishConfig()
            config.learning_rate = learning_rate
            config.batch_size = batch_size
            config.num_epochs = num_epochs
            config.validation_split = 0.2
            config.test_split = 0.1
            config.early_stopping_patience = 8

            # Device setup
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            config.device = str(device)

            print(f"🎮 Device: {device}")

            # Data preparation
            data = list(zip(positions, features, sf_moves, magnus_moves, evaluations))

            # Split data
            test_size = config.test_split
            val_size = config.validation_split / (1 - test_size)

            data_train_val, data_test = train_test_split(
                data, test_size=test_size, random_state=42
            )
            data_train, data_val = train_test_split(
                data_train_val, test_size=val_size, random_state=42
            )

            def unpack_data(split_data):
                return list(zip(*split_data))

            train_dataset = MagnusDataset(*unpack_data(data_train))
            val_dataset = MagnusDataset(*unpack_data(data_val))
            test_dataset = MagnusDataset(*unpack_data(data_test))

            # Share vocabulary
            val_dataset.move_to_idx = train_dataset.move_to_idx
            val_dataset.idx_to_move = train_dataset.idx_to_move
            val_dataset.vocab_size = train_dataset.vocab_size

            test_dataset.move_to_idx = train_dataset.move_to_idx
            test_dataset.idx_to_move = train_dataset.idx_to_move
            test_dataset.vocab_size = train_dataset.vocab_size

            # Create data loaders
            train_loader = DataLoader(
                train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
            )
            val_loader = DataLoader(
                val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
            )
            test_loader = DataLoader(
                test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
            )

            print(f"📚 Dataset:")
            print(f"   Train: {len(train_dataset):,}")
            print(f"   Validation: {len(val_dataset):,}")
            print(f"   Test: {len(test_dataset):,}")
            print(f"   Vocabulary: {train_dataset.vocab_size:,} moves")

            # Create model
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
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=5, factor=0.5
            )

            # Training history
            history = {
                "train_loss": [],
                "val_loss": [],
                "train_move_acc": [],
                "val_move_acc": [],
                "learning_rate": [],
            }

            best_val_loss = float("inf")
            best_val_acc = 0
            patience_counter = 0

            print(f"\n🔥 Starting training...")
            training_start_time = time.time()

            for epoch in range(config.num_epochs):
                # Training phase
                model.train()
                train_losses = []
                train_move_correct = 0
                train_total = 0

                train_pbar = tqdm(
                    train_loader,
                    desc=f"Epoch {epoch+1:2d}/{config.num_epochs}",
                    leave=False,
                )

                for batch in train_pbar:
                    optimizer.zero_grad()

                    positions = batch["position"].to(device)
                    features = batch["features"].to(device)
                    magnus_moves = batch["magnus_move"].squeeze().to(device)
                    evaluations = batch["evaluation"].squeeze().to(device)

                    # Forward pass
                    move_logits, eval_adjustment = model(positions, features)

                    # Losses
                    move_loss = move_criterion(move_logits, magnus_moves)
                    eval_loss = eval_criterion(
                        eval_adjustment.squeeze(), evaluations / 1000.0
                    )
                    total_loss = move_loss + 0.1 * eval_loss

                    # Backward pass with gradient clipping
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    # Metrics
                    train_losses.append(total_loss.item())
                    _, predicted = torch.max(move_logits, 1)
                    train_move_correct += (predicted == magnus_moves).sum().item()
                    train_total += magnus_moves.size(0)

                    # Update progress bar
                    current_acc = (
                        train_move_correct / train_total if train_total > 0 else 0
                    )
                    train_pbar.set_postfix(
                        {
                            "loss": f"{total_loss.item():.4f}",
                            "acc": f"{current_acc:.3f}",
                        }
                    )

                # Validation phase
                model.eval()
                val_losses = []
                val_move_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch in val_loader:
                        positions = batch["position"].to(device)
                        features = batch["features"].to(device)
                        magnus_moves = batch["magnus_move"].squeeze().to(device)
                        evaluations = batch["evaluation"].squeeze().to(device)

                        move_logits, eval_adjustment = model(positions, features)

                        move_loss = move_criterion(move_logits, magnus_moves)
                        eval_loss = eval_criterion(
                            eval_adjustment.squeeze(), evaluations / 1000.0
                        )
                        total_loss = move_loss + 0.1 * eval_loss

                        val_losses.append(total_loss.item())

                        _, predicted = torch.max(move_logits, 1)
                        val_move_correct += (predicted == magnus_moves).sum().item()
                        val_total += magnus_moves.size(0)

                # Calculate metrics
                train_loss = np.mean(train_losses)
                val_loss = np.mean(val_losses)
                train_move_acc = train_move_correct / train_total
                val_move_acc = val_move_correct / val_total
                current_lr = optimizer.param_groups[0]["lr"]

                # Update history
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["train_move_acc"].append(train_move_acc)
                history["val_move_acc"].append(val_move_acc)
                history["learning_rate"].append(current_lr)

                # Log to MLflow
                if self.enable_mlflow:
                    mlflow.log_metrics(
                        {
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "train_accuracy": train_move_acc,
                            "val_accuracy": val_move_acc,
                            "learning_rate": current_lr,
                        },
                        step=epoch,
                    )

                # Calculate ETA
                elapsed_time = time.time() - training_start_time
                epochs_completed = epoch + 1
                avg_time_per_epoch = elapsed_time / epochs_completed
                eta = (config.num_epochs - epochs_completed) * avg_time_per_epoch

                print(
                    f"Epoch {epoch+1:2d}: "
                    f"Loss {val_loss:.4f} | "
                    f"Acc {val_move_acc:.3f} | "
                    f"LR {current_lr:.2e} | "
                    f"ETA {eta/60:.0f}m"
                )

                # Learning rate scheduling
                scheduler.step(val_loss)

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = val_move_acc
                    patience_counter = 0

                    # Save model checkpoint
                    model_save_dir = Path(f"models/magnus_mlflow_{run_name}")
                    model_save_dir.mkdir(parents=True, exist_ok=True)

                    checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_accuracy": val_move_acc,
                        "config": config.__dict__,
                        "dataset_info": {
                            "vocab_size": train_dataset.vocab_size,
                            "move_to_idx": train_dataset.move_to_idx,
                            "feature_names": train_dataset.feature_names,
                        },
                    }

                    model_path = model_save_dir / "best_model.pth"
                    torch.save(checkpoint, model_path)
                    print(f"         ✅ Best model saved!")

                else:
                    patience_counter += 1
                    if patience_counter >= config.early_stopping_patience:
                        print(f"         🛑 Early stopping at epoch {epoch+1}")
                        break

            training_time = time.time() - training_start_time

            # Final evaluation on test set
            print(f"\n🧪 Final evaluation...")
            model.eval()
            test_move_correct = 0
            test_total = 0

            with torch.no_grad():
                for batch in test_loader:
                    positions = batch["position"].to(device)
                    features = batch["features"].to(device)
                    magnus_moves = batch["magnus_move"].squeeze().to(device)

                    move_logits, _ = model(positions, features)
                    _, predicted = torch.max(move_logits, 1)
                    test_move_correct += (predicted == magnus_moves).sum().item()
                    test_total += magnus_moves.size(0)

            test_acc = test_move_correct / test_total

            # Log final results to MLflow
            if self.enable_mlflow:
                mlflow.log_metrics(
                    {
                        "final_test_accuracy": test_acc,
                        "best_val_accuracy": best_val_acc,
                        "training_time_minutes": training_time / 60,
                    }
                )

                # Log model to MLflow
                mlflow.pytorch.log_model(model, "model")

                # Save training curves
                self.plot_and_log_curves(history, model_save_dir)

            # Results summary
            print(f"\n🎉 Training Completed!")
            print(f"   Final Test Accuracy: {test_acc:.3f}")
            print(f"   Best Validation Accuracy: {best_val_acc:.3f}")
            print(f"   Training Time: {training_time/60:.1f} minutes")
            print(f"   Model Saved: {model_path}")

            if self.enable_mlflow:
                print(f"   MLflow Run: {run.info.run_id}")

            return model, test_acc, history

        finally:
            if self.enable_mlflow:
                mlflow.end_run()

    def plot_and_log_curves(self, history, save_dir):
        """Plot and log training curves"""

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Loss curves
        axes[0, 0].plot(history["train_loss"], label="Train")
        axes[0, 0].plot(history["val_loss"], label="Validation")
        axes[0, 0].set_title("Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy curves
        axes[0, 1].plot(history["train_move_acc"], label="Train")
        axes[0, 1].plot(history["val_move_acc"], label="Validation")
        axes[0, 1].set_title("Move Accuracy")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Learning rate
        axes[1, 0].plot(history["learning_rate"])
        axes[1, 0].set_title("Learning Rate")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Learning Rate")
        axes[1, 0].set_yscale("log")
        axes[1, 0].grid(True)

        # Final accuracy comparison
        final_train_acc = history["train_move_acc"][-1]
        final_val_acc = history["val_move_acc"][-1]
        axes[1, 1].bar(["Train", "Validation"], [final_train_acc, final_val_acc])
        axes[1, 1].set_title("Final Accuracy")
        axes[1, 1].set_ylabel("Accuracy")
        axes[1, 1].grid(True, axis="y")

        plt.tight_layout()

        # Save plot
        plot_path = save_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")

        # Log to MLflow
        if self.enable_mlflow:
            mlflow.log_artifact(str(plot_path))

        plt.close()


def quick_training_session():
    """Run a quick training session with good default parameters"""

    trainer = MLflowMagnusTrainer()

    # Train with conservative settings that should work well
    model, accuracy, history = trainer.train_model(
        learning_rate=0.0001,  # Conservative learning rate
        batch_size=512,  # Good for M3 Pro GPU
        num_epochs=20,  # Reasonable number of epochs
        run_name="quick_session",
    )

    return model, accuracy, history


def hyperparameter_search():
    """Run multiple experiments with different hyperparameters"""

    trainer = MLflowMagnusTrainer()

    # Define search space
    experiments = [
        {"lr": 0.001, "batch": 256, "name": "high_lr_small_batch"},
        {"lr": 0.0005, "batch": 512, "name": "med_lr_med_batch"},
        {"lr": 0.0001, "batch": 512, "name": "low_lr_med_batch"},
        {"lr": 0.00005, "batch": 1024, "name": "very_low_lr_large_batch"},
    ]

    results = []

    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"🔬 Experiment: {exp['name']}")
        print(f"{'='*60}")

        try:
            model, accuracy, history = trainer.train_model(
                learning_rate=exp["lr"],
                batch_size=exp["batch"],
                num_epochs=15,  # Shorter for search
                run_name=exp["name"],
            )

            results.append(
                {
                    "name": exp["name"],
                    "lr": exp["lr"],
                    "batch_size": exp["batch"],
                    "accuracy": accuracy,
                }
            )

            print(f"✅ {exp['name']}: {accuracy:.3f} accuracy")

        except Exception as e:
            print(f"❌ {exp['name']} failed: {e}")

    # Print summary
    print(f"\n🏆 HYPERPARAMETER SEARCH RESULTS:")
    print(f"{'='*60}")
    for result in sorted(results, key=lambda x: x["accuracy"], reverse=True):
        print(
            f"{result['name']:25} | "
            f"LR: {result['lr']:8.6f} | "
            f"Batch: {result['batch_size']:4d} | "
            f"Acc: {result['accuracy']:.3f}"
        )

    return results


if __name__ == "__main__":
    print("🎯 Magnus Carlsen MLOps Training")
    print("Choose an option:")
    print("1. Quick training session (recommended)")
    print("2. Hyperparameter search")
    print("3. Custom training")

    choice = input("Enter choice (1-3): ").strip()

    if choice == "1":
        print("\n🚀 Starting quick training session...")
        quick_training_session()

    elif choice == "2":
        print("\n🔬 Starting hyperparameter search...")
        hyperparameter_search()

    elif choice == "3":
        print("\n⚙️ Custom training...")
        lr = float(input("Learning rate (e.g., 0.0001): "))
        batch = int(input("Batch size (e.g., 512): "))
        epochs = int(input("Number of epochs (e.g., 20): "))
        name = input("Run name: ")

        trainer = MLflowMagnusTrainer()
        trainer.train_model(lr, batch, epochs, name)

    else:
        print("❌ Invalid choice")

    # Show MLflow UI instructions
    if MLFLOW_AVAILABLE:
        print(f"\n📊 View results in MLflow UI:")
        print(f"   cd {Path.cwd()}")
        print(f"   mlflow ui")
        print(f"   Open: http://localhost:5000")
