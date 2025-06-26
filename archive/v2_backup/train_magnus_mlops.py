#!/usr/bin/env python3
"""
Magnus Carlsen Neural Network Training with MLOps Integration

Features:
- MLflow experiment tracking
- Model versioning and registry
- Real-time metrics monitoring
- Data drift detection with Evidently
- Hyperparameter optimization
- Model comparison and evaluation
"""

import sys
import time
import json
import pickle
import uuid
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import numpy as np
import pandas as pd
from datetime import datetime

# Add the project directory to path
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# MLOps imports
try:
    import mlflow
    import mlflow.pytorch
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    print("⚠️  MLflow not installed. Install with: pip install mlflow")
    MLFLOW_AVAILABLE = False

try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, RegressionPreset

    EVIDENTLY_AVAILABLE = True
except ImportError:
    print("⚠️  Evidently not installed. Install with: pip install evidently")
    EVIDENTLY_AVAILABLE = False

from stockfish_magnus_trainer import (
    StockfishConfig,
    MagnusDataset,
    MagnusStyleModel,
    plot_training_curves,
)


class MLOpsConfig:
    """Configuration for MLOps features"""

    def __init__(self):
        self.experiment_name = "magnus_carlsen_chess_engine"
        self.run_name = f"m3_pro_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.model_name = "magnus_carlsen_model"
        self.tracking_uri = "./mlruns"  # Local MLflow tracking
        self.enable_mlflow = MLFLOW_AVAILABLE
        self.enable_evidently = EVIDENTLY_AVAILABLE
        self.log_every_n_batches = 50
        self.save_checkpoints = True
        self.auto_log = True


class MagnusMLOpsTrainer:
    """Magnus Carlsen training with MLOps integration"""

    def __init__(self, config: Optional[MLOpsConfig] = None):
        self.mlops_config = config or MLOpsConfig()
        self.client = None
        self.run_id = None

        # Setup MLflow
        if self.mlops_config.enable_mlflow:
            self.setup_mlflow()

    def setup_mlflow(self):
        """Initialize MLflow tracking"""
        try:
            mlflow.set_tracking_uri(self.mlops_config.tracking_uri)

            # Create or get experiment
            try:
                experiment_id = mlflow.create_experiment(
                    self.mlops_config.experiment_name
                )
            except:
                experiment = mlflow.get_experiment_by_name(
                    self.mlops_config.experiment_name
                )
                experiment_id = experiment.experiment_id

            mlflow.set_experiment(self.mlops_config.experiment_name)
            self.client = MlflowClient()

            print(f"🔬 MLflow tracking initialized")
            print(f"   Experiment: {self.mlops_config.experiment_name}")
            print(f"   Tracking URI: {self.mlops_config.tracking_uri}")

        except Exception as e:
            print(f"⚠️  MLflow setup failed: {e}")
            self.mlops_config.enable_mlflow = False

    def load_extracted_positions(self):
        """Load pre-extracted positions from pickle file"""
        data_path = Path("magnus_extracted_positions_m3_pro.pkl")

        if not data_path.exists():
            raise FileNotFoundError(
                f"❌ Pre-extracted positions not found: {data_path}\n"
                f"   Please run extract_positions_m3_pro.py first!"
            )

        print(f"📂 Loading pre-extracted positions from {data_path}...")
        print(f"   File size: {data_path.stat().st_size / (1024*1024):.1f} MB")

        with open(data_path, "rb") as f:
            data = pickle.load(f)

        print(f"✅ Loaded {len(data['positions']):,} positions")
        print(f"   Extraction time: {data['extraction_time']/3600:.1f} hours")
        print(f"   Source: {data['metadata']['pgn_file']}")

        return data

    def log_experiment_params(self, config, device, dataset_info):
        """Log experiment parameters to MLflow"""
        if not self.mlops_config.enable_mlflow:
            return

        params = {
            "device": str(device),
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
            "validation_split": config.validation_split,
            "test_split": config.test_split,
            "early_stopping_patience": config.early_stopping_patience,
            "hardware": "M3 Pro",
            "total_positions": dataset_info["total_positions"],
            "train_size": dataset_info["train_size"],
            "val_size": dataset_info["val_size"],
            "test_size": dataset_info["test_size"],
            "vocab_size": dataset_info["vocab_size"],
            "model_parameters": dataset_info["model_parameters"],
        }

        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, epoch, metrics):
        """Log training metrics to MLflow"""
        if not self.mlops_config.enable_mlflow:
            return

        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=epoch)

    def create_data_drift_report(self, train_data, val_data):
        """Create data drift report using Evidently"""
        if not self.mlops_config.enable_evidently:
            return None

        try:
            # Convert to DataFrame for Evidently
            train_df = pd.DataFrame(train_data)
            val_df = pd.DataFrame(val_data)

            # Create drift report
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=train_df, current_data=val_df)

            # Save report
            report_path = Path(f"reports/data_drift_{self.mlops_config.run_name}.html")
            report_path.parent.mkdir(exist_ok=True)
            report.save_html(str(report_path))

            print(f"📊 Data drift report saved: {report_path}")
            return report_path

        except Exception as e:
            print(f"⚠️  Data drift report failed: {e}")
            return None

    def train_with_mlops(
        self,
        learning_rates=[0.001, 0.0005, 0.0001],
        batch_sizes=[256, 512, 1024],
        architectures=["default"],
        max_experiments=3,
    ):
        """
        Train Magnus model with hyperparameter optimization and MLOps tracking
        """

        print("🚀 M3 PRO MAGNUS TRAINING WITH MLOPS")
        print("=" * 60)

        # Load pre-extracted data
        extracted_data = self.load_extracted_positions()

        positions = extracted_data["positions"]
        features = extracted_data["features"]
        sf_moves = extracted_data["stockfish_moves"]
        magnus_moves = extracted_data["magnus_moves"]
        evaluations = extracted_data["evaluations"]

        # Prepare data splits
        data = list(zip(positions, features, sf_moves, magnus_moves, evaluations))

        best_model = None
        best_score = 0
        experiment_results = []

        # Hyperparameter search
        experiment_count = 0
        for lr in learning_rates:
            for batch_size in batch_sizes:
                if experiment_count >= max_experiments:
                    break

                experiment_count += 1

                print(f"\n🔬 Experiment {experiment_count}/{max_experiments}")
                print(f"   Learning Rate: {lr}")
                print(f"   Batch Size: {batch_size}")

                if self.mlops_config.enable_mlflow:
                    with mlflow.start_run(
                        run_name=f"{self.mlops_config.run_name}_exp{experiment_count}"
                    ):
                        model, score, history = self.run_single_experiment(
                            data, extracted_data, lr, batch_size
                        )

                        # Log final results
                        mlflow.log_metric("final_accuracy", score)

                else:
                    model, score, history = self.run_single_experiment(
                        data, extracted_data, lr, batch_size
                    )

                experiment_results.append(
                    {
                        "experiment": experiment_count,
                        "learning_rate": lr,
                        "batch_size": batch_size,
                        "final_accuracy": score,
                        "model": model,
                        "history": history,
                    }
                )

                if score > best_score:
                    best_score = score
                    best_model = model
                    print(f"🏆 New best model! Accuracy: {score:.3f}")

                if experiment_count >= max_experiments:
                    break

        # Save best model with MLflow
        if self.mlops_config.enable_mlflow and best_model:
            self.register_best_model(best_model, best_score)

        # Create comparison report
        self.create_experiment_comparison(experiment_results)

        return best_model, experiment_results

    def run_single_experiment(self, data, extracted_data, learning_rate, batch_size):
        """Run a single training experiment"""

        # Training configuration
        config = StockfishConfig()
        config.learning_rate = learning_rate
        config.batch_size = batch_size
        config.num_epochs = 30  # Fewer epochs for faster iteration
        config.validation_split = 0.2
        config.test_split = 0.1
        config.early_stopping_patience = 5

        # Device setup
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        config.device = str(device)

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
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == "mps"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "mps"),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "mps"),
        )

        # Create model
        model = MagnusStyleModel(
            config, train_dataset.vocab_size, len(train_dataset.feature_names)
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())

        # Log experiment parameters
        dataset_info = {
            "total_positions": len(data),
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
            "vocab_size": train_dataset.vocab_size,
            "model_parameters": total_params,
        }

        self.log_experiment_params(config, device, dataset_info)

        # Training setup
        move_criterion = nn.CrossEntropyLoss()
        eval_criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5
        )

        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_move_acc": [],
            "val_move_acc": [],
            "train_eval_loss": [],
            "val_eval_loss": [],
        }

        best_val_loss = float("inf")
        patience_counter = 0

        # Training loop
        training_start_time = time.time()

        for epoch in range(config.num_epochs):
            # Training phase
            model.train()
            train_losses = []
            train_move_correct = 0
            train_eval_losses = []
            train_total = 0

            for batch_idx, batch in enumerate(train_loader):
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

                # Backward
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )  # Gradient clipping
                optimizer.step()

                # Metrics
                train_losses.append(total_loss.item())
                train_eval_losses.append(eval_loss.item())

                _, predicted = torch.max(move_logits, 1)
                train_move_correct += (predicted == magnus_moves).sum().item()
                train_total += magnus_moves.size(0)

                # Log every N batches
                if batch_idx % self.mlops_config.log_every_n_batches == 0:
                    batch_acc = (
                        train_move_correct / train_total if train_total > 0 else 0
                    )
                    if self.mlops_config.enable_mlflow:
                        mlflow.log_metric(
                            "batch_loss",
                            total_loss.item(),
                            step=epoch * len(train_loader) + batch_idx,
                        )
                        mlflow.log_metric(
                            "batch_accuracy",
                            batch_acc,
                            step=epoch * len(train_loader) + batch_idx,
                        )

            # Validation phase
            model.eval()
            val_losses = []
            val_move_correct = 0
            val_eval_losses = []
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
                    val_eval_losses.append(eval_loss.item())

                    _, predicted = torch.max(move_logits, 1)
                    val_move_correct += (predicted == magnus_moves).sum().item()
                    val_total += magnus_moves.size(0)

            # Calculate metrics
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            train_move_acc = train_move_correct / train_total
            val_move_acc = val_move_correct / val_total
            train_eval_loss = np.mean(train_eval_losses)
            val_eval_loss = np.mean(val_eval_losses)

            # Update history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_move_acc"].append(train_move_acc)
            history["val_move_acc"].append(val_move_acc)
            history["train_eval_loss"].append(train_eval_loss)
            history["val_eval_loss"].append(val_eval_loss)

            # Log metrics
            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_move_acc,
                "val_accuracy": val_move_acc,
                "train_eval_loss": train_eval_loss,
                "val_eval_loss": val_eval_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            self.log_metrics(epoch, metrics)

            # Log epoch results
            print(
                f"  Epoch {epoch+1:2d}: "
                f"Loss {val_loss:.4f} | "
                f"Acc {val_move_acc:.3f} | "
                f"LR {optimizer.param_groups[0]['lr']:.2e}"
            )

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    print(f"    🛑 Early stopping at epoch {epoch+1}")
                    break

        # Final evaluation
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

        return model, test_acc, history

    def register_best_model(self, model, score):
        """Register the best model in MLflow Model Registry"""
        if not self.mlops_config.enable_mlflow:
            return

        try:
            # Log the model
            mlflow.pytorch.log_model(
                model, "model", registered_model_name=self.mlops_config.model_name
            )

            # Add model version tags
            latest_version = self.client.get_latest_versions(
                self.mlops_config.model_name, stages=["None"]
            )[0]

            self.client.set_model_version_tag(
                self.mlops_config.model_name,
                latest_version.version,
                "accuracy",
                f"{score:.4f}",
            )

            self.client.set_model_version_tag(
                self.mlops_config.model_name,
                latest_version.version,
                "hardware",
                "M3 Pro",
            )

            print(
                f"🏆 Best model registered: {self.mlops_config.model_name} v{latest_version.version}"
            )

        except Exception as e:
            print(f"⚠️  Model registration failed: {e}")

    def create_experiment_comparison(self, results):
        """Create comparison report of all experiments"""

        # Create comparison DataFrame
        comparison_data = []
        for result in results:
            comparison_data.append(
                {
                    "Experiment": result["experiment"],
                    "Learning Rate": result["learning_rate"],
                    "Batch Size": result["batch_size"],
                    "Final Accuracy": result["final_accuracy"],
                }
            )

        df = pd.DataFrame(comparison_data)

        # Save comparison
        comparison_path = Path(
            f"reports/experiment_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        comparison_path.parent.mkdir(exist_ok=True)
        df.to_csv(comparison_path, index=False)

        # Create visualization
        plt.figure(figsize=(12, 8))

        # Accuracy comparison
        plt.subplot(2, 2, 1)
        plt.bar(df["Experiment"], df["Final Accuracy"])
        plt.title("Final Accuracy by Experiment")
        plt.xlabel("Experiment")
        plt.ylabel("Accuracy")

        # Learning rate vs accuracy
        plt.subplot(2, 2, 2)
        plt.scatter(df["Learning Rate"], df["Final Accuracy"], s=100, alpha=0.7)
        plt.xscale("log")
        plt.title("Learning Rate vs Accuracy")
        plt.xlabel("Learning Rate")
        plt.ylabel("Accuracy")

        # Batch size vs accuracy
        plt.subplot(2, 2, 3)
        plt.scatter(df["Batch Size"], df["Final Accuracy"], s=100, alpha=0.7)
        plt.title("Batch Size vs Accuracy")
        plt.xlabel("Batch Size")
        plt.ylabel("Accuracy")

        plt.tight_layout()

        plot_path = comparison_path.with_suffix(".png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"📊 Experiment comparison saved:")
        print(f"   Data: {comparison_path}")
        print(f"   Plot: {plot_path}")

        # Log to MLflow
        if self.mlops_config.enable_mlflow:
            try:
                mlflow.log_artifact(str(comparison_path))
                mlflow.log_artifact(str(plot_path))
            except:
                pass

        return df


def main():
    """Main training function with MLOps"""

    # MLOps configuration
    mlops_config = MLOpsConfig()

    # Create trainer
    trainer = MagnusMLOpsTrainer(mlops_config)

    print(f"🎯 Magnus Carlsen Training with MLOps")
    print(f"   MLflow: {'✅' if mlops_config.enable_mlflow else '❌'}")
    print(f"   Evidently: {'✅' if mlops_config.enable_evidently else '❌'}")
    print()

    # Run training with hyperparameter optimization
    best_model, results = trainer.train_with_mlops(
        learning_rates=[0.001, 0.0005, 0.0001],
        batch_sizes=[256, 512] if torch.backends.mps.is_available() else [128, 256],
        max_experiments=6,
    )

    print(f"\n🎉 MLOps Training Completed!")
    print(f"   Best Model Accuracy: {max(r['final_accuracy'] for r in results):.3f}")
    print(f"   Total Experiments: {len(results)}")

    if mlops_config.enable_mlflow:
        print(
            f"   MLflow UI: mlflow ui --backend-store-uri {mlops_config.tracking_uri}"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
