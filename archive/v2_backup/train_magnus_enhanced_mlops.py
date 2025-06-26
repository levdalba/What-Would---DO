#!/usr/bin/env python3
"""
Enhanced Magnus Carlsen Training with Comprehensive MLOps

This script provides enterprise-grade MLOps capabilities:
- MLflow experiment tracking with comprehensive metrics
- Hyperparameter optimization with Optuna
- Model versioning and artifact management
- Performance monitoring and alerting
- Automated model comparison and selection
- Multiple training runs for experimentation
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
import seaborn as sns

# MLOps imports
try:
    import mlflow
    import mlflow.pytorch
    from mlflow.tracking import MlflowClient
    import optuna
    from optuna.integration.mlflow import MLflowCallback

    MLOPS_AVAILABLE = True
    print("✅ MLOps stack available (MLflow + Optuna)")
except ImportError as e:
    MLOPS_AVAILABLE = False
    print(f"⚠️  MLOps dependencies missing: {e}")
    print("   Install with: pip install mlflow optuna")

from stockfish_magnus_trainer import (
    StockfishConfig,
    MagnusDataset,
    MagnusStyleModel,
    plot_training_curves,
)


class MagnusMLOpsTrainer:
    """
    Enterprise-grade Magnus chess training with comprehensive MLOps

    Features:
    - Experiment tracking with MLflow
    - Hyperparameter optimization with Optuna
    - Model versioning and comparison
    - Performance monitoring
    - Automated model selection
    """

    def __init__(self, experiment_name: str = "magnus_chess_mlops_enhanced"):
        self.experiment_name = experiment_name
        self.enable_mlops = MLOPS_AVAILABLE
        self.tracking_uri = "./mlruns"
        self.model_registry_uri = "./model_registry"

        # Performance thresholds
        self.min_accuracy_threshold = 0.85
        self.min_improvement_threshold = 0.01

        if self.enable_mlops:
            self.setup_mlflow()
            self.client = MlflowClient()

    def setup_mlflow(self):
        """Initialize MLflow with model registry"""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)

            # Create experiment if it doesn't exist
            try:
                mlflow.create_experiment(
                    self.experiment_name,
                    artifact_location=str(Path("./mlruns") / "artifacts"),
                )
            except:
                pass  # Experiment already exists

            mlflow.set_experiment(self.experiment_name)
            print(f"🔬 MLflow experiment: {self.experiment_name}")
            print(f"📊 Tracking URI: {self.tracking_uri}")

        except Exception as e:
            print(f"⚠️  MLflow setup failed: {e}")
            self.enable_mlops = False

    def load_extracted_positions(self) -> Dict[str, Any]:
        """Load pre-extracted positions with validation"""
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

        # Validate data structure
        required_keys = [
            "positions",
            "features",
            "stockfish_moves",
            "magnus_moves",
            "evaluations",
        ]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key in data: {key}")

        print(f"✅ Loaded {len(data['positions']):,} positions")
        print(
            f"   Features: {len(data['features'][0]) if data['features'] else 0} per position"
        )

        return data

    def prepare_datasets(self, data: Dict[str, Any], config: StockfishConfig) -> Tuple:
        """Prepare train/val/test datasets with comprehensive statistics"""

        positions = data["positions"]
        features = data["features"]
        sf_moves = data["stockfish_moves"]
        magnus_moves = data["magnus_moves"]
        evaluations = data["evaluations"]

        # Combine all data
        combined_data = list(
            zip(positions, features, sf_moves, magnus_moves, evaluations)
        )

        print(f"📊 Data Statistics:")
        print(f"   Total positions: {len(combined_data):,}")
        print(f"   Feature vector size: {len(features[0]) if features else 0}")
        print(f"   Evaluation range: [{min(evaluations):.2f}, {max(evaluations):.2f}]")

        # Split data
        test_size = config.test_split
        val_size = config.validation_split / (1 - test_size)

        data_train_val, data_test = train_test_split(
            combined_data, test_size=test_size, random_state=42, shuffle=True
        )
        data_train, data_val = train_test_split(
            data_train_val, test_size=val_size, random_state=42, shuffle=True
        )

        def unpack_data(split_data):
            return list(zip(*split_data))

        # Create datasets
        train_dataset = MagnusDataset(*unpack_data(data_train))
        val_dataset = MagnusDataset(*unpack_data(data_val))
        test_dataset = MagnusDataset(*unpack_data(data_test))

        # Share vocabulary across splits
        val_dataset.move_to_idx = train_dataset.move_to_idx
        val_dataset.idx_to_move = train_dataset.idx_to_move
        val_dataset.vocab_size = train_dataset.vocab_size

        test_dataset.move_to_idx = train_dataset.move_to_idx
        test_dataset.idx_to_move = train_dataset.idx_to_move
        test_dataset.vocab_size = train_dataset.vocab_size

        print(f"📚 Dataset splits:")
        print(
            f"   Train: {len(train_dataset):,} ({len(train_dataset)/len(combined_data)*100:.1f}%)"
        )
        print(
            f"   Validation: {len(val_dataset):,} ({len(val_dataset)/len(combined_data)*100:.1f}%)"
        )
        print(
            f"   Test: {len(test_dataset):,} ({len(test_dataset)/len(combined_data)*100:.1f}%)"
        )
        print(f"   Move vocabulary: {train_dataset.vocab_size:,} unique moves")

        return train_dataset, val_dataset, test_dataset

    def evaluate_model_comprehensive(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: torch.device,
        criterion_move: nn.Module,
        criterion_eval: nn.Module,
    ) -> Dict[str, float]:
        """Comprehensive model evaluation with multiple metrics"""

        model.eval()
        total_loss = 0
        move_predictions = []
        move_targets = []
        eval_predictions = []
        eval_targets = []

        with torch.no_grad():
            for batch in data_loader:
                features, magnus_moves, evaluations = [x.to(device) for x in batch]

                move_logits, eval_preds = model(features)

                # Calculate losses
                move_loss = criterion_move(move_logits, magnus_moves)
                eval_loss = criterion_eval(eval_preds.squeeze(), evaluations)
                total_loss += (move_loss + eval_loss).item()

                # Store predictions for detailed analysis
                move_predictions.extend(torch.argmax(move_logits, dim=1).cpu().numpy())
                move_targets.extend(magnus_moves.cpu().numpy())
                eval_predictions.extend(eval_preds.squeeze().cpu().numpy())
                eval_targets.extend(evaluations.cpu().numpy())

        # Calculate comprehensive metrics
        move_accuracy = accuracy_score(move_targets, move_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            move_targets, move_predictions, average="weighted", zero_division=0
        )

        # Evaluation metrics
        eval_mse = np.mean((np.array(eval_predictions) - np.array(eval_targets)) ** 2)
        eval_mae = np.mean(np.abs(np.array(eval_predictions) - np.array(eval_targets)))
        eval_corr = (
            np.corrcoef(eval_predictions, eval_targets)[0, 1]
            if len(eval_predictions) > 1
            else 0
        )

        return {
            "total_loss": total_loss / len(data_loader),
            "move_accuracy": move_accuracy,
            "move_precision": precision,
            "move_recall": recall,
            "move_f1": f1,
            "eval_mse": eval_mse,
            "eval_mae": eval_mae,
            "eval_correlation": eval_corr,
        }

    def train_single_model(
        self, hyperparams: Dict[str, Any], run_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train a single model with comprehensive tracking

        Args:
            hyperparams: Dictionary of hyperparameters
            run_name: Custom name for this training run

        Returns:
            Dictionary with training results and metrics
        """

        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"magnus_m3_pro_{timestamp}_{str(uuid.uuid4())[:8]}"

        print(f"\n🚀 Training Magnus Model: {run_name}")
        print(f"   Hyperparameters: {hyperparams}")

        # Load data
        data = self.load_extracted_positions()

        # Setup configuration
        config = StockfishConfig()
        for key, value in hyperparams.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Device setup
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        config.device = str(device)
        print(f"🎮 Device: {device}")

        # Prepare datasets
        train_dataset, val_dataset, test_dataset = self.prepare_datasets(data, config)

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

        # Create model
        model = MagnusStyleModel(
            config, train_dataset.vocab_size, len(train_dataset.feature_names)
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"🧠 Model Architecture:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Memory footprint: ~{total_params * 4 / (1024*1024):.1f} MB")

        # Start MLflow run
        results = {}
        if self.enable_mlops:
            with mlflow.start_run(run_name=run_name) as run:
                results["run_id"] = run.info.run_id
                print(f"🔬 MLflow run ID: {run.info.run_id}")

                # Log comprehensive parameters
                mlflow.log_params(
                    {
                        **hyperparams,
                        "device": str(device),
                        "total_params": total_params,
                        "trainable_params": trainable_params,
                        "train_size": len(train_dataset),
                        "val_size": len(val_dataset),
                        "test_size": len(test_dataset),
                        "vocab_size": train_dataset.vocab_size,
                        "feature_count": len(train_dataset.feature_names),
                        "hardware": "M3 Pro",
                        "pytorch_version": torch.__version__,
                    }
                )

                results.update(
                    self._train_model_core(
                        model, train_loader, val_loader, test_loader, config, device
                    )
                )

                # Log final metrics
                mlflow.log_metrics(
                    {
                        "final_train_accuracy": results["final_train_accuracy"],
                        "final_val_accuracy": results["final_val_accuracy"],
                        "final_test_accuracy": results["final_test_accuracy"],
                        "training_time_minutes": results["training_time"] / 60,
                        "best_epoch": results["best_epoch"],
                    }
                )

                # Save model artifacts
                model_path = f"models/magnus_mlflow_{run.info.run_id}"
                Path(model_path).mkdir(parents=True, exist_ok=True)

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config.__dict__,
                        "hyperparams": hyperparams,
                        "vocab_size": train_dataset.vocab_size,
                        "feature_names": train_dataset.feature_names,
                        "results": results,
                    },
                    f"{model_path}/model.pth",
                )

                # Log model to MLflow
                mlflow.pytorch.log_model(
                    model,
                    "model",
                    registered_model_name="MagnusChessM3Pro",
                    conda_env={
                        "channels": ["pytorch", "conda-forge"],
                        "dependencies": [
                            "python=3.12",
                            "pytorch>=2.0",
                            "scikit-learn>=1.3",
                            {"pip": ["chess>=1.11", "python-chess>=1.999"]},
                        ],
                    },
                )

                print(f"💾 Model saved to: {model_path}")

        else:
            # Train without MLflow
            results.update(
                self._train_model_core(
                    model, train_loader, val_loader, test_loader, config, device
                )
            )

        return results

    def _train_model_core(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: StockfishConfig,
        device: torch.device,
    ) -> Dict[str, Any]:
        """Core training logic with comprehensive metrics tracking"""

        # Training setup
        move_criterion = nn.CrossEntropyLoss()
        eval_criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, min_lr=1e-7
        )

        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "test_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "test_accuracy": [],
            "learning_rate": [],
            "epoch_time": [],
        }

        best_val_accuracy = 0
        best_epoch = 0
        patience_counter = 0
        training_start_time = time.time()

        print(f"\n🔥 Starting training for {config.num_epochs} epochs...")

        for epoch in range(config.num_epochs):
            epoch_start_time = time.time()

            # Training phase
            model.train()
            train_losses = []
            train_correct = 0
            train_total = 0

            train_pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1:2d}/{config.num_epochs}",
                leave=False,
            )

            for batch in train_pbar:
                features, magnus_moves, evaluations = [x.to(device) for x in batch]

                optimizer.zero_grad()
                move_logits, eval_preds = model(features)

                move_loss = move_criterion(move_logits, magnus_moves)
                eval_loss = eval_criterion(eval_preds.squeeze(), evaluations)
                total_loss = move_loss + eval_loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

            # Validation evaluation
            val_metrics = self.evaluate_model_comprehensive(
                model, val_loader, device, move_criterion, eval_criterion
            )

            # Test evaluation (for monitoring, not selection)
            test_metrics = self.evaluate_model_comprehensive(
                model, test_loader, device, move_criterion, eval_criterion
            )

            # Update learning rate
            scheduler.step(val_metrics["total_loss"])
            current_lr = optimizer.param_groups[0]["lr"]

            # Record history
            epoch_time = time.time() - epoch_start_time
            train_loss = np.mean(train_losses)
            train_accuracy = train_correct / train_total

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["total_loss"])
            history["test_loss"].append(test_metrics["total_loss"])
            history["train_accuracy"].append(train_accuracy)
            history["val_accuracy"].append(val_metrics["move_accuracy"])
            history["test_accuracy"].append(test_metrics["move_accuracy"])
            history["learning_rate"].append(current_lr)
            history["epoch_time"].append(epoch_time)

            # Check for best model
            if val_metrics["move_accuracy"] > best_val_accuracy:
                best_val_accuracy = val_metrics["move_accuracy"]
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            # Log metrics to MLflow
            if self.enable_mlops:
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_metrics["total_loss"],
                        "test_loss": test_metrics["total_loss"],
                        "train_accuracy": train_accuracy,
                        "val_accuracy": val_metrics["move_accuracy"],
                        "test_accuracy": test_metrics["move_accuracy"],
                        "val_precision": val_metrics["move_precision"],
                        "val_recall": val_metrics["move_recall"],
                        "val_f1": val_metrics["move_f1"],
                        "val_eval_mse": val_metrics["eval_mse"],
                        "val_eval_correlation": val_metrics["eval_correlation"],
                        "learning_rate": current_lr,
                        "epoch_time": epoch_time,
                    },
                    step=epoch,
                )

            # Print progress
            print(
                f"Epoch {epoch+1:2d}/{config.num_epochs} | "
                f"Train: {train_accuracy:.3f} | "
                f"Val: {val_metrics['move_accuracy']:.3f} | "
                f"Test: {test_metrics['move_accuracy']:.3f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Early stopping
            if patience_counter >= config.early_stopping_patience:
                print(f"⏹️  Early stopping triggered after {epoch+1} epochs")
                break

        total_training_time = time.time() - training_start_time

        # Final evaluation
        final_test_metrics = self.evaluate_model_comprehensive(
            model, test_loader, device, move_criterion, eval_criterion
        )

        print(f"\n✅ Training completed!")
        print(f"   Total time: {total_training_time/60:.1f} minutes")
        print(
            f"   Best validation accuracy: {best_val_accuracy:.4f} (epoch {best_epoch+1})"
        )
        print(f"   Final test accuracy: {final_test_metrics['move_accuracy']:.4f}")
        print(f"   Final test F1: {final_test_metrics['move_f1']:.4f}")

        return {
            "history": history,
            "final_train_accuracy": history["train_accuracy"][-1],
            "final_val_accuracy": history["val_accuracy"][-1],
            "final_test_accuracy": final_test_metrics["move_accuracy"],
            "best_val_accuracy": best_val_accuracy,
            "best_epoch": best_epoch,
            "training_time": total_training_time,
            "final_test_metrics": final_test_metrics,
        }

    def run_hyperparameter_study(self, n_trials: int = 10) -> Optional[Dict[str, Any]]:
        """
        Run hyperparameter optimization study with Optuna

        Args:
            n_trials: Number of trials for optimization

        Returns:
            Best hyperparameters and study results
        """

        if not self.enable_mlops:
            print("⚠️  MLOps not available - skipping hyperparameter study")
            return None

        print(f"\n🔍 Starting hyperparameter optimization with {n_trials} trials...")

        def objective(trial):
            """Optuna objective function"""

            # Define hyperparameter search space
            hyperparams = {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-5, 1e-2, log=True
                ),
                "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
                "num_epochs": trial.suggest_int("num_epochs", 20, 50),
                "early_stopping_patience": trial.suggest_int(
                    "early_stopping_patience", 5, 15
                ),
            }

            try:
                # Train model with these hyperparameters
                results = self.train_single_model(
                    hyperparams, run_name=f"optuna_trial_{trial.number}"
                )

                # Return metric to optimize (validation accuracy)
                return results["best_val_accuracy"]

            except Exception as e:
                print(f"❌ Trial {trial.number} failed: {e}")
                return 0.0  # Return poor score for failed trials

        # Create and run study
        study = optuna.create_study(
            direction="maximize",
            study_name=f"magnus_hyperopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

        # Add MLflow callback for Optuna
        mlflc = MLflowCallback(
            tracking_uri=self.tracking_uri, metric_name="validation_accuracy"
        )

        study.optimize(objective, n_trials=n_trials, callbacks=[mlflc])

        print(f"\n🏆 Hyperparameter optimization completed!")
        print(f"   Best validation accuracy: {study.best_value:.4f}")
        print(f"   Best hyperparameters: {study.best_params}")

        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "study": study,
        }

    def run_multiple_experiments(
        self, experiments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Run multiple training experiments with different configurations

        Args:
            experiments: List of experiment configurations

        Returns:
            List of results for each experiment
        """

        print(f"\n🧪 Running {len(experiments)} experiments...")

        all_results = []

        for i, exp_config in enumerate(experiments):
            print(f"\n{'='*60}")
            print(
                f"Experiment {i+1}/{len(experiments)}: {exp_config.get('name', f'Experiment_{i+1}')}"
            )
            print(f"{'='*60}")

            hyperparams = exp_config.get("hyperparams", {})
            run_name = exp_config.get(
                "name", f"experiment_{i+1}_{datetime.now().strftime('%H%M%S')}"
            )

            try:
                results = self.train_single_model(hyperparams, run_name)
                results["experiment_config"] = exp_config
                results["experiment_index"] = i
                all_results.append(results)

                print(f"✅ Experiment {i+1} completed successfully!")
                print(f"   Final test accuracy: {results['final_test_accuracy']:.4f}")

            except Exception as e:
                print(f"❌ Experiment {i+1} failed: {e}")
                all_results.append(
                    {
                        "experiment_config": exp_config,
                        "experiment_index": i,
                        "error": str(e),
                        "final_test_accuracy": 0.0,
                    }
                )

        # Summary
        print(f"\n📊 Experiment Summary:")
        print(f"{'='*60}")

        successful_results = [r for r in all_results if "error" not in r]
        if successful_results:
            best_result = max(
                successful_results, key=lambda x: x["final_test_accuracy"]
            )

            print(f"🏆 Best performing experiment:")
            print(f"   Name: {best_result['experiment_config'].get('name', 'Unknown')}")
            print(f"   Test accuracy: {best_result['final_test_accuracy']:.4f}")
            print(f"   Training time: {best_result['training_time']/60:.1f} minutes")

            if self.enable_mlops:
                print(f"   MLflow run ID: {best_result.get('run_id', 'N/A')}")

        return all_results


def main():
    """Main execution with multiple training scenarios"""

    print("🎯 Magnus Carlsen Enhanced MLOps Training")
    print("=" * 60)

    # Initialize trainer
    trainer = MagnusMLOpsTrainer("magnus_chess_m3_pro_enhanced")

    # Define different experiment configurations
    experiments = [
        {
            "name": "baseline_fast",
            "description": "Quick baseline training",
            "hyperparams": {
                "learning_rate": 0.001,
                "batch_size": 512,
                "num_epochs": 15,
                "early_stopping_patience": 5,
            },
        },
        {
            "name": "high_lr_experiment",
            "description": "Higher learning rate experiment",
            "hyperparams": {
                "learning_rate": 0.003,
                "batch_size": 512,
                "num_epochs": 20,
                "early_stopping_patience": 8,
            },
        },
        {
            "name": "large_batch_experiment",
            "description": "Large batch size experiment",
            "hyperparams": {
                "learning_rate": 0.0005,
                "batch_size": 1024,
                "num_epochs": 25,
                "early_stopping_patience": 8,
            },
        },
        {
            "name": "conservative_training",
            "description": "Conservative training with patience",
            "hyperparams": {
                "learning_rate": 0.0001,
                "batch_size": 256,
                "num_epochs": 40,
                "early_stopping_patience": 12,
            },
        },
    ]

    # Run multiple experiments
    results = trainer.run_multiple_experiments(experiments)

    # Optionally run hyperparameter optimization
    print(f"\n🤔 Would you like to run hyperparameter optimization? (5 trials)")
    print("   This will take additional time but may find better hyperparameters...")

    # For demo purposes, we'll skip the interactive part and just run a small study
    print("🔍 Running small hyperparameter study...")
    hp_results = trainer.run_hyperparameter_study(n_trials=3)

    if hp_results:
        print(f"\n🎯 Hyperparameter optimization found:")
        print(f"   Best accuracy: {hp_results['best_value']:.4f}")
        print(f"   Best params: {hp_results['best_params']}")

    # Final summary
    print(f"\n🎉 All experiments completed!")
    print(f"📈 View results in MLflow UI:")
    print(f"   cd {Path.cwd()}")
    print(f"   mlflow ui --host 127.0.0.1 --port 5000")
    print(f"   Open: http://127.0.0.1:5000")

    print(f"\n📊 Quick Stats:")
    successful_runs = [r for r in results if "error" not in r]
    if successful_runs:
        accuracies = [r["final_test_accuracy"] for r in successful_runs]
        print(f"   Experiments run: {len(results)}")
        print(f"   Successful runs: {len(successful_runs)}")
        print(f"   Accuracy range: {min(accuracies):.4f} - {max(accuracies):.4f}")
        print(f"   Average accuracy: {np.mean(accuracies):.4f}")


if __name__ == "__main__":
    main()
