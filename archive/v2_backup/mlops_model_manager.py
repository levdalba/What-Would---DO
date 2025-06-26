#!/usr/bin/env python3
"""
MLOps Model Manager for Magnus Chess AI
Comprehensive model versioning, storage, and lifecycle management
"""

import os
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import pickle
import torch
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import pandas as pd


class MagnusModelManager:
    """Comprehensive model management with MLflow and local backup"""

    def __init__(self, base_dir: str = "./models_vault"):
        self.base_dir = Path(base_dir)
        self.client = MlflowClient()
        self.setup_directories()

    def setup_directories(self):
        """Create directory structure for model storage"""
        directories = [
            self.base_dir,
            self.base_dir / "checkpoints",
            self.base_dir / "artifacts",
            self.base_dir / "metadata",
            self.base_dir / "exports",
            self.base_dir / "backups",
        ]

        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)

        print(f"📁 Model vault initialized at: {self.base_dir}")

    def save_model_comprehensive(
        self,
        model: torch.nn.Module,
        model_name: str,
        experiment_name: str,
        metrics: Dict[str, float],
        hyperparameters: Dict[str, Any],
        training_data_info: Dict[str, Any],
        model_architecture: str,
        notes: str = "",
    ) -> str:
        """Save model with comprehensive metadata and versioning"""

        # Generate unique model ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_hash = self._calculate_model_hash(model)
        model_id = f"{model_name}_{timestamp}_{model_hash[:8]}"

        # Create model directory
        model_dir = self.base_dir / "checkpoints" / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save PyTorch model
        model_path = model_dir / "model.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_architecture": model_architecture,
                "timestamp": timestamp,
                "model_id": model_id,
            },
            model_path,
        )

        # Save comprehensive metadata
        metadata = {
            "model_id": model_id,
            "model_name": model_name,
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "model_hash": model_hash,
            "architecture": model_architecture,
            "metrics": metrics,
            "hyperparameters": hyperparameters,
            "training_data": training_data_info,
            "notes": notes,
            "model_size_mb": os.path.getsize(model_path) / (1024 * 1024),
            "pytorch_version": torch.__version__,
            "parameters_count": sum(p.numel() for p in model.parameters()),
        }

        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        # Log to MLflow with enhanced information
        with mlflow.start_run(experiment_id=self._get_experiment_id(experiment_name)):
            # Log model to MLflow
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name=f"magnus_{model_name}",
                metadata={"model_id": model_id, "vault_path": str(model_dir)},
            )

            # Log all metrics
            mlflow.log_metrics(metrics)
            mlflow.log_params(hyperparameters)

            # Log additional metadata
            mlflow.log_params(
                {
                    "model_id": model_id,
                    "architecture": model_architecture,
                    "parameters_count": metadata["parameters_count"],
                    "model_size_mb": metadata["model_size_mb"],
                }
            )

            # Log artifacts
            mlflow.log_artifact(str(metadata_path), "metadata")
            mlflow.log_artifact(str(model_path), "checkpoint")

            # Set tags
            mlflow.set_tags(
                {
                    "model_type": "magnus_chess_ai",
                    "model_id": model_id,
                    "architecture": model_architecture,
                    "auto_saved": "true",
                }
            )

            run_id = mlflow.active_run().info.run_id

        # Create model registry entry
        self._update_model_registry(metadata)

        # Create backup
        self._create_backup(model_dir, model_id)

        print(f"✅ Model saved comprehensively:")
        print(f"   Model ID: {model_id}")
        print(f"   Local path: {model_dir}")
        print(f"   MLflow run: {run_id}")
        print(f"   Metrics: {metrics}")

        return model_id

    def _calculate_model_hash(self, model: torch.nn.Module) -> str:
        """Calculate hash of model parameters for versioning"""
        model_str = str(model.state_dict())
        return hashlib.md5(model_str.encode()).hexdigest()

    def _get_experiment_id(self, experiment_name: str) -> str:
        """Get or create MLflow experiment"""
        try:
            experiment = self.client.get_experiment_by_name(experiment_name)
            return experiment.experiment_id
        except:
            return mlflow.create_experiment(experiment_name)

    def _update_model_registry(self, metadata: Dict[str, Any]):
        """Update local model registry"""
        registry_path = self.base_dir / "metadata" / "model_registry.json"

        if registry_path.exists():
            with open(registry_path, "r") as f:
                registry = json.load(f)
        else:
            registry = {"models": []}

        registry["models"].append(metadata)
        registry["last_updated"] = datetime.now().isoformat()

        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2, default=str)

    def _create_backup(self, model_dir: Path, model_id: str):
        """Create compressed backup of model"""
        backup_path = self.base_dir / "backups" / f"{model_id}.tar.gz"
        shutil.make_archive(str(backup_path.with_suffix("")), "gztar", model_dir)
        print(f"📦 Backup created: {backup_path}")

    def list_models(self, model_type: str = None) -> pd.DataFrame:
        """List all saved models with their metadata"""
        registry_path = self.base_dir / "metadata" / "model_registry.json"

        if not registry_path.exists():
            return pd.DataFrame()

        with open(registry_path, "r") as f:
            registry = json.load(f)

        df = pd.DataFrame(registry["models"])

        if model_type:
            df = df[df["model_name"].str.contains(model_type, case=False)]

        # Sort by timestamp (newest first)
        if not df.empty:
            df = df.sort_values("timestamp", ascending=False)

        return df

    def load_model(self, model_id: str, model_class: type) -> torch.nn.Module:
        """Load model by ID"""
        model_dir = self.base_dir / "checkpoints" / model_id
        model_path = model_dir / "model.pth"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_id}")

        # Load metadata to get architecture info
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location="cpu")

        # You'll need to instantiate the model with the correct parameters
        # This is a simplified version - you might need to store more architecture info
        print(f"📥 Loading model: {model_id}")
        print(f"   Architecture: {metadata['architecture']}")
        print(f"   Parameters: {metadata['parameters_count']:,}")
        print(f"   Metrics: {metadata['metrics']}")

        return checkpoint  # Return checkpoint for now, user can load into model

    def compare_models(self, model_ids: List[str]) -> pd.DataFrame:
        """Compare multiple models by their metrics"""
        registry_path = self.base_dir / "metadata" / "model_registry.json"

        with open(registry_path, "r") as f:
            registry = json.load(f)

        models_data = []
        for model in registry["models"]:
            if model["model_id"] in model_ids:
                row = {
                    "model_id": model["model_id"],
                    "model_name": model["model_name"],
                    "architecture": model["architecture"],
                    "parameters": model["parameters_count"],
                    "timestamp": model["timestamp"],
                }
                # Add all metrics
                for metric, value in model["metrics"].items():
                    row[f"metric_{metric}"] = value

                models_data.append(row)

        return pd.DataFrame(models_data)

    def export_model(self, model_id: str, export_format: str = "onnx") -> str:
        """Export model to different formats"""
        model_dir = self.base_dir / "checkpoints" / model_id
        export_dir = self.base_dir / "exports" / model_id
        export_dir.mkdir(parents=True, exist_ok=True)

        if export_format == "onnx":
            # ONNX export would go here
            export_path = export_dir / f"{model_id}.onnx"
            print(f"📤 ONNX export not implemented yet: {export_path}")
        elif export_format == "torchscript":
            # TorchScript export would go here
            export_path = export_dir / f"{model_id}.pt"
            print(f"📤 TorchScript export not implemented yet: {export_path}")

        return str(export_path)

    def cleanup_old_models(self, keep_top_n: int = 10):
        """Clean up old models, keeping only the best performing ones"""
        df = self.list_models()

        if len(df) <= keep_top_n:
            print(f"📊 Only {len(df)} models found, no cleanup needed")
            return

        # Sort by test accuracy (or your preferred metric)
        if "metrics" in df.columns:
            # This is simplified - you'd want to extract the actual metric values
            models_to_remove = df.iloc[keep_top_n:]

            for _, model in models_to_remove.iterrows():
                model_id = model["model_id"]
                model_dir = self.base_dir / "checkpoints" / model_id
                backup_path = self.base_dir / "backups" / f"{model_id}.tar.gz"

                if model_dir.exists():
                    shutil.rmtree(model_dir)
                    print(f"🗑️ Removed old model: {model_id}")

                # Keep backup but could remove if needed

    def get_best_model(self, metric: str = "test_accuracy") -> Optional[str]:
        """Get the best performing model by a specific metric"""
        registry_path = self.base_dir / "metadata" / "model_registry.json"

        if not registry_path.exists():
            return None

        with open(registry_path, "r") as f:
            registry = json.load(f)

        best_model = None
        best_score = -1

        for model in registry["models"]:
            if metric in model["metrics"]:
                score = model["metrics"][metric]
                if score > best_score:
                    best_score = score
                    best_model = model["model_id"]

        if best_model:
            print(f"🏆 Best model by {metric}: {best_model} (score: {best_score:.4f})")

        return best_model

    def generate_model_report(self) -> str:
        """Generate comprehensive model performance report"""
        df = self.list_models()

        if df.empty:
            return "No models found in registry"

        report = []
        report.append("🔬 MAGNUS CHESS AI - MODEL PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append(f"📊 Total models: {len(df)}")
        report.append(
            f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}"
        )
        report.append("")

        # Group by architecture
        if "architecture" in df.columns:
            arch_summary = df.groupby("architecture").size()
            report.append("🏗️ Models by architecture:")
            for arch, count in arch_summary.items():
                report.append(f"   {arch}: {count} models")
            report.append("")

        # Best performers
        best_model = self.get_best_model("test_accuracy")
        if best_model:
            report.append(f"🏆 Best performing model: {best_model}")

        report.append("")
        report.append("📈 Recent models (top 5):")
        for _, model in df.head().iterrows():
            report.append(
                f"   {model['model_id']}: {model['model_name']} ({model['timestamp']})"
            )

        return "\n".join(report)


def integrate_with_training(
    model_manager: MagnusModelManager,
    model: torch.nn.Module,
    model_name: str,
    experiment_name: str,
    final_metrics: Dict[str, float],
    config: Dict[str, Any],
) -> str:
    """Integration function for training scripts"""

    # Extract training data info
    training_data_info = {
        "train_size": config.get("train_size", 0),
        "val_size": config.get("val_size", 0),
        "test_size": config.get("test_size", 0),
        "vocab_size": config.get("vocab_size", 0),
        "data_source": config.get("data_source", "magnus_games"),
    }

    # Extract hyperparameters (remove non-serializable items)
    hyperparameters = {
        k: v for k, v in config.items() if isinstance(v, (int, float, str, bool, list))
    }

    # Save model with comprehensive tracking
    model_id = model_manager.save_model_comprehensive(
        model=model,
        model_name=model_name,
        experiment_name=experiment_name,
        metrics=final_metrics,
        hyperparameters=hyperparameters,
        training_data_info=training_data_info,
        model_architecture=type(model).__name__,
        notes=f"Training completed with {final_metrics.get('test_accuracy', 0):.4f} accuracy",
    )

    return model_id


if __name__ == "__main__":
    # Demo/test the model manager
    manager = MagnusModelManager()

    print(manager.generate_model_report())

    models_df = manager.list_models()
    if not models_df.empty:
        print("\n📋 All models:")
        print(models_df[["model_id", "model_name", "timestamp"]].to_string(index=False))
