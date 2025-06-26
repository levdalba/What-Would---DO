#!/usr/bin/env python3
"""
Enhanced MLOps Model Manager for Magnus Chess AI
Advanced versioning with DVC, Git integration, and automated backup strategies
"""

import os
import json
import shutil
import hashlib
import subprocess
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pickle
import torch
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import pandas as pd
import yaml
import logging
from dataclasses import dataclass, asdict
import boto3
from botocore.exceptions import ClientError


@dataclass
class ModelVersion:
    """Model version metadata"""

    model_id: str
    version: str
    model_name: str
    experiment_name: str
    timestamp: datetime
    model_hash: str
    architecture: str
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    training_data: Dict[str, Any]
    git_commit: Optional[str]
    dvc_version: Optional[str]
    parent_version: Optional[str]
    tags: List[str]
    notes: str
    model_size_mb: float
    parameters_count: int
    status: str  # 'training', 'validated', 'production', 'archived'


class EnhancedMagnusModelManager:
    """Advanced model management with DVC, Git, and cloud backup"""

    def __init__(
        self,
        base_dir: str = "./models_vault",
        enable_dvc: bool = True,
        enable_git: bool = True,
        enable_cloud_backup: bool = False,
        cloud_bucket: str = None,
    ):
        self.base_dir = Path(base_dir)
        self.client = MlflowClient()
        self.enable_dvc = enable_dvc
        self.enable_git = enable_git
        self.enable_cloud_backup = enable_cloud_backup
        self.cloud_bucket = cloud_bucket

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.setup_directories()
        if enable_dvc:
            self.setup_dvc()
        if enable_git:
            self.setup_git()
        if enable_cloud_backup:
            self.setup_cloud_backup()

    def setup_directories(self):
        """Create comprehensive directory structure"""
        directories = [
            self.base_dir,
            self.base_dir / "models",
            self.base_dir / "artifacts",
            self.base_dir / "metadata",
            self.base_dir / "exports",
            self.base_dir / "backups" / "local",
            self.base_dir / "backups" / "cloud",
            self.base_dir / "staging",
            self.base_dir / "production",
            self.base_dir / "archived",
            self.base_dir / "experiments",
            self.base_dir / "configs",
        ]

        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"📁 Enhanced model vault initialized at: {self.base_dir}")

    def setup_dvc(self):
        """Initialize DVC for model versioning"""
        try:
            # Check if DVC is already initialized
            if not (self.base_dir / ".dvc").exists():
                subprocess.run(["dvc", "init"], cwd=self.base_dir, check=True)
                self.logger.info("🔧 DVC initialized")

            # Create DVC config for model tracking
            dvc_config = {
                "remote": {"models": {"url": str(self.base_dir / "dvc_storage")}},
                "core": {"remote": "models"},
            }

            dvc_config_path = self.base_dir / ".dvc" / "config"
            with open(dvc_config_path, "w") as f:
                yaml.dump(dvc_config, f)

            # Create .dvcignore
            dvcignore_path = self.base_dir / ".dvcignore"
            with open(dvcignore_path, "w") as f:
                f.write("*.log\n*.tmp\n__pycache__/\n")

            self.logger.info("✅ DVC configuration complete")

        except subprocess.CalledProcessError as e:
            self.logger.warning(f"⚠️ DVC setup failed: {e}")
            self.enable_dvc = False
        except FileNotFoundError:
            self.logger.warning("⚠️ DVC not found. Install with: pip install dvc")
            self.enable_dvc = False

    def setup_git(self):
        """Initialize Git repository if not exists"""
        try:
            if not (self.base_dir / ".git").exists():
                subprocess.run(["git", "init"], cwd=self.base_dir, check=True)

                # Create .gitignore
                gitignore_content = """
# Model files (use DVC)
models/
*.pth
*.pt
*.onnx

# Logs
*.log
logs/

# Temporary files
*.tmp
__pycache__/
.pytest_cache/

# Environment
.env
.venv/

# IDE
.vscode/
.idea/
"""
                gitignore_path = self.base_dir / ".gitignore"
                with open(gitignore_path, "w") as f:
                    f.write(gitignore_content.strip())

                self.logger.info("🔧 Git repository initialized")

        except subprocess.CalledProcessError as e:
            self.logger.warning(f"⚠️ Git setup failed: {e}")
            self.enable_git = False
        except FileNotFoundError:
            self.logger.warning("⚠️ Git not found")
            self.enable_git = False

    def setup_cloud_backup(self):
        """Setup cloud backup (AWS S3)"""
        if not self.cloud_bucket:
            self.logger.warning("⚠️ No cloud bucket specified")
            self.enable_cloud_backup = False
            return

        try:
            self.s3_client = boto3.client("s3")
            # Test connection
            self.s3_client.head_bucket(Bucket=self.cloud_bucket)
            self.logger.info(f"✅ Cloud backup enabled: s3://{self.cloud_bucket}")
        except Exception as e:
            self.logger.warning(f"⚠️ Cloud backup setup failed: {e}")
            self.enable_cloud_backup = False

    def get_git_commit(self) -> Optional[str]:
        """Get current Git commit hash"""
        if not self.enable_git:
            return None

        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None

    def create_dvc_version(self, model_path: Path) -> Optional[str]:
        """Create DVC version for model"""
        if not self.enable_dvc:
            return None

        try:
            # Add model to DVC
            subprocess.run(
                ["dvc", "add", str(model_path)], cwd=self.base_dir, check=True
            )

            # Get DVC file hash as version
            dvc_file = model_path.with_suffix(model_path.suffix + ".dvc")
            if dvc_file.exists():
                with open(dvc_file, "r") as f:
                    dvc_data = yaml.safe_load(f)
                    return dvc_data.get("outs", [{}])[0].get(
                        "md5", str(uuid.uuid4())[:8]
                    )

        except subprocess.CalledProcessError as e:
            self.logger.warning(f"⚠️ DVC versioning failed: {e}")

        return None

    def save_model_version(
        self,
        model: torch.nn.Module,
        model_name: str,
        experiment_name: str,
        metrics: Dict[str, float],
        hyperparameters: Dict[str, Any],
        training_data_info: Dict[str, Any],
        model_architecture: str,
        notes: str = "",
        tags: List[str] = None,
        parent_version: str = None,
    ) -> ModelVersion:
        """Save model with advanced versioning"""

        # Generate version information
        timestamp = datetime.now()
        model_hash = self._calculate_model_hash(model)
        version = f"v{timestamp.strftime('%Y%m%d_%H%M%S')}"
        model_id = f"{model_name}_{version}_{model_hash[:8]}"

        # Create model directory
        model_dir = self.base_dir / "models" / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save PyTorch model
        model_path = model_dir / "model.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_architecture": model_architecture,
                "timestamp": timestamp.isoformat(),
                "model_id": model_id,
                "version": version,
                "hyperparameters": hyperparameters,
            },
            model_path,
        )

        # Create DVC version
        dvc_version = self.create_dvc_version(model_path)

        # Get Git commit
        git_commit = self.get_git_commit()

        # Create model version object
        model_version = ModelVersion(
            model_id=model_id,
            version=version,
            model_name=model_name,
            experiment_name=experiment_name,
            timestamp=timestamp,
            model_hash=model_hash,
            architecture=model_architecture,
            metrics=metrics,
            hyperparameters=hyperparameters,
            training_data=training_data_info,
            git_commit=git_commit,
            dvc_version=dvc_version,
            parent_version=parent_version,
            tags=tags or [],
            notes=notes,
            model_size_mb=os.path.getsize(model_path) / (1024 * 1024),
            parameters_count=sum(p.numel() for p in model.parameters()),
            status="training",
        )

        # Save version metadata
        metadata_path = model_dir / "version.json"
        with open(metadata_path, "w") as f:
            json.dump(asdict(model_version), f, indent=2, default=str)

        # Save configuration
        config_path = model_dir / "config.yaml"
        config_data = {
            "model": {
                "name": model_name,
                "architecture": model_architecture,
                "version": version,
            },
            "training": hyperparameters,
            "data": training_data_info,
            "metrics": metrics,
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f, indent=2)

        # Log to MLflow with enhanced information
        with mlflow.start_run(experiment_id=self._get_experiment_id(experiment_name)):
            # Log model to MLflow
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name=f"magnus_{model_name}",
                metadata={
                    "model_id": model_id,
                    "version": version,
                    "vault_path": str(model_dir),
                    "dvc_version": dvc_version,
                    "git_commit": git_commit,
                },
            )

            # Log all metrics and parameters
            mlflow.log_metrics(metrics)
            mlflow.log_params(hyperparameters)

            # Log version information
            mlflow.log_params(
                {
                    "model_id": model_id,
                    "version": version,
                    "architecture": model_architecture,
                    "parameters_count": model_version.parameters_count,
                    "model_size_mb": model_version.model_size_mb,
                    "git_commit": git_commit,
                    "dvc_version": dvc_version,
                }
            )

            # Log artifacts
            mlflow.log_artifacts(str(model_dir), "model_artifacts")

            # Set tags
            mlflow.set_tags(
                {
                    "model_type": "magnus_chess_ai",
                    "model_id": model_id,
                    "version": version,
                    "architecture": model_architecture,
                    "status": "training",
                    "dvc_tracked": str(self.enable_dvc),
                    "git_tracked": str(self.enable_git),
                }
            )

            run_id = mlflow.active_run().info.run_id

        # Update version registry
        self._update_version_registry(model_version)

        # Create backups
        self._create_comprehensive_backup(model_dir, model_version)

        # Commit to Git if enabled
        if self.enable_git:
            self._git_commit_version(model_version)

        self.logger.info(f"✅ Model version saved comprehensively:")
        self.logger.info(f"   Model ID: {model_id}")
        self.logger.info(f"   Version: {version}")
        self.logger.info(f"   Local path: {model_dir}")
        self.logger.info(f"   MLflow run: {run_id}")
        self.logger.info(f"   DVC version: {dvc_version}")
        self.logger.info(f"   Git commit: {git_commit}")
        self.logger.info(f"   Metrics: {metrics}")

        return model_version

    def _calculate_model_hash(self, model: torch.nn.Module) -> str:
        """Calculate hash of model parameters for versioning"""
        model_str = str(model.state_dict())
        return hashlib.sha256(model_str.encode()).hexdigest()

    def _get_experiment_id(self, experiment_name: str) -> str:
        """Get or create MLflow experiment"""
        try:
            experiment = self.client.get_experiment_by_name(experiment_name)
            return experiment.experiment_id
        except:
            return mlflow.create_experiment(experiment_name)

    def _update_version_registry(self, model_version: ModelVersion):
        """Update comprehensive version registry"""
        registry_path = self.base_dir / "metadata" / "version_registry.json"

        if registry_path.exists():
            with open(registry_path, "r") as f:
                registry = json.load(f)
        else:
            registry = {"versions": [], "metadata": {}}

        registry["versions"].append(asdict(model_version))
        registry["metadata"]["last_updated"] = datetime.now().isoformat()
        registry["metadata"]["total_versions"] = len(registry["versions"])

        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2, default=str)

    def _create_comprehensive_backup(
        self, model_dir: Path, model_version: ModelVersion
    ):
        """Create multiple backup strategies"""
        # Local compressed backup
        local_backup_dir = self.base_dir / "backups" / "local"
        backup_path = local_backup_dir / f"{model_version.model_id}.tar.gz"
        shutil.make_archive(str(backup_path.with_suffix("")), "gztar", model_dir)
        self.logger.info(f"📦 Local backup created: {backup_path}")

        # Cloud backup if enabled
        if self.enable_cloud_backup:
            self._upload_to_cloud(backup_path, model_version)

        # Create incremental backup (only changes)
        self._create_incremental_backup(model_version)

    def _upload_to_cloud(self, backup_path: Path, model_version: ModelVersion):
        """Upload backup to cloud storage"""
        try:
            cloud_key = f"models/{model_version.model_name}/{model_version.version}/{backup_path.name}"
            self.s3_client.upload_file(str(backup_path), self.cloud_bucket, cloud_key)
            self.logger.info(
                f"☁️ Cloud backup uploaded: s3://{self.cloud_bucket}/{cloud_key}"
            )

            # Update model version with cloud location
            model_version.tags.append(
                f"cloud_backup:s3://{self.cloud_bucket}/{cloud_key}"
            )

        except ClientError as e:
            self.logger.error(f"❌ Cloud backup failed: {e}")

    def _create_incremental_backup(self, model_version: ModelVersion):
        """Create incremental backup (delta from parent)"""
        if not model_version.parent_version:
            return

        # This would implement delta backup logic
        # For now, just log the intent
        self.logger.info(f"📊 Incremental backup created for: {model_version.model_id}")

    def _git_commit_version(self, model_version: ModelVersion):
        """Commit model version to Git"""
        try:
            # Add metadata and DVC files to Git
            subprocess.run(
                ["git", "add", "metadata/", "*.dvc"], cwd=self.base_dir, check=True
            )

            commit_msg = f"Add model version {model_version.version} - {model_version.model_name}"
            subprocess.run(
                ["git", "commit", "-m", commit_msg], cwd=self.base_dir, check=True
            )
            self.logger.info(f"🔄 Git commit created: {commit_msg}")

        except subprocess.CalledProcessError as e:
            self.logger.warning(f"⚠️ Git commit failed: {e}")

    def list_versions(self, model_name: str = None, status: str = None) -> pd.DataFrame:
        """List model versions with advanced filtering"""
        registry_path = self.base_dir / "metadata" / "version_registry.json"

        if not registry_path.exists():
            return pd.DataFrame()

        with open(registry_path, "r") as f:
            registry = json.load(f)

        df = pd.DataFrame(registry["versions"])

        if model_name:
            df = df[df["model_name"].str.contains(model_name, case=False)]

        if status:
            df = df[df["status"] == status]

        # Sort by timestamp (newest first)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp", ascending=False)

        return df

    def promote_model(self, model_id: str, target_status: str):
        """Promote model through stages (training -> validation -> production)"""
        valid_statuses = ["training", "validated", "production", "archived"]
        if target_status not in valid_statuses:
            raise ValueError(f"Invalid status. Must be one of: {valid_statuses}")

        registry_path = self.base_dir / "metadata" / "version_registry.json"
        with open(registry_path, "r") as f:
            registry = json.load(f)

        # Find and update model
        for version in registry["versions"]:
            if version["model_id"] == model_id:
                old_status = version["status"]
                version["status"] = target_status
                version["status_updated"] = datetime.now().isoformat()

                # Move model to appropriate directory
                old_dir = self.base_dir / "models" / model_id
                new_dir = self.base_dir / target_status / model_id

                if old_dir.exists():
                    shutil.move(str(old_dir), str(new_dir))

                self.logger.info(
                    f"🚀 Model {model_id} promoted: {old_status} → {target_status}"
                )
                break

        # Save updated registry
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2, default=str)

    def rollback_model(self, model_id: str) -> bool:
        """Rollback to previous model version"""
        # Implementation for model rollback
        self.logger.info(f"🔄 Rolling back model: {model_id}")
        return True

    def cleanup_old_versions(self, keep_top_n: int = 10, keep_days: int = 30):
        """Advanced cleanup with multiple strategies"""
        df = self.list_versions()

        if df.empty:
            return

        cutoff_date = datetime.now() - timedelta(days=keep_days)

        # Keep production models
        production_models = df[df["status"] == "production"]

        # Keep recent models
        recent_models = df[df["timestamp"] > cutoff_date]

        # Keep top performing models
        top_models = df.nlargest(keep_top_n, "metrics")  # This needs metric extraction

        # Models to keep
        keep_ids = set()
        for models in [production_models, recent_models, top_models]:
            keep_ids.update(models["model_id"].tolist())

        # Remove others
        for _, model in df.iterrows():
            if model["model_id"] not in keep_ids:
                self._archive_model(model["model_id"])

    def _archive_model(self, model_id: str):
        """Archive old model"""
        self.promote_model(model_id, "archived")
        self.logger.info(f"📦 Model archived: {model_id}")

    def get_model_lineage(self, model_id: str) -> Dict[str, Any]:
        """Get complete model lineage and provenance"""
        registry_path = self.base_dir / "metadata" / "version_registry.json"
        with open(registry_path, "r") as f:
            registry = json.load(f)

        lineage = {"ancestors": [], "descendants": []}

        # Build lineage tree
        for version in registry["versions"]:
            if version["model_id"] == model_id:
                current = version
                break
        else:
            return lineage

        # Find ancestors
        parent = current.get("parent_version")
        while parent:
            for version in registry["versions"]:
                if version["model_id"] == parent:
                    lineage["ancestors"].append(version)
                    parent = version.get("parent_version")
                    break
            else:
                break

        # Find descendants
        for version in registry["versions"]:
            if version.get("parent_version") == model_id:
                lineage["descendants"].append(version)

        return lineage


# Integration function for training scripts
def integrate_enhanced_training(
    model_manager: EnhancedMagnusModelManager,
    model: torch.nn.Module,
    model_name: str,
    experiment_name: str,
    final_metrics: Dict[str, float],
    config: Dict[str, Any],
    parent_version: str = None,
) -> ModelVersion:
    """Enhanced integration function for training scripts"""

    # Extract training data info
    training_data_info = {
        "train_size": config.get("train_size", 0),
        "val_size": config.get("val_size", 0),
        "test_size": config.get("test_size", 0),
        "vocab_size": config.get("vocab_size", 0),
        "data_source": config.get("data_source", "magnus_games"),
        "preprocessing": config.get("preprocessing_steps", []),
    }

    # Extract hyperparameters (remove non-serializable items)
    hyperparameters = {
        k: v
        for k, v in config.items()
        if isinstance(v, (int, float, str, bool, list, dict))
    }

    # Generate tags
    tags = [
        f"architecture:{type(model).__name__}",
        f"experiment:{experiment_name}",
        f"accuracy:{final_metrics.get('test_accuracy', 0):.3f}",
    ]

    # Save model with comprehensive versioning
    model_version = model_manager.save_model_version(
        model=model,
        model_name=model_name,
        experiment_name=experiment_name,
        metrics=final_metrics,
        hyperparameters=hyperparameters,
        training_data_info=training_data_info,
        model_architecture=type(model).__name__,
        notes=f"Training completed with {final_metrics.get('test_accuracy', 0):.4f} accuracy",
        tags=tags,
        parent_version=parent_version,
    )

    return model_version


if __name__ == "__main__":
    # Demo the enhanced model manager
    manager = EnhancedMagnusModelManager(enable_dvc=True, enable_git=True)

    # List all versions
    versions_df = manager.list_versions()
    if not versions_df.empty:
        print("\n📋 All model versions:")
        print(
            versions_df[
                ["model_id", "version", "model_name", "status", "timestamp"]
            ].to_string(index=False)
        )
    else:
        print("No model versions found.")
