#!/usr/bin/env python3
"""
DVC Integration Setup for Magnus Chess AI Models
Additional layer of model versioning with Git + DVC
"""

import os
import subprocess
import json
from pathlib import Path


def setup_dvc_integration():
    """Setup DVC for model versioning"""

    print("🔧 Setting up DVC integration...")

    # Check if we're in a git repo
    try:
        result = subprocess.run(["git", "status"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Not in a git repository. Please initialize git first:")
            print("   git init")
            print("   git add .")
            print("   git commit -m 'Initial commit'")
            return False
    except FileNotFoundError:
        print("❌ Git not found. Please install Git first.")
        return False

    # Check if DVC is installed
    try:
        result = subprocess.run(["dvc", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("📦 Installing DVC...")
            subprocess.run(["pip", "install", "dvc"], check=True)
    except FileNotFoundError:
        print("📦 Installing DVC...")
        try:
            subprocess.run(["pip", "install", "dvc"], check=True)
        except subprocess.CalledProcessError:
            print("❌ Failed to install DVC. Please install manually:")
            print("   pip install dvc")
            return False

    # Initialize DVC if not already done
    if not Path(".dvc").exists():
        print("🔄 Initializing DVC...")
        subprocess.run(["dvc", "init"], check=True)
        print("✅ DVC initialized")

    # Setup DVC tracking for models
    models_dir = Path("models_vault")
    if models_dir.exists():
        dvc_file = Path("models_vault.dvc")
        if not dvc_file.exists():
            print("📂 Adding models to DVC tracking...")
            subprocess.run(["dvc", "add", "models_vault"], check=True)
            print("✅ Models added to DVC")
        else:
            print("✅ Models already tracked by DVC")

    # Setup DVC remote (local for now, can be cloud later)
    dvc_remote_dir = Path("../dvc_remote")
    dvc_remote_dir.mkdir(exist_ok=True)

    try:
        # Add remote if it doesn't exist
        subprocess.run(
            ["dvc", "remote", "add", "local_storage", str(dvc_remote_dir.absolute())],
            check=False,
            capture_output=True,
        )
        subprocess.run(["dvc", "remote", "default", "local_storage"], check=True)
        print(f"✅ DVC remote configured: {dvc_remote_dir.absolute()}")
    except subprocess.CalledProcessError:
        print("⚠️ DVC remote already configured")

    # Create DVC pipeline for model training
    create_dvc_pipeline()

    print("\n🎉 DVC setup completed!")
    print("📋 Next steps:")
    print("   1. git add .dvc models_vault.dvc dvc.yaml")
    print("   2. git commit -m 'Add DVC tracking'")
    print("   3. dvc push  # Push models to remote storage")
    print("   4. Use 'dvc repro' to reproduce training pipeline")

    return True


def create_dvc_pipeline():
    """Create DVC pipeline for reproducible training"""

    pipeline = {
        "stages": {
            "extract_data": {
                "cmd": "python extract_positions_m3_pro.py",
                "deps": ["../carlsen-games.pgn"],
                "outs": ["magnus_extracted_positions_m3_pro.pkl"],
            },
            "train_enhanced": {
                "cmd": "python train_enhanced_magnus.py",
                "deps": [
                    "magnus_extracted_positions_m3_pro.pkl",
                    "train_enhanced_magnus.py",
                    "stockfish_magnus_trainer.py",
                ],
                "outs": ["models_vault/checkpoints"],
                "metrics": ["mlruns"],
                "params": {
                    "learning_rate": 0.001,
                    "batch_size": 128,
                    "num_epochs": 40,
                    "min_move_count": 15,
                },
            },
            "train_fast": {
                "cmd": "python train_fast_magnus.py",
                "deps": [
                    "magnus_extracted_positions_m3_pro.pkl",
                    "train_fast_magnus.py",
                ],
                "outs": ["models_vault/checkpoints"],
                "metrics": ["mlruns"],
            },
        }
    }

    with open("dvc.yaml", "w") as f:
        import yaml

        yaml.dump(pipeline, f, default_flow_style=False)

    print("✅ DVC pipeline created (dvc.yaml)")


def commit_model_version(model_id: str, message: str = ""):
    """Commit a specific model version with DVC"""

    if not Path(".dvc").exists():
        print("❌ DVC not initialized. Run setup_dvc_integration() first.")
        return False

    try:
        # Add changes to DVC
        subprocess.run(["dvc", "add", "models_vault"], check=True)

        # Commit to git
        commit_msg = f"Add model {model_id}"
        if message:
            commit_msg += f": {message}"

        subprocess.run(["git", "add", "models_vault.dvc"], check=True)
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)

        # Tag the commit
        tag_name = f"model-{model_id}"
        subprocess.run(["git", "tag", tag_name], check=True)

        # Push to DVC remote
        subprocess.run(["dvc", "push"], check=True)

        print(f"✅ Model {model_id} committed and versioned")
        print(f"🏷️ Tagged as: {tag_name}")

        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Error committing model: {e}")
        return False


def load_model_version(tag_name: str):
    """Load a specific model version by git tag"""

    try:
        # Checkout the specific tag
        subprocess.run(["git", "checkout", tag_name], check=True)

        # Pull the model data from DVC
        subprocess.run(["dvc", "checkout"], check=True)
        subprocess.run(["dvc", "pull"], check=True)

        print(f"✅ Loaded model version: {tag_name}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Error loading model version: {e}")
        return False


def list_model_versions():
    """List all available model versions"""

    try:
        result = subprocess.run(
            ["git", "tag", "--list", "model-*"],
            capture_output=True,
            text=True,
            check=True,
        )

        tags = result.stdout.strip().split("\n") if result.stdout.strip() else []

        if tags:
            print("📋 Available model versions:")
            for tag in tags:
                print(f"   🏷️ {tag}")
        else:
            print("📋 No model versions found")

        return tags

    except subprocess.CalledProcessError:
        print("❌ Error listing model versions")
        return []


def setup_cloud_storage(storage_type: str = "s3", **kwargs):
    """Setup cloud storage for DVC (S3, GCS, Azure, etc.)"""

    print(f"☁️ Setting up {storage_type} storage for DVC...")

    if storage_type == "s3":
        bucket = kwargs.get("bucket", "magnus-chess-models")
        region = kwargs.get("region", "us-west-2")

        print("📋 To complete S3 setup:")
        print(f"   1. Create S3 bucket: {bucket}")
        print("   2. Configure AWS credentials")
        print("   3. Run: dvc remote add s3_storage s3://{bucket}/models")
        print("   4. Run: dvc remote default s3_storage")

    elif storage_type == "gcs":
        bucket = kwargs.get("bucket", "magnus-chess-models")

        print("📋 To complete GCS setup:")
        print(f"   1. Create GCS bucket: {bucket}")
        print("   2. Configure GCP credentials")
        print("   3. Run: dvc remote add gcs_storage gs://{bucket}/models")
        print("   4. Run: dvc remote default gcs_storage")

    print("💡 After setup, use 'dvc push' to sync models to cloud")


if __name__ == "__main__":
    print("🔧 DVC Integration for Magnus Chess AI")
    print("=" * 40)

    if setup_dvc_integration():
        print("\n🎉 DVC setup successful!")
        print("\n💡 Usage examples:")
        print("   python dvc_integration.py  # This script")
        print("   dvc repro                  # Reproduce training pipeline")
        print("   dvc push                   # Push models to remote")
        print("   dvc pull                   # Pull models from remote")
        print("   dvc checkout               # Restore model files")

        # Show model versions
        list_model_versions()
    else:
        print("\n❌ DVC setup failed. Please check requirements.")
