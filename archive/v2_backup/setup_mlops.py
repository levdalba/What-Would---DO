#!/usr/bin/env python3
"""
Setup script for Magnus MLOps training
"""

import subprocess
import sys
from pathlib import Path


def check_and_install_packages():
    """Check and install required packages"""

    required_packages = ["mlflow", "evidently", "plotly", "seaborn"]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - missing")

    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install"] + missing_packages
            )
            print("✅ Installation completed!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Installation failed: {e}")
            return False

    return True


def check_data_file():
    """Check if extracted positions exist"""

    data_path = Path("magnus_extracted_positions_m3_pro.pkl")

    if data_path.exists():
        size_mb = data_path.stat().st_size / (1024 * 1024)
        print(f"✅ Extracted positions found: {size_mb:.1f} MB")
        return True
    else:
        print(f"❌ Extracted positions not found: {data_path}")
        print(f"   Run extract_positions_m3_pro.py first!")
        return False


def setup_directories():
    """Create necessary directories"""

    directories = ["models", "mlruns", "reports", "logs"]

    for dir_name in directories:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
        print(f"✅ Directory: {dir_name}")


def main():
    """Main setup function"""

    print("🔧 Magnus MLOps Training Setup")
    print("=" * 40)

    # Check packages
    print("\n📦 Checking packages...")
    if not check_and_install_packages():
        return False

    # Check data
    print("\n📂 Checking data files...")
    if not check_data_file():
        return False

    # Setup directories
    print("\n📁 Setting up directories...")
    setup_directories()

    # Check PyTorch MPS
    print("\n🎮 Checking hardware...")
    try:
        import torch

        if torch.backends.mps.is_available():
            print("✅ M3 Pro GPU (MPS) available")
        else:
            print("⚠️  MPS not available, will use CPU")
    except Exception as e:
        print(f"❌ PyTorch check failed: {e}")

    print("\n🎉 Setup completed!")
    print("\n🚀 Ready to train! Run:")
    print("   python train_magnus_simple_mlops.py")

    return True


if __name__ == "__main__":
    main()
