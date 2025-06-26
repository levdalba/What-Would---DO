#!/usr/bin/env python3
"""
Comprehensive training script to test different model architectures
"""

import sys
from pathlib import Path
import subprocess
import time

# Add project directory to path
sys.path.append(str(Path(__file__).parent))

from train_magnus_fixed_mlops import MagnusMLOpsFixed


def run_enhanced_training():
    """Run the enhanced training script"""
    print("🚀 Running Enhanced Magnus Training...")
    try:
        result = subprocess.run([
            sys.executable, "train_enhanced_magnus.py"
        ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print("✅ Enhanced training completed successfully")
            print("📊 Enhanced Results:")
            # Parse output for accuracy
            output_lines = result.stdout.split('\n')
            for line in output_lines[-20:]:  # Last 20 lines
                if 'Test Accuracy' in line or 'Top-' in line:
                    print(f"   {line.strip()}")
            return True
        else:
            print(f"❌ Enhanced training failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("⏰ Enhanced training timed out")
        return False
    except Exception as e:
        print(f"❌ Enhanced training error: {e}")
        return False


def run_advanced_training():
    """Run the advanced training script"""
    print("🚀 Running Advanced Magnus Training...")
    try:
        result = subprocess.run([
            sys.executable, "train_advanced_magnus.py"
        ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print("✅ Advanced training completed successfully")
            print("📊 Advanced Results:")
            # Parse output for accuracy
            output_lines = result.stdout.split('\n')
            for line in output_lines[-20:]:  # Last 20 lines
                if 'Test Accuracy' in line or 'Advanced Results' in line:
                    print(f"   {line.strip()}")
            return True
        else:
            print(f"❌ Advanced training failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("⏰ Advanced training timed out")
        return False
    except Exception as e:
        print(f"❌ Advanced training error: {e}")
        return False


def run_baseline_configs():
    """Run baseline configurations for comparison"""
    print("🚀 Running baseline configurations...")

    trainer = MagnusMLOpsFixed("magnus_baseline_comparison")

    # Optimized configurations based on previous results
    configs = [
        {
            "learning_rate": 0.0001,
            "batch_size": 256,
            "num_epochs": 20,
            "name": "baseline_optimized",
        },
        {
            "learning_rate": 0.0005,
            "batch_size": 128,
            "num_epochs": 30,
            "name": "baseline_aggressive",
        },
    ]

    results = []

    for i, config in enumerate(configs, 1):
        print(f"\n📊 Running baseline config {i}/{len(configs)}: {config['name']}")
        
        try:
            result = trainer.train_magnus_style(
                learning_rate=config["learning_rate"],
                batch_size=config["batch_size"],
                num_epochs=config["num_epochs"],
            )

            if result:
                test_accuracy = result.get("test_accuracy", 0)
                results.append({
                    "config": config["name"],
                    "accuracy": test_accuracy,
                    "type": "baseline"
                })
                print(f"✅ Baseline {i} completed - Test accuracy: {test_accuracy:.4f}")

        except Exception as e:
            print(f"❌ Baseline {i} error: {e}")

    return results


def main():
    """Run comprehensive training comparison"""
    print("🔬 Magnus Chess AI - Comprehensive Training Comparison")
    print("=" * 60)
    
    all_results = []
    
    # 1. Run baseline configurations
    print("\n1️⃣ BASELINE CONFIGURATIONS")
    print("-" * 40)
    baseline_results = run_baseline_configs()
    all_results.extend(baseline_results)
    
    # 2. Run enhanced training
    print("\n2️⃣ ENHANCED MODEL TRAINING")
    print("-" * 40)
    enhanced_success = run_enhanced_training()
    if enhanced_success:
        all_results.append({
            "config": "enhanced_model",
            "accuracy": "See console output",
            "type": "enhanced"
        })
    
    # 3. Run advanced training
    print("\n3️⃣ ADVANCED MODEL TRAINING")
    print("-" * 40)
    advanced_success = run_advanced_training()
    if advanced_success:
        all_results.append({
            "config": "advanced_model",
            "accuracy": "See console output", 
            "type": "advanced"
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("� COMPREHENSIVE TRAINING SUMMARY")
    print("=" * 60)
    
    print("\n🎯 Training Results:")
    for result in all_results:
        print(f"   {result['type'].upper()}: {result['config']} - {result['accuracy']}")
    
    print(f"\n🔬 View all results in MLflow UI: http://127.0.0.1:5000")
    print("\n📈 Expected Improvements:")
    print("   Baseline: ~1.0-1.5% accuracy")
    print("   Enhanced: ~8-12% accuracy (10x improvement)")
    print("   Advanced: ~12-18% accuracy (target: 15%+)")
    
    print("\n🏆 Key Improvements in Advanced Model:")
    print("   ✅ Multi-head attention mechanism")
    print("   ✅ Advanced feature extraction (8 chess features)")
    print("   ✅ Weighted focal loss for imbalanced classes")
    print("   ✅ Gradient clipping and batch normalization")
    print("   ✅ OneCycle learning rate scheduler")
    print("   ✅ Residual connections and dropout")


if __name__ == "__main__":
    main()
