#!/usr/bin/env python3
"""
Magnus Chess AI - Model Management Dashboard
Clean and working dashboard for model management
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add the v2 directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from mlops_enhanced_manager import EnhancedMagnusModelManager

    ENHANCED_AVAILABLE = True
except ImportError:
    try:
        from mlops_model_manager import MagnusModelManager

        ENHANCED_AVAILABLE = False
    except ImportError:
        print("❌ No model manager available. Please check your setup.")
        sys.exit(1)


class ModelDashboard:
    """Clean dashboard for model management"""

    def __init__(self):
        # Initialize model manager
        if ENHANCED_AVAILABLE:
            self.manager = EnhancedMagnusModelManager()
        else:
            self.manager = MagnusModelManager()

    def run_console(self):
        """Run console version of dashboard"""
        print("🔬 MAGNUS CHESS AI - MODEL MANAGEMENT DASHBOARD")
        print("=" * 70)

        # Generate and display comprehensive report
        if hasattr(self.manager, "generate_model_report"):
            report = self.manager.generate_model_report()
            print(report)

        # Get all models
        if ENHANCED_AVAILABLE:
            models_df = self.manager.list_versions()
        else:
            models_df = self.manager.list_models()

        if models_df.empty:
            print("\n❌ No models found in registry")
            return

        print("\n📊 DETAILED MODEL COMPARISON")
        print("-" * 50)

        # Display top 10 models with key metrics
        display_columns = [
            "model_id",
            "model_name",
            "architecture",
            "timestamp",
        ]

        available_columns = [col for col in display_columns if col in models_df.columns]

        if available_columns:
            print("\n📋 Recent Models:")
            top_models = models_df.head(10)[available_columns]
            print(top_models.to_string(index=False))

        # Display best performers
        print("\n🏆 BEST PERFORMERS:")
        if hasattr(self.manager, "get_best_model"):
            best_accuracy = self.manager.get_best_model("test_accuracy")
            if best_accuracy:
                print(f"   🥇 Best Accuracy: {best_accuracy}")

        # Display storage information
        print("\n💾 STORAGE INFORMATION:")
        models_vault = Path("./models_vault")
        if models_vault.exists():
            # Calculate storage usage
            total_size = sum(
                f.stat().st_size for f in models_vault.rglob("*") if f.is_file()
            )
            total_size_mb = total_size / (1024 * 1024)

            checkpoint_count = len(list((models_vault / "checkpoints").glob("*")))
            backup_count = len(list((models_vault / "backups").glob("*.tar.gz")))

            print(f"   📁 Total storage: {total_size_mb:.1f} MB")
            print(f"   💾 Checkpoints: {checkpoint_count}")
            print(f"   📦 Backups: {backup_count}")

        print(f"\n🔗 MLflow UI: http://127.0.0.1:5000")
        print(f"📂 Models vault: {models_vault.absolute()}")


def cleanup_old_models():
    """Interactive cleanup of old models"""
    if ENHANCED_AVAILABLE:
        manager = EnhancedMagnusModelManager()
        models_df = manager.list_versions()
    else:
        manager = MagnusModelManager()
        models_df = manager.list_models()

    if len(models_df) <= 5:
        print("📊 Only a few models found, no cleanup recommended")
        return

    print(f"📊 Found {len(models_df)} models")
    response = input("🗑️ Clean up old models? Keep only top 10? (y/N): ")

    if response.lower() == "y":
        if hasattr(manager, "cleanup_old_versions"):
            manager.cleanup_old_versions(keep_top_n=10)
        elif hasattr(manager, "cleanup_old_models"):
            manager.cleanup_old_models(keep_top_n=10)
        print("✅ Cleanup completed")


def compare_specific_models():
    """Compare specific models interactively"""
    if ENHANCED_AVAILABLE:
        manager = EnhancedMagnusModelManager()
        models_df = manager.list_versions()
    else:
        manager = MagnusModelManager()
        models_df = manager.list_models()

    if models_df.empty:
        print("❌ No models found")
        return

    print("\n📋 Available models:")
    for i, (_, model) in enumerate(models_df.head(10).iterrows()):
        print(f"   {i}: {model['model_id']} ({model['model_name']})")

    try:
        indices = input("\n🔍 Enter model indices to compare (e.g., 0,1,2): ").split(
            ","
        )
        selected_models = [models_df.iloc[int(i.strip())]["model_id"] for i in indices]

        if hasattr(manager, "compare_models"):
            comparison_df = manager.compare_models(selected_models)
            if not comparison_df.empty:
                print("\n📊 MODEL COMPARISON:")
                print(comparison_df.to_string(index=False))
            else:
                print("❌ No comparison data available")
        else:
            print("❌ Model comparison not available with current manager")

    except (ValueError, IndexError):
        print("❌ Invalid selection")


def export_model_interactive():
    """Interactive model export"""
    if ENHANCED_AVAILABLE:
        manager = EnhancedMagnusModelManager()
        models_df = manager.list_versions()
    else:
        manager = MagnusModelManager()
        models_df = manager.list_models()

    if models_df.empty:
        print("❌ No models found")
        return

    print("\n📋 Available models:")
    for i, (_, model) in enumerate(models_df.head(5).iterrows()):
        model_id = model["model_id"]
        model_name = model["model_name"]
        timestamp = model["timestamp"]
        print(f"   {i}: {model_id} ({model_name}) - {timestamp}")

    try:
        index = int(input("\n📤 Select model to export (index): "))
        model_id = models_df.iloc[index]["model_id"]

        print(f"🔄 Exporting model: {model_id}")
        if hasattr(manager, "export_model"):
            export_path = manager.export_model(model_id, "onnx")
            print(f"✅ Model exported to: {export_path}")
        else:
            print("📋 Export formats available: ONNX, TorchScript")
            print("💡 Export functionality ready for implementation")

    except (ValueError, IndexError):
        print("❌ Invalid selection")


def display_model_dashboard():
    """Display the main dashboard"""
    dashboard = ModelDashboard()
    dashboard.run_console()


def main():
    """Main dashboard interface"""
    while True:
        print("\n" + "=" * 50)
        print("🎮 MAGNUS MODEL MANAGEMENT MENU")
        print("=" * 50)
        print("1. 📊 View Model Dashboard")
        print("2. 🔍 Compare Specific Models")
        print("3. 📤 Export Model")
        print("4. 🗑️ Cleanup Old Models")
        print("5. 🚪 Exit")

        choice = input("\n🎯 Select option (1-5): ").strip()

        if choice == "1":
            display_model_dashboard()
        elif choice == "2":
            compare_specific_models()
        elif choice == "3":
            export_model_interactive()
        elif choice == "4":
            cleanup_old_models()
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard closed by user")
    except Exception as e:
        print(f"\n❌ Dashboard error: {e}")
        print("💡 Check that training has been run and models exist")
