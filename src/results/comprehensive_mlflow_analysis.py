#!/usr/bin/env python3
"""
Comprehensive MLflow Data Recovery and Analysis for Presentation
Combines current and archived MLflow data to show complete experiment history
"""

import sys
import os
import shutil
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Add the project directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import mlflow
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️ MLflow not available")


class ComprehensiveMLflowAnalyzer:
    """Analyze and present complete MLflow experiment history"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.current_mlruns = self.project_root / "mlruns"
        self.archived_mlruns = self.project_root / "archive" / "v2_backup" / "mlruns"
        self.results_dir = Path(__file__).parent

    def backup_current_mlruns(self):
        """Backup current MLflow data before merging"""
        backup_dir = self.project_root / "mlruns_current_backup"
        if self.current_mlruns.exists():
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            shutil.copytree(self.current_mlruns, backup_dir)
            print(f"✅ Current MLflow data backed up to: {backup_dir}")

    def merge_archived_data(self):
        """Merge archived MLflow data with current for comprehensive view"""

        if not self.archived_mlruns.exists():
            print("❌ No archived MLflow data found")
            return False

        print("🔄 Merging archived MLflow data...")

        # Copy archived experiments to current mlruns
        for exp_dir in self.archived_mlruns.iterdir():
            if exp_dir.is_dir() and exp_dir.name not in [".trash", "models"]:
                target_dir = self.current_mlruns / exp_dir.name
                if not target_dir.exists():
                    shutil.copytree(exp_dir, target_dir)
                    print(f"📊 Restored experiment: {exp_dir.name}")
                else:
                    print(f"⚠️ Experiment {exp_dir.name} already exists, skipping")

        # Copy archived models
        archived_models = self.archived_mlruns / "models"
        current_models = self.current_mlruns / "models"

        if archived_models.exists():
            if not current_models.exists():
                current_models.mkdir(parents=True)

            for model_dir in archived_models.iterdir():
                if model_dir.is_dir():
                    target_model = current_models / model_dir.name
                    if not target_model.exists():
                        shutil.copytree(model_dir, target_model)
                        print(f"💾 Restored model: {model_dir.name}")

        return True

    def analyze_comprehensive_data(self):
        """Analyze the complete merged MLflow data"""

        if not MLFLOW_AVAILABLE:
            print("❌ MLflow not available")
            return None

        try:
            # Set tracking URI to merged data
            tracking_uri = f"file://{self.current_mlruns.absolute()}"
            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient()

            print(f"📊 Analyzing comprehensive MLflow data: {tracking_uri}")

            # Get all experiments
            experiments = client.search_experiments()

            all_runs_data = []
            experiment_summary = {}

            for exp in experiments:
                if exp.name.lower() == "default":
                    continue

                print(f"\\n🔬 Analyzing experiment: {exp.name}")
                runs = client.search_runs(experiment_ids=[exp.experiment_id])

                finished_runs = [r for r in runs if r.info.status == "FINISHED"]
                failed_runs = [r for r in runs if r.info.status == "FAILED"]

                experiment_summary[exp.name] = {
                    "total_runs": len(runs),
                    "finished_runs": len(finished_runs),
                    "failed_runs": len(failed_runs),
                }

                print(f"   📈 Total runs: {len(runs)}")
                print(f"   ✅ Finished: {len(finished_runs)}")
                print(f"   ❌ Failed: {len(failed_runs)}")

                for run in finished_runs:
                    metrics = run.data.metrics
                    params = run.data.params

                    # Extract metrics with various naming patterns
                    test_acc = (
                        metrics.get("final_test_accuracy")
                        or metrics.get("test_accuracy")
                        or metrics.get("final_test_accuracy_top1")
                        or 0
                    )

                    test_top3 = (
                        metrics.get("final_test_accuracy_top3")
                        or metrics.get("test_top3_accuracy")
                        or 0
                    )

                    test_top5 = (
                        metrics.get("final_test_accuracy_top5")
                        or metrics.get("test_top5_accuracy")
                        or 0
                    )

                    training_time = (
                        metrics.get("training_time_minutes")
                        or metrics.get("training_time")
                        or 0
                    )

                    best_val_acc = metrics.get("best_val_accuracy", 0)

                    # Extract parameters
                    run_data = {
                        "experiment": exp.name,
                        "run_id": run.info.run_id,
                        "run_name": run.data.tags.get("mlflow.runName", "unnamed"),
                        "test_accuracy": test_acc,
                        "test_top3_accuracy": test_top3,
                        "test_top5_accuracy": test_top5,
                        "training_time_minutes": training_time,
                        "best_val_accuracy": best_val_acc,
                        "learning_rate": float(params.get("learning_rate", 0)),
                        "batch_size": (
                            int(params.get("batch_size", 0))
                            if params.get("batch_size")
                            else 0
                        ),
                        "num_epochs": (
                            int(params.get("num_epochs", 0))
                            if params.get("num_epochs")
                            else 0
                        ),
                        "parameters_count": int(
                            params.get(
                                "parameters_count", params.get("total_params", 0)
                            )
                        ),
                        "model_architecture": params.get(
                            "model_architecture", "unknown"
                        ),
                        "device": params.get("device", "unknown"),
                        "start_time": pd.to_datetime(run.info.start_time, unit="ms"),
                        "end_time": pd.to_datetime(run.info.end_time, unit="ms"),
                        "duration_minutes": (run.info.end_time - run.info.start_time)
                        / (1000 * 60),
                    }

                    all_runs_data.append(run_data)

            if not all_runs_data:
                print("❌ No finished runs found")
                return None

            df = pd.DataFrame(all_runs_data)

            print(f"\\n📈 COMPREHENSIVE ANALYSIS SUMMARY")
            print("=" * 60)
            print(f"🔬 Total experiments: {len(experiments)-1}")  # Exclude Default
            print(f"🏃 Total successful runs: {len(df)}")
            print(
                f"📅 Date range: {df['start_time'].min().strftime('%Y-%m-%d')} to {df['start_time'].max().strftime('%Y-%m-%d')}"
            )

            # Best performance across all experiments
            best_run = df.loc[df["test_accuracy"].idxmax()]
            print(f"\\n🏆 BEST PERFORMING RUN (ALL TIME)")
            print("-" * 40)
            print(f"Experiment: {best_run['experiment']}")
            print(f"Run Name: {best_run['run_name']}")
            print(f"Test Accuracy: {best_run['test_accuracy']:.4f}")
            print(f"Top-3 Accuracy: {best_run['test_top3_accuracy']:.4f}")
            print(f"Top-5 Accuracy: {best_run['test_top5_accuracy']:.4f}")
            print(f"Training Time: {best_run['training_time_minutes']:.2f} minutes")
            print(f"Parameters: {best_run['parameters_count']:,}")

            # Experiment performance summary
            print(f"\\n📊 EXPERIMENT PERFORMANCE RANKING")
            print("-" * 60)
            exp_performance = (
                df.groupby("experiment")
                .agg(
                    {
                        "test_accuracy": ["count", "mean", "max", "std"],
                        "training_time_minutes": "mean",
                        "parameters_count": "first",
                    }
                )
                .round(4)
            )

            # Flatten column names
            exp_performance.columns = [
                "runs",
                "mean_acc",
                "max_acc",
                "std_acc",
                "avg_time",
                "params",
            ]
            exp_performance = exp_performance.sort_values("max_acc", ascending=False)

            for i, (exp_name, row) in enumerate(exp_performance.iterrows(), 1):
                print(f"{i:2d}. {exp_name}")
                print(
                    f"    Best: {row['max_acc']:.4f} | Avg: {row['mean_acc']:.4f} ± {row['std_acc']:.4f}"
                )
                print(
                    f"    Runs: {row['runs']} | Avg Time: {row['avg_time']:.1f}min | Params: {row['params']:,}"
                )

            return df, experiment_summary

        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return None, None

    def create_presentation_visualizations(self, df, experiment_summary):
        """Create comprehensive visualizations for presentation"""

        if df is None or df.empty:
            print("❌ No data for visualization")
            return

        # Set style for presentation
        plt.style.use("default")
        sns.set_palette("husl")
        plt.rcParams.update(
            {"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12}
        )

        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 16))

        # 1. Experiment Overview (Top row)
        ax1 = plt.subplot(3, 4, 1)
        exp_counts = pd.Series(experiment_summary).apply(lambda x: x["total_runs"])
        exp_counts.plot(kind="bar", ax=ax1, color="skyblue")
        ax1.set_title("Total Runs per Experiment")
        ax1.set_ylabel("Number of Runs")
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # 2. Success Rate
        ax2 = plt.subplot(3, 4, 2)
        success_rates = pd.Series(experiment_summary).apply(
            lambda x: (
                x["finished_runs"] / x["total_runs"] * 100 if x["total_runs"] > 0 else 0
            )
        )
        success_rates.plot(kind="bar", ax=ax2, color="lightgreen")
        ax2.set_title("Success Rate by Experiment (%)")
        ax2.set_ylabel("Success Rate (%)")
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        # 3. Performance Distribution
        ax3 = plt.subplot(3, 4, 3)
        df.boxplot(column="test_accuracy", by="experiment", ax=ax3)
        ax3.set_title("Accuracy Distribution by Experiment")
        ax3.set_ylabel("Test Accuracy")
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

        # 4. Top-K Accuracy Comparison
        ax4 = plt.subplot(3, 4, 4)
        topk_data = df[
            ["test_accuracy", "test_top3_accuracy", "test_top5_accuracy"]
        ].mean()
        topk_data.plot(kind="bar", ax=ax4, color=["blue", "green", "orange"])
        ax4.set_title("Average Top-K Accuracy")
        ax4.set_ylabel("Accuracy")
        ax4.set_xticklabels(["Top-1", "Top-3", "Top-5"], rotation=0)

        # 5. Performance vs Model Size (Middle row)
        ax5 = plt.subplot(3, 4, 5)
        scatter = ax5.scatter(
            df["parameters_count"],
            df["test_accuracy"],
            c=df["training_time_minutes"],
            cmap="viridis",
            alpha=0.7,
            s=60,
        )
        ax5.set_title("Performance vs Model Size")
        ax5.set_xlabel("Parameters Count")
        ax5.set_ylabel("Test Accuracy")
        ax5.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))
        plt.colorbar(scatter, ax=ax5, label="Training Time (min)")

        # 6. Training Efficiency
        ax6 = plt.subplot(3, 4, 6)
        efficiency = df["test_accuracy"] / (
            df["training_time_minutes"] + 1
        )  # +1 to avoid division by zero
        ax6.scatter(
            df["training_time_minutes"],
            df["test_accuracy"],
            c=efficiency,
            cmap="RdYlGn",
            alpha=0.7,
            s=60,
        )
        ax6.set_title("Training Time vs Accuracy")
        ax6.set_xlabel("Training Time (minutes)")
        ax6.set_ylabel("Test Accuracy")

        # 7. Hyperparameter Impact - Learning Rate
        ax7 = plt.subplot(3, 4, 7)
        if df["learning_rate"].nunique() > 1:
            lr_groups = df.groupby("learning_rate")["test_accuracy"].mean().sort_index()
            lr_groups.plot(kind="bar", ax=ax7, color="coral")
            ax7.set_title("Learning Rate Impact")
            ax7.set_xlabel("Learning Rate")
            ax7.set_ylabel("Mean Accuracy")
        else:
            ax7.text(
                0.5,
                0.5,
                "Single LR Used",
                ha="center",
                va="center",
                transform=ax7.transAxes,
            )
            ax7.set_title("Learning Rate Analysis")

        # 8. Timeline of Experiments
        ax8 = plt.subplot(3, 4, 8)
        df_sorted = df.sort_values("start_time")
        ax8.plot(
            df_sorted["start_time"],
            df_sorted["test_accuracy"],
            "o-",
            alpha=0.7,
            markersize=4,
        )
        ax8.set_title("Performance Progress Over Time")
        ax8.set_xlabel("Date")
        ax8.set_ylabel("Test Accuracy")
        plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45)

        # 9. Best Runs per Experiment (Bottom row)
        ax9 = plt.subplot(3, 4, 9)
        best_per_exp = (
            df.groupby("experiment")["test_accuracy"].max().sort_values(ascending=False)
        )
        best_per_exp.plot(kind="bar", ax=ax9, color="gold")
        ax9.set_title("Best Performance per Experiment")
        ax9.set_ylabel("Best Test Accuracy")
        plt.setp(ax9.xaxis.get_majorticklabels(), rotation=45)

        # 10. Parameter Count Distribution
        ax10 = plt.subplot(3, 4, 10)
        df["parameters_count"].hist(bins=20, ax=ax10, color="lightblue", alpha=0.7)
        ax10.set_title("Model Size Distribution")
        ax10.set_xlabel("Parameters Count")
        ax10.set_ylabel("Frequency")
        ax10.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))

        # 11. Training Duration Distribution
        ax11 = plt.subplot(3, 4, 11)
        df["training_time_minutes"].hist(
            bins=20, ax=ax11, color="lightcoral", alpha=0.7
        )
        ax11.set_title("Training Time Distribution")
        ax11.set_xlabel("Training Time (minutes)")
        ax11.set_ylabel("Frequency")

        # 12. Summary Statistics Table
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis("off")

        # Create summary stats
        summary_stats = [
            ["Total Experiments", f"{len(experiment_summary)}"],
            ["Total Runs", f"{len(df)}"],
            ["Best Accuracy", f"{df['test_accuracy'].max():.4f}"],
            ["Avg Accuracy", f"{df['test_accuracy'].mean():.4f}"],
            ["Best Top-3", f"{df['test_top3_accuracy'].max():.4f}"],
            ["Best Top-5", f"{df['test_top5_accuracy'].max():.4f}"],
            ["Largest Model", f"{df['parameters_count'].max():,} params"],
            ["Longest Training", f"{df['training_time_minutes'].max():.1f} min"],
        ]

        table = ax12.table(
            cellText=summary_stats,
            colLabels=["Metric", "Value"],
            cellLoc="left",
            loc="center",
            colWidths=[0.6, 0.4],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax12.set_title("Summary Statistics", pad=20)

        plt.suptitle(
            "Magnus Chess AI - Comprehensive MLflow Experiment Analysis\\nStockfish Engine + Magnus Fine-tuned Neural Network",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()

        # Save high-quality plot for presentation
        plot_file = self.results_dir / "comprehensive_mlflow_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"\\n📊 Comprehensive visualization saved: {plot_file}")

        # Also save as PDF for presentation
        pdf_file = self.results_dir / "comprehensive_mlflow_analysis.pdf"
        plt.savefig(pdf_file, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"📊 PDF version saved: {pdf_file}")

        plt.show()

    def save_presentation_data(self, df, experiment_summary):
        """Save comprehensive data for presentation"""

        if df is None:
            return

        # Save detailed CSV
        csv_file = self.results_dir / "comprehensive_mlflow_data.csv"
        df.to_csv(csv_file, index=False)
        print(f"💾 Comprehensive data saved: {csv_file}")

        # Save experiment summary
        summary_file = self.results_dir / "experiment_summary.json"
        with open(summary_file, "w") as f:
            json.dump(experiment_summary, f, indent=2)
        print(f"💾 Experiment summary saved: {summary_file}")

        # Create presentation-ready summary
        presentation_summary = {
            "total_experiments": len(experiment_summary),
            "total_runs": len(df),
            "best_accuracy": float(df["test_accuracy"].max()),
            "avg_accuracy": float(df["test_accuracy"].mean()),
            "best_top3_accuracy": float(df["test_top3_accuracy"].max()),
            "best_top5_accuracy": float(df["test_top5_accuracy"].max()),
            "date_range": {
                "start": df["start_time"].min().isoformat(),
                "end": df["start_time"].max().isoformat(),
            },
            "model_size_range": {
                "min": int(df["parameters_count"].min()),
                "max": int(df["parameters_count"].max()),
                "avg": int(df["parameters_count"].mean()),
            },
            "top_performing_experiments": df.groupby("experiment")["test_accuracy"]
            .max()
            .sort_values(ascending=False)
            .head(5)
            .to_dict(),
        }

        presentation_file = self.results_dir / "presentation_summary.json"
        with open(presentation_file, "w") as f:
            json.dump(presentation_summary, f, indent=2)
        print(f"💾 Presentation summary saved: {presentation_file}")


def main():
    """Main function to recover and analyze comprehensive MLflow data"""

    analyzer = ComprehensiveMLflowAnalyzer()

    print("🎯 Magnus Chess AI - Comprehensive MLflow Data Recovery")
    print("=" * 70)

    # Step 1: Backup current data
    print("\\n1️⃣ Backing up current MLflow data...")
    analyzer.backup_current_mlruns()

    # Step 2: Merge archived data
    print("\\n2️⃣ Merging archived experiment data...")
    success = analyzer.merge_archived_data()

    if not success:
        print("❌ Could not merge archived data")
        return

    # Step 3: Comprehensive analysis
    print("\\n3️⃣ Analyzing comprehensive experiment data...")
    df, experiment_summary = analyzer.analyze_comprehensive_data()

    if df is None:
        print("❌ No data to analyze")
        return

    # Step 4: Create presentation visualizations
    print("\\n4️⃣ Creating presentation visualizations...")
    analyzer.create_presentation_visualizations(df, experiment_summary)

    # Step 5: Save presentation data
    print("\\n5️⃣ Saving presentation-ready data...")
    analyzer.save_presentation_data(df, experiment_summary)

    print("\\n✅ Comprehensive MLflow analysis complete!")
    print("🎯 Ready for presentation with full experiment history!")


if __name__ == "__main__":
    main()
