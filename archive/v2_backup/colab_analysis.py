#!/usr/bin/env python3
"""
Performance analysis for Magnus Carlsen training on Google Colab runtimes
"""

import sys
from pathlib import Path

# Add the project directory to path
sys.path.append(str(Path(__file__).parent))

from stockfish_magnus_trainer import StockfishConfig
import chess.pgn


def analyze_colab_runtimes():
    """Analyze performance across different Google Colab runtime configurations"""

    print("🚀 GOOGLE COLAB RUNTIME PERFORMANCE ANALYSIS")
    print("=" * 80)
    print("Magnus Carlsen Style Chess Engine Training")
    print("=" * 80)

    # Hardware configurations for different Colab runtimes
    colab_configs = {
        "CPU": {
            "name": "CPU Runtime (Free)",
            "cpu_cores": 2,
            "cpu_type": "Intel Xeon (2.3 GHz)",
            "ram_gb": 12.7,
            "gpu_cores": 0,
            "gpu_type": "None",
            "disk_gb": 78,
            "cost": "Free",
            "time_limit": "12 hours",
            "parallel_efficiency": 45.0,  # Low due to limited cores
            "optimal_threads": 2,
        },
        "GPU_T4": {
            "name": "GPU Runtime - T4 (Free)",
            "cpu_cores": 2,
            "cpu_type": "Intel Xeon (2.3 GHz)",
            "ram_gb": 12.7,
            "gpu_cores": 2560,
            "gpu_type": "Tesla T4 (16GB)",
            "disk_gb": 78,
            "cost": "Free",
            "time_limit": "12 hours",
            "parallel_efficiency": 45.0,
            "optimal_threads": 2,
        },
        "GPU_V100": {
            "name": "GPU Runtime - V100 (Pro)",
            "cpu_cores": 4,
            "cpu_type": "Intel Xeon (2.3 GHz)",
            "ram_gb": 25.5,
            "gpu_cores": 5120,
            "gpu_type": "Tesla V100 (16GB)",
            "disk_gb": 166,
            "cost": "$9.99/month",
            "time_limit": "24 hours",
            "parallel_efficiency": 60.0,
            "optimal_threads": 3,
        },
        "GPU_A100": {
            "name": "GPU Runtime - A100 (Pro+)",
            "cpu_cores": 4,
            "cpu_type": "Intel Xeon (2.3 GHz)",
            "ram_gb": 83.5,
            "gpu_cores": 6912,
            "gpu_type": "Tesla A100 (40GB)",
            "disk_gb": 166,
            "cost": "$49.99/month",
            "time_limit": "24 hours",
            "parallel_efficiency": 65.0,
            "optimal_threads": 4,
        },
        "TPU_v2": {
            "name": "TPU Runtime - v2 (Free)",
            "cpu_cores": 4,
            "cpu_type": "Intel Xeon (2.3 GHz)",
            "ram_gb": 12.7,
            "gpu_cores": 0,
            "gpu_type": "TPU v2 (8 cores)",
            "disk_gb": 78,
            "cost": "Free",
            "time_limit": "12 hours",
            "parallel_efficiency": 55.0,
            "optimal_threads": 3,
        },
    }

    # Reference performance (M3 Pro benchmark)
    m3_pro_sequential_rate = 2.66  # positions/second
    m3_pro_parallel_rate = 15.00  # positions/second

    # Dataset analysis
    print("📊 DATASET ANALYSIS:")
    print("=" * 50)

    # Quick estimate based on our previous analysis
    estimated_magnus_games = 6615
    avg_positions_per_game = 20
    total_positions = estimated_magnus_games * avg_positions_per_game

    print(f"   Magnus Carlsen games: {estimated_magnus_games:,}")
    print(f"   Estimated positions: {total_positions:,}")
    print(f"   Training epochs: 50")
    print(f"   Analysis time per position: 0.5 seconds")

    print(f"\n⚡ PERFORMANCE COMPARISON:")
    print("=" * 80)
    print(
        f"{'Runtime':<20} {'CPU':<8} {'RAM':<8} {'GPU':<15} {'Cost':<12} {'Time Limit'}"
    )
    print("-" * 80)

    for key, config in colab_configs.items():
        gpu_info = config["gpu_type"] if config["gpu_cores"] > 0 else "None"
        print(
            f"{config['name']:<20} {config['cpu_cores']}c{'':<5} {config['ram_gb']:.1f}GB{'':<2} {gpu_info:<15} {config['cost']:<12} {config['time_limit']}"
        )

    print(f"\n⏱️  TIME ESTIMATES:")
    print("=" * 80)
    print(
        f"{'Runtime':<25} {'Extract':<10} {'Train':<10} {'Total':<10} {'Within Limit?':<15} {'Feasible?'}"
    )
    print("-" * 85)

    results = {}

    for key, config in colab_configs.items():
        # CPU performance estimation
        cpu_performance_factor = (
            config["cpu_cores"] / 12
        )  # Relative to M3 Pro (12 cores)
        memory_factor = min(
            config["ram_gb"] / 18, 1.2
        )  # Memory advantage capped at 20%

        # Parallel processing capabilities
        threads = config["optimal_threads"]
        efficiency = config["parallel_efficiency"] / 100

        # Sequential and parallel rates
        sequential_rate = (
            m3_pro_sequential_rate * cpu_performance_factor * memory_factor
        )
        parallel_speedup = threads * efficiency
        parallel_rate = sequential_rate * parallel_speedup

        # Time calculations for data extraction
        extraction_hours = total_positions / parallel_rate / 3600

        # Training time estimation based on GPU
        if config["gpu_cores"] == 0:  # CPU only
            training_hours = 8.0  # Much slower without GPU
        elif "T4" in config["gpu_type"]:
            training_hours = 3.0  # T4 is decent
        elif "V100" in config["gpu_type"]:
            training_hours = 1.5  # V100 is fast
        elif "A100" in config["gpu_type"]:
            training_hours = 0.8  # A100 is very fast
        elif "TPU" in config["gpu_type"]:
            training_hours = 4.0  # TPU not optimal for this workload
        else:
            training_hours = 6.0  # Default

        total_hours = extraction_hours + training_hours

        # Check if within time limit
        time_limit_hours = float(config["time_limit"].split()[0])
        within_limit = total_hours <= time_limit_hours

        # Feasibility assessment
        if total_hours <= time_limit_hours * 0.8:  # 80% of time limit
            feasible = "✅ Excellent"
        elif within_limit:
            feasible = "⚠️ Tight"
        else:
            feasible = "❌ Too long"

        results[key] = {
            "extraction_hours": extraction_hours,
            "training_hours": training_hours,
            "total_hours": total_hours,
            "within_limit": within_limit,
            "feasible": feasible,
            "parallel_rate": parallel_rate,
        }

        print(
            f"{config['name']:<25} {extraction_hours:.1f}h{'':<6} {training_hours:.1f}h{'':<6} {total_hours:.1f}h{'':<6} {str(within_limit):<15} {feasible}"
        )

    print(f"\n🎯 DETAILED RUNTIME ANALYSIS:")
    print("=" * 80)

    for key, config in colab_configs.items():
        result = results[key]
        print(f"\n📱 {config['name'].upper()}")
        print(
            f"   Hardware: {config['cpu_cores']} CPU cores, {config['ram_gb']}GB RAM, {config['gpu_type']}"
        )
        print(f"   Cost: {config['cost']}")
        print(f"   Time Limit: {config['time_limit']}")
        print(f"   Optimal Threads: {config['optimal_threads']}")
        print(f"   Analysis Rate: {result['parallel_rate']:.1f} positions/second")
        print(f"   Data Extraction: {result['extraction_hours']:.1f} hours")
        print(f"   Model Training: {result['training_hours']:.1f} hours")
        print(f"   Total Time: {result['total_hours']:.1f} hours")
        print(f"   Status: {result['feasible']}")

        # Specific recommendations
        if key == "CPU":
            print(f"   ⚠️  CPU-only is very slow. Consider GPU runtime.")
            print(f"   💡 Reduce dataset size with config.max_games = 1000")

        elif key == "GPU_T4":
            print(f"   ✅ Good free option! T4 GPU accelerates training significantly.")
            print(f"   💡 Perfect for prototyping and initial experiments.")

        elif key == "GPU_V100":
            print(f"   🚀 Excellent balance of performance and cost!")
            print(f"   💡 Recommended for serious development.")

        elif key == "GPU_A100":
            print(f"   🔥 Maximum performance! Fastest training possible.")
            print(f"   💡 Best for production and multiple experiments.")

        elif key == "TPU_v2":
            print(f"   ⚠️  TPU not optimal for Stockfish analysis (CPU-bound).")
            print(f"   💡 Better for pure neural network training tasks.")

    print(f"\n🏆 RECOMMENDATIONS BY USE CASE:")
    print("=" * 50)

    print(f"\n🆓 **FREE TIER (Experimenting)**")
    print(f"   Best Choice: GPU T4 Runtime")
    print(f"   Time: ~{results['GPU_T4']['total_hours']:.1f} hours")
    print(f"   Strategy: Use smaller dataset first (config.max_games = 1000)")
    print(f"   Pros: Free, good GPU acceleration")
    print(f"   Cons: 12-hour limit, may need multiple sessions")

    print(f"\n💰 **COLAB PRO ($9.99/month)**")
    print(f"   Best Choice: GPU V100 Runtime")
    print(f"   Time: ~{results['GPU_V100']['total_hours']:.1f} hours")
    print(f"   Strategy: Full dataset, single session")
    print(f"   Pros: 24-hour limit, faster GPU, more RAM")
    print(f"   Cons: Monthly cost")

    print(f"\n🚀 **COLAB PRO+ ($49.99/month)**")
    print(f"   Best Choice: GPU A100 Runtime")
    print(f"   Time: ~{results['GPU_A100']['total_hours']:.1f} hours")
    print(f"   Strategy: Multiple experiments, hyperparameter tuning")
    print(f"   Pros: Fastest performance, massive RAM")
    print(f"   Cons: Higher cost")

    print(f"\n⚙️  CONFIGURATION RECOMMENDATIONS:")
    print("=" * 50)

    print(f"\n🆓 **For Free Tier (T4 GPU):**")
    print(f"```python")
    print(f"config.max_games = 1000        # Reduced dataset")
    print(f"config.max_threads = 2         # Limited CPU cores")
    print(f"config.batch_size = 128        # Smaller batches")
    print(f"config.num_epochs = 30         # Fewer epochs")
    print(f"config.analysis_time = 0.3     # Faster analysis")
    print(f"```")

    print(f"\n💰 **For Colab Pro (V100 GPU):**")
    print(f"```python")
    print(f"config.max_games = None        # Full dataset")
    print(f"config.max_threads = 3         # More CPU cores")
    print(f"config.batch_size = 256        # Standard batches")
    print(f"config.num_epochs = 50         # Full training")
    print(f"config.analysis_time = 0.5     # Balanced analysis")
    print(f"```")

    print(f"\n🚀 **For Colab Pro+ (A100 GPU):**")
    print(f"```python")
    print(f"config.max_games = None        # Full dataset")
    print(f"config.max_threads = 4         # Maximum CPU cores")
    print(f"config.batch_size = 512        # Large batches (more memory)")
    print(f"config.num_epochs = 50         # Full training")
    print(f"config.analysis_time = 0.5     # Balanced analysis")
    print(f"```")

    print(f"\n💾 MEMORY USAGE ESTIMATES:")
    print("=" * 50)

    for key, config in colab_configs.items():
        threads = config["optimal_threads"]
        thread_memory = 0.256 * threads  # GB per thread
        dataset_memory = 0.4  # GB (estimated)
        model_memory = 1.5 if config["gpu_cores"] > 0 else 0.5  # GB
        total_memory = thread_memory + dataset_memory + model_memory
        utilization = total_memory / config["ram_gb"] * 100

        status = (
            "✅ Good"
            if utilization < 60
            else "⚠️ Tight" if utilization < 80 else "❌ Risk"
        )

        print(
            f"{config['name']:<25} {total_memory:.1f}GB / {config['ram_gb']:.1f}GB ({utilization:.0f}%) {status}"
        )

    print(f"\n🔧 SETUP INSTRUCTIONS FOR COLAB:")
    print("=" * 50)
    print(
        f"""
1. **Install Dependencies:**
   ```bash
   !apt-get update && apt-get install -y stockfish
   !pip install python-chess torch torchvision torchaudio
   !pip install numpy pandas scikit-learn matplotlib seaborn tqdm
   ```

2. **Upload Magnus PGN File:**
   ```python
   from google.colab import files
   uploaded = files.upload()  # Upload carlsen-games.pgn
   ```

3. **Clone Repository:**
   ```bash
   !git clone https://github.com/yourusername/What-Would---DO.git
   %cd What-Would---DO/Backend/data_processing/v2
   ```

4. **Configure Stockfish Path:**
   ```python
   config.stockfish_path = "/usr/games/stockfish"  # Colab default path
   ```

5. **Start Training:**
   ```python
   python stockfish_magnus_trainer.py
   ```
"""
    )

    print(f"\n⚠️  IMPORTANT COLAB LIMITATIONS:")
    print("=" * 50)
    print(
        f"🕒 **Session Timeouts:** Free tier disconnects after 12 hours of inactivity"
    )
    print(f"💾 **Storage:** Files deleted when session ends (download models!)")
    print(f"🔄 **GPU Availability:** Free tier may queue during peak hours")
    print(f"📊 **Resource Limits:** Usage caps apply to prevent abuse")
    print(f"💰 **Costs:** Pro/Pro+ have compute unit limits")

    print(f"\n🎯 FINAL RECOMMENDATION:")
    print("=" * 50)

    best_free = min(
        [k for k in results.keys() if colab_configs[k]["cost"] == "Free"],
        key=lambda x: results[x]["total_hours"],
    )
    best_paid = min(
        [k for k in results.keys() if colab_configs[k]["cost"] != "Free"],
        key=lambda x: results[x]["total_hours"],
    )

    print(f"🆓 **Best Free Option:** {colab_configs[best_free]['name']}")
    print(f"   Time: {results[best_free]['total_hours']:.1f} hours")
    print(f"   Perfect for: Learning, prototyping, small experiments")

    print(f"\n💰 **Best Paid Option:** {colab_configs[best_paid]['name']}")
    print(f"   Time: {results[best_paid]['total_hours']:.1f} hours")
    print(f"   Perfect for: Production training, multiple experiments")

    print(f"\n🎮 **Compared to Local Hardware:**")
    print(f"   M3 Pro (benchmark): ~4.5 hours")
    print(f"   M2 Max (estimated): ~3.1 hours")
    print(f"   Best Colab: ~{results[best_paid]['total_hours']:.1f} hours")

    if results[best_paid]["total_hours"] < 4.5:
        print(f"   🏆 Colab Pro+ A100 is FASTER than local M3 Pro!")
    else:
        print(f"   🏠 Local hardware is faster for this specific task")


if __name__ == "__main__":
    analyze_colab_runtimes()
