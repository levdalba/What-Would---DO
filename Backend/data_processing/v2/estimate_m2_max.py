#!/usr/bin/env python3
"""
Performance estimation for M2 Max vs M3 Pro configurations
"""

import sys
from pathlib import Path

# Add the project directory to path
sys.path.append(str(Path(__file__).parent))

from stockfish_magnus_trainer import StockfishConfig
import chess.pgn


def estimate_m2_max_performance():
    """Estimate performance for M2 Max 32GB vs M3 Pro configurations"""

    print("🚀 PERFORMANCE COMPARISON: M2 Max vs M3 Pro")
    print("=" * 80)

    # Hardware configurations
    m3_pro_config = {
        "name": 'M3 Pro 16" (Current Benchmark)',
        "cpu_cores": 12,
        "gpu_cores": 18,
        "ram_gb": 18,
        "memory_bandwidth": "150 GB/s",
        "cpu_base_freq": "~4.0 GHz",
        "parallel_efficiency": 70.5,  # From benchmark
        "optimal_threads": 8,
    }

    m2_max_config = {
        "name": "M2 Max (User's System)",
        "cpu_cores": 12,
        "gpu_cores": 32,
        "ram_gb": 32,
        "memory_bandwidth": "400 GB/s",
        "cpu_base_freq": "~3.7 GHz",
        "parallel_efficiency": 75.0,  # Estimated (more memory = better efficiency)
        "optimal_threads": 10,  # Can handle more threads with more memory
    }

    print("🖥️  HARDWARE COMPARISON:")
    print("=" * 50)
    print(f"{'Specification':<20} {'M3 Pro':<15} {'M2 Max':<15} {'Advantage'}")
    print("-" * 65)
    print(
        f"{'CPU Cores':<20} {m3_pro_config['cpu_cores']:<15} {m2_max_config['cpu_cores']:<15} {'Same'}"
    )
    print(
        f"{'GPU Cores':<20} {m3_pro_config['gpu_cores']:<15} {m2_max_config['gpu_cores']:<15} {'M2 Max +78%'}"
    )
    print(
        f"{'RAM':<20} {m3_pro_config['ram_gb']}GB{'':<10} {m2_max_config['ram_gb']}GB{'':<10} {'M2 Max +78%'}"
    )
    print(
        f"{'Memory B/W':<20} {m3_pro_config['memory_bandwidth']:<15} {m2_max_config['memory_bandwidth']:<15} {'M2 Max +167%'}"
    )
    print(
        f"{'CPU Freq':<20} {m3_pro_config['cpu_base_freq']:<15} {m2_max_config['cpu_base_freq']:<15} {'M3 Pro +8%'}"
    )

    # Count games in dataset
    pgn_path = Path("../carlsen-games.pgn")
    if not pgn_path.exists():
        print("\n❌ Full dataset not found. Using quarter dataset for estimation...")
        pgn_path = Path("../carlsen-games-quarter.pgn")
        is_quarter = True
    else:
        is_quarter = False

    # Quick count (sample first 100 games)
    magnus_games = 0
    sample_games = 0

    with open(pgn_path, "r") as pgn_file:
        while sample_games < 100:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            sample_games += 1
            headers = game.headers
            white_player = headers.get("White", "")
            black_player = headers.get("Black", "")

            if "Carlsen" in white_player or "Carlsen" in black_player:
                magnus_games += 1

    # Estimate total Magnus games
    if is_quarter:
        estimated_total_games = (magnus_games / sample_games) * 1658 * 4  # Quarter * 4
    else:
        estimated_total_games = (magnus_games / sample_games) * 6615  # Full dataset

    avg_positions_per_game = 20
    total_positions = int(estimated_total_games * avg_positions_per_game)

    print(f"\n📊 DATASET ANALYSIS:")
    print(f"   Dataset: {'Quarter (x4)' if is_quarter else 'Full'}")
    print(f"   Estimated Magnus games: {estimated_total_games:,.0f}")
    print(f"   Estimated positions: {total_positions:,}")

    print(f"\n⚡ PERFORMANCE ESTIMATES:")
    print("=" * 50)

    # M3 Pro performance (from benchmark)
    m3_sequential_rate = 2.66  # positions/second
    m3_parallel_rate = 15.00  # positions/second with 8 threads
    m3_speedup = 5.64

    # M2 Max estimates
    # CPU performance: slightly slower base freq (-8%) but compensated by better memory (+167%)
    # Net CPU effect: +15% due to memory bandwidth advantage in CPU-bound tasks
    cpu_performance_factor = 1.15

    # More threads possible due to more memory
    memory_advantage = m2_max_config["ram_gb"] / m3_pro_config["ram_gb"]  # 1.78x
    thread_efficiency_improvement = 1.065  # 6.5% better efficiency with more memory

    # M2 Max estimates
    m2_sequential_rate = m3_sequential_rate * cpu_performance_factor
    m2_parallel_rate = (
        m3_parallel_rate * cpu_performance_factor * thread_efficiency_improvement
    )
    m2_speedup = m2_parallel_rate / m2_sequential_rate
    m2_threads = m2_max_config["optimal_threads"]

    # Time calculations
    m3_sequential_hours = total_positions / m3_sequential_rate / 3600
    m3_parallel_hours = total_positions / m3_parallel_rate / 3600

    m2_sequential_hours = total_positions / m2_sequential_rate / 3600
    m2_parallel_hours = total_positions / m2_parallel_rate / 3600

    # Training estimates (GPU advantage for M2 Max)
    gpu_advantage = m2_max_config["gpu_cores"] / m3_pro_config["gpu_cores"]  # 1.78x
    m3_training_hours = 2.0  # From previous estimates
    m2_training_hours = m3_training_hours / gpu_advantage  # Faster training

    print(f"{'Metric':<25} {'M3 Pro':<15} {'M2 Max':<15} {'Improvement'}")
    print("-" * 70)
    print(
        f"{'Sequential Rate':<25} {m3_sequential_rate:.2f} pos/s{'':<4} {m2_sequential_rate:.2f} pos/s{'':<4} {m2_sequential_rate/m3_sequential_rate:.1f}x"
    )
    print(
        f"{'Parallel Rate':<25} {m3_parallel_rate:.2f} pos/s{'':<4} {m2_parallel_rate:.2f} pos/s{'':<4} {m2_parallel_rate/m3_parallel_rate:.1f}x"
    )
    print(
        f"{'Optimal Threads':<25} {m3_pro_config['optimal_threads']:<15} {m2_threads:<15} {m2_threads/m3_pro_config['optimal_threads']:.1f}x"
    )
    print(
        f"{'Speedup':<25} {m3_speedup:.1f}x{'':<11} {m2_speedup:.1f}x{'':<11} +{((m2_speedup-m3_speedup)/m3_speedup*100):.1f}%"
    )

    print(f"\n⏱️  TIME ESTIMATES:")
    print("=" * 50)
    print(f"{'Phase':<25} {'M3 Pro':<15} {'M2 Max':<15} {'Time Saved'}")
    print("-" * 70)
    print(
        f"{'Data Extraction':<25} {m3_parallel_hours:.1f}h{'':<11} {m2_parallel_hours:.1f}h{'':<11} {m3_parallel_hours-m2_parallel_hours:.1f}h"
    )
    print(
        f"{'Model Training':<25} {m3_training_hours:.1f}h{'':<11} {m2_training_hours:.1f}h{'':<11} {m3_training_hours-m2_training_hours:.1f}h"
    )

    total_m3_hours = m3_parallel_hours + m3_training_hours
    total_m2_hours = m2_parallel_hours + m2_training_hours
    total_saved = total_m3_hours - total_m2_hours
    overall_speedup = total_m3_hours / total_m2_hours

    print(
        f"{'TOTAL PROJECT':<25} {total_m3_hours:.1f}h{'':<11} {total_m2_hours:.1f}h{'':<11} {total_saved:.1f}h"
    )
    print(
        f"{'Overall Speedup':<25} {'':<15} {overall_speedup:.1f}x{'':<11} {((overall_speedup-1)*100):.0f}% faster"
    )

    print(f"\n🎯 M2 MAX ADVANTAGES:")
    print("=" * 50)
    print(
        f"✅ **Memory**: {m2_max_config['ram_gb']}GB vs {m3_pro_config['ram_gb']}GB (+78% more)"
    )
    print(f"✅ **Memory Bandwidth**: 400 GB/s vs 150 GB/s (+167% faster)")
    print(
        f"✅ **GPU Cores**: {m2_max_config['gpu_cores']} vs {m3_pro_config['gpu_cores']} (+78% more)"
    )
    print(
        f"✅ **More Threads**: Can efficiently run {m2_threads} vs {m3_pro_config['optimal_threads']} threads"
    )
    print(
        f"✅ **Better Efficiency**: {m2_max_config['parallel_efficiency']:.1f}% vs {m3_pro_config['parallel_efficiency']:.1f}%"
    )
    print(f"✅ **Training Speed**: {gpu_advantage:.1f}x faster neural network training")

    print(f"\n📈 RECOMMENDED CONFIGURATION FOR M2 MAX:")
    print("=" * 50)
    print(f"```python")
    print(f"config.max_threads = {m2_threads}  # Optimal for M2 Max")
    print(f"config.batch_size = 384     # Larger batches (more memory)")
    print(f"config.stockfish_hash = 512  # Per thread (more memory available)")
    print(f"config.analysis_time = 0.4   # Slightly faster (better CPU)")
    print(f"config.use_parallel_analysis = True")
    print(f"```")

    print(f"\n🚀 PERFORMANCE SUMMARY FOR M2 MAX:")
    print("=" * 50)
    print(
        f"📊 **Data Extraction**: ~{m2_parallel_hours:.1f} hours ({m2_parallel_rate:.1f} pos/sec)"
    )
    print(
        f"🧠 **Model Training**: ~{m2_training_hours:.1f} hours ({gpu_advantage:.1f}x GPU advantage)"
    )
    print(
        f"⏱️  **Total Time**: ~{total_m2_hours:.1f} hours ({total_m2_hours/24:.1f} days)"
    )
    print(f"🎯 **Overall Speedup**: {overall_speedup:.1f}x faster than M3 Pro")
    print(f"💰 **Time Saved**: {total_saved:.1f} hours compared to M3 Pro")

    # Memory usage estimate
    thread_memory = 512 * m2_threads / 1024  # GB
    dataset_memory = total_positions * 4 * 768 / (1024**3)  # GB (rough estimate)
    model_memory = 2.0  # GB
    total_memory_usage = thread_memory + dataset_memory + model_memory

    print(f"\n💾 MEMORY USAGE ESTIMATE:")
    print(f"   Stockfish threads: ~{thread_memory:.1f}GB")
    print(f"   Dataset: ~{dataset_memory:.1f}GB")
    print(f"   Model + Training: ~{model_memory:.1f}GB")
    print(f"   Total: ~{total_memory_usage:.1f}GB of {m2_max_config['ram_gb']}GB")
    print(
        f"   Memory utilization: {total_memory_usage/m2_max_config['ram_gb']*100:.1f}%"
    )

    if total_memory_usage / m2_max_config["ram_gb"] < 0.8:
        print(f"   ✅ Excellent memory headroom!")
    else:
        print(f"   ⚠️  Consider reducing batch size if memory issues occur")

    print(f"\n🏁 CONCLUSION:")
    print(
        f"Your M2 Max will complete the full Magnus training in ~{total_m2_hours:.1f} hours"
    )
    print(
        f"That's {overall_speedup:.1f}x faster than M3 Pro, saving {total_saved:.1f} hours!"
    )
    if total_m2_hours < 8:
        print(
            f"🌙 Perfect for overnight training - start before bed, wake up to Magnus AI!"
        )
    elif total_m2_hours < 24:
        print(f"📅 Easily doable in a single day with some breaks")
    else:
        print(f"📅 Will take about {total_m2_hours/24:.1f} days")


if __name__ == "__main__":
    estimate_m2_max_performance()
