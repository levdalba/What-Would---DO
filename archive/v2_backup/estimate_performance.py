#!/usr/bin/env python3
"""
Performance estimation for full Magnus dataset with parallel processing
"""

import sys
from pathlib import Path

# Add the project directory to path
sys.path.append(str(Path(__file__).parent))

from stockfish_magnus_trainer import StockfishConfig
import chess.pgn


def estimate_full_dataset_performance():
    """Estimate performance improvement for full Magnus dataset"""

    print("📈 PERFORMANCE ESTIMATION: Full Magnus Dataset")
    print("=" * 70)

    # Count games in full dataset
    pgn_path = Path("../carlsen-games.pgn")
    if not pgn_path.exists():
        print("❌ Full dataset not found. Using quarter dataset for estimation...")
        pgn_path = Path("../carlsen-games-quarter.pgn")

    print(f"📂 Analyzing dataset: {pgn_path}")

    # Count total games
    total_games = 0
    magnus_games = 0

    with open(pgn_path, "r") as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            total_games += 1

            # Check if Magnus is playing
            headers = game.headers
            white_player = headers.get("White", "")
            black_player = headers.get("Black", "")

            if "Carlsen" in white_player or "Carlsen" in black_player:
                magnus_games += 1

    print(f"📊 Dataset Analysis:")
    print(f"   Total games: {total_games:,}")
    print(f"   Magnus games: {magnus_games:,}")
    print(f"   Magnus ratio: {magnus_games/total_games*100:.1f}%")

    # Performance estimates based on benchmark
    sequential_rate = 2.66  # positions/second from benchmark
    parallel_rate = 15.00  # positions/second from benchmark
    speedup = 5.64

    # Estimate positions (conservative)
    avg_positions_per_game = 20  # Conservative estimate
    total_positions = magnus_games * avg_positions_per_game

    print(f"\n⏱️  Performance Estimates:")
    print(f"   Estimated positions to analyze: {total_positions:,}")

    # Time estimates
    sequential_time_hours = total_positions / sequential_rate / 3600
    parallel_time_hours = total_positions / parallel_rate / 3600
    time_saved_hours = sequential_time_hours - parallel_time_hours

    print(f"\n🔄 Sequential Processing:")
    print(
        f"   Estimated time: {sequential_time_hours:.1f} hours ({sequential_time_hours/24:.1f} days)"
    )
    print(f"   Rate: {sequential_rate:.2f} positions/second")

    print(f"\n🚀 Parallel Processing (8 threads):")
    print(
        f"   Estimated time: {parallel_time_hours:.1f} hours ({parallel_time_hours/24:.1f} days)"
    )
    print(f"   Rate: {parallel_rate:.2f} positions/second")
    print(f"   Speedup: {speedup:.1f}x faster")
    print(f"   Efficiency: {speedup/8*100:.1f}% (vs 8 threads)")

    print(f"\n💰 Time Savings:")
    print(
        f"   Time saved: {time_saved_hours:.1f} hours ({time_saved_hours/24:.1f} days)"
    )
    print(f"   Reduction: {(1-parallel_time_hours/sequential_time_hours)*100:.1f}%")

    # Memory usage estimates
    print(f"\n💾 Resource Usage (M3 Pro):")
    print(f"   CPU cores used: 8 of 12 (67%)")
    print(f"   Threads per core: 1 (optimal for CPU-bound tasks)")
    print(f"   Memory per thread: ~256MB (Stockfish hash)")
    print(f"   Total memory: ~2GB (plus dataset)")
    print(f"   Remaining cores: 4 (for system + training)")

    # Training estimates
    training_time_estimate = 2.0  # hours, based on previous runs
    total_time_sequential = sequential_time_hours + training_time_estimate
    total_time_parallel = parallel_time_hours + training_time_estimate

    print(f"\n🎯 Total Project Time Estimate:")
    print(
        f"   Sequential total: {total_time_sequential:.1f} hours ({total_time_sequential/24:.1f} days)"
    )
    print(
        f"   Parallel total: {total_time_parallel:.1f} hours ({total_time_parallel/24:.1f} days)"
    )
    print(f"   Project speedup: {total_time_sequential/total_time_parallel:.1f}x")

    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    print(f"   ✅ Use parallel processing for {speedup:.1f}x speedup")
    print(f"   ✅ Run overnight: ~{parallel_time_hours:.1f} hours for data extraction")
    print(f"   ✅ Monitor memory usage during peak load")
    print(f"   ✅ Keep analysis_time=0.5s for quality/speed balance")
    print(f"   ✅ Consider running on full dataset - very feasible!")

    if "quarter" in str(pgn_path):
        full_estimate = parallel_time_hours * 4
        print(f"\n📏 Full Dataset Estimate (if using quarter dataset):")
        print(
            f"   Full dataset parallel time: ~{full_estimate:.1f} hours ({full_estimate/24:.1f} days)"
        )


if __name__ == "__main__":
    estimate_full_dataset_performance()
