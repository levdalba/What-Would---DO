#!/usr/bin/env python3
"""
Test script to benchmark parallel vs sequential Stockfish analysis
"""

import sys
import time
from pathlib import Path

# Add the project directory to path
sys.path.append(str(Path(__file__).parent))

from stockfish_magnus_trainer import StockfishConfig, StockfishMagnusTrainer


def benchmark_extraction_methods():
    """Benchmark parallel vs sequential data extraction"""

    # Test with a small subset first
    config = StockfishConfig(
        pgn_file="../carlsen-games-quarter.pgn",  # Smaller file for testing
        max_games=10,  # Limit to 10 games for quick test
        max_positions_per_game=20,  # Limit positions per game
        analysis_time=0.3,  # Faster analysis for benchmarking
        analysis_depth=15,
        max_threads=8,  # Use 8 threads for M3 Pro
        device="mps",
    )

    print("=" * 60)
    print("BENCHMARKING: Sequential vs Parallel Stockfish Analysis")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - PGN file: {config.pgn_file}")
    print(f"  - Max games: {config.max_games}")
    print(f"  - Max positions per game: {config.max_positions_per_game}")
    print(f"  - Analysis time: {config.analysis_time}s")
    print(f"  - Analysis depth: {config.analysis_depth}")
    print(f"  - Max threads: {config.max_threads}")
    print(f"  - Device: {config.device}")
    print()

    trainer = StockfishMagnusTrainer(config)

    # Test sequential processing
    print("🔄 Testing SEQUENTIAL processing...")
    config.use_parallel_analysis = False
    start_time = time.time()

    try:
        positions_seq, features_seq, sf_moves_seq, magnus_moves_seq, evals_seq = (
            trainer.extract_magnus_games_data()
        )
        sequential_time = time.time() - start_time
        sequential_positions = len(positions_seq)

        print(f"✅ Sequential completed!")
        print(f"   Time: {sequential_time:.2f} seconds")
        print(f"   Positions extracted: {sequential_positions}")
        print(f"   Rate: {sequential_positions/sequential_time:.2f} positions/second")

    except Exception as e:
        print(f"❌ Sequential failed: {e}")
        sequential_time = float("inf")
        sequential_positions = 0

    print()

    # Test parallel processing
    print("🚀 Testing PARALLEL processing...")
    config.use_parallel_analysis = True
    start_time = time.time()

    try:
        positions_par, features_par, sf_moves_par, magnus_moves_par, evals_par = (
            trainer.extract_magnus_games_data_parallel()
        )
        parallel_time = time.time() - start_time
        parallel_positions = len(positions_par)

        print(f"✅ Parallel completed!")
        print(f"   Time: {parallel_time:.2f} seconds")
        print(f"   Positions extracted: {parallel_positions}")
        print(f"   Rate: {parallel_positions/parallel_time:.2f} positions/second")

        # Calculate speedup
        if sequential_time != float("inf") and parallel_time > 0:
            speedup = sequential_time / parallel_time
            print(f"   🎯 SPEEDUP: {speedup:.2f}x faster!")

    except Exception as e:
        print(f"❌ Parallel failed: {e}")
        parallel_time = float("inf")
        parallel_positions = 0

    print()
    print("=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    if sequential_time != float("inf"):
        print(f"Sequential: {sequential_time:.2f}s ({sequential_positions} positions)")
    else:
        print(f"Sequential: FAILED")

    if parallel_time != float("inf"):
        print(f"Parallel:   {parallel_time:.2f}s ({parallel_positions} positions)")
        if sequential_time != float("inf"):
            speedup = sequential_time / parallel_time
            print(f"Speedup:    {speedup:.2f}x")
            print(
                f"Efficiency: {speedup/config.max_threads*100:.1f}% (vs {config.max_threads} threads)"
            )
    else:
        print(f"Parallel: FAILED")

    print()

    # Data validation
    if (
        sequential_time != float("inf")
        and parallel_time != float("inf")
        and sequential_positions > 0
        and parallel_positions > 0
    ):

        print("🔍 DATA VALIDATION")
        if sequential_positions == parallel_positions:
            print(f"✅ Same number of positions extracted: {sequential_positions}")
        else:
            print(
                f"⚠️  Different position counts: seq={sequential_positions}, par={parallel_positions}"
            )

        # Quick spot check on a few moves
        if len(sf_moves_seq) > 0 and len(sf_moves_par) > 0:
            check_indices = [
                0,
                min(5, len(sf_moves_seq) - 1),
                min(10, len(sf_moves_seq) - 1),
            ]
            all_match = True
            for i in check_indices:
                if i < len(sf_moves_seq) and i < len(sf_moves_par):
                    if sf_moves_seq[i] != sf_moves_par[i]:
                        all_match = False
                        break

            if all_match:
                print(f"✅ Spot check: Stockfish moves match at test indices")
            else:
                print(
                    f"⚠️  Spot check: Some Stockfish moves differ (may be normal due to analysis variations)"
                )

    print("\n🏁 Benchmark complete!")


if __name__ == "__main__":
    benchmark_extraction_methods()
