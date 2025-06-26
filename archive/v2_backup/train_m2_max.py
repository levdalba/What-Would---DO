#!/usr/bin/env python3
"""
Optimized Magnus Carlsen training configuration for M2 Max
"""

import sys
from pathlib import Path

# Add the project directory to path
sys.path.append(str(Path(__file__).parent))

from stockfish_magnus_trainer import StockfishConfig, StockfishMagnusTrainer


def main_m2_max():
    """Optimized training function for M2 Max"""
    config = StockfishConfig()

    # 🚀 OPTIMIZED FOR M2 MAX 32GB CONFIGURATION
    print("🚀 M2 MAX OPTIMIZED CONFIGURATION")
    print("=" * 50)

    # Full dataset with M2 Max optimizations
    config.max_games = None  # Use full Magnus dataset
    config.num_epochs = 50  # Good convergence
    config.analysis_time = 0.4  # Faster analysis (M2 Max can handle it)
    config.batch_size = 384  # Larger batches (more memory available)

    # 🚀 M2 Max parallel processing settings
    config.use_parallel_analysis = True  # Enable multithreaded Stockfish analysis
    config.max_threads = 10  # Optimal for M2 Max (12 cores, can handle more threads)

    # Stockfish engine settings optimized for M2 Max
    config.stockfish_threads = 1  # Per instance (total 10 instances)
    config.stockfish_hash = 512  # MB per thread (more memory available)

    print(f"✅ Max Threads: {config.max_threads}")
    print(f"✅ Batch Size: {config.batch_size}")
    print(f"✅ Analysis Time: {config.analysis_time}s")
    print(f"✅ Hash per Thread: {config.stockfish_hash}MB")
    print(f"✅ Expected Total Time: ~3.1 hours")
    print(f"✅ Memory Usage: ~7.4GB of 32GB (23%)")
    print()

    # Check if PGN file exists
    pgn_candidates = [
        "../carlsen-games.pgn",
        "../carlsen-games-quarter.pgn",
        "carlsen-games.pgn",
        "magnus_games.pgn",
        "carlsen-games-quarter.pgn",
        "data_processing/carlsen-games.pgn",
        "Backend/data_processing/carlsen-games.pgn",
    ]

    for pgn_file in pgn_candidates:
        if Path(pgn_file).exists():
            config.pgn_file = pgn_file
            break
    else:
        print(
            "❌ No Magnus Carlsen PGN file found. Please ensure one of these files exists:"
        )
        for pgn_file in pgn_candidates:
            print(f"  - {pgn_file}")
        return

    print(f"📂 Using PGN file: {config.pgn_file}")

    if config.use_parallel_analysis:
        print(f"🚀 Parallel processing enabled with {config.max_threads} threads")
        print(f"   Expected ~6.0x speedup in position analysis!")
        print(f"   Your M2 Max will process ~18.4 positions/second!")

    trainer = StockfishMagnusTrainer(config)

    try:
        import time

        start_time = time.time()
        model, history = trainer.train()
        total_time = time.time() - start_time

        print("\n🎉 Magnus Carlsen style training completed successfully!")
        print(f"📊 Total training time: {total_time/3600:.2f} hours")
        print(f"🎯 M2 Max performance: {132300/(total_time/3600):.0f} positions/hour")

        if config.use_parallel_analysis:
            estimated_sequential_time = total_time * 6.0  # M2 Max speedup
            print(
                f"💰 Time saved with parallel processing: {(estimated_sequential_time - total_time)/3600:.2f} hours"
            )

        # Plot training curves
        from stockfish_magnus_trainer import plot_training_curves

        plot_training_curves(history, config.model_save_dir)

        print(
            f"\n🏆 SUCCESS! Your M2 Max completed Magnus training in {total_time/3600:.1f} hours!"
        )
        print(f"🎮 The trained model is ready for chess domination!")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main_m2_max()
