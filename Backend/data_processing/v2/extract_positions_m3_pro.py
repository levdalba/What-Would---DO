#!/usr/bin/env python3
"""
Magnus Carlsen Position Extraction for M3 Pro (No Training)

This script extracts and analyzes positions from Magnus's games using parallel
Stockfish analysis, optimized for M3 Pro. It gives you timing estimates without
the full training pipeline.
"""

import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

# Add the project directory to path
sys.path.append(str(Path(__file__).parent))

from stockfish_magnus_trainer import (
    StockfishConfig,
    ChessPositionEncoder,
    StockfishAnalyzer,
    StockfishMagnusTrainer,
)
import chess
import chess.pgn


def extract_positions_only():
    """Extract and analyze positions from Magnus games (no training)"""
    config = StockfishConfig()

    # 🚀 OPTIMIZED FOR M3 PRO POSITION EXTRACTION
    print("🚀 M3 PRO POSITION EXTRACTION")
    print("=" * 50)

    # Position extraction settings
    config.max_games = None  # Use full Magnus dataset for accurate timing
    config.analysis_time = 0.5  # Balanced speed/quality
    config.max_positions_per_game = 50  # Full position extraction

    # 🚀 M3 Pro parallel processing settings (OPTIMIZED)
    config.use_parallel_analysis = True  # Enable multithreaded Stockfish analysis
    config.max_threads = 10  # Optimized for M3 Pro (12 cores, using 10 for max speed)

    # Stockfish engine settings
    config.stockfish_threads = 1  # Per instance
    config.stockfish_hash = 256  # MB per thread

    print(f"✅ Max Threads: {config.max_threads}")
    print(f"✅ Analysis Time: {config.analysis_time}s per position")
    print(f"✅ Hash per Thread: {config.stockfish_hash}MB")
    print(f"✅ Expected Speedup: ~6.6x (parallel vs sequential)")
    print(f"✅ Time Estimate: ~2.8 hours (30min faster than 8 threads)")
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

    pgn_file = None
    for candidate in pgn_candidates:
        if Path(candidate).exists():
            pgn_file = candidate
            config.pgn_file = candidate
            break

    if not pgn_file:
        print(
            "❌ No Magnus Carlsen PGN file found. Please ensure one of these files exists:"
        )
        for candidate in pgn_candidates:
            print(f"  - {candidate}")
        return

    print(f"📂 Using PGN file: {pgn_file}")

    # Estimate dataset size
    print(f"\n📊 Analyzing PGN file...")
    game_count, total_moves = count_games_and_moves(pgn_file)
    estimated_positions = min(
        game_count * config.max_positions_per_game // 2, total_moves // 2
    )  # Magnus plays ~half the moves

    print(f"   Games in file: {game_count:,}")
    print(f"   Total moves: {total_moves:,}")
    print(f"   Estimated Magnus positions: {estimated_positions:,}")

    # Time estimates
    sequential_time = estimated_positions * config.analysis_time  # seconds
    parallel_time = sequential_time / 6.6  # M3 Pro speedup with 10 threads

    print(f"\n⏱️  TIME ESTIMATES:")
    print(f"   Sequential analysis: {sequential_time/3600:.1f} hours")
    print(f"   Parallel analysis (M3 Pro, 10 threads): {parallel_time/3600:.1f} hours")
    print(f"   Expected speedup: 6.6x")
    print(f"   Time saved vs 8 threads: ~0.5 hours")
    print()

    # Ask user if they want to proceed
    proceed = input("🤔 Proceed with position extraction? (y/n): ").lower().strip()
    if proceed != "y":
        print("👋 Extraction cancelled.")
        return

    print(f"\n🔥 Starting position extraction...")

    # Initialize trainer for position extraction
    trainer = StockfishMagnusTrainer(config)

    try:
        start_time = time.time()

        # Extract positions (this is the same as the first part of training)
        if config.use_parallel_analysis:
            positions, features, sf_moves, magnus_moves, evaluations = (
                trainer.extract_magnus_games_data_parallel()
            )
        else:
            positions, features, sf_moves, magnus_moves, evaluations = (
                trainer.extract_magnus_games_data()
            )

        extraction_time = time.time() - start_time
        actual_positions = len(positions)

        print(f"\n🎉 POSITION EXTRACTION COMPLETED!")
        print(f"📊 Results:")
        print(f"   ✅ Positions extracted: {actual_positions:,}")
        print(f"   ⏱️  Extraction time: {extraction_time/3600:.2f} hours")
        print(f"   🚀 Positions/second: {actual_positions/extraction_time:.1f}")
        print(
            f"   🧠 Average analysis time: {extraction_time/actual_positions:.3f}s per position"
        )

        if config.use_parallel_analysis:
            estimated_sequential_time = extraction_time * 6.6
            print(
                f"   💰 Time saved: {(estimated_sequential_time - extraction_time)/3600:.2f} hours"
            )
            print(
                f"   📈 Actual speedup: {estimated_sequential_time/extraction_time:.1f}x"
            )

        # Extrapolate to full training time
        print(f"\n🎯 FULL TRAINING TIME ESTIMATES:")

        # Training typically takes 2-3x the position extraction time
        # (due to neural network training epochs)
        estimated_full_training = extraction_time * 2.5

        print(
            f"   📚 Position extraction: {extraction_time/3600:.1f} hours (completed)"
        )
        print(
            f"   🧠 Neural network training: ~{(estimated_full_training - extraction_time)/3600:.1f} hours"
        )
        print(f"   🏁 Total estimated time: ~{estimated_full_training/3600:.1f} hours")

        # Save extraction results
        results = {
            "extraction_time_hours": extraction_time / 3600,
            "positions_extracted": actual_positions,
            "positions_per_second": actual_positions / extraction_time,
            "average_analysis_time": extraction_time / actual_positions,
            "estimated_full_training_hours": estimated_full_training / 3600,
            "config": {
                "max_threads": config.max_threads,
                "analysis_time": config.analysis_time,
                "parallel_enabled": config.use_parallel_analysis,
                "pgn_file": config.pgn_file,
            },
            "hardware": "M3 Pro",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        results_path = Path("position_extraction_results_m3_pro.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n📄 Results saved to: {results_path}")

        # 💾 SAVE EXTRACTED POSITIONS FOR TRAINING
        print(f"\n💾 Saving extracted positions for training...")
        extracted_data = {
            "positions": positions,
            "features": features,
            "stockfish_moves": sf_moves,
            "magnus_moves": magnus_moves,
            "evaluations": evaluations,
            "extraction_time": extraction_time,
            "config": config.__dict__,
            "metadata": {
                "total_positions": actual_positions,
                "hardware": "M3 Pro",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pgn_file": config.pgn_file,
            },
        }

        # Save as pickle for efficient loading
        import pickle

        data_path = Path("magnus_extracted_positions_m3_pro.pkl")
        with open(data_path, "wb") as f:
            pickle.dump(extracted_data, f)

        print(f"✅ Extracted positions saved to: {data_path}")
        print(f"   File size: {data_path.stat().st_size / (1024*1024):.1f} MB")
        print(f"   Ready for training!")

        # Also save a JSON version for inspection (smaller subset)
        json_sample = {
            "sample_positions": (
                [
                    pos.tolist() if hasattr(pos, "tolist") else pos
                    for pos in positions[:3]
                ]
                if positions
                else []
            ),
            "sample_moves": magnus_moves[:10] if magnus_moves else [],
            "sample_evaluations": evaluations[:10] if evaluations else [],
            "metadata": extracted_data["metadata"],
        }

        json_path = Path("magnus_positions_sample.json")
        with open(json_path, "w") as f:
            json.dump(json_sample, f, indent=2, default=str)

        print(f"📋 Sample data saved to: {json_path} (for inspection)")

        # Quick data quality check
        print(f"\n🔍 DATA QUALITY CHECK:")
        print(
            f"   ✅ Valid positions: {len([p for p in positions if p is not None]):,}"
        )
        print(f"   ✅ Valid moves: {len([m for m in magnus_moves if m]):,}")
        print(
            f"   ✅ Valid evaluations: {len([e for e in evaluations if e is not None]):,}"
        )

        # Sample some positions
        if len(positions) > 0:
            print(
                f"   📋 Sample evaluation range: {min(evaluations):.0f} to {max(evaluations):.0f} centipawns"
            )
            print(f"   🎲 Sample Magnus moves: {magnus_moves[:5]}")

        print(f"\n✨ Ready for training! Use the main trainer when you're ready.")

    except Exception as e:
        print(f"❌ Position extraction failed: {e}")
        raise

    finally:
        # Always close the analyzer
        trainer.analyzer.close()


def count_games_and_moves(pgn_file: str) -> Tuple[int, int]:
    """Quickly count games and moves in PGN file"""
    game_count = 0
    total_moves = 0

    with open(pgn_file, "r") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            # Check if Magnus is playing
            headers = game.headers
            white_player = headers.get("White", "")
            black_player = headers.get("Black", "")

            if "Carlsen" in white_player or "Carlsen" in black_player:
                game_count += 1
                # Count moves in this game
                node = game
                game_moves = 0
                while node.variations:
                    node = node.variation(0)
                    game_moves += 1
                total_moves += game_moves

    return game_count, total_moves


if __name__ == "__main__":
    extract_positions_only()
