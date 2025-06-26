#!/usr/bin/env python3
"""
Test script for Stockfish Magnus Trainer

This script runs a quick test of the Stockfish Magnus training system
with a small dataset to verify everything works correctly.
"""

import sys
import logging
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent))

from stockfish_magnus_trainer import StockfishConfig, StockfishMagnusTrainer


def test_stockfish_magnus_trainer():
    """Test the Stockfish Magnus training system"""

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    logger.info("🚀 Starting Stockfish Magnus Trainer Test")
    logger.info("=" * 60)

    # Create test configuration
    config = StockfishConfig()
    config.max_games = 5  # Very small test
    config.num_epochs = 3
    config.analysis_time = 0.2  # Fast analysis
    config.max_positions_per_game = 10
    config.batch_size = 8
    config.model_save_dir = "test_magnus_model"

    # Check for PGN files
    pgn_candidates = [
        "../carlsen-games-quarter.pgn",
        "../carlsen-games.pgn",
        "../../data_processing/carlsen-games-quarter.pgn",
        "../../data_processing/carlsen-games.pgn",
        "carlsen-games.pgn",
        "magnus_games.pgn",
    ]

    pgn_found = False
    for pgn_file in pgn_candidates:
        if Path(pgn_file).exists():
            config.pgn_file = pgn_file
            pgn_found = True
            logger.info(f"✅ Found PGN file: {pgn_file}")
            break

    if not pgn_found:
        logger.error("❌ No Magnus PGN file found. Available candidates:")
        for pgn_file in pgn_candidates:
            logger.error(
                f"   - {pgn_file} ({'exists' if Path(pgn_file).exists() else 'not found'})"
            )
        return False

    try:
        # Test trainer initialization
        logger.info("🔧 Initializing trainer...")
        trainer = StockfishMagnusTrainer(config)
        logger.info("✅ Trainer initialized successfully")

        # Test data extraction (just a few positions)
        logger.info("📊 Testing data extraction...")
        positions, features, sf_moves, magnus_moves, evaluations = (
            trainer.extract_magnus_games_data()
        )

        if len(positions) == 0:
            logger.error("❌ No positions extracted. Check PGN file and filters.")
            return False

        logger.info(f"✅ Extracted {len(positions)} positions for testing")
        logger.info(f"   - Sample feature names: {list(features[0].keys())[:5]}...")
        logger.info(f"   - Sample moves: SF={sf_moves[0]}, Magnus={magnus_moves[0]}")
        logger.info(f"   - Sample evaluation: {evaluations[0]}")

        # Test dataset creation
        logger.info("🎯 Testing dataset creation...")
        train_loader, val_loader, test_loader = trainer.create_datasets()
        logger.info(
            f"✅ Datasets created - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}"
        )

        # Test model creation
        logger.info("🧠 Testing model creation...")
        info_path = Path(config.model_save_dir) / "dataset_info.json"
        import json

        with open(info_path, "r") as f:
            dataset_info = json.load(f)

        from stockfish_magnus_trainer import MagnusStyleModel

        model = MagnusStyleModel(
            config, dataset_info["vocab_size"], dataset_info["feature_dim"]
        )
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"✅ Model created with {param_count:,} parameters")

        # Test one training step
        logger.info("🏋️ Testing one training step...")
        import torch
        import torch.nn as nn

        device = torch.device(config.device)
        model = model.to(device)

        # Get one batch
        batch = next(iter(train_loader))
        positions = batch["position"].to(device)
        features = batch["features"].to(device)
        magnus_moves = batch["magnus_move"].squeeze().to(device)

        # Forward pass
        move_logits, eval_adjustment = model(positions, features)
        logger.info(
            f"✅ Forward pass successful - Output shapes: {move_logits.shape}, {eval_adjustment.shape}"
        )

        # Test loss computation
        move_criterion = nn.CrossEntropyLoss()
        move_loss = move_criterion(move_logits, magnus_moves)
        logger.info(
            f"✅ Loss computation successful - Move loss: {move_loss.item():.4f}"
        )

        logger.info(
            "🎉 All tests passed! The Stockfish Magnus trainer is working correctly."
        )
        logger.info("=" * 60)
        logger.info("Next steps:")
        logger.info("1. Run: python stockfish_magnus_trainer.py")
        logger.info("2. Increase max_games for full training")
        logger.info("3. Adjust hyperparameters as needed")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Cleanup
        try:
            trainer.analyzer.close()
        except:
            pass


if __name__ == "__main__":
    success = test_stockfish_magnus_trainer()
    sys.exit(0 if success else 1)
