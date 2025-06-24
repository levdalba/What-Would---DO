#!/usr/bin/env python3
"""
Debug the dimension issues in the Stockfish Magnus trainer
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent))

from stockfish_magnus_trainer import StockfishConfig, MagnusStyleModel


def debug_model_dimensions():
    """Debug the model dimension issues"""

    print("🔍 Debugging Model Dimensions")
    print("=" * 50)

    # Create test configuration
    config = StockfishConfig()
    vocab_size = 25
    feature_dim = 28

    print(f"Vocab size: {vocab_size}")
    print(f"Feature dim: {feature_dim}")

    # Create model
    model = MagnusStyleModel(config, vocab_size, feature_dim)

    # Test input dimensions
    batch_size = 8
    position_input = torch.randn(batch_size, 768)  # 64 squares * 12 pieces
    feature_input = torch.randn(batch_size, feature_dim)

    print(f"\nInput shapes:")
    print(f"Position: {position_input.shape}")
    print(f"Features: {feature_input.shape}")

    # Test each component
    print(f"\n🧠 Testing board encoder...")
    board_encoding = model.board_encoder(position_input)
    print(f"Board encoding shape: {board_encoding.shape}")

    print(f"\n🎯 Testing feature encoder...")
    if hasattr(model.feature_encoder, "weight"):
        feature_encoding = model.feature_encoder(feature_input)
        print(f"Feature encoding shape: {feature_encoding.shape}")

        combined = torch.cat([board_encoding, feature_encoding], dim=1)
        print(f"Combined shape: {combined.shape}")
    else:
        print("Feature encoder is Identity")
        combined = board_encoding
        print(f"Combined shape (board only): {combined.shape}")

    print(f"\n🎲 Testing move predictor...")
    try:
        move_logits = model.move_predictor(combined)
        print(f"Move logits shape: {move_logits.shape}")
    except Exception as e:
        print(f"❌ Move predictor failed: {e}")
        print(f"Expected input: {model.move_predictor[0].in_features}")
        print(f"Actual input: {combined.shape[1]}")

    print(f"\n📊 Testing eval predictor...")
    try:
        eval_adjustment = model.eval_adjustment(combined)
        print(f"Eval adjustment shape: {eval_adjustment.shape}")
    except Exception as e:
        print(f"❌ Eval predictor failed: {e}")

    print(f"\n🎉 Testing full forward pass...")
    try:
        move_logits, eval_adjustment = model(position_input, feature_input)
        print(f"✅ Success! Shapes: {move_logits.shape}, {eval_adjustment.shape}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")


if __name__ == "__main__":
    debug_model_dimensions()
