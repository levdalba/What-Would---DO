#!/usr/bin/env python3
"""
Test the trained Magnus model on sample positions
"""

import torch
import chess
import numpy as np
import json
import pickle
from pathlib import Path
import sys

# Add path for imports
sys.path.append(str(Path(__file__).parent))

from stockfish_magnus_trainer import ChessPositionEncoder, MagnusStyleModel


def test_magnus_model():
    """Test the trained Magnus model"""
    print("🧪 Testing Magnus Carlsen Model")
    print("=" * 40)

    # Load the trained model
    model_path = "models/magnus_m3_pro_trained/best_magnus_model.pth"

    if not Path(model_path).exists():
        print("❌ Model file not found!")
        return

    print(f"📂 Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    # Get model info
    dataset_info = checkpoint.get("dataset_info", {})
    vocab_size = dataset_info.get("vocab_size", 1000)
    feature_dim = dataset_info.get("feature_dim", 20)
    move_to_idx = dataset_info.get("move_to_idx", {})
    idx_to_move = {idx: move for move, idx in move_to_idx.items()}

    print(f"   Vocabulary size: {vocab_size}")
    print(f"   Feature dimension: {feature_dim}")

    # Create model with correct dimensions
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Use the actual feature dimension from the checkpoint
    actual_feature_dim = (
        28
        if "feature_encoder.0.weight" in checkpoint["model_state_dict"]
        else feature_dim
    )
    model = MagnusStyleModel(None, vocab_size, actual_feature_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"   Device: {device}")
    print("✅ Model loaded successfully!")

    # Test on some famous positions
    encoder = ChessPositionEncoder()

    test_positions = [
        # Starting position
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        # Sicilian Defense
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
        # Queen's Gambit
        "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2",
    ]

    position_names = ["Starting Position", "Sicilian Defense", "Queen's Gambit"]

    print(f"\n🎯 Testing on sample positions:")

    for i, (fen, name) in enumerate(zip(test_positions, position_names)):
        print(f"\n{i+1}. {name}")
        print(f"   FEN: {fen}")

        try:
            board = chess.Board(fen)

            # Encode position
            position_encoding = encoder.encode_board_for_nn(board)
            position_features = encoder.extract_position_features(board)

            # Convert to tensors
            position_tensor = (
                torch.FloatTensor(position_encoding).unsqueeze(0).to(device)
            )

            # Create features tensor with correct dimension
            feature_names = [
                "turn",
                "move_number",
                "halfmove_clock",
                "white_kingside_castle",
                "white_queenside_castle",
                "black_kingside_castle",
                "black_queenside_castle",
                "en_passant",
                "white_material",
                "black_material",
                "material_imbalance",
                "white_pawns",
                "black_pawns",
                "white_knights",
                "black_knights",
                "white_bishops",
                "black_bishops",
                "white_rooks",
                "black_rooks",
                "white_queens",
                "black_queens",
                "legal_moves",
                "white_king_attackers",
                "black_king_attackers",
                "king_distance",
                "white_center_control",
                "black_center_control",
                "game_phase",
            ]
            feature_array = np.array(
                [position_features.get(fname, 0) for fname in feature_names]
            )
            features_tensor = torch.FloatTensor(feature_array).unsqueeze(0).to(device)

            # Get prediction
            with torch.no_grad():
                move_logits, eval_adjustment = model(position_tensor, features_tensor)

                # Get top 3 predicted moves
                top_probs, top_indices = torch.topk(
                    torch.softmax(move_logits, dim=1), 3
                )

                print(f"   Top 3 predicted moves:")
                for j, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
                    move_str = idx_to_move.get(idx.item(), f"move_{idx.item()}")
                    print(f"     {j+1}. {move_str} ({prob.item():.1%})")

                # Evaluation
                eval_pred = (
                    eval_adjustment[0].item() * 1000
                )  # Convert back to centipawns
                print(f"   Position evaluation: {eval_pred:+.0f} centipawns")

        except Exception as e:
            print(f"   ❌ Error testing position: {e}")

    print(f"\n✅ Model testing completed!")


if __name__ == "__main__":
    test_magnus_model()
