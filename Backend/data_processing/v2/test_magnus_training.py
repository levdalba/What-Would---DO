"""
Simple test script to validate Magnus training approach
This script tests the core functionality without requiring all dependencies
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import chess.pgn
import numpy as np
from pathlib import Path
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_pgn_loading():
    """Test loading and parsing Magnus Carlsen games"""
    pgn_path = Path("../carlsen-games-quarter.pgn")

    if not pgn_path.exists():
        logger.error(f"PGN file not found: {pgn_path}")
        return False

    logger.info(f"Testing PGN loading from {pgn_path}")

    games_found = 0
    magnus_games = 0
    positions_extracted = 0

    try:
        with open(pgn_path, "r") as pgn_file:
            while games_found < 10:  # Test first 10 games
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                games_found += 1
                headers = game.headers

                # Check if Magnus is playing
                if "Carlsen" in headers.get("White", "") or "Carlsen" in headers.get(
                    "Black", ""
                ):
                    magnus_games += 1
                    magnus_color = (
                        chess.WHITE
                        if "Carlsen" in headers.get("White", "")
                        else chess.BLACK
                    )

                    logger.info(
                        f"Game {games_found}: Magnus as {'White' if magnus_color == chess.WHITE else 'Black'}"
                    )
                    logger.info(f"Result: {headers.get('Result', '*')}")

                    # Count Magnus's moves
                    board = game.board()
                    for move in game.mainline_moves():
                        if board.turn == magnus_color:
                            positions_extracted += 1
                        board.push(move)

        logger.info(f"Successfully loaded {games_found} games")
        logger.info(f"Found {magnus_games} games with Magnus")
        logger.info(f"Extracted {positions_extracted} Magnus positions")

        return games_found > 0 and magnus_games > 0

    except Exception as e:
        logger.error(f"Error loading PGN: {e}")
        return False


def test_board_encoding():
    """Test chess board encoding functionality"""
    logger.info("Testing board encoding...")

    try:
        # Create a test position
        board = chess.Board(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        )

        # Simple encoding function (without external dependencies)
        def encode_board_simple(board):
            """Simple board encoding for testing"""
            encoding = np.zeros((8, 8, 12), dtype=np.float32)

            piece_map = {
                (chess.PAWN, chess.WHITE): 0,
                (chess.KNIGHT, chess.WHITE): 1,
                (chess.BISHOP, chess.WHITE): 2,
                (chess.ROOK, chess.WHITE): 3,
                (chess.QUEEN, chess.WHITE): 4,
                (chess.KING, chess.WHITE): 5,
                (chess.PAWN, chess.BLACK): 6,
                (chess.KNIGHT, chess.BLACK): 7,
                (chess.BISHOP, chess.BLACK): 8,
                (chess.ROOK, chess.BLACK): 9,
                (chess.QUEEN, chess.BLACK): 10,
                (chess.KING, chess.BLACK): 11,
            }

            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece:
                    row, col = divmod(square, 8)
                    piece_idx = piece_map[(piece.piece_type, piece.color)]
                    encoding[row, col, piece_idx] = 1.0

            return encoding

        # Test encoding
        encoding = encode_board_simple(board)
        logger.info(f"Board encoding shape: {encoding.shape}")
        logger.info(f"Non-zero elements: {np.sum(encoding > 0)}")

        # Should have 32 pieces (non-zero elements)
        piece_count = np.sum(encoding > 0)
        if piece_count == 32:
            logger.info("✓ Board encoding test passed")
            return True
        else:
            logger.error(f"✗ Expected 32 pieces, got {piece_count}")
            return False

    except Exception as e:
        logger.error(f"Board encoding test failed: {e}")
        return False


def test_move_extraction():
    """Test move extraction from games"""
    logger.info("Testing move extraction...")

    try:
        # Create a simple game
        game_pgn = """
[Event "Test Game"]
[White "Carlsen, Magnus"]
[Black "Opponent"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0
"""

        from io import StringIO

        pgn_io = StringIO(game_pgn)
        game = chess.pgn.read_game(pgn_io)

        magnus_moves = []
        board = game.board()

        for move in game.mainline_moves():
            if board.turn == chess.WHITE:  # Magnus is white
                magnus_moves.append(move.uci())
            board.push(move)

        expected_moves = ["e2e4", "g1f3", "f1b5", "b5a4", "e1g1"]

        logger.info(f"Extracted Magnus moves: {magnus_moves}")
        logger.info(f"Expected moves: {expected_moves}")

        if magnus_moves == expected_moves:
            logger.info("✓ Move extraction test passed")
            return True
        else:
            logger.error("✗ Move extraction test failed")
            return False

    except Exception as e:
        logger.error(f"Move extraction test failed: {e}")
        return False


def test_basic_training_pipeline():
    """Test basic training pipeline without ML dependencies"""
    logger.info("Testing basic training pipeline...")

    try:
        # Simulate training data structure
        positions = [np.random.random((8, 8, 12)) for _ in range(100)]
        features = [np.random.random(15) for _ in range(100)]
        moves = [f"move_{i}" for i in range(100)]
        outcomes = [np.random.choice([0.0, 0.5, 1.0]) for _ in range(100)]

        logger.info(f"Generated {len(positions)} training samples")

        # Test data splitting (simple version)
        train_size = int(0.7 * len(positions))
        val_size = int(0.2 * len(positions))
        test_size = len(positions) - train_size - val_size

        train_data = positions[:train_size]
        val_data = positions[train_size : train_size + val_size]
        test_data = positions[train_size + val_size :]

        logger.info(
            f"Split data: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}"
        )

        # Test move vocabulary creation
        unique_moves = list(set(moves))
        move_to_idx = {move: idx for idx, move in enumerate(unique_moves)}
        vocab_size = len(move_to_idx)

        logger.info(f"Vocabulary size: {vocab_size}")

        if (
            len(train_data) > 0
            and len(val_data) > 0
            and len(test_data) > 0
            and vocab_size > 0
        ):
            logger.info("✓ Basic training pipeline test passed")
            return True
        else:
            logger.error("✗ Basic training pipeline test failed")
            return False

    except Exception as e:
        logger.error(f"Basic training pipeline test failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("Starting Magnus training validation tests...")

    tests = [
        ("PGN Loading", test_pgn_loading),
        ("Board Encoding", test_board_encoding),
        ("Move Extraction", test_move_extraction),
        ("Training Pipeline", test_basic_training_pipeline),
    ]

    results = {}

    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")

        try:
            result = test_func()
            results[test_name] = result
            status = "PASSED" if result else "FAILED"
            logger.info(f"Test {test_name}: {status}")
        except Exception as e:
            results[test_name] = False
            logger.error(f"Test {test_name} crashed: {e}")

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name}: {status}")

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! Ready for Magnus training.")
        logger.info("\nNext steps:")
        logger.info(
            "1. Install training dependencies: pip install -r training_requirements.txt"
        )
        logger.info("2. Run full training: python improved_magnus_training.py")
        logger.info("3. Monitor training progress and adjust hyperparameters")
    else:
        logger.error("❌ Some tests failed. Please fix issues before proceeding.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
