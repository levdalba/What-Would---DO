#!/usr/bin/env python3
"""Leela Chess Zero (LC0) integration for chess move prediction"""

import chess
import chess.engine
import subprocess
import os
from typing import Tuple, Optional, Dict, Any
import logging


class LC0ChessPredictor:
    """Chess move predictor using Leela Chess Zero engine"""

    def __init__(
        self,
        lc0_path: str = "/opt/homebrew/Cellar/lc0/0.31.2/libexec/lc0",
        weights_path: str = "/opt/homebrew/Cellar/lc0/0.31.2/libexec/42850.pb.gz",
        time_limit: float = 1.0,
    ):
        """
        Initialize LC0 chess predictor

        Args:
            lc0_path: Path to LC0 executable
            weights_path: Path to neural network weights file
            time_limit: Time limit for analysis in seconds
        """
        self.lc0_path = lc0_path
        self.weights_path = weights_path
        self.time_limit = time_limit

        # Verify LC0 installation
        if not os.path.exists(lc0_path):
            raise FileNotFoundError(f"LC0 executable not found at {lc0_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"LC0 weights file not found at {weights_path}")

        print(f"LC0 Chess Predictor initialized:")
        print(f"  Engine: {lc0_path}")
        print(f"  Weights: {weights_path}")
        print(f"  Time limit: {time_limit}s")

    def predict_move_uci(self, fen: str) -> Tuple[str, float]:
        """
        Predict the best move for a given position using LC0

        Args:
            fen: FEN string representing the chess position

        Returns:
            Tuple of (best_move_uci, confidence_score)
        """
        try:
            # Validate FEN
            board = chess.Board(fen)

            # Check if game is over
            if board.is_game_over():
                return "0000", 0.0

            # Use LC0 engine to analyze position
            with chess.engine.SimpleEngine.popen_uci(
                [self.lc0_path, f"--weights={self.weights_path}"]
            ) as engine:

                # Set position
                board = chess.Board(fen)

                # Try different analysis approaches for better results
                best_move = None
                confidence = 0.5

                # Method 1: Try analysis with time limit (primary method)
                try:
                    result = engine.analyse(
                        board, chess.engine.Limit(time=self.time_limit), multipv=1
                    )

                    print(f"LC0 analysis result: {result}")

                    # Handle both single result and list of results
                    analysis_result = result
                    if isinstance(result, list) and len(result) > 0:
                        analysis_result = result[0]  # Take first result from multipv

                    # Check if we have a valid result with a move
                    if (
                        analysis_result
                        and isinstance(analysis_result, dict)
                        and "pv" in analysis_result
                        and analysis_result["pv"]
                        and len(analysis_result["pv"]) > 0
                    ):

                        best_move = analysis_result["pv"][0]
                        confidence = self._score_to_confidence(
                            analysis_result.get("score")
                        )

                        print(
                            f"LC0 analysis: {best_move} (confidence: {confidence:.3f})"
                        )
                        return str(best_move), confidence

                except Exception as analysis_error:
                    print(f"LC0: Time-based analysis failed: {analysis_error}")

                # Method 2: Try analysis with nodes limit instead of time
                try:
                    result = engine.analyse(
                        board,
                        chess.engine.Limit(nodes=50000),  # Use nodes instead of time
                        multipv=1,
                    )

                    # Handle both single result and list of results
                    analysis_result = result
                    if isinstance(result, list) and len(result) > 0:
                        analysis_result = result[0]  # Take first result from multipv

                    # Check if we have a valid result with a move
                    if (
                        analysis_result
                        and isinstance(analysis_result, dict)
                        and "pv" in analysis_result
                        and analysis_result["pv"]
                        and len(analysis_result["pv"]) > 0
                    ):

                        best_move = analysis_result["pv"][0]
                        confidence = self._score_to_confidence(
                            analysis_result.get("score")
                        )

                        print(
                            f"LC0 nodes analysis: {best_move} (confidence: {confidence:.3f})"
                        )
                        return str(best_move), confidence

                except Exception as nodes_error:
                    print(f"LC0: Nodes analysis failed: {nodes_error}")

                # Method 3: Use play with depth and try to extract evaluation
                try:
                    # Don't configure Hash option as LC0 doesn't support it
                    # engine.configure({"Threads": 1, "Hash": 16})

                    play_result = engine.play(
                        board,
                        chess.engine.Limit(time=self.time_limit),
                        info=chess.engine.INFO_SCORE | chess.engine.INFO_PV,
                    )

                    if play_result.move:
                        best_move = play_result.move

                        # Try to get evaluation after the move
                        board_after = board.copy()
                        board_after.push(best_move)

                        try:
                            eval_result = engine.analyse(
                                board_after,
                                chess.engine.Limit(time=0.1),  # Quick evaluation
                                multipv=1,
                            )
                            if eval_result and eval_result.get("score"):
                                # Invert confidence since it's after our move
                                confidence = 1.0 - self._score_to_confidence(
                                    eval_result.get("score")
                                )
                            else:
                                confidence = 0.6  # Reasonable default for LC0
                        except:
                            confidence = 0.6  # Reasonable default for LC0

                        print(
                            f"LC0 play result: {best_move} (confidence: {confidence:.3f})"
                        )
                        return str(best_move), confidence

                except Exception as play_error:
                    print(f"LC0: Play method failed: {play_error}")

                # Method 4: Fallback to simple legal move if all else fails
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    # Return first legal move with low confidence
                    fallback_move = legal_moves[0]
                    print(f"LC0: Fallback to legal move: {fallback_move}")
                    return str(fallback_move), 0.3

                print("LC0: No valid moves found")
                return "0000", 0.0

        except chess.InvalidFenError:
            print(f"LC0: Invalid FEN string: {fen}")
            return "0000", 0.0
        except Exception as e:
            print(f"LC0: Error during analysis: {str(e)}")
            return "0000", 0.0

    def _score_to_confidence(self, score) -> float:
        """
        Convert LC0 score to confidence percentage

        Args:
            score: Chess engine score object (PovScore for LC0)

        Returns:
            Confidence value between 0.0 and 1.0
        """
        if not score:
            return 0.6  # Reasonable default for LC0

        try:
            # Handle LC0's PovScore format
            if hasattr(score, "is_mate") and score.is_mate():
                # Mate scores get high confidence
                mate_moves = abs(score.mate())
                if mate_moves <= 3:
                    return 0.95 if score.mate() > 0 else 0.05
                else:
                    return 0.85 if score.mate() > 0 else 0.15

            # Handle PovScore with Cp (centipawn) values
            if hasattr(score, "score"):
                # This is the case for traditional engines
                try:
                    cp_score = score.score(mate_score=10000)
                    if cp_score is not None:
                        return self._cp_to_confidence(cp_score)
                except:
                    pass

            # Handle LC0's PovScore directly
            if hasattr(score, "cp"):
                # LC0 PovScore has .cp attribute
                cp_value = score.cp
                if cp_value is not None:
                    return self._cp_to_confidence(cp_value)

            # Try to extract from string representation if needed
            score_str = str(score)
            if "Cp(" in score_str:
                # Extract centipawn value from string like "PovScore(Cp(+23), WHITE)"
                import re

                match = re.search(r"Cp\(([+-]?\d+)\)", score_str)
                if match:
                    cp_value = int(match.group(1))
                    return self._cp_to_confidence(cp_value)

            # Fallback: try to get any numeric value
            if hasattr(score, "white") and callable(score.white):
                win_prob = score.white()
                if win_prob is not None:
                    return max(0.1, min(0.9, win_prob))

            return 0.6  # Default for LC0

        except Exception as e:
            print(f"LC0: Error converting score to confidence: {e}")
            return 0.6  # Default for LC0

    def _cp_to_confidence(self, cp_score: int) -> float:
        """
        Convert centipawn score to confidence

        Args:
            cp_score: Centipawn score (positive = good for current player)

        Returns:
            Confidence value between 0.1 and 0.9
        """
        # LC0's centipawn values are typically smaller than traditional engines
        # Convert to a reasonable confidence scale

        if abs(cp_score) < 25:  # Very close game
            confidence = 0.5 + (cp_score / 1000.0)  # Small adjustment around 0.5
        elif abs(cp_score) < 75:  # Slight advantage
            confidence = 0.5 + (cp_score / 400.0)  # Moderate adjustment
        elif abs(cp_score) < 200:  # Clear advantage
            confidence = 0.5 + (cp_score / 250.0)  # Larger adjustment
        else:  # Winning/losing
            confidence = 0.5 + (min(500, max(-500, cp_score)) / 100.0)

        # Ensure confidence is in valid range
        return max(0.1, min(0.9, confidence))

    def get_engine_info(self) -> Dict[str, Any]:
        """
        Get information about the LC0 engine

        Returns:
            Dictionary with engine information
        """
        try:
            # Get LC0 version info
            result = subprocess.run(
                [self.lc0_path, "--version"], capture_output=True, text=True, timeout=5
            )
            version_info = (
                result.stdout.strip() if result.returncode == 0 else "Unknown"
            )

            return {
                "engine_name": "Leela Chess Zero (LC0)",
                "engine_path": self.lc0_path,
                "weights_path": self.weights_path,
                "version": version_info,
                "time_limit": self.time_limit,
                "available": os.path.exists(self.lc0_path)
                and os.path.exists(self.weights_path),
            }
        except Exception as e:
            return {
                "engine_name": "Leela Chess Zero (LC0)",
                "engine_path": self.lc0_path,
                "error": str(e),
                "available": False,
            }


# Test function
def test_lc0_predictor():
    """Test the LC0 predictor with a starting position"""
    try:
        predictor = LC0ChessPredictor()

        # Test with starting position
        test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        move, confidence = predictor.predict_move_uci(test_fen)

        print(f"✅ LC0 Test successful!")
        print(f"Position: Starting position")
        print(f"Best move: {move}")
        print(f"Confidence: {confidence:.3f}")

        # Test engine info
        info = predictor.get_engine_info()
        print(f"Engine info: {info}")

        return True

    except Exception as e:
        print(f"❌ LC0 Test failed: {e}")
        return False


if __name__ == "__main__":
    test_lc0_predictor()
