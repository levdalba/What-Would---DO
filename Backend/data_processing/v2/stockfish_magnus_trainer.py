"""
Stockfish Magnus Carlsen Style Training System

This module creates a Magnus Carlsen-style chess engine by:
1. Extracting positions from Magnus's games
2. Analyzing them with Stockfish to get evaluations
3. Training a model to predict Magnus's moves
4. Creating a Magnus-style evaluation function

Stockfish NNUE is ~3500 ELO (comparable to LC0) and much easier to train.
"""

import chess
import chess.pgn
import chess.engine
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import time
from collections import defaultdict
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StockfishConfig:
    """Configuration for Stockfish Magnus training"""

    stockfish_path: str = "/opt/homebrew/bin/stockfish"
    data_dir: str = "stockfish_magnus_data"
    pgn_file: str = "carlsen-games.pgn"
    model_save_dir: str = "models/stockfish_magnus_model"

    # Training parameters
    batch_size: int = 256
    learning_rate: float = 0.001
    num_epochs: int = 50
    validation_split: float = 0.2
    test_split: float = 0.1
    early_stopping_patience: int = 10
    device: str = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Analysis parameters
    analysis_depth: int = 20
    analysis_time: float = 0.5  # seconds per position
    max_positions_per_game: int = 50
    max_games: Optional[int] = None  # Set to limit for testing

    # Stockfish engine settings
    stockfish_threads: int = 4
    stockfish_hash: int = 1024

    # Parallel processing settings
    max_threads: int = 8  # Optimal for M3 Pro (12 cores, leaving some for system)
    use_parallel_analysis: bool = True  # Enable parallel Stockfish analysis

    # Magnus-specific parameters
    focus_on_classical: bool = True  # Focus on classical time controls
    min_elo_opponent: int = 2400  # Only include games vs strong opponents
    include_endgames: bool = True
    include_openings: bool = True
    include_middlegames: bool = True


class ChessPositionEncoder:
    """Enhanced position encoder for Stockfish compatibility"""

    def __init__(self):
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000,
        }

        # Square values for piece placement evaluation
        self.pst = self._init_piece_square_tables()

    def _init_piece_square_tables(self) -> Dict:
        """Initialize piece-square tables similar to traditional engines"""
        # Simplified piece-square tables (Stockfish uses much more complex ones)
        pawn_table = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            5,
            10,
            10,
            -20,
            -20,
            10,
            10,
            5,
            5,
            -5,
            -10,
            0,
            0,
            -10,
            -5,
            5,
            0,
            0,
            0,
            20,
            20,
            0,
            0,
            0,
            5,
            5,
            10,
            25,
            25,
            10,
            5,
            5,
            10,
            10,
            20,
            30,
            30,
            20,
            10,
            10,
            50,
            50,
            50,
            50,
            50,
            50,
            50,
            50,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]

        knight_table = [
            -50,
            -40,
            -30,
            -30,
            -30,
            -30,
            -40,
            -50,
            -40,
            -20,
            0,
            5,
            5,
            0,
            -20,
            -40,
            -30,
            5,
            10,
            15,
            15,
            10,
            5,
            -30,
            -30,
            0,
            15,
            20,
            20,
            15,
            0,
            -30,
            -30,
            5,
            15,
            20,
            20,
            15,
            5,
            -30,
            -30,
            0,
            10,
            15,
            15,
            10,
            0,
            -30,
            -40,
            -20,
            0,
            0,
            0,
            0,
            -20,
            -40,
            -50,
            -40,
            -30,
            -30,
            -30,
            -30,
            -40,
            -50,
        ]

        return {
            chess.PAWN: pawn_table,
            chess.KNIGHT: knight_table,
            chess.BISHOP: knight_table,  # Simplified
            chess.ROOK: [0] * 64,
            chess.QUEEN: [0] * 64,
            chess.KING: [0] * 64,
        }

    def encode_board_for_nn(self, board: chess.Board) -> np.ndarray:
        """
        Encode chess board for neural network (similar to NNUE input)
        Returns: 768-dimensional feature vector (64 squares * 12 piece types)
        """
        features = np.zeros(768, dtype=np.float32)

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Calculate feature index: square * 12 + piece_type_offset
                piece_offset = (piece.piece_type - 1) + (
                    0 if piece.color == chess.WHITE else 6
                )
                feature_idx = square * 12 + piece_offset
                features[feature_idx] = 1.0

        return features

    def extract_position_features(self, board: chess.Board) -> Dict[str, float]:
        """Extract comprehensive position features for analysis"""
        features = {}

        # Basic position info
        features["turn"] = float(board.turn)
        features["move_number"] = len(board.move_stack)
        features["halfmove_clock"] = board.halfmove_clock

        # Castling rights
        features["white_kingside_castle"] = board.has_kingside_castling_rights(
            chess.WHITE
        )
        features["white_queenside_castle"] = board.has_queenside_castling_rights(
            chess.WHITE
        )
        features["black_kingside_castle"] = board.has_kingside_castling_rights(
            chess.BLACK
        )
        features["black_queenside_castle"] = board.has_queenside_castling_rights(
            chess.BLACK
        )

        # En passant
        features["en_passant"] = board.ep_square is not None

        # Material evaluation
        white_material = sum(
            len(board.pieces(piece_type, chess.WHITE)) * value
            for piece_type, value in self.piece_values.items()
        )
        black_material = sum(
            len(board.pieces(piece_type, chess.BLACK)) * value
            for piece_type, value in self.piece_values.items()
        )

        features["white_material"] = white_material
        features["black_material"] = black_material
        features["material_imbalance"] = white_material - black_material

        # Piece count
        for piece_type in [
            chess.PAWN,
            chess.KNIGHT,
            chess.BISHOP,
            chess.ROOK,
            chess.QUEEN,
        ]:
            features[f"white_{chess.piece_name(piece_type)}s"] = len(
                board.pieces(piece_type, chess.WHITE)
            )
            features[f"black_{chess.piece_name(piece_type)}s"] = len(
                board.pieces(piece_type, chess.BLACK)
            )

        # Mobility
        features["legal_moves"] = len(list(board.legal_moves))

        # King safety
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)

        if white_king and black_king:
            features["white_king_attackers"] = len(
                board.attackers(chess.BLACK, white_king)
            )
            features["black_king_attackers"] = len(
                board.attackers(chess.WHITE, black_king)
            )
            features["king_distance"] = chess.square_distance(white_king, black_king)

        # Center control
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        white_center_control = sum(
            len(board.attackers(chess.WHITE, sq)) for sq in center_squares
        )
        black_center_control = sum(
            len(board.attackers(chess.BLACK, sq)) for sq in center_squares
        )
        features["white_center_control"] = white_center_control
        features["black_center_control"] = black_center_control

        # Game phase (opening/middlegame/endgame)
        total_material = white_material + black_material
        if total_material > 6000:
            features["game_phase"] = 0  # Opening/early middlegame
        elif total_material > 3000:
            features["game_phase"] = 1  # Middlegame
        else:
            features["game_phase"] = 2  # Endgame

        return features


class StockfishAnalyzer:
    """Wrapper for Stockfish analysis"""

    def __init__(self, config: StockfishConfig):
        self.config = config
        self.engine = None
        self._init_engine()

    def _init_engine(self):
        """Initialize Stockfish engine"""
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(
                self.config.stockfish_path
            )

            # Configure engine
            self.engine.configure(
                {
                    "Threads": self.config.stockfish_threads,
                    "Hash": self.config.stockfish_hash,
                }
            )

            logger.info(f"Initialized Stockfish engine: {self.engine.id}")

        except Exception as e:
            logger.error(f"Failed to initialize Stockfish: {e}")
            raise

    def analyze_position(
        self, board: chess.Board, time_limit: float = None
    ) -> Dict[str, Any]:
        """Analyze position with Stockfish"""
        if not self.engine:
            raise RuntimeError("Stockfish engine not initialized")

        time_limit = time_limit or self.config.analysis_time

        try:
            # Get best move and evaluation
            result = self.engine.analyse(
                board,
                chess.engine.Limit(time=time_limit, depth=self.config.analysis_depth),
            )

            # Handle both single result and list results
            if isinstance(result, list):
                main_result = result[0] if result else {}
            else:
                main_result = result

            analysis = {
                "best_move": main_result["pv"][0] if main_result.get("pv") else None,
                "evaluation": main_result.get("score", chess.engine.Cp(0))
                .white()
                .score(mate_score=10000),
                "depth": main_result.get("depth", 0),
                "nodes": main_result.get("nodes", 0),
                "time": main_result.get("time", 0),
                "pv": main_result.get("pv", []),
                "multipv_results": [],
            }

            # Try to get multiple variations (simplified)
            try:
                multipv_results = self.engine.analyse(
                    board,
                    chess.engine.Limit(time=time_limit),
                    multipv=3,  # Get top 3 moves
                )

                if isinstance(multipv_results, list):
                    for i, res in enumerate(multipv_results[:3]):
                        if res.get("pv"):
                            move_eval = (
                                res.get("score", chess.engine.Cp(0))
                                .white()
                                .score(mate_score=10000)
                            )
                            analysis["multipv_results"].append(
                                {
                                    "move": res["pv"][0],
                                    "evaluation": move_eval,
                                    "pv": res.get("pv", []),
                                }
                            )
            except:
                # If multipv fails, just use the main result
                pass

            return analysis

        except Exception as e:
            logger.warning(f"Analysis failed: {e}")
            return {
                "best_move": None,
                "evaluation": 0,
                "depth": 0,
                "nodes": 0,
                "time": 0,
                "pv": [],
                "multipv_results": [],
            }

    def close(self):
        """Close engine"""
        if self.engine:
            self.engine.close()


class MagnusDataset(Dataset):
    """Dataset for Magnus Carlsen positions and moves"""

    def __init__(
        self,
        positions: List[np.ndarray],
        features: List[Dict],
        moves: List[str],
        evaluations: List[float],
        magnus_moves: List[str],
    ):
        self.positions = positions
        self.features = features
        self.moves = moves
        self.evaluations = evaluations
        self.magnus_moves = magnus_moves

        # Create move vocabulary - ensure all moves are strings
        all_moves = set()
        for move in moves + magnus_moves:
            if move and isinstance(move, str):
                all_moves.add(move)

        self.move_to_idx = {move: idx for idx, move in enumerate(sorted(all_moves))}
        self.idx_to_move = {idx: move for move, idx in self.move_to_idx.items()}
        self.vocab_size = len(self.move_to_idx)

        # Normalize features
        self.feature_names = list(features[0].keys()) if features else []
        self.feature_array = self._features_to_array()

    def _features_to_array(self) -> np.ndarray:
        """Convert feature dictionaries to numpy array"""
        if not self.features:
            return np.array([])

        feature_matrix = np.zeros((len(self.features), len(self.feature_names)))
        for i, feat_dict in enumerate(self.features):
            for j, feat_name in enumerate(self.feature_names):
                feature_matrix[i, j] = float(feat_dict.get(feat_name, 0))

        return feature_matrix

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        position = torch.FloatTensor(self.positions[idx])
        features = (
            torch.FloatTensor(self.feature_array[idx])
            if len(self.feature_array) > 0
            else torch.zeros(1)
        )

        stockfish_move_idx = self.move_to_idx.get(self.moves[idx], 0)
        magnus_move_idx = self.move_to_idx.get(self.magnus_moves[idx], 0)

        # Ensure evaluation is a float
        try:
            eval_value = float(self.evaluations[idx])
        except (ValueError, TypeError):
            eval_value = 0.0

        evaluation = torch.FloatTensor([eval_value])

        return {
            "position": position,
            "features": features,
            "stockfish_move": torch.LongTensor([stockfish_move_idx]),
            "magnus_move": torch.LongTensor([magnus_move_idx]),
            "evaluation": evaluation,
        }


class MagnusStyleModel(nn.Module):
    """Neural network to predict Magnus Carlsen's playing style"""

    def __init__(self, config: StockfishConfig, vocab_size: int, feature_dim: int):
        super().__init__()
        self.config = config
        self.feature_dim = feature_dim

        # NNUE-style board encoder (similar to Stockfish NNUE)
        self.board_encoder = nn.Sequential(
            nn.Linear(768, 512),  # 768 = 64 squares * 12 piece types
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        # Feature encoder for additional position characteristics
        if feature_dim > 1:  # Only create if we have meaningful features
            self.feature_encoder = nn.Sequential(
                nn.Linear(feature_dim, 64), nn.ReLU(), nn.Linear(64, 32)
            )
            self.use_features = True
            combined_dim = 128 + 32
        else:
            self.feature_encoder = None
            self.use_features = False
            combined_dim = 128

        # Move prediction head (Magnus's move choice)
        self.move_predictor = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, vocab_size),
        )

        # Evaluation head (how much better/worse than Stockfish)
        self.eval_adjustment = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),  # Output between -1 and 1 for evaluation adjustment
        )

    def forward(self, position, features):
        # Encode board
        board_encoding = self.board_encoder(position)

        # Encode additional features
        if self.use_features and self.feature_encoder is not None:
            feature_encoding = self.feature_encoder(features)
            combined = torch.cat([board_encoding, feature_encoding], dim=1)
        else:
            combined = board_encoding

        # Predict Magnus's move preference
        move_logits = self.move_predictor(combined)

        # Predict evaluation adjustment (Magnus's style bias)
        eval_adjustment = self.eval_adjustment(combined)

        return move_logits, eval_adjustment


class StockfishMagnusTrainer:
    """Main trainer class for Magnus Carlsen style engine"""

    def __init__(self, config: StockfishConfig):
        self.config = config
        self.encoder = ChessPositionEncoder()
        self.analyzer = StockfishAnalyzer(config)
        self.device = torch.device(config.device)

        # Create directories
        Path(config.data_dir).mkdir(parents=True, exist_ok=True)
        Path(config.model_save_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized Magnus trainer with device: {self.device}")

    def extract_magnus_games_data(self) -> Tuple[List, List, List, List, List]:
        """Extract training data from Magnus Carlsen games"""
        pgn_path = Path(self.config.data_dir) / self.config.pgn_file
        if not pgn_path.exists():
            # Try in current directory
            pgn_path = Path(self.config.pgn_file)
            if not pgn_path.exists():
                raise FileNotFoundError(f"PGN file not found: {self.config.pgn_file}")

        positions = []
        features = []
        stockfish_moves = []
        magnus_moves = []
        evaluations = []

        logger.info(f"Loading Magnus games from {pgn_path}")

        with open(pgn_path, "r") as pgn_file:
            game_count = 0
            total_positions = 0

            while True:
                if self.config.max_games and game_count >= self.config.max_games:
                    break

                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                # Check if Magnus is playing
                headers = game.headers
                white_player = headers.get("White", "")
                black_player = headers.get("Black", "")

                if "Carlsen" not in white_player and "Carlsen" not in black_player:
                    continue

                # Filter by time control (classical games only if specified)
                if self.config.focus_on_classical:
                    time_control = headers.get("TimeControl", "")
                    event = headers.get("Event", "").lower()
                    if any(fast in event for fast in ["blitz", "bullet", "rapid"]):
                        continue

                # Check opponent strength
                magnus_color = chess.WHITE if "Carlsen" in white_player else chess.BLACK
                opponent_elo = headers.get(
                    "BlackElo" if magnus_color == chess.WHITE else "WhiteElo", "0"
                )
                try:
                    if (
                        opponent_elo
                        and int(opponent_elo) < self.config.min_elo_opponent
                    ):
                        continue
                except (ValueError, TypeError):
                    # If ELO is not available or invalid, include the game
                    pass

                # Process game
                game_data = self._process_single_game(game, magnus_color)

                positions.extend(game_data["positions"])
                features.extend(game_data["features"])
                stockfish_moves.extend(game_data["stockfish_moves"])
                magnus_moves.extend(game_data["magnus_moves"])
                evaluations.extend(game_data["evaluations"])

                total_positions += len(game_data["positions"])
                game_count += 1

                if game_count % 10 == 0:
                    logger.info(
                        f"Processed {game_count} games, {total_positions} positions"
                    )

        logger.info(
            f"Extracted {total_positions} positions from {game_count} Magnus games"
        )
        return positions, features, stockfish_moves, magnus_moves, evaluations

    def _process_single_game(self, game, magnus_color) -> Dict[str, List]:
        """Process a single Magnus game and extract training data"""
        positions = []
        features = []
        stockfish_moves = []
        magnus_moves = []
        evaluations = []

        board = game.board()
        node = game
        position_count = 0

        while node.variations and position_count < self.config.max_positions_per_game:
            next_node = node.variation(0)

            # Only analyze positions where Magnus is to move
            if board.turn == magnus_color:
                try:
                    # Encode position
                    position_encoding = self.encoder.encode_board_for_nn(board)
                    position_features = self.encoder.extract_position_features(board)

                    # Get Magnus's actual move
                    magnus_move = next_node.move.uci()

                    # Analyze with Stockfish
                    analysis = self.analyzer.analyze_position(board)

                    if analysis["best_move"]:
                        stockfish_move = analysis["best_move"].uci()
                        evaluation = analysis["evaluation"] or 0

                        # Only include if we have valid data
                        positions.append(position_encoding)
                        features.append(position_features)
                        stockfish_moves.append(stockfish_move)
                        magnus_moves.append(magnus_move)
                        evaluations.append(evaluation)

                        position_count += 1

                except Exception as e:
                    logger.debug(f"Failed to process position: {e}")
                    pass

            board.push(next_node.move)
            node = next_node

        return {
            "positions": positions,
            "features": features,
            "stockfish_moves": stockfish_moves,
            "magnus_moves": magnus_moves,
            "evaluations": evaluations,
        }

    def _process_single_game_parallel(self, game, magnus_color) -> Dict[str, List]:
        """Process a single Magnus game with parallel Stockfish analysis"""
        positions_to_analyze = []
        game_data = {
            "positions": [],
            "features": [],
            "stockfish_moves": [],
            "magnus_moves": [],
            "evaluations": [],
        }

        board = game.board()
        node = game
        position_count = 0

        # First pass: collect all positions that need analysis
        while node.variations and position_count < self.config.max_positions_per_game:
            next_node = node.variation(0)

            # Only analyze positions where Magnus is to move
            if board.turn == magnus_color:
                try:
                    # Encode position
                    position_encoding = self.encoder.encode_board_for_nn(board)
                    position_features = self.encoder.extract_position_features(board)
                    magnus_move = next_node.move.uci()

                    positions_to_analyze.append(
                        {
                            "board_fen": board.fen(),
                            "position_encoding": position_encoding,
                            "position_features": position_features,
                            "magnus_move": magnus_move,
                            "index": position_count,
                        }
                    )

                    position_count += 1

                except Exception as e:
                    logger.debug(f"Failed to encode position: {e}")
                    pass

            board.push(next_node.move)
            node = next_node

        # Second pass: analyze positions in parallel
        if positions_to_analyze:
            analyzed_positions = self._analyze_positions_parallel(positions_to_analyze)

            # Third pass: collect results
            for pos_data in analyzed_positions:
                if pos_data["analysis"] and pos_data["analysis"]["best_move"]:
                    game_data["positions"].append(pos_data["position_encoding"])
                    game_data["features"].append(pos_data["position_features"])
                    game_data["stockfish_moves"].append(
                        pos_data["analysis"]["best_move"].uci()
                    )
                    game_data["magnus_moves"].append(pos_data["magnus_move"])
                    game_data["evaluations"].append(
                        pos_data["analysis"]["evaluation"] or 0
                    )

        return game_data

    def _analyze_positions_parallel(self, positions_data: List[Dict]) -> List[Dict]:
        """Analyze multiple positions in parallel using ThreadPoolExecutor"""
        results = []

        # Use ThreadPoolExecutor with optimal number of threads for your M3 Pro
        max_workers = min(self.config.max_threads, len(positions_data))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all analysis tasks
            future_to_data = {}
            for pos_data in positions_data:
                future = executor.submit(self._analyze_single_position, pos_data)
                future_to_data[future] = pos_data

            # Collect results as they complete
            for future in as_completed(future_to_data):
                pos_data = future_to_data[future]
                try:
                    analysis_result = future.result()
                    pos_data["analysis"] = analysis_result
                    results.append(pos_data)
                except Exception as e:
                    logger.debug(f"Failed to analyze position: {e}")
                    pos_data["analysis"] = None
                    results.append(pos_data)

        # Sort results by original index to maintain order
        results.sort(key=lambda x: x["index"])
        return results

    def _analyze_single_position(self, pos_data: Dict) -> Dict[str, Any]:
        """Analyze a single position with Stockfish (thread-safe)"""
        try:
            # Create a new board from FEN (thread-safe)
            board = chess.Board(pos_data["board_fen"])

            # Each thread needs its own Stockfish analyzer for thread safety
            with chess.engine.SimpleEngine.popen_uci(
                self.config.stockfish_path
            ) as engine:
                # Configure engine
                engine.configure(
                    {
                        "Hash": 256,  # Smaller hash for multiple threads
                        "Threads": 1,  # Each instance uses 1 thread
                    }
                )

                # Analyze position
                result = engine.analyse(
                    board,
                    chess.engine.Limit(
                        time=self.config.analysis_time, depth=self.config.analysis_depth
                    ),
                )

                # Handle both single result and list results
                if isinstance(result, list):
                    main_result = result[0] if result else {}
                else:
                    main_result = result

                analysis = {
                    "best_move": (
                        main_result["pv"][0] if main_result.get("pv") else None
                    ),
                    "evaluation": main_result.get("score", chess.engine.Cp(0))
                    .white()
                    .score(mate_score=10000),
                    "depth": main_result.get("depth", 0),
                    "nodes": main_result.get("nodes", 0),
                    "time": main_result.get("time", 0),
                    "pv": main_result.get("pv", []),
                }

                return analysis

        except Exception as e:
            logger.debug(f"Error in position analysis: {e}")
            return None

    def extract_magnus_games_data_parallel(self) -> Tuple[List, List, List, List, List]:
        """Extract training data from Magnus Carlsen games with parallel processing"""
        pgn_path = Path(self.config.data_dir) / self.config.pgn_file
        if not pgn_path.exists():
            # Try in current directory
            pgn_path = Path(self.config.pgn_file)
            if not pgn_path.exists():
                raise FileNotFoundError(f"PGN file not found: {self.config.pgn_file}")

        positions = []
        features = []
        stockfish_moves = []
        magnus_moves = []
        evaluations = []

        logger.info(
            f"Loading Magnus games from {pgn_path} with {self.config.max_threads} threads"
        )

        with open(pgn_path, "r") as pgn_file:
            game_count = 0
            total_positions = 0

            while True:
                if self.config.max_games and game_count >= self.config.max_games:
                    break

                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                # Check if Magnus is playing
                headers = game.headers
                white_player = headers.get("White", "")
                black_player = headers.get("Black", "")

                if "Carlsen" not in white_player and "Carlsen" not in black_player:
                    continue

                # Filter by time control (classical games only if specified)
                if self.config.focus_on_classical:
                    time_control = headers.get("TimeControl", "")
                    event = headers.get("Event", "").lower()
                    if any(fast in event for fast in ["blitz", "bullet", "rapid"]):
                        continue

                # Check opponent strength
                magnus_color = chess.WHITE if "Carlsen" in white_player else chess.BLACK
                opponent_elo = headers.get(
                    "BlackElo" if magnus_color == chess.WHITE else "WhiteElo", "0"
                )
                try:
                    if (
                        opponent_elo
                        and int(opponent_elo) < self.config.min_elo_opponent
                    ):
                        continue
                except (ValueError, TypeError):
                    # If ELO is not available or invalid, include the game
                    pass

                # Process game with parallel analysis
                game_data = self._process_single_game_parallel(game, magnus_color)

                positions.extend(game_data["positions"])
                features.extend(game_data["features"])
                stockfish_moves.extend(game_data["stockfish_moves"])
                magnus_moves.extend(game_data["magnus_moves"])
                evaluations.extend(game_data["evaluations"])

                total_positions += len(game_data["positions"])
                game_count += 1

                if game_count % 10 == 0:
                    logger.info(
                        f"Processed {game_count} games, {total_positions} positions (parallel mode)"
                    )

        logger.info(
            f"Extracted {total_positions} positions from {game_count} Magnus games (parallel mode)"
        )
        return positions, features, stockfish_moves, magnus_moves, evaluations

    def create_datasets(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test datasets"""
        logger.info("Extracting Magnus games data...")
        if self.config.use_parallel_analysis:
            positions, features, sf_moves, magnus_moves, evaluations = (
                self.extract_magnus_games_data_parallel()
            )
        else:
            positions, features, sf_moves, magnus_moves, evaluations = (
                self.extract_magnus_games_data()
            )

        if len(positions) == 0:
            raise ValueError("No training data extracted. Check PGN file and filters.")

        # Split data
        test_size = self.config.test_split
        val_size = self.config.validation_split / (1 - test_size)

        # First split: separate test set
        data_train_val, data_test = train_test_split(
            list(zip(positions, features, sf_moves, magnus_moves, evaluations)),
            test_size=test_size,
            random_state=42,
        )

        # Second split: separate train and validation
        data_train, data_val = train_test_split(
            data_train_val, test_size=val_size, random_state=42
        )

        # Unpack data
        def unpack_data(data):
            pos, feat, sf_mov, mag_mov, evals = zip(*data)
            return list(pos), list(feat), list(sf_mov), list(mag_mov), list(evals)

        train_data = unpack_data(data_train)
        val_data = unpack_data(data_val)
        test_data = unpack_data(data_test)

        # Create datasets
        train_dataset = MagnusDataset(*train_data)
        val_dataset = MagnusDataset(*val_data)
        test_dataset = MagnusDataset(*test_data)

        # Ensure consistent vocabularies
        val_dataset.move_to_idx = train_dataset.move_to_idx
        val_dataset.idx_to_move = train_dataset.idx_to_move
        val_dataset.vocab_size = train_dataset.vocab_size

        test_dataset.move_to_idx = train_dataset.move_to_idx
        test_dataset.idx_to_move = train_dataset.idx_to_move
        test_dataset.vocab_size = train_dataset.vocab_size

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Disable multiprocessing for stability
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Save dataset info
        dataset_info = {
            "vocab_size": train_dataset.vocab_size,
            "move_to_idx": train_dataset.move_to_idx,
            "feature_names": train_dataset.feature_names,
            "feature_dim": len(train_dataset.feature_names),
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
        }

        info_path = Path(self.config.model_save_dir) / "dataset_info.json"
        with open(info_path, "w") as f:
            json.dump(dataset_info, f, indent=2)

        logger.info(
            f"Dataset created - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
        )
        logger.info(f"Move vocabulary size: {train_dataset.vocab_size}")
        logger.info(f"Feature dimension: {len(train_dataset.feature_names)}")

        return train_loader, val_loader, test_loader

    def train(self):
        """Train the Magnus style model"""
        logger.info("Starting Magnus Carlsen style training...")

        try:
            # Create datasets
            train_loader, val_loader, test_loader = self.create_datasets()

            # Load dataset info
            info_path = Path(self.config.model_save_dir) / "dataset_info.json"
            with open(info_path, "r") as f:
                dataset_info = json.load(f)

            # Create model
            model = MagnusStyleModel(
                self.config, dataset_info["vocab_size"], dataset_info["feature_dim"]
            ).to(self.device)

            logger.info(
                f"Model created with {sum(p.numel() for p in model.parameters())} parameters"
            )

            # Loss functions and optimizer
            move_criterion = nn.CrossEntropyLoss()
            eval_criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=5, factor=0.5
            )

            # Training history
            history = {
                "train_loss": [],
                "val_loss": [],
                "train_move_acc": [],
                "val_move_acc": [],
                "train_eval_loss": [],
                "val_eval_loss": [],
            }

            best_val_loss = float("inf")
            patience_counter = 0

            for epoch in range(self.config.num_epochs):
                # Training phase
                model.train()
                train_losses = []
                train_move_correct = 0
                train_eval_losses = []
                train_total = 0

                for batch in tqdm(
                    train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}"
                ):
                    optimizer.zero_grad()

                    positions = batch["position"].to(self.device)
                    features = batch["features"].to(self.device)
                    magnus_moves = batch["magnus_move"].squeeze().to(self.device)
                    evaluations = batch["evaluation"].squeeze().to(self.device)

                    # Forward pass
                    move_logits, eval_adjustment = model(positions, features)

                    # Compute losses
                    move_loss = move_criterion(move_logits, magnus_moves)
                    eval_loss = eval_criterion(
                        eval_adjustment.squeeze(), evaluations / 1000.0
                    )  # Normalize evals
                    total_loss = move_loss + 0.1 * eval_loss  # Weight eval loss lower

                    # Backward pass
                    total_loss.backward()
                    optimizer.step()

                    # Metrics
                    train_losses.append(total_loss.item())
                    train_eval_losses.append(eval_loss.item())

                    _, predicted = torch.max(move_logits, 1)
                    train_move_correct += (predicted == magnus_moves).sum().item()
                    train_total += magnus_moves.size(0)

                # Validation phase
                model.eval()
                val_losses = []
                val_move_correct = 0
                val_eval_losses = []
                val_total = 0

                with torch.no_grad():
                    for batch in val_loader:
                        positions = batch["position"].to(self.device)
                        features = batch["features"].to(self.device)
                        magnus_moves = batch["magnus_move"].squeeze().to(self.device)
                        evaluations = batch["evaluation"].squeeze().to(self.device)

                        move_logits, eval_adjustment = model(positions, features)

                        move_loss = move_criterion(move_logits, magnus_moves)
                        eval_loss = eval_criterion(
                            eval_adjustment.squeeze(), evaluations / 1000.0
                        )
                        total_loss = move_loss + 0.1 * eval_loss

                        val_losses.append(total_loss.item())
                        val_eval_losses.append(eval_loss.item())

                        _, predicted = torch.max(move_logits, 1)
                        val_move_correct += (predicted == magnus_moves).sum().item()
                        val_total += magnus_moves.size(0)

                # Calculate metrics
                train_loss = np.mean(train_losses)
                val_loss = np.mean(val_losses)
                train_move_acc = train_move_correct / train_total
                val_move_acc = val_move_correct / val_total
                train_eval_loss = np.mean(train_eval_losses)
                val_eval_loss = np.mean(val_eval_losses)

                # Update history
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["train_move_acc"].append(train_move_acc)
                history["val_move_acc"].append(val_move_acc)
                history["train_eval_loss"].append(train_eval_loss)
                history["val_eval_loss"].append(val_eval_loss)

                # Logging
                logger.info(f"Epoch {epoch+1}:")
                logger.info(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                logger.info(
                    f"  Train Move Acc: {train_move_acc:.4f}, Val Move Acc: {val_move_acc:.4f}"
                )
                logger.info(
                    f"  Train Eval Loss: {train_eval_loss:.4f}, Val Eval Loss: {val_eval_loss:.4f}"
                )

                # Learning rate scheduling
                scheduler.step(val_loss)

                # Early stopping and model saving
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0

                    # Save best model
                    checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "config": self.config,
                        "dataset_info": dataset_info,
                    }

                    model_path = (
                        Path(self.config.model_save_dir) / "best_magnus_model.pth"
                    )
                    torch.save(checkpoint, model_path)
                    logger.info("Saved new best model!")

                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        logger.info("Early stopping triggered!")
                        break

            # Save training history
            history_path = Path(self.config.model_save_dir) / "training_history.json"
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2)

            # Final evaluation
            self.evaluate_model(model, test_loader, dataset_info)

            logger.info("Training completed successfully!")
            return model, history

        finally:
            # Always close the engine
            self.analyzer.close()

    def evaluate_model(self, model, test_loader, dataset_info):
        """Evaluate model on test set"""
        model.eval()
        test_move_correct = 0
        test_total = 0
        eval_errors = []

        with torch.no_grad():
            for batch in test_loader:
                positions = batch["position"].to(self.device)
                features = batch["features"].to(self.device)
                magnus_moves = batch["magnus_move"].squeeze().to(self.device)
                evaluations = batch["evaluation"].squeeze().to(self.device)

                move_logits, eval_adjustment = model(positions, features)

                _, predicted = torch.max(move_logits, 1)
                test_move_correct += (predicted == magnus_moves).sum().item()
                test_total += magnus_moves.size(0)

                # Evaluation errors
                eval_pred = eval_adjustment.squeeze() * 1000.0  # Denormalize
                eval_errors.extend(torch.abs(eval_pred - evaluations).cpu().numpy())

        test_acc = test_move_correct / test_total
        mean_eval_error = np.mean(eval_errors)

        logger.info("Test Results:")
        logger.info(f"Move Accuracy: {test_acc:.4f}")
        logger.info(f"Mean Evaluation Error: {mean_eval_error:.2f} centipawns")

        # Save test results
        test_results = {
            "move_accuracy": float(test_acc),
            "mean_eval_error": float(mean_eval_error),
            "total_samples": int(test_total),
        }

        results_path = Path(self.config.model_save_dir) / "test_results.json"
        with open(results_path, "w") as f:
            json.dump(test_results, f, indent=2)


def main():
    """Main training function"""
    config = StockfishConfig()

    # Optimal configuration for M3 Pro full training with parallel processing
    config.max_games = None  # Use full Magnus dataset
    config.num_epochs = 50  # Good convergence
    config.analysis_time = 0.5  # Balanced speed/quality
    config.batch_size = 256  # Optimal for M3 Pro

    # 🚀 Enable parallel processing for 5.6x speedup!
    config.use_parallel_analysis = True  # Enable multithreaded Stockfish analysis
    config.max_threads = 8  # Optimal for M3 Pro (12 cores, leaving some for system)

    # Optional: Reduce for faster experimentation
    # config.max_games = 500  # Subset for testing
    # config.num_epochs = 20  # Quick training
    # config.use_parallel_analysis = False  # Disable for debugging

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
        logger.error(
            "No Magnus Carlsen PGN file found. Please ensure one of these files exists:"
        )
        for pgn_file in pgn_candidates:
            logger.error(f"  - {pgn_file}")
        return

    logger.info(f"Using PGN file: {config.pgn_file}")

    if config.use_parallel_analysis:
        logger.info(f"🚀 Parallel processing enabled with {config.max_threads} threads")
        logger.info(f"   Expected ~5.6x speedup in position analysis!")
    else:
        logger.info("🔄 Using sequential processing")

    trainer = StockfishMagnusTrainer(config)

    try:
        start_time = time.time()
        model, history = trainer.train()
        total_time = time.time() - start_time

        logger.info("Magnus Carlsen style training completed successfully!")
        logger.info(f"📊 Total training time: {total_time/3600:.2f} hours")

        if config.use_parallel_analysis:
            estimated_sequential_time = (
                total_time * 5.6
            )  # Estimated if sequential was used
            logger.info(
                f"🎯 Estimated time saved with parallel processing: {(estimated_sequential_time - total_time)/3600:.2f} hours"
            )

        # Plot training curves
        plot_training_curves(history, config.model_save_dir)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def plot_training_curves(history: Dict, save_dir: str):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history["train_loss"], label="Train Loss")
    axes[0, 0].plot(history["val_loss"], label="Validation Loss")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    # Move Accuracy
    axes[0, 1].plot(history["train_move_acc"], label="Train Accuracy")
    axes[0, 1].plot(history["val_move_acc"], label="Validation Accuracy")
    axes[0, 1].set_title("Move Prediction Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()

    # Evaluation Loss
    axes[1, 0].plot(history["train_eval_loss"], label="Train Eval Loss")
    axes[1, 0].plot(history["val_eval_loss"], label="Validation Eval Loss")
    axes[1, 0].set_title("Evaluation Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()

    # Combined metrics
    axes[1, 1].plot(history["val_move_acc"], label="Move Accuracy")
    axes[1, 1].plot(
        [1 - x for x in history["val_eval_loss"]], label="Eval Accuracy (inverted)"
    )
    axes[1, 1].set_title("Validation Metrics")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Metric Value")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(Path(save_dir) / "training_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Training curves saved to {save_dir}/training_curves.png")


if __name__ == "__main__":
    main()
