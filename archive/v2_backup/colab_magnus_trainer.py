"""
Magnus Carlsen Style Chess Engine Training for Google Colab

This is a complete, standalone script for training a Magnus Carlsen-style chess engine
on Google Colab. It includes all necessary components:
- Data extraction from Magnus's games
- Parallel Stockfish analysis (optimized for Colab)
- Neural network training
- Model evaluation and saving

Usage in Colab:
1. Upload your Magnus Carlsen PGN file
2. Run this script
3. Download the trained model

Optimized for Colab Free (reduced dataset), Pro (V100), and Pro+ (A100) tiers.
"""

import os
import subprocess
import sys
import time
import json
import logging
import pickle
import zipfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


# Install required packages if not available
def install_packages():
    """Install required packages in Colab environment"""
    packages = [
        "chess",
        "torch",
        "torchvision",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "tqdm",
    ]

    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# Install packages first
install_packages()

import chess
import chess.pgn
import chess.engine
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Detect Colab environment
def is_colab():
    try:
        import google.colab

        return True
    except ImportError:
        return False


def install_stockfish():
    """Install Stockfish in Colab environment"""
    if is_colab():
        print("Installing Stockfish for Colab...")
        subprocess.run(["apt-get", "update"], check=False, capture_output=True)
        subprocess.run(
            ["apt-get", "install", "-y", "stockfish"], check=False, capture_output=True
        )
        return "/usr/games/stockfish"
    else:
        # Local installation paths
        possible_paths = [
            "/opt/homebrew/bin/stockfish",
            "/usr/local/bin/stockfish",
            "/usr/bin/stockfish",
            "stockfish",
        ]
        for path in possible_paths:
            if (
                os.path.exists(path)
                or subprocess.run(["which", path], capture_output=True).returncode == 0
            ):
                return path
        raise FileNotFoundError("Stockfish not found. Please install Stockfish.")


@dataclass
class ColabConfig:
    """Configuration optimized for Google Colab"""

    # Colab-specific paths
    stockfish_path: str = "/usr/games/stockfish"  # Default Colab path
    data_dir: str = "/content/magnus_data"
    pgn_file: str = "/content/carlsen-games.pgn"  # Upload location in Colab
    model_save_dir: str = "/content/magnus_model"

    # Training parameters (optimized for Colab)
    batch_size: int = 128  # Smaller for Colab memory limits
    learning_rate: float = 0.001
    num_epochs: int = 30  # Reduced for time limits
    validation_split: float = 0.2
    test_split: float = 0.1
    early_stopping_patience: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Analysis parameters (optimized for Colab)
    analysis_depth: int = 15  # Reduced for speed
    analysis_time: float = 0.3  # Faster analysis
    max_positions_per_game: int = 30  # Fewer positions per game
    max_games: Optional[int] = 1000  # Limit for Colab Free tier

    # Stockfish engine settings (Colab optimized)
    stockfish_threads: int = 1  # Single thread per engine
    stockfish_hash: int = 256  # Smaller hash for Colab

    # Parallel processing (Colab optimized)
    max_threads: int = 4  # Conservative for Colab
    use_parallel_analysis: bool = True

    # Magnus-specific parameters
    focus_on_classical: bool = True
    min_elo_opponent: int = 2300  # Slightly lower for more data
    include_endgames: bool = True
    include_openings: bool = True
    include_middlegames: bool = True

    # Colab tier configurations
    colab_tier: str = "free"  # 'free', 'pro', 'pro_plus'


def configure_for_colab_tier(config: ColabConfig, tier: str = "auto"):
    """Configure settings based on Colab tier"""

    # Auto-detect tier based on available resources
    if tier == "auto":
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            if "A100" in gpu_name:
                tier = "pro_plus"
            elif any(gpu in gpu_name for gpu in ["V100", "T4", "P100"]):
                tier = "pro"
            else:
                tier = "free"
        else:
            tier = "free"

    config.colab_tier = tier

    if tier == "pro_plus":  # A100 GPU
        config.max_games = None  # Full dataset
        config.batch_size = 256
        config.num_epochs = 50
        config.max_threads = 8
        config.analysis_time = 0.5
        config.analysis_depth = 20
        config.max_positions_per_game = 50
        print("🚀 Configured for Colab Pro+ (A100) - Full training")

    elif tier == "pro":  # V100/T4 GPU
        config.max_games = 3000  # Large subset
        config.batch_size = 192
        config.num_epochs = 40
        config.max_threads = 6
        config.analysis_time = 0.4
        config.analysis_depth = 18
        config.max_positions_per_game = 40
        print("⚡ Configured for Colab Pro (V100/T4) - Large subset training")

    else:  # Free tier
        config.max_games = 1000  # Reduced dataset
        config.batch_size = 128
        config.num_epochs = 25
        config.max_threads = 4
        config.analysis_time = 0.3
        config.analysis_depth = 15
        config.max_positions_per_game = 30
        print("💡 Configured for Colab Free tier - Reduced dataset")

    print(
        f"Configuration: {config.max_games or 'All'} games, {config.num_epochs} epochs, {config.max_threads} threads"
    )


class ChessPositionEncoder:
    """Enhanced position encoder for neural network training"""

    def __init__(self):
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000,
        }

    def encode_board_for_nn(self, board: chess.Board) -> np.ndarray:
        """Encode chess board for neural network (NNUE-style)"""
        features = np.zeros(768, dtype=np.float32)  # 64 squares * 12 piece types

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_offset = (piece.piece_type - 1) + (
                    0 if piece.color == chess.WHITE else 6
                )
                feature_idx = square * 12 + piece_offset
                features[feature_idx] = 1.0

        return features

    def extract_position_features(self, board: chess.Board) -> Dict[str, float]:
        """Extract comprehensive position features"""
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

        # Piece counts
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

        # Game phase
        total_material = white_material + black_material
        if total_material > 6000:
            features["game_phase"] = 0  # Opening/early middlegame
        elif total_material > 3000:
            features["game_phase"] = 1  # Middlegame
        else:
            features["game_phase"] = 2  # Endgame

        return features


class StockfishAnalyzer:
    """Stockfish analyzer optimized for Colab"""

    def __init__(self, config: ColabConfig):
        self.config = config
        self.engine = None
        self._init_engine()

    def _init_engine(self):
        """Initialize Stockfish engine"""
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(
                self.config.stockfish_path
            )
            self.engine.configure(
                {
                    "Threads": self.config.stockfish_threads,
                    "Hash": self.config.stockfish_hash,
                }
            )
            logger.info(f"Initialized Stockfish: {self.engine.id}")
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
            result = self.engine.analyse(
                board,
                chess.engine.Limit(time=time_limit, depth=self.config.analysis_depth),
            )

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
            }

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

        # Create move vocabulary
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

    def __init__(self, config: ColabConfig, vocab_size: int, feature_dim: int):
        super().__init__()
        self.config = config
        self.feature_dim = feature_dim

        # NNUE-style board encoder
        self.board_encoder = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        # Feature encoder
        if feature_dim > 1:
            self.feature_encoder = nn.Sequential(
                nn.Linear(feature_dim, 64), nn.ReLU(), nn.Linear(64, 32)
            )
            self.use_features = True
            combined_dim = 128 + 32
        else:
            self.feature_encoder = None
            self.use_features = False
            combined_dim = 128

        # Move prediction head
        self.move_predictor = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, vocab_size),
        )

        # Evaluation head
        self.eval_adjustment = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, position, features):
        board_encoding = self.board_encoder(position)

        if self.use_features and self.feature_encoder is not None:
            feature_encoding = self.feature_encoder(features)
            combined = torch.cat([board_encoding, feature_encoding], dim=1)
        else:
            combined = board_encoding

        move_logits = self.move_predictor(combined)
        eval_adjustment = self.eval_adjustment(combined)

        return move_logits, eval_adjustment


class ColabMagnusTrainer:
    """Magnus Carlsen trainer optimized for Google Colab"""

    def __init__(self, config: ColabConfig):
        self.config = config
        self.encoder = ChessPositionEncoder()
        self.analyzer = StockfishAnalyzer(config)
        self.device = torch.device(config.device)

        # Create directories
        Path(config.data_dir).mkdir(parents=True, exist_ok=True)
        Path(config.model_save_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized Magnus trainer for Colab with device: {self.device}")

    def _analyze_single_position(self, pos_data: Dict) -> Dict[str, Any]:
        """Analyze a single position with Stockfish (thread-safe for Colab)"""
        try:
            board = chess.Board(pos_data["board_fen"])

            with chess.engine.SimpleEngine.popen_uci(
                self.config.stockfish_path
            ) as engine:
                engine.configure(
                    {
                        "Hash": 128,  # Smaller hash for Colab
                        "Threads": 1,
                    }
                )

                result = engine.analyse(
                    board,
                    chess.engine.Limit(
                        time=self.config.analysis_time, depth=self.config.analysis_depth
                    ),
                )

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

    def _analyze_positions_parallel(self, positions_data: List[Dict]) -> List[Dict]:
        """Analyze multiple positions in parallel (Colab optimized)"""
        results = []
        max_workers = min(self.config.max_threads, len(positions_data))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_data = {}
            for pos_data in positions_data:
                future = executor.submit(self._analyze_single_position, pos_data)
                future_to_data[future] = pos_data

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

        results.sort(key=lambda x: x["index"])
        return results

    def _process_single_game(self, game, magnus_color) -> Dict[str, List]:
        """Process a single Magnus game with parallel analysis"""
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

        # Collect positions for analysis
        while node.variations and position_count < self.config.max_positions_per_game:
            next_node = node.variation(0)

            if board.turn == magnus_color:
                try:
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

        # Analyze positions in parallel
        if positions_to_analyze:
            if self.config.use_parallel_analysis:
                analyzed_positions = self._analyze_positions_parallel(
                    positions_to_analyze
                )
            else:
                analyzed_positions = []
                for pos_data in positions_to_analyze:
                    pos_data["analysis"] = self._analyze_single_position(pos_data)
                    analyzed_positions.append(pos_data)

            # Collect results
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

    def extract_magnus_games_data(self) -> Tuple[List, List, List, List, List]:
        """Extract training data from Magnus Carlsen games"""
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

                # Filter by time control
                if self.config.focus_on_classical:
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

    def create_datasets(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test datasets"""
        logger.info("Extracting Magnus games data...")
        positions, features, sf_moves, magnus_moves, evaluations = (
            self.extract_magnus_games_data()
        )

        if len(positions) == 0:
            raise ValueError("No training data extracted. Check PGN file and filters.")

        # Split data
        test_size = self.config.test_split
        val_size = self.config.validation_split / (1 - test_size)

        data_train_val, data_test = train_test_split(
            list(zip(positions, features, sf_moves, magnus_moves, evaluations)),
            test_size=test_size,
            random_state=42,
        )

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
            num_workers=0,
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

        return train_loader, val_loader, test_loader

    def train(self):
        """Train the Magnus style model"""
        logger.info("Starting Magnus Carlsen style training on Colab...")

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
            logger.info(f"Training on device: {self.device}")

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
                    )
                    total_loss = move_loss + 0.1 * eval_loss

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

                eval_pred = eval_adjustment.squeeze() * 1000.0
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


def create_download_package(config: ColabConfig):
    """Create a downloadable package with trained model and results"""
    package_path = "/content/magnus_model_package.zip"

    with zipfile.ZipFile(package_path, "w") as zipf:
        model_dir = Path(config.model_save_dir)

        # Add all model files
        for file_path in model_dir.rglob("*"):
            if file_path.is_file():
                zipf.write(file_path, file_path.relative_to(model_dir.parent))

    logger.info(f"Model package created: {package_path}")
    logger.info("Download this file to get your trained Magnus model!")

    if is_colab():
        from google.colab import files

        files.download(package_path)


def main():
    """Main training function for Colab"""
    print("🏁 Starting Magnus Carlsen Style Chess Engine Training for Colab")
    print("=" * 60)

    # Install Stockfish
    stockfish_path = install_stockfish()
    print(f"✅ Stockfish installed at: {stockfish_path}")

    # Create configuration
    config = ColabConfig()
    config.stockfish_path = stockfish_path

    # Auto-configure for Colab tier
    configure_for_colab_tier(config, tier="auto")

    # Check for PGN file
    if not Path(config.pgn_file).exists():
        print(f"❌ PGN file not found at {config.pgn_file}")
        print("Please upload your Magnus Carlsen games PGN file and run again.")
        print("You can upload files using:")
        print("  from google.colab import files")
        print("  uploaded = files.upload()")
        return

    print(f"✅ Found PGN file: {config.pgn_file}")

    # Create trainer
    trainer = ColabMagnusTrainer(config)

    try:
        start_time = time.time()

        print("\n🚀 Starting training...")
        model, history = trainer.train()

        total_time = time.time() - start_time

        print(f"\n🎉 Training completed successfully!")
        print(f"⏱️  Total training time: {total_time/3600:.2f} hours")
        print(f"🎯 Configuration: {config.colab_tier} tier")

        # Plot training curves
        plot_training_curves(history, config.model_save_dir)

        # Create download package
        create_download_package(config)

        print("\n📦 Your Magnus Carlsen model is ready!")
        print("The model package has been automatically downloaded.")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        logger.error(f"Training failed: {e}")
        raise


# Instructions for Colab usage
COLAB_INSTRUCTIONS = """
🚀 MAGNUS CARLSEN STYLE CHESS ENGINE TRAINING FOR COLAB

STEP-BY-STEP INSTRUCTIONS:

1. UPLOAD YOUR PGN FILE:
   - Upload your Magnus Carlsen games PGN file to Colab
   - Name it 'carlsen-games.pgn' or update the path in config

2. RUN THIS SCRIPT:
   - Simply run: main()
   - The script will auto-detect your Colab tier and optimize settings

3. WAIT FOR TRAINING:
   - Free tier: ~2-3 hours (reduced dataset)
   - Pro tier: ~4-6 hours (large dataset) 
   - Pro+ tier: ~6-8 hours (full dataset)

4. DOWNLOAD YOUR MODEL:
   - The trained model will be automatically downloaded as a zip file
   - Contains the model, training history, and evaluation results

COLAB TIER OPTIMIZATIONS:
- Free: 1000 games, 25 epochs, 4 threads
- Pro: 3000 games, 40 epochs, 6 threads  
- Pro+: Full dataset, 50 epochs, 8 threads

The script automatically detects your GPU and optimizes accordingly!
"""

if __name__ == "__main__":
    if is_colab():
        print(COLAB_INSTRUCTIONS)
    main()
