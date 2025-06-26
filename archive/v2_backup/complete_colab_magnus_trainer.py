"""
Google Colab Magnus Carlsen Chess Engine Trainer - Complete Version

This is a comprehensive, standalone Python script for training a Magnus Carlsen-style
chess engine in Google Colab. It includes automatic environment setup, parallel
Stockfish analysis, and model training.

HOW TO USE IN COLAB:
1. Upload this file to Colab
2. Upload your Magnus Carlsen PGN file (name it: carlsen-games.pgn)
3. Run this script with: !python complete_colab_magnus_trainer.py
4. Download the trained model package

EXPECTED TRAINING TIMES:
- Free Colab: 1-2 hours (reduced dataset: 100 games)
- Colab Pro (V100): ~20 hours (full dataset)
- Colab Pro+ (A100): ~14 hours (full dataset)

FEATURES:
- Automatic package installation
- Stockfish auto-install and configuration
- Parallel position analysis (optimized for Colab)
- Smart memory management
- Progress tracking with ETA
- Downloadable model package
- Training visualization
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm
import json
import pickle
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

print("🚀 Magnus Carlsen Chess Engine Trainer for Google Colab")
print("=" * 65)

# Stage 1: Install Required Packages
print("📦 Installing required packages...")


def install_package(package_name):
    """Install a single package with error handling"""
    try:
        __import__(package_name)
        print(f"✓ {package_name} already installed")
        return True
    except ImportError:
        try:
            print(f"Installing {package_name}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"✓ {package_name} installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package_name}: {e}")
            return False


# Install core packages
required_packages = [
    "chess",
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "tqdm",
]

failed_packages = []
for package in required_packages:
    if not install_package(package):
        failed_packages.append(package)

if failed_packages:
    print(f"❌ Failed to install: {failed_packages}")
    print("Please install them manually and restart")
    sys.exit(1)

print("✓ All packages installed successfully!")

# Stage 2: Install and Configure Stockfish
print("\n🏰 Installing Stockfish...")


def install_stockfish():
    """Install Stockfish engine with multiple fallback methods"""
    # Check if already installed
    stockfish_paths = [
        "/usr/local/bin/stockfish",
        "/usr/bin/stockfish",
        "/usr/games/stockfish",
        "./stockfish/stockfish-ubuntu-x86-64-avx2",
    ]

    for path in stockfish_paths:
        if os.path.exists(path):
            print(f"✓ Found Stockfish at: {path}")
            return path

    print("Installing Stockfish...")

    try:
        # Method 1: apt-get
        subprocess.run(["apt-get", "update"], check=True, capture_output=True)
        subprocess.run(
            ["apt-get", "install", "-y", "stockfish"], check=True, capture_output=True
        )

        # Find installed path
        result = subprocess.run(["which", "stockfish"], capture_output=True, text=True)
        if result.returncode == 0:
            path = result.stdout.strip()
            print(f"✓ Stockfish installed via apt-get at: {path}")
            return path

    except Exception as e:
        print(f"apt-get installation failed: {e}")

    try:
        # Method 2: Download binary
        print("Downloading Stockfish binary...")
        subprocess.run(
            [
                "wget",
                "-q",
                "https://github.com/official-stockfish/Stockfish/releases/download/sf_16/stockfish-ubuntu-x86-64-avx2.tar",
            ],
            check=True,
        )

        subprocess.run(["tar", "-xf", "stockfish-ubuntu-x86-64-avx2.tar"], check=True)

        # Make executable and copy
        sf_binary = "stockfish/stockfish-ubuntu-x86-64-avx2"
        subprocess.run(["chmod", "+x", sf_binary], check=True)

        if not os.path.exists("/usr/local/bin"):
            os.makedirs("/usr/local/bin", exist_ok=True)

        subprocess.run(["cp", sf_binary, "/usr/local/bin/stockfish"], check=True)

        print("✓ Stockfish installed via download")
        return "/usr/local/bin/stockfish"

    except Exception as e:
        print(f"❌ Failed to install Stockfish: {e}")
        print("Please install Stockfish manually")
        return None


stockfish_path = install_stockfish()
if not stockfish_path:
    print("❌ Could not install Stockfish. Training cannot proceed.")
    sys.exit(1)

# Stage 3: Detect Colab Environment
print("\n🖥️  Detecting environment...")


def detect_environment():
    """Detect Colab tier and configure accordingly"""
    # Check if in Colab
    try:
        import google.colab

        is_colab = True
        print("✓ Running in Google Colab")
    except ImportError:
        is_colab = False
        print("ℹ️ Not running in Google Colab")

    # Detect GPU
    gpu_info = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
            print(f"✓ GPU detected: {gpu_info}")
        else:
            print("ℹ️ No GPU detected")
    except:
        print("ℹ️ Could not check GPU")

    # Configure based on hardware
    if gpu_info and "A100" in gpu_info:
        tier = "pro_plus"
        config = {
            "max_threads": 8,
            "batch_size": 512,
            "analysis_time": 0.5,
            "analysis_depth": 20,
            "max_games": None,
            "description": "Colab Pro+ (A100) - Full Dataset",
        }
    elif gpu_info and ("V100" in gpu_info or "T4" in gpu_info):
        tier = "pro"
        config = {
            "max_threads": 6,
            "batch_size": 256,
            "analysis_time": 0.4,
            "analysis_depth": 18,
            "max_games": None,
            "description": "Colab Pro - Full Dataset",
        }
    else:
        tier = "free"
        config = {
            "max_threads": 2,
            "batch_size": 64,
            "analysis_time": 0.2,
            "analysis_depth": 15,
            "max_games": 100,
            "description": "Free Colab - Reduced Dataset",
        }

    config.update({"is_colab": is_colab, "gpu_info": gpu_info, "tier": tier})

    print(f"🚀 Configuration: {config['description']}")
    print(f"   Threads: {config['max_threads']}")
    print(f"   Batch size: {config['batch_size']}")
    if config["max_games"]:
        print(f"   Max games: {config['max_games']}")

    return config


env_config = detect_environment()

# Now import heavy packages
print("\n📚 Loading libraries...")
import chess
import chess.pgn
import chess.engine
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("✓ All libraries loaded")

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ColabConfig:
    """Training configuration optimized for Colab"""

    # Environment
    stockfish_path: str = stockfish_path
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    pgn_file: str = "carlsen-games.pgn"
    data_dir: str = "magnus_data"
    model_save_dir: str = "magnus_model"

    # Training (from environment detection)
    batch_size: int = env_config["batch_size"]
    learning_rate: float = 0.001
    num_epochs: int = 25 if env_config["tier"] == "free" else 40
    validation_split: float = 0.2
    test_split: float = 0.1
    early_stopping_patience: int = 6

    # Analysis (from environment detection)
    analysis_depth: int = env_config["analysis_depth"]
    analysis_time: float = env_config["analysis_time"]
    max_positions_per_game: int = 25 if env_config["tier"] == "free" else 40
    max_games: Optional[int] = env_config["max_games"]

    # Stockfish
    stockfish_threads: int = 1
    stockfish_hash: int = 256

    # Parallel processing (from environment detection)
    max_threads: int = env_config["max_threads"]

    # Magnus filters
    focus_on_classical: bool = True
    min_elo_opponent: int = 2400

    def __post_init__(self):
        # Create directories
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.model_save_dir).mkdir(parents=True, exist_ok=True)


class ChessPositionEncoder:
    """Chess position encoder for neural network"""

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
        """Encode board as 768-dimensional vector (64 squares * 12 piece types)"""
        features = np.zeros(768, dtype=np.float32)

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
        """Extract tactical and strategic features"""
        features = {}

        # Basic info
        features["turn"] = float(board.turn)
        features["move_number"] = len(board.move_stack)
        features["halfmove_clock"] = board.halfmove_clock

        # Castling rights
        features["white_kingside_castle"] = float(
            board.has_kingside_castling_rights(chess.WHITE)
        )
        features["white_queenside_castle"] = float(
            board.has_queenside_castling_rights(chess.WHITE)
        )
        features["black_kingside_castle"] = float(
            board.has_kingside_castling_rights(chess.BLACK)
        )
        features["black_queenside_castle"] = float(
            board.has_queenside_castling_rights(chess.BLACK)
        )

        # Material evaluation
        white_material = sum(
            len(board.pieces(pt, chess.WHITE)) * val
            for pt, val in self.piece_values.items()
        )
        black_material = sum(
            len(board.pieces(pt, chess.BLACK)) * val
            for pt, val in self.piece_values.items()
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
        features["white_center_control"] = sum(
            len(board.attackers(chess.WHITE, sq)) for sq in center_squares
        )
        features["black_center_control"] = sum(
            len(board.attackers(chess.BLACK, sq)) for sq in center_squares
        )

        # Game phase
        total_material = white_material + black_material
        if total_material > 6000:
            features["game_phase"] = 0  # Opening
        elif total_material > 3000:
            features["game_phase"] = 1  # Middlegame
        else:
            features["game_phase"] = 2  # Endgame

        return features


class MagnusDataset(Dataset):
    """PyTorch dataset for Magnus positions"""

    def __init__(self, positions, features, moves, evaluations, magnus_moves):
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

        # Convert features to array
        self.feature_names = list(features[0].keys()) if features else []
        self.feature_array = self._features_to_array()

    def _features_to_array(self):
        """Convert feature dicts to numpy array"""
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

        return {
            "position": position,
            "features": features,
            "stockfish_move": torch.LongTensor([stockfish_move_idx]),
            "magnus_move": torch.LongTensor([magnus_move_idx]),
            "evaluation": torch.FloatTensor([eval_value]),
        }


class MagnusStyleModel(nn.Module):
    """Neural network to learn Magnus Carlsen's playing style"""

    def __init__(self, vocab_size: int, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim

        # Board position encoder (NNUE-style)
        self.board_encoder = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        # Position features encoder
        if feature_dim > 1:
            self.feature_encoder = nn.Sequential(
                nn.Linear(feature_dim, 64), nn.ReLU(), nn.Linear(64, 32)
            )
            combined_dim = 128 + 32
            self.use_features = True
        else:
            self.feature_encoder = None
            combined_dim = 128
            self.use_features = False

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

        # Evaluation adjustment head
        self.eval_predictor = nn.Sequential(
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
        eval_adjustment = self.eval_predictor(combined)

        return move_logits, eval_adjustment


class ColabMagnusTrainer:
    """Main trainer class optimized for Google Colab"""

    def __init__(self, config: ColabConfig):
        self.config = config
        self.encoder = ChessPositionEncoder()
        self.device = torch.device(config.device)

        print(f"\n🧠 Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Stockfish: {config.stockfish_path}")
        print(f"   Threads: {config.max_threads}")

    def _analyze_position(self, pos_data: Dict) -> Optional[Dict]:
        """Analyze single position with Stockfish"""
        try:
            board = chess.Board(pos_data["board_fen"])

            with chess.engine.SimpleEngine.popen_uci(
                self.config.stockfish_path
            ) as engine:
                engine.configure(
                    {
                        "Hash": self.config.stockfish_hash,
                        "Threads": self.config.stockfish_threads,
                    }
                )

                result = engine.analyse(
                    board,
                    chess.engine.Limit(
                        time=self.config.analysis_time, depth=self.config.analysis_depth
                    ),
                )

                if isinstance(result, list):
                    result = result[0] if result else {}

                return {
                    "best_move": result["pv"][0] if result.get("pv") else None,
                    "evaluation": result.get("score", chess.engine.Cp(0))
                    .white()
                    .score(mate_score=10000),
                    "depth": result.get("depth", 0),
                    "nodes": result.get("nodes", 0),
                }

        except Exception as e:
            logger.debug(f"Analysis error: {e}")
            return None

    def _analyze_positions_parallel(self, positions_data: List[Dict]) -> List[Dict]:
        """Analyze positions in parallel"""
        results = []

        with ThreadPoolExecutor(max_workers=self.config.max_threads) as executor:
            future_to_data = {
                executor.submit(self._analyze_position, pos_data): pos_data
                for pos_data in positions_data
            }

            for future in as_completed(future_to_data):
                pos_data = future_to_data[future]
                try:
                    analysis = future.result()
                    pos_data["analysis"] = analysis
                    results.append(pos_data)
                except Exception as e:
                    logger.debug(f"Position analysis failed: {e}")
                    pos_data["analysis"] = None
                    results.append(pos_data)

        # Restore original order
        results.sort(key=lambda x: x["index"])
        return results

    def _process_game(self, game, magnus_color) -> Dict:
        """Process single Magnus game"""
        positions_to_analyze = []

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
                    logger.debug(f"Position encoding error: {e}")

            board.push(next_node.move)
            node = next_node

        # Analyze positions
        game_data = {
            "positions": [],
            "features": [],
            "stockfish_moves": [],
            "magnus_moves": [],
            "evaluations": [],
        }

        if positions_to_analyze:
            analyzed = self._analyze_positions_parallel(positions_to_analyze)

            for pos_data in analyzed:
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

    def extract_training_data(self) -> Tuple[List, List, List, List, List]:
        """Extract training data from Magnus games"""
        # Find PGN file
        pgn_candidates = [
            self.config.pgn_file,
            "carlsen-games.pgn",
            "magnus_games.pgn",
            "carlsen-games-quarter.pgn",
        ]

        pgn_path = None
        for candidate in pgn_candidates:
            if Path(candidate).exists():
                pgn_path = Path(candidate)
                break

        if not pgn_path:
            raise FileNotFoundError(
                f"No Magnus PGN file found. Upload one of: {pgn_candidates}"
            )

        print(f"\n📖 Loading games from {pgn_path.name}")

        all_positions = []
        all_features = []
        all_stockfish_moves = []
        all_magnus_moves = []
        all_evaluations = []

        with open(pgn_path, "r") as pgn_file:
            games_processed = 0
            total_positions = 0

            pbar = tqdm(desc="Processing games", unit="games")

            while True:
                if self.config.max_games and games_processed >= self.config.max_games:
                    break

                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                headers = game.headers
                white_player = headers.get("White", "")
                black_player = headers.get("Black", "")

                # Check if Magnus is playing
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
                game_data = self._process_game(game, magnus_color)

                # Add to dataset
                all_positions.extend(game_data["positions"])
                all_features.extend(game_data["features"])
                all_stockfish_moves.extend(game_data["stockfish_moves"])
                all_magnus_moves.extend(game_data["magnus_moves"])
                all_evaluations.extend(game_data["evaluations"])

                total_positions += len(game_data["positions"])
                games_processed += 1

                pbar.update(1)
                pbar.set_postfix(
                    {
                        "positions": total_positions,
                        "avg_pos/game": (
                            f"{total_positions/games_processed:.1f}"
                            if games_processed > 0
                            else "0"
                        ),
                    }
                )

            pbar.close()

        print(f"✓ Extracted {total_positions} positions from {games_processed} games")
        return (
            all_positions,
            all_features,
            all_stockfish_moves,
            all_magnus_moves,
            all_evaluations,
        )

    def create_datasets(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train/validation/test datasets"""
        positions, features, sf_moves, magnus_moves, evaluations = (
            self.extract_training_data()
        )

        if len(positions) == 0:
            raise ValueError("No training data extracted!")

        # Split data
        test_size = self.config.test_split
        val_size = self.config.validation_split / (1 - test_size)

        data = list(zip(positions, features, sf_moves, magnus_moves, evaluations))

        data_train_val, data_test = train_test_split(
            data, test_size=test_size, random_state=42
        )
        data_train, data_val = train_test_split(
            data_train_val, test_size=val_size, random_state=42
        )

        def unpack_data(split_data):
            return list(zip(*split_data))

        train_dataset = MagnusDataset(*unpack_data(data_train))
        val_dataset = MagnusDataset(*unpack_data(data_val))
        test_dataset = MagnusDataset(*unpack_data(data_test))

        # Share vocabulary
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

        print(f"📊 Dataset created:")
        print(f"   Train: {len(train_dataset)} positions")
        print(f"   Validation: {len(val_dataset)} positions")
        print(f"   Test: {len(test_dataset)} positions")
        print(f"   Move vocabulary: {train_dataset.vocab_size} moves")

        return train_loader, val_loader, test_loader

    def train(self):
        """Train the Magnus style model"""
        print(f"\n🔥 Starting Magnus training...")
        print(f"   Epochs: {self.config.num_epochs}")
        print(f"   Batch size: {self.config.batch_size}")
        print(f"   Learning rate: {self.config.learning_rate}")

        # Create datasets
        train_loader, val_loader, test_loader = self.create_datasets()

        # Load dataset info
        info_path = Path(self.config.model_save_dir) / "dataset_info.json"
        with open(info_path, "r") as f:
            dataset_info = json.load(f)

        # Create model
        model = MagnusStyleModel(
            dataset_info["vocab_size"], dataset_info["feature_dim"]
        ).to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Model parameters: {total_params:,}")

        # Loss and optimizer
        move_criterion = nn.CrossEntropyLoss()
        eval_criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=4, factor=0.5
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

        print(f"\n📈 Training progress:")

        for epoch in range(self.config.num_epochs):
            # Training phase
            model.train()
            train_losses = []
            train_move_correct = 0
            train_eval_losses = []
            train_total = 0

            train_pbar = tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}"
            )

            for batch in train_pbar:
                optimizer.zero_grad()

                positions = batch["position"].to(self.device)
                features = batch["features"].to(self.device)
                magnus_moves = batch["magnus_move"].squeeze().to(self.device)
                evaluations = batch["evaluation"].squeeze().to(self.device)

                # Forward pass
                move_logits, eval_adjustment = model(positions, features)

                # Losses
                move_loss = move_criterion(move_logits, magnus_moves)
                eval_loss = eval_criterion(
                    eval_adjustment.squeeze(), evaluations / 1000.0
                )
                total_loss = move_loss + 0.1 * eval_loss

                # Backward
                total_loss.backward()
                optimizer.step()

                # Metrics
                train_losses.append(total_loss.item())
                train_eval_losses.append(eval_loss.item())

                _, predicted = torch.max(move_logits, 1)
                train_move_correct += (predicted == magnus_moves).sum().item()
                train_total += magnus_moves.size(0)

                # Update progress bar
                train_pbar.set_postfix(
                    {
                        "loss": f"{total_loss.item():.4f}",
                        "acc": f"{train_move_correct/train_total:.3f}",
                    }
                )

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

            # Log epoch results
            print(
                f"   Epoch {epoch+1:2d}: Loss {val_loss:.4f} | Move Acc {val_move_acc:.3f} | LR {optimizer.param_groups[0]['lr']:.2e}"
            )

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "config": self.config,
                    "dataset_info": dataset_info,
                    "env_config": env_config,
                }

                model_path = Path(self.config.model_save_dir) / "best_magnus_model.pth"
                torch.save(checkpoint, model_path)
                print(f"   ✓ New best model saved!")

            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    print(f"   Early stopping at epoch {epoch+1}")
                    break

        # Save training history
        history_path = Path(self.config.model_save_dir) / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        # Final evaluation
        self.evaluate_model(model, test_loader, dataset_info)

        return model, history

    def evaluate_model(self, model, test_loader, dataset_info):
        """Evaluate model on test set"""
        print(f"\n🧪 Final evaluation...")

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

        print(f"📊 Test Results:")
        print(f"   Move Accuracy: {test_acc:.1%}")
        print(f"   Eval Error: {mean_eval_error:.0f} centipawns")
        print(f"   Test Samples: {test_total:,}")

        # Save results
        test_results = {
            "move_accuracy": float(test_acc),
            "mean_eval_error": float(mean_eval_error),
            "total_samples": int(test_total),
        }

        results_path = Path(self.config.model_save_dir) / "test_results.json"
        with open(results_path, "w") as f:
            json.dump(test_results, f, indent=2)


def plot_training_curves(history: Dict, save_dir: str):
    """Plot and save training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss curves
    axes[0, 0].plot(history["train_loss"], label="Train", color="blue")
    axes[0, 0].plot(history["val_loss"], label="Validation", color="red")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Move accuracy
    axes[0, 1].plot(history["train_move_acc"], label="Train", color="blue")
    axes[0, 1].plot(history["val_move_acc"], label="Validation", color="red")
    axes[0, 1].set_title("Move Prediction Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Evaluation loss
    axes[1, 0].plot(history["train_eval_loss"], label="Train", color="blue")
    axes[1, 0].plot(history["val_eval_loss"], label="Validation", color="red")
    axes[1, 0].set_title("Evaluation Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Summary
    best_val_acc = max(history["val_move_acc"])
    final_val_acc = history["val_move_acc"][-1]

    axes[1, 1].text(
        0.1,
        0.8,
        f"Best Val Accuracy: {best_val_acc:.1%}",
        transform=axes[1, 1].transAxes,
        fontsize=12,
        fontweight="bold",
    )
    axes[1, 1].text(
        0.1,
        0.6,
        f"Final Val Accuracy: {final_val_acc:.1%}",
        transform=axes[1, 1].transAxes,
        fontsize=12,
    )
    axes[1, 1].text(
        0.1,
        0.4,
        f"Total Epochs: {len(history['train_loss'])}",
        transform=axes[1, 1].transAxes,
        fontsize=12,
    )
    axes[1, 1].set_title("Training Summary")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(Path(save_dir) / "training_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("✓ Training curves saved")


def create_download_package():
    """Create downloadable package"""
    import zipfile

    print("\n📦 Creating download package...")

    zip_path = "magnus_model_complete.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add model files
        model_dir = Path("magnus_model")
        if model_dir.exists():
            for file_path in model_dir.rglob("*"):
                if file_path.is_file():
                    zipf.write(
                        file_path, f"magnus_model/{file_path.relative_to(model_dir)}"
                    )

        # Add readme
        readme_content = f"""# Magnus Carlsen Style Chess Engine

This package contains a trained neural network that mimics Magnus Carlsen's playing style.

## Files:
- best_magnus_model.pth: Trained PyTorch model
- dataset_info.json: Dataset and vocabulary information
- training_history.json: Training metrics over time
- test_results.json: Final evaluation results
- training_curves.png: Training visualization

## Environment:
- Trained on: {env_config['description']}
- GPU: {env_config['gpu_info'] or 'CPU'}
- Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}

## Usage:
Load the model with:
```python
import torch
checkpoint = torch.load('best_magnus_model.pth')
model_state = checkpoint['model_state_dict']
# Initialize your model and load state_dict
```

Generated by Colab Magnus Trainer v1.0
"""

        zipf.writestr("README.md", readme_content)

    print(f"✓ Package created: {zip_path}")
    return zip_path


def main():
    """Main training function"""
    print(f"\n🎯 Magnus Carlsen Chess Engine Training")
    print(f"   Configuration: {env_config['description']}")
    print(
        f"   Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}"
    )

    # Check for PGN file
    pgn_files = ["carlsen-games.pgn", "magnus_games.pgn", "carlsen-games-quarter.pgn"]
    pgn_found = None

    for pgn_file in pgn_files:
        if Path(pgn_file).exists():
            pgn_found = pgn_file
            break

    if not pgn_found:
        print(f"\n❌ ERROR: No Magnus Carlsen PGN file found!")
        print(f"   Please upload one of these files:")
        for pgn_file in pgn_files:
            print(f"   - {pgn_file}")
        print(f"\n💡 Upload using the file browser in the left panel")
        return

    print(f"✓ Found PGN file: {pgn_found}")

    # Show training time estimate
    if env_config["tier"] == "free":
        print(f"\n⚠️  FREE COLAB - Training time: ~1-2 hours (reduced dataset)")
        print(f"   Using max {env_config['max_games']} games")
    elif env_config["tier"] == "pro":
        print(f"\n🚀 COLAB PRO - Training time: ~20 hours (full dataset)")
    elif env_config["tier"] == "pro_plus":
        print(f"\n🚀 COLAB PRO+ - Training time: ~14 hours (full dataset)")

    # Initialize and train
    config = ColabConfig()
    config.pgn_file = pgn_found

    trainer = ColabMagnusTrainer(config)

    try:
        start_time = time.time()

        model, history = trainer.train()

        training_time = time.time() - start_time
        hours = training_time / 3600

        print(f"\n🎉 TRAINING COMPLETED!")
        print(f"   Total time: {hours:.1f} hours")

        # Create visualizations
        plot_training_curves(history, config.model_save_dir)

        # Create download package
        package_path = create_download_package()

        print(f"\n📁 FILES READY FOR DOWNLOAD:")
        print(f"   📦 Complete package: {package_path}")
        print(f"   🧠 Model file: magnus_model/best_magnus_model.pth")
        print(f"   📊 Results: magnus_model/test_results.json")
        print(f"   📈 Curves: magnus_model/training_curves.png")

        print(f"\n📥 TO DOWNLOAD:")
        print(f"   1. Click the file browser (📁) in the left panel")
        print(f"   2. Right-click on '{package_path}'")
        print(f"   3. Select 'Download'")

        # Show final results
        results_path = Path(config.model_save_dir) / "test_results.json"
        if results_path.exists():
            with open(results_path, "r") as f:
                results = json.load(f)

            print(f"\n🏆 FINAL PERFORMANCE:")
            print(f"   Move Accuracy: {results['move_accuracy']:.1%}")
            print(f"   Evaluation Error: {results['mean_eval_error']:.0f} centipawns")
            print(f"   Test Positions: {results['total_samples']:,}")

        print(f"\n✨ Your Magnus Carlsen-style chess engine is ready!")

    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
