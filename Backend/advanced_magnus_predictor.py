#!/usr/bin/env python3
"""
Advanced Magnus Model Backend Integration
Loads and serves the latest trained advanced Magnus model for FastAPI
"""

import sys
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import chess
import chess.pgn
import yaml
import json
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


class AdvancedChessFeatureExtractor:
    """Extract advanced chess features for better move prediction"""

    def __init__(self):
        self.piece_values = {
            "p": 1,
            "n": 3,
            "b": 3,
            "r": 5,
            "q": 9,
            "k": 0,
            "P": 1,
            "N": 3,
            "B": 3,
            "R": 5,
            "Q": 9,
            "K": 0,
        }

    def extract_features(self, position_data):
        """Extract comprehensive position features"""
        features = []

        # Basic piece counts and material balance
        white_material = sum(
            self.piece_values.get(p, 0) for p in str(position_data) if p.isupper()
        )
        black_material = sum(
            self.piece_values.get(p, 0) for p in str(position_data) if p.islower()
        )
        material_balance = white_material - black_material

        # Feature vector
        features.extend(
            [
                white_material / 39.0,  # Normalized material (max = Q+2R+2B+2N+8P)
                black_material / 39.0,
                material_balance / 39.0,
                abs(material_balance) / 39.0,  # Material imbalance magnitude
            ]
        )

        # Game phase estimation (opening/middlegame/endgame)
        total_material = white_material + black_material
        game_phase = total_material / 78.0  # 0 = endgame, 1 = opening
        features.extend(
            [
                game_phase,
                1 - game_phase,  # Endgame indicator
                min(game_phase * 2, 1),  # Opening indicator
                max(0, min((game_phase - 0.3) * 2, 1)),  # Middlegame indicator
            ]
        )

        return np.array(features, dtype=np.float32)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for position encoding"""

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.size(0)

        # Linear transformations
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)

        # Concatenate heads
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )
        output = self.W_o(context)

        return output.mean(dim=1)  # Global average pooling


class AdvancedMagnusModel(nn.Module):
    """Advanced Magnus model architecture matching the trained model"""

    def __init__(self, vocab_size: int, feature_dim: int = 8):
        super().__init__()
        self.vocab_size = vocab_size

        # Advanced board encoder with residual connections
        self.board_encoder = nn.Sequential(
            nn.Linear(768, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        # Multi-head attention mechanism for board understanding
        self.board_attention = MultiHeadAttention(256, 8)

        # Advanced feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Combined feature processing
        combined_dim = 256 + 32
        self.feature_combiner = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Move prediction with multiple paths
        self.move_predictor = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, vocab_size),
        )

        # Evaluation head
        self.eval_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, position, features):
        # Process board position
        board_enc = self.board_encoder(position)

        # Apply attention (reshape for attention if needed)
        if len(board_enc.shape) == 2:
            board_enc_reshaped = board_enc.unsqueeze(1)  # Add sequence dimension
            board_att = self.board_attention(board_enc_reshaped)
        else:
            board_att = self.board_attention(board_enc)

        # Process additional features
        feature_enc = self.feature_encoder(features)

        # Combine features
        combined = torch.cat([board_att, feature_enc], dim=1)
        combined = self.feature_combiner(combined)

        # Predictions
        move_logits = self.move_predictor(combined)
        eval_pred = self.eval_predictor(combined)

        return move_logits, eval_pred


class AdvancedMagnusPredictor:
    """Advanced Magnus model predictor for FastAPI backend"""

    def __init__(self, model_path: Optional[str] = None):
        self.device = self._get_device()
        self.model = None
        self.move_to_idx = {}
        self.idx_to_move = {}
        self.vocab_size = 0
        self.model_config = {}
        self.feature_extractor = AdvancedChessFeatureExtractor()

        # Default to latest MLflow model if no path provided
        if model_path is None:
            model_path = self._get_latest_mlflow_model()

        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            print(f"⚠️ Model path not found: {model_path}")

    def _get_device(self):
        """Get the best available device"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def _get_latest_mlflow_model(self):
        """Get the latest MLflow model path"""
        # Try multiple possible paths
        possible_paths = [
            project_root
            / "mlruns"
            / "427589957554434254"
            / "cbb3fccf10b64db5a8985add8bcac5ef"
            / "artifacts"
            / "model_artifacts",
            Path(__file__).parent.parent
            / "mlruns"
            / "427589957554434254"
            / "cbb3fccf10b64db5a8985add8bcac5ef"
            / "artifacts"
            / "model_artifacts",
            Path(
                "/Users/levandalbashvili/Documents/GitHub/What-Would---DO/mlruns/427589957554434254/cbb3fccf10b64db5a8985add8bcac5ef/artifacts/model_artifacts"
            ),
        ]

        for path in possible_paths:
            if path.exists():
                print(f"✅ Found model at: {path}")
                return str(path)

        print(f"❌ Model not found in any of these paths:")
        for path in possible_paths:
            print(f"   - {path}")
        return None

    def load_model(self, model_path: str):
        """Load the trained model"""
        try:
            model_path = Path(model_path)

            # Load configuration
            config_file = model_path / "config.yaml"
            if config_file.exists():
                with open(config_file, "r") as f:
                    self.model_config = yaml.safe_load(f)
                print(f"✅ Loaded model config: {config_file}")

            # Load version info
            version_file = model_path / "version.json"
            if version_file.exists():
                with open(version_file, "r") as f:
                    version_info = json.load(f)
                print(f"✅ Model version: {version_info.get('model_id', 'unknown')}")

            # Load the model state dict
            model_file = model_path / "model.pth"
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_file}")

            checkpoint = torch.load(model_file, map_location=self.device)

            # Extract model components
            if "model_state_dict" in checkpoint:
                model_state = checkpoint["model_state_dict"]
                self.move_to_idx = checkpoint.get("move_to_idx", {})
                self.idx_to_move = checkpoint.get("idx_to_move", {})
                self.vocab_size = checkpoint.get("vocab_size", len(self.move_to_idx))
            else:
                # Handle direct state dict
                model_state = checkpoint

                # Try to load vocabulary from config
                vocab_size = self.model_config.get("vocab_size", 2000)  # Default
                self.vocab_size = vocab_size

            # Check if vocabulary is missing and create it
            if not self.move_to_idx or len(self.move_to_idx) == 0:
                print(
                    "⚠️ Move vocabulary not found in checkpoint, creating from chess games"
                )
                # Get vocab size from model architecture
                vocab_size = self.model_config.get("data", {}).get("vocab_size", 945)
                self.vocab_size = vocab_size
                self._create_vocabulary_from_games()

            # Initialize model
            vocab_size = self.model_config.get("data", {}).get("vocab_size", 945)
            feature_dim = 8  # The saved model was trained with 8 features
            self.vocab_size = vocab_size
            self.model = AdvancedMagnusModel(self.vocab_size, feature_dim).to(
                self.device
            )

            # Load state dict
            self.model.load_state_dict(model_state)
            self.model.eval()

            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"✅ Advanced Magnus model loaded successfully!")
            print(f"   Device: {self.device}")
            print(f"   Parameters: {total_params:,}")
            print(f"   Vocabulary size: {self.vocab_size}")
            print(f"   Model path: {model_path}")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None

    def _create_vocabulary_from_games(self):
        """Create vocabulary from actual chess games (like the training data)"""
        print("🔧 Creating vocabulary from Magnus Carlsen games...")

        moves = set()

        # Try to load moves from available PGN files
        pgn_paths = [
            Path(__file__).parent / "data_processing" / "carlsen-games-quarter.pgn",
            Path(__file__).parent / "data_processing" / "carlsen-games.pgn",
        ]

        games_processed = 0
        for pgn_path in pgn_paths:
            if pgn_path.exists():
                print(f"📖 Reading moves from {pgn_path.name}...")
                try:
                    with open(pgn_path, "r") as f:
                        while True:
                            game = chess.pgn.read_game(f)
                            if game is None:
                                break

                            # Extract all moves from the game
                            board = game.board()
                            for move in game.mainline_moves():
                                moves.add(move.uci())
                                board.push(move)

                            games_processed += 1
                            if games_processed % 100 == 0:
                                print(
                                    f"   Processed {games_processed} games, {len(moves)} unique moves"
                                )

                            # Limit games processed to avoid too long loading
                            if games_processed >= 500:
                                break

                    if moves:
                        break  # We have enough moves from this file

                except Exception as e:
                    print(f"   ⚠️ Error reading {pgn_path}: {e}")
                    continue

        # If we couldn't read from PGN files, fall back to comprehensive UCI generation
        if not moves:
            print("📝 Falling back to comprehensive UCI move generation...")
            moves = self._generate_comprehensive_uci_moves()

        # Convert to sorted list and limit to vocab_size
        moves_list = sorted(list(moves))
        if len(moves_list) > self.vocab_size:
            # Keep the most common/basic moves first
            basic_moves = []
            promotion_moves = []
            other_moves = []

            for move in moves_list:
                if len(move) == 5 and move[4] in "qrbn":  # Promotion
                    promotion_moves.append(move)
                elif len(move) == 4:  # Basic move
                    basic_moves.append(move)
                else:
                    other_moves.append(move)

            # Prioritize basic moves, then promotions, then others
            moves_list = (basic_moves + promotion_moves + other_moves)[
                : self.vocab_size
            ]

        # Pad if needed
        while len(moves_list) < self.vocab_size:
            moves_list.append(f"null_move_{len(moves_list)}")

        self.move_to_idx = {move: idx for idx, move in enumerate(moves_list)}
        self.idx_to_move = {idx: move for move, idx in self.move_to_idx.items()}

        print(
            f"✅ Created vocabulary with {len(self.move_to_idx)} moves from {games_processed} games"
        )
        print(f"   Sample moves: {moves_list[:10]}")
        print(f"   Last moves: {moves_list[-10:]}")

    def _generate_comprehensive_uci_moves(self):
        """Generate comprehensive UCI moves as fallback"""
        moves = set()
        files = "abcdefgh"
        ranks = "12345678"

        # All possible square-to-square moves
        for from_file in files:
            for from_rank in ranks:
                for to_file in files:
                    for to_rank in ranks:
                        from_sq = from_file + from_rank
                        to_sq = to_file + to_rank
                        if from_sq != to_sq:
                            moves.add(from_sq + to_sq)

        # Pawn promotions
        promotion_pieces = ["q", "r", "b", "n"]
        for from_file in files:
            for to_file in files:
                # White promotions (rank 7 to 8)
                for piece in promotion_pieces:
                    moves.add(f"{from_file}7{to_file}8{piece}")
                # Black promotions (rank 2 to 1)
                for piece in promotion_pieces:
                    moves.add(f"{from_file}2{to_file}1{piece}")

        return moves

    def board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        """Convert chess board to tensor representation"""
        # Create 768-dimensional board representation (8x8x12)
        board_tensor = np.zeros((8, 8, 12), dtype=np.float32)

        piece_map = {
            chess.PAWN: 0,
            chess.ROOK: 1,
            chess.KNIGHT: 2,
            chess.BISHOP: 3,
            chess.QUEEN: 4,
            chess.KING: 5,
        }

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                rank, file = divmod(square, 8)
                piece_type = piece_map[piece.piece_type]
                color_offset = 0 if piece.color == chess.WHITE else 6
                board_tensor[rank, file, piece_type + color_offset] = 1.0

        return torch.FloatTensor(board_tensor.flatten())

    def extract_features(self, board: chess.Board) -> torch.Tensor:
        """Extract advanced features from the board position"""
        # Get FEN string for the feature extractor
        fen = board.fen()

        # Use the advanced feature extractor
        features = self.feature_extractor.extract_features(fen)

        return torch.FloatTensor(features)

    def predict_moves(self, board: chess.Board, top_k: int = 5) -> List[Dict[str, Any]]:
        """Predict top-k moves prioritizing best moves with Magnus style flavor"""
        if self.model is None:
            return [{"move": "e2e4", "confidence": 0.5, "evaluation": 0.0}]

        try:
            # Get legal moves first
            legal_moves = list(board.legal_moves)

            if not legal_moves:
                return []

            # Strategy: Start with chess engine quality, then add Magnus flavor
            predictions = []

            # Get quick engine analysis for all legal moves
            try:
                import chess.engine

                with chess.engine.SimpleEngine.popen_uci(
                    "/opt/homebrew/bin/stockfish"
                ) as engine:
                    # Analyze current position briefly
                    main_info = engine.analyse(board, chess.engine.Limit(time=0.1))

                    for legal_move in legal_moves:
                        # Make the move and evaluate
                        board_copy = board.copy()
                        board_copy.push(legal_move)

                        try:
                            # Quick evaluation
                            move_info = engine.analyse(
                                board_copy, chess.engine.Limit(time=0.03)
                            )
                            move_score = move_info.get(
                                "score",
                                chess.engine.PovScore(
                                    chess.engine.Cp(0), board_copy.turn
                                ),
                            )

                            # Calculate move quality based on engine
                            if move_score.is_mate():
                                if move_score.mate() > 0:
                                    engine_quality = 0.95
                                else:
                                    engine_quality = 0.05
                            else:
                                # Get centipawn evaluation from the side to move's perspective
                                cp_score = move_score.white().score(mate_score=10000)
                                if not board.turn:  # Black to move
                                    cp_score = -cp_score

                                # Convert to quality score (0.1 to 0.9)
                                engine_quality = max(
                                    0.1, min(0.9, 0.5 + cp_score / 300)
                                )

                        except:
                            engine_quality = 0.5  # Neutral if evaluation fails

                        # Add Magnus style bonus (small influence)
                        magnus_bonus = 0.0
                        move_uci = legal_move.uci()

                        # Check if move is in Magnus's vocabulary
                        if move_uci in self.move_to_idx:
                            try:
                                with torch.no_grad():
                                    position_tensor = (
                                        self.board_to_tensor(board)
                                        .unsqueeze(0)
                                        .to(self.device)
                                    )
                                    features_tensor = (
                                        self.extract_features(board)
                                        .unsqueeze(0)
                                        .to(self.device)
                                    )
                                    move_logits, _ = self.model(
                                        position_tensor, features_tensor
                                    )
                                    move_probs = F.softmax(move_logits, dim=1)

                                    idx = self.move_to_idx[move_uci]
                                    magnus_style_score = float(
                                        move_probs[0, idx].item()
                                    )
                                    magnus_bonus = (
                                        magnus_style_score * 0.1
                                    )  # Only 10% influence
                            except:
                                magnus_bonus = 0.0

                        # Apply chess heuristics
                        heuristic_bonus = self._calculate_heuristic_bonus(
                            board, legal_move
                        )

                        # Final score: 80% engine quality, 10% Magnus style, 10% heuristics
                        final_confidence = (
                            0.8 * engine_quality
                            + 0.1 * magnus_bonus
                            + 0.1 * heuristic_bonus
                        )

                        predictions.append(
                            {
                                "move": move_uci,
                                "confidence": final_confidence,
                                "evaluation": (
                                    cp_score if "cp_score" in locals() else 0.0
                                ),
                                "engine_quality": engine_quality,
                                "magnus_bonus": magnus_bonus,
                                "heuristic_bonus": heuristic_bonus,
                                "is_legal": True,
                                "approach": "engine_primary",
                            }
                        )

            except Exception as e:
                print(f"Engine analysis failed, using heuristics only: {e}")
                # Fallback to heuristics-based approach
                for legal_move in legal_moves:
                    move_uci = legal_move.uci()

                    # Base quality from heuristics
                    heuristic_score = self._calculate_comprehensive_heuristic_score(
                        board, legal_move
                    )

                    # Small Magnus style influence
                    magnus_bonus = 0.0
                    if move_uci in self.move_to_idx:
                        try:
                            with torch.no_grad():
                                position_tensor = (
                                    self.board_to_tensor(board)
                                    .unsqueeze(0)
                                    .to(self.device)
                                )
                                features_tensor = (
                                    self.extract_features(board)
                                    .unsqueeze(0)
                                    .to(self.device)
                                )
                                move_logits, _ = self.model(
                                    position_tensor, features_tensor
                                )
                                move_probs = F.softmax(move_logits, dim=1)

                                idx = self.move_to_idx[move_uci]
                                magnus_style_score = float(move_probs[0, idx].item())
                                magnus_bonus = (
                                    magnus_style_score * 0.2
                                )  # Slightly higher without engine
                        except:
                            magnus_bonus = 0.0

                    final_confidence = 0.8 * heuristic_score + 0.2 * magnus_bonus

                    predictions.append(
                        {
                            "move": move_uci,
                            "confidence": final_confidence,
                            "evaluation": 0.0,
                            "heuristic_score": heuristic_score,
                            "magnus_bonus": magnus_bonus,
                            "is_legal": True,
                            "approach": "heuristic_primary",
                        }
                    )

            # Sort by confidence and return top-k
            predictions.sort(key=lambda x: x["confidence"], reverse=True)
            return predictions[:top_k]

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            # Return safe defaults with legal moves
            legal_moves = list(board.legal_moves)
            if legal_moves:
                return [
                    {
                        "move": legal_moves[i % len(legal_moves)].uci(),
                        "confidence": max(0.15 - i * 0.02, 0.05),
                        "evaluation": 0.0,
                        "error": str(e),
                        "approach": "fallback",
                    }
                    for i in range(min(top_k, len(legal_moves)))
                ]
            else:
                return [
                    {
                        "move": "e2e4",
                        "confidence": 0.1,
                        "evaluation": 0.0,
                        "error": str(e),
                    }
                ]

    def _calculate_heuristic_bonus(self, board: chess.Board, move: chess.Move) -> float:
        """Calculate a small heuristic bonus for the move"""
        bonus = 0.0
        piece = board.piece_at(move.from_square)

        if piece:
            # Center control
            center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
            if move.to_square in center_squares:
                bonus += 0.05

            # Piece development in opening
            if (
                piece.piece_type in [chess.KNIGHT, chess.BISHOP]
                and board.fullmove_number <= 10
            ):
                bonus += 0.03

            # Captures
            if board.is_capture(move):
                captured = board.piece_at(move.to_square)
                if captured:
                    piece_values = {
                        chess.PAWN: 1,
                        chess.KNIGHT: 3,
                        chess.BISHOP: 3,
                        chess.ROOK: 5,
                        chess.QUEEN: 9,
                    }
                    if piece_values.get(captured.piece_type, 0) >= piece_values.get(
                        piece.piece_type, 0
                    ):
                        bonus += 0.04

            # Checks
            if board.gives_check(move):
                bonus += 0.02

            # Castling
            if board.is_castling(move) and board.fullmove_number <= 15:
                bonus += 0.06

        return min(bonus, 0.15)  # Cap the bonus

    def _calculate_comprehensive_heuristic_score(
        self, board: chess.Board, move: chess.Move
    ) -> float:
        """Calculate a comprehensive heuristic score for a move (used when engine is unavailable)"""
        score = 0.5  # Base score
        piece = board.piece_at(move.from_square)

        if piece:
            # Piece values and basic principles
            piece_values = {
                chess.PAWN: 1,
                chess.KNIGHT: 3,
                chess.BISHOP: 3,
                chess.ROOK: 5,
                chess.QUEEN: 9,
                chess.KING: 0,
            }

            # Center control (major bonus)
            center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
            extended_center = [
                chess.C3,
                chess.C4,
                chess.C5,
                chess.C6,
                chess.D3,
                chess.D6,
                chess.E3,
                chess.E6,
                chess.F3,
                chess.F4,
                chess.F5,
                chess.F6,
            ]

            if move.to_square in center_squares:
                score += 0.15
            elif move.to_square in extended_center:
                score += 0.08

            # Opening principles
            if board.fullmove_number <= 10:
                if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                    score += 0.12  # Develop pieces
                elif (
                    piece.piece_type == chess.PAWN and move.to_square in center_squares
                ):
                    score += 0.10  # Central pawns

            # Captures (evaluate by material gain)
            if board.is_capture(move):
                captured = board.piece_at(move.to_square)
                if captured:
                    material_gain = piece_values.get(
                        captured.piece_type, 0
                    ) - piece_values.get(piece.piece_type, 0)
                    if material_gain >= 0:
                        score += min(0.2, 0.05 + material_gain * 0.02)
                    else:
                        score -= 0.1  # Bad capture

            # Castling
            if board.is_castling(move):
                score += 0.15

            # Checks (can be good or bad)
            if board.gives_check(move):
                score += 0.05  # Modest bonus for checks

            # Avoid moving same piece twice in opening
            if board.fullmove_number <= 8:
                # Check if this piece has moved before
                moves_history = list(board.move_stack)
                piece_moved_before = any(
                    m.from_square == move.from_square for m in moves_history[-6:]
                )
                if piece_moved_before and piece.piece_type != chess.PAWN:
                    score -= 0.08

        return max(0.1, min(0.9, score))  # Clamp between 0.1 and 0.9

    def predict_moves_with_engine_guidance(
        self,
        board: chess.Board,
        top_k: int = 5,
        engine_path: str = "/opt/homebrew/bin/stockfish",
    ) -> List[Dict[str, Any]]:
        """Predict moves combining Magnus style with engine guidance for better quality"""
        try:
            import chess.engine

            # Get Magnus-style predictions first
            magnus_predictions = self.predict_moves(
                board, top_k * 2
            )  # Get more candidates

            # Analyze with engine
            with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
                # Get top engine moves
                info = engine.analyse(
                    board, chess.engine.Limit(time=0.1), multipv=top_k
                )

                enhanced_predictions = []

                for pred in magnus_predictions:
                    move_uci = pred["move"]
                    try:
                        move = chess.Move.from_uci(move_uci)
                        if move in board.legal_moves:
                            # Get engine evaluation of this move
                            board_copy = board.copy()
                            board_copy.push(move)

                            try:
                                eval_info = engine.analyse(
                                    board_copy, chess.engine.Limit(time=0.05)
                                )
                                score = eval_info.get("score")

                                # Convert engine score to confidence adjustment
                                engine_confidence = 0.5  # Base
                                if score:
                                    if score.is_mate():
                                        if score.mate() > 0:
                                            engine_confidence = 0.95
                                        else:
                                            engine_confidence = 0.05
                                    else:
                                        cp_score = score.white().score(mate_score=10000)
                                        if board.turn == chess.BLACK:
                                            cp_score = -cp_score
                                        # Convert centipawn to confidence (better moves get higher confidence)
                                        engine_confidence = max(
                                            0.1, min(0.9, 0.5 + cp_score / 500)
                                        )

                                # Blend Magnus style with engine evaluation
                                magnus_weight = 0.6  # 60% Magnus style
                                engine_weight = 0.4  # 40% engine evaluation

                                blended_confidence = (
                                    magnus_weight * pred["confidence"]
                                    + engine_weight * engine_confidence
                                )

                                enhanced_predictions.append(
                                    {
                                        "move": move_uci,
                                        "confidence": blended_confidence,
                                        "evaluation": pred.get("evaluation", 0.0),
                                        "magnus_confidence": pred["confidence"],
                                        "engine_confidence": engine_confidence,
                                        "style": "magnus_engine_hybrid",
                                    }
                                )

                            except:
                                # If engine analysis fails, use original prediction
                                enhanced_predictions.append(pred)
                    except:
                        continue

                # Sort by blended confidence
                enhanced_predictions.sort(key=lambda x: x["confidence"], reverse=True)
                return enhanced_predictions[:top_k]

        except Exception as e:
            print(f"Engine guidance failed, falling back to Magnus-only: {e}")
            return self.predict_moves(board, top_k)

    def _apply_chess_heuristics(
        self, board: chess.Board, predictions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply chess heuristics to improve prediction quality"""

        for pred in predictions:
            move_uci = pred["move"]
            try:
                move = chess.Move.from_uci(move_uci)
                confidence_boost = 0.0

                # Boost confidence for good chess principles
                piece = board.piece_at(move.from_square)
                if piece:
                    # Center control (e4, e5, d4, d5)
                    center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
                    if move.to_square in center_squares:
                        confidence_boost += 0.02

                    # Piece development (knights and bishops)
                    if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                        if board.fullmove_number <= 10:  # Opening phase
                            confidence_boost += 0.03

                    # Captures are generally good
                    if board.is_capture(move):
                        captured_piece = board.piece_at(move.to_square)
                        if captured_piece:
                            # Higher value captures get more boost
                            piece_values = {
                                chess.PAWN: 1,
                                chess.KNIGHT: 3,
                                chess.BISHOP: 3,
                                chess.ROOK: 5,
                                chess.QUEEN: 9,
                            }
                            capture_value = piece_values.get(
                                captured_piece.piece_type, 0
                            )
                            attacking_value = piece_values.get(piece.piece_type, 0)
                            if capture_value >= attacking_value:  # Good trades
                                confidence_boost += 0.04

                    # Checks can be good (but not always)
                    if board.gives_check(move):
                        confidence_boost += 0.02

                    # Castling is usually good in opening/middlegame
                    if board.is_castling(move) and board.fullmove_number <= 15:
                        confidence_boost += 0.05

                # Apply the boost
                pred["confidence"] = min(0.95, pred["confidence"] + confidence_boost)
                pred["heuristic_boost"] = confidence_boost

            except Exception as e:
                # If we can't analyze the move, keep original confidence
                pred["heuristic_boost"] = 0.0

        return predictions

    def is_loaded(self) -> bool:
        """Check if the model is successfully loaded"""
        return self.model is not None


# Global instance for FastAPI
_magnus_predictor = None


def get_magnus_predictor() -> AdvancedMagnusPredictor:
    """Get the global Magnus predictor instance"""
    global _magnus_predictor
    if _magnus_predictor is None:
        _magnus_predictor = AdvancedMagnusPredictor()
    return _magnus_predictor


def test_predictor():
    """Test the predictor with a simple position"""
    predictor = AdvancedMagnusPredictor()

    if predictor.is_loaded():
        board = chess.Board()
        predictions = predictor.predict_moves(board, top_k=3)

        print("🧪 Test Predictions:")
        for i, pred in enumerate(predictions, 1):
            print(f"  {i}. {pred['move']} (confidence: {pred['confidence']:.3f})")
    else:
        print("❌ Predictor not loaded")


if __name__ == "__main__":
    test_predictor()
