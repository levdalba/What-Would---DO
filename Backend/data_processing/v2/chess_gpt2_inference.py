import os
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
import chess


# GPT-2 based chess model that matches the saved weights
class ChessGPT2Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Create GPT-2 config with the right parameters
        config = GPT2Config(
            vocab_size=50257,  # Standard GPT-2 vocab size
            n_positions=1024,
            n_ctx=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
        )
        self.base_model = GPT2Model(config)
        self.classifier = nn.Linear(768, 4096)  # 64*64 possible moves

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        # Use the last token's hidden state
        logits = self.classifier(outputs.last_hidden_state[:, -1, :])
        return logits


class ChessGPT2Predictor:
    def __init__(self, model_path=None, device=None):
        # Set device
        self.device = device or torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # Initialize model
        self.model = ChessGPT2Model().to(self.device)

        # Load model weights if path provided
        if model_path and os.path.exists(model_path):
            try:
                from safetensors.torch import load_file

                state_dict = load_file(model_path)
                self.model.load_state_dict(state_dict, strict=False)
                print(f"Model loaded successfully from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load model weights: {e}")
                print("Using untrained model")
        else:
            print(
                "No model path provided or file doesn't exist. Using untrained model."
            )

        self.model.eval()

    def fen_to_token_ids(self, fen, max_length=128):
        """Convert FEN to token IDs using chess-aware tokenization"""
        # Split FEN into components
        fen_parts = fen.split()
        board_state = fen_parts[0] if len(fen_parts) > 0 else fen

        # Create a vocabulary for chess pieces and symbols
        chess_vocab = {
            "r": 1,
            "n": 2,
            "b": 3,
            "q": 4,
            "k": 5,
            "p": 6,  # black pieces
            "R": 7,
            "N": 8,
            "B": 9,
            "Q": 10,
            "K": 11,
            "P": 12,  # white pieces
            "/": 13,
            "1": 14,
            "2": 15,
            "3": 16,
            "4": 17,
            "5": 18,
            "6": 19,
            "7": 20,
            "8": 21,
            " ": 22,
            "w": 23,
            "b": 24,
            "K": 25,
            "Q": 26,
            "k": 27,
            "q": 28,
            "-": 29,
        }

        tokens = []
        for char in fen[:max_length]:
            token_id = chess_vocab.get(
                char, ord(char) % 1000 + 100
            )  # fallback for unknown chars
            tokens.append(token_id)

        # Pad to max_length
        tokens.extend([0] * (max_length - len(tokens)))
        return torch.tensor(tokens[:max_length], dtype=torch.long)

    def decode_move(self, index):
        """Convert move index back to chess.Move"""
        from_sq = index // 64
        to_sq = index % 64
        return chess.Move(from_sq, to_sq)

    def predict_move(self, fen):
        """Predict the best move for a given FEN position"""
        try:
            # Tokenize FEN
            input_ids = self.fen_to_token_ids(fen).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # Get model predictions
                logits = self.model(input_ids)
                probabilities = torch.softmax(logits, dim=-1)

                # Create chess board for legal move validation
                board = chess.Board(fen)
                legal_moves = list(board.legal_moves)

                if not legal_moves:
                    return None, 0.0

                # Find the highest probability legal move
                best_move = None
                best_confidence = 0.0

                # Sort probabilities in descending order
                sorted_probs, sorted_indices = torch.sort(
                    probabilities[0], descending=True
                )

                # Check top predictions for legal moves
                for prob, idx in zip(
                    sorted_probs[:100], sorted_indices[:100]
                ):  # Check top 100
                    candidate_move = self.decode_move(idx.item())
                    if candidate_move in legal_moves:
                        best_move = candidate_move
                        best_confidence = prob.item() * 100
                        break

                # If no legal move found in top predictions, use first legal move
                if best_move is None:
                    best_move = legal_moves[0]
                    best_confidence = 0.1  # Low confidence fallback

                return best_move, best_confidence

        except Exception as e:
            print(f"Error in prediction: {e}")
            # Fallback: return a random legal move
            board = chess.Board(fen)
            legal_moves = list(board.legal_moves)
            if legal_moves:
                return legal_moves[0], 0.0
            return None, 0.0

    def predict_move_uci(self, fen):
        """Predict move and return in UCI format"""
        move, confidence = self.predict_move(fen)
        if move:
            return move.uci(), confidence
        return "0000", 0.0
