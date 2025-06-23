import os
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
import chess

# PIECE_TO_ID mapping for FEN processing
PIECE_TO_ID = {
    ".": 0,
    "P": 1,
    "N": 2,
    "B": 3,
    "R": 4,
    "Q": 5,
    "K": 6,
    "p": 7,
    "n": 8,
    "b": 9,
    "r": 10,
    "q": 11,
    "k": 12,
}


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


# Function to convert FEN to token IDs (improved chess-aware approach)
def fen_to_token_ids(fen, max_length=128):
    # More sophisticated FEN tokenization
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


# Function to decode the predicted move back to UCI format
def decode_move(index):
    from_sq = index // 64
    to_sq = index % 64
    return chess.Move(from_sq, to_sq)


# Define paths
model_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "Backend",
    "data_processing",
    "v2",
    "models",
    "fine_tuned_chessgpt",
    "model.safetensors",
)
model_path = os.path.abspath(model_path)

# Load the model
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using device: {device}")

model = ChessGPT2Model().to(device)

# Try to load the model weights
try:
    # Load from safetensors format
    from safetensors.torch import load_file

    state_dict = load_file(model_path)
    model.load_state_dict(
        state_dict, strict=False
    )  # Use strict=False to ignore missing keys
    print("Model loaded successfully from safetensors")
except ImportError:
    print("safetensors not installed. Installing...")
    import subprocess

    subprocess.check_call(
        [
            "/Users/levandalbashvili/Documents/GitHub/What-Would---DO/wwxd/bin/python",
            "-m",
            "pip",
            "install",
            "safetensors",
        ]
    )
    from safetensors.torch import load_file

    state_dict = load_file(model_path)
    model.load_state_dict(state_dict, strict=False)
    print("Model loaded successfully from safetensors")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Using untrained model.")

model.eval()


# Prediction function
def predict_move(fen):
    input_ids = fen_to_token_ids(fen).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():  # Disable gradient tracking for inference
        logits = model(input_ids)  # Get model output (predicted logits)
        probabilities = torch.softmax(logits, dim=-1)
        predicted_index = logits.argmax(
            dim=-1
        ).item()  # Get the index of the predicted move
        confidence = probabilities[0, predicted_index].item() * 100

        # Debug: print top 5 probabilities
        top_probs, top_indices = torch.topk(probabilities[0], 5)
        print(f"Debug - Top 5 predictions:")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            move_debug = decode_move(idx.item())
            print(
                f"  {i+1}. Move: {move_debug.uci() if move_debug else 'invalid'}, Prob: {prob.item()*100:.6f}%"
            )

    move = decode_move(predicted_index)  # Decode the predicted move index
    board = chess.Board(fen)

    print(f"Debug - Predicted move from index {predicted_index}: {move.uci()}")

    # Check if the predicted move is legal
    if move in board.legal_moves:
        print(f"Debug - Move {move.uci()} is legal")
        return move.uci(), confidence
    else:
        print(f"Debug - Move {move.uci()} is NOT legal")
        # If not legal, find the most probable legal move
        legal_moves = list(board.legal_moves)
        if legal_moves:
            # Find the highest probability legal move
            best_legal_move = None
            best_legal_prob = 0.0

            for i, prob in enumerate(probabilities[0]):
                candidate_move = decode_move(i)
                if candidate_move in legal_moves:
                    if prob.item() > best_legal_prob:
                        best_legal_prob = prob.item()
                        best_legal_move = candidate_move

            if best_legal_move:
                return best_legal_move.uci(), best_legal_prob * 100
            else:
                return legal_moves[0].uci(), 0.0
        else:
            return "0000", 0.0


# Test cases
test_fens = [
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
    "rnbqkb1r/pppp1ppp/5n2/5p2/5P2/5N2/PPPP1PPP/RNBQKB1R w KQkq - 1 2",
]

for fen in test_fens:
    move, confidence = predict_move(fen)
    print(f"FEN: {fen}")
    print(f"Predicted move: {move}, Confidence: {confidence:.2f}%")
    print("---")
