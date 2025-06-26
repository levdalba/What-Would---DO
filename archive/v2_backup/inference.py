import torch
import chess
import torch.nn as nn

# PIECE_TO_ID mapping for FEN processing
PIECE_TO_ID = {
    '.': 0,
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
}

class MagnusTransformer(nn.Module):
    def __init__(self, vocab_size=15, seq_len=65, num_moves=4096, d_model=128, nhead=8, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, d_model))  # learned positional encoding

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model * seq_len, num_moves)

    def forward(self, x):
        x = self.embedding(x)  # (batch, seq_len, d_model)
        x = x + self.pos_encoding.unsqueeze(0)  # Add position encoding
        x = x.permute(1, 0, 2)  # (seq_len, batch, d_model)
        x = self.transformer(x)  # (seq_len, batch, d_model)
        x = x.permute(1, 0, 2).reshape(x.shape[1], -1)  # (batch, seq_len * d_model)
        out = self.fc(x)  # (batch, num_moves)
        return out
    
# Function to convert FEN to tokens
def fen_to_tokens(fen):
    board = chess.Board(fen)
    tokens = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        tokens.append(PIECE_TO_ID[piece.symbol()] if piece else 0)
    tokens.append(13 if board.turn == chess.WHITE else 14)
    return torch.tensor(tokens, dtype=torch.long)

# Function to encode a move into an integer
def encode_move(move_str):
    try:
        move = chess.Move.from_uci(move_str)
        return move.from_square * 64 + move.to_square  # 0–4095
    except ValueError:
        return None  # If the move is invalid, return None

# Function to decode the predicted move back to UCI format
def decode_move(index):
    from_sq = index // 64
    to_sq = index % 64
    return chess.Move(from_sq, to_sq)

# Load the fine-tuned model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MagnusTransformer().to(device)
model.load_state_dict(torch.load("data_processing/v2/magnus_transformer_finetuned.pth"))
model.eval()

# Function to predict the next move given a FEN string
def predict_move(fen):
    x = fen_to_tokens(fen).unsqueeze(0).to(device)  # Add batch dimension and move to device
    with torch.no_grad():  # Disable gradient tracking for inference
        logits = model(x)  # Get model output (predicted logits)
        predicted_index = logits.argmax(dim=1).item()  # Get the index of the predicted move
    
    move = decode_move(predicted_index)  # Decode the predicted move index
    return move

# Example to input current board FEN and predict the move
def input_and_predict_move():
    fen = input("Enter the current board in FEN format: ").strip()  # Take FEN string as input
    predicted_move = predict_move(fen)
    print(f"Predicted move: {predicted_move.uci()}")

# Call the function to input FEN and predict the move
input_and_predict_move()
