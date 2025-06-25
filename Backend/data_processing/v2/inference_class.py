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
    def __init__(self, vocab_size=15, seq_len=65, num_moves=4096, d_model=256, nhead=16, num_layers=6, dim_feedforward=8192, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)  # match saved model
        self.pos = nn.Parameter(torch.randn(seq_len, d_model))  # match saved model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.tr = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  # match saved model

        self.fc = nn.Linear(d_model * seq_len, num_moves)  # [4096, 16640] if 256x65

    def forward(self, x):
        x = self.embed(x) + self.pos.unsqueeze(0)  # (batch, seq_len, d_model)
        x = self.tr(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)

class ChessModel:
    def __init__(self, model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MagnusTransformer().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))  # Safe load
        self.model.eval()

    def fen_to_tokens(self, fen):
        board = chess.Board(fen)
        tokens = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            tokens.append(PIECE_TO_ID[piece.symbol()] if piece else 0)
        tokens.append(13 if board.turn == chess.WHITE else 14)
        return torch.tensor(tokens, dtype=torch.long)

    def encode_move(self, move_str):
        try:
            move = chess.Move.from_uci(move_str)
            return move.from_square * 64 + move.to_square
        except ValueError:
            return None

    def decode_move(self, index):
        from_sq = index // 64
        to_sq = index % 64
        return chess.Move(from_sq, to_sq)

    def predict_move(self, fen):
        x = self.fen_to_tokens(fen).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            predicted_index = logits.argmax(dim=1).item()
        return self.decode_move(predicted_index)

    @staticmethod
    def input_and_predict_move(input_board):
        model_path = "data_processing/v2/models/magnus_transformer_finetuned_4th_general_all_carlsen.pth"
        chess_model = ChessModel(model_path)
        predicted_move = chess_model.predict_move(input_board)
        print(f"Predicted move: {predicted_move.uci()}")
