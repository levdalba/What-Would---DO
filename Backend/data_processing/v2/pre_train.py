import torch.nn as nn
import torch
import chess.pgn
# from io import StringIO
# from torch.utils.data import Dataset
# from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import chess

PIECE_TO_ID = {
    '.': 0,
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
}

# Function to convert FEN to tokens (same as before)
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
    except ValueError:  # In case the move string is not valid UCI
        return None  # Invalid move, return None

# Function to process the data from the text file
def process_data(file_path):
    with open(file_path, "r") as f:
        data = f.readlines()
    
    positions = []
    
    for line in data:
        # Strip any extra whitespace and split by space
        parts = line.strip().split()
        
        # Debug: Print the line and its parts to see how it's split
        # print(f"Line: {line.strip()}")
        # print(f"Parts: {parts}")
        
        if len(parts) != 2:  # Skip invalid lines with less than 2 parts
            print(f"Skipping invalid line: {line.strip()}")
            continue
        
        fen, move = parts[0], parts[1]
        
        # Only add valid moves
        if encode_move(move) is not None:
            positions.append((fen, move))
        else:
            print(f"Invalid move skipped: {move}")
    
    # Debug: Print how many valid positions were processed
    print(f"Loaded {len(positions)} valid samples from {file_path}.")
    return positions


# Load the datasets
general_data = process_data("data_processing/v2/training_data/general-4th-data.txt")
carlsen_data = process_data("data_processing/v2/training_data/all_formatted_carlson_data.txt")


# Dataset class for training
class MagnusTransformerDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.samples = data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fen, move = self.samples[idx]
        x = fen_to_tokens(fen)
        y = encode_move(move)
        return x, torch.tensor(y)

# Model definition (same as before)
class MagnusTransformer(nn.Module):
    def __init__(self, vocab_size=15, seq_len=65, num_moves=4096, d_model=128, nhead=8, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, d_model))  # learned positional encoding

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model * seq_len, num_moves)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x)  # (batch, seq_len, d_model)
        x = x + self.pos_encoding.unsqueeze(0)  # Add position encoding
        x = x.permute(1, 0, 2)  # (seq_len, batch, d_model)
        x = self.transformer(x)  # (seq_len, batch, d_model)
        x = x.permute(1, 0, 2).reshape(x.shape[1], -1)  # (batch, seq_len * d_model)
        out = self.fc(x)  # (batch, num_moves)
        return out

# Pretrain the model on the general dataset
print(f"Pretraining on {len(general_data)} samples from the general dataset.")
train_data = MagnusTransformerDataset(general_data)
print(f"Training on {len(train_data)} samples from the general dataset.")
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = MagnusTransformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

scaler = torch.cuda.amp.GradScaler()

# Pretraining loop
for epoch in range(1):
    model.train()
    total_loss = 0
    step = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for x, y in loop:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            out = model(x)
            loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        step += 1
        total_loss += loss.item()
        loop.set_postfix(step=step, loss=loss.item())

# Save the pretrained model
torch.save(model.state_dict(), "data_processing/v2/models/magnus_transformer_pretrained_4th_general_all_carlsen.pth")
