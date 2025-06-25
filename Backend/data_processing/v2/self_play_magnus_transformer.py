import chess
import chess.pgn
import datetime
import random
from inference_class import ChessModel

# Load model
model_path = "data_processing/v2/models/magnus_transformer_finetuned.pth"
model = ChessModel(model_path)

# Create a legal random starting position
board = chess.Board()
for _ in range(random.randint(4, 12)):  # play a few random legal moves
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        break
    board.push(random.choice(legal_moves))

starting_fen = board.fen()
print(f"Starting self-play game from position:\n{starting_fen}\n")

# PGN setup
game = chess.pgn.Game()
game.headers["Event"] = "MagnusTransformer Self-Play (Random Start)"
game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
game.setup(board)
node = game

fen = starting_fen
turn = 0
max_turns = 100  # number of moves total (50 by each side)

while turn < max_turns:
    move = model.predict_move(fen)
    print(f"{'White' if turn % 2 == 0 else 'Black'} plays: {move.uci()}")

    try:
        move_obj = chess.Move.from_uci(move.uci())
        node = node.add_variation(move_obj)
    except:
        print(f"[!] Could not add move {move.uci()} to PGN — skipping")
        break

    # Don't update board or fen — just blindly continue
    turn += 1

# Save PGN
game.headers["Result"] = "*"
with open("data_processing/v2/self_play_game_random.pgn", "w") as f:
    f.write(str(game))

print("Game saved to data_processing/v2/self_play_game_random.pgn")
