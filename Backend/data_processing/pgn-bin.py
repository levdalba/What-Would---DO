import chess.pgn
import os
import numpy as np
import chess
import struct
from tqdm import tqdm
import tensorflow as tf
import glob


def parse_magnus_pgn(pgn_file_path):
    positions = []
    with open(pgn_file_path, 'r') as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            if "Carlsen" not in game.headers.get("White", "") and "Carlsen" not in game.headers.get("Black", ""):
                continue

            magnus_color = "white" if "Carlsen" in game.headers["White"] else "black"
            board = game.board()
            node = game
            while node.variations:
                next_node = node.variation(0)
                if (board.turn == chess.WHITE and magnus_color == "white") or \
                   (board.turn == chess.BLACK and magnus_color == "black"):
                    positions.append((board.fen(), next_node.move.uci()))
                board.push(next_node.move)
                node = next_node
    return positions

positions = parse_magnus_pgn("data-processing/carlsen-games-quarter.pgn")
print(f"Extracted {len(positions)} positions where Magnus moved.")


def board_to_planes(board):
    planes = np.zeros((112, 8, 8), dtype=np.float32)
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = 7 - (square // 8), square % 8
            planes[piece_map[piece.symbol()], row, col] = 1.0
    planes[12] = 1.0 if board.turn == chess.WHITE else 0.0
    return planes

def move_to_policy(board, move_uci):
    policy = np.zeros(1858, dtype=np.float32)
    move = chess.Move.from_uci(move_uci)
    legal_moves = list(board.legal_moves)
    if move in legal_moves:
        policy[legal_moves.index(move)] = 1.0
    return policy

def save_binpack(positions, output_dir="data-processing/content/training_data"):
    os.makedirs(output_dir, exist_ok=True)
    for i, (fen, move) in tqdm(enumerate(positions), total=len(positions), desc="Saving binpack"):
        board = chess.Board(fen)
        planes = board_to_planes(board)
        policy = move_to_policy(board, move)
        value = 0.0
        with open(f"{output_dir}/game_{i}.bin", "wb") as f:
            f.write(struct.pack('f' * planes.size, *planes.flatten()))
            f.write(struct.pack('f' * policy.size, *policy.flatten()))
            f.write(struct.pack('f', value))

save_binpack(positions)




def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def write_tfrecord(bin_files_pattern="data-processing/content/training_data/*.bin", output_file="data-processing/content/tfrecords/magnus.tfrecord"):
    os.makedirs("data-processing/content/tfrecords", exist_ok=True)  # ← make sure this matches the output_file path
    bin_files = glob.glob(bin_files_pattern)
    with tf.io.TFRecordWriter(output_file) as writer:
        for bin_file in tqdm(bin_files, desc="Converting to TFRecord", unit="file"):
            with open(bin_file, "rb") as f:
                data = f.read()
                planes = np.frombuffer(data[:112*8*8*4], dtype=np.float32)
                policy = np.frombuffer(data[112*8*8*4:112*8*8*4 + 1858*4], dtype=np.float32)
                value = np.frombuffer(data[-4:], dtype=np.float32)
                example = tf.train.Example(features=tf.train.Features(feature={
                    'planes': _float_feature(planes),
                    'policy': _float_feature(policy),
                    'value': _float_feature(value)
                }))
                writer.write(example.SerializeToString())

# Run it
write_tfrecord()

