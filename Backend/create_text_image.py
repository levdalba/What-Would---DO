#!/usr/bin/env python3
"""Create a simple text-based chessboard that's easier for AI to read"""

from PIL import Image, ImageDraw, ImageFont
import os


def create_text_chessboard():
    """Create a chessboard with text labels for pieces"""
    # Create larger image
    img = Image.new("RGB", (800, 800), "white")
    draw = ImageDraw.Draw(img)

    # Board colors
    light_color = (240, 217, 181)  # Light squares
    dark_color = (181, 136, 99)  # Dark squares

    square_size = 100

    # Draw the board
    for row in range(8):
        for col in range(8):
            x1 = col * square_size
            y1 = row * square_size
            x2 = x1 + square_size
            y2 = y1 + square_size

            color = dark_color if (row + col) % 2 == 1 else light_color
            draw.rectangle([x1, y1, x2, y2], fill=color)

            # Add coordinate labels
            coord = f"{chr(ord('a') + col)}{8 - row}"
            draw.text((x1 + 5, y1 + 5), coord, fill="gray", font=None)

    # Use text labels for pieces (easier for AI to read)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 36)
        coord_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
    except:
        font = ImageFont.load_default()
        coord_font = ImageFont.load_default()

    # Starting position with text labels
    pieces = {
        # Black pieces (row 0 = 8th rank)
        (0, 0): "bR",
        (1, 0): "bN",
        (2, 0): "bB",
        (3, 0): "bQ",
        (4, 0): "bK",
        (5, 0): "bB",
        (6, 0): "bN",
        (7, 0): "bR",
        # Black pawns (row 1 = 7th rank)
        (0, 1): "bP",
        (1, 1): "bP",
        (2, 1): "bP",
        (3, 1): "bP",
        (4, 1): "bP",
        (5, 1): "bP",
        (6, 1): "bP",
        (7, 1): "bP",
        # White pieces (row 7 = 1st rank)
        (0, 7): "wR",
        (1, 7): "wN",
        (2, 7): "wB",
        (3, 7): "wQ",
        (4, 7): "wK",
        (5, 7): "wB",
        (6, 7): "wN",
        (7, 7): "wR",
        # White pawns (row 6 = 2nd rank)
        (0, 6): "wP",
        (1, 6): "wP",
        (2, 6): "wP",
        (3, 6): "wP",
        (4, 6): "wP",
        (5, 6): "wP",
        (6, 6): "wP",
        (7, 6): "wP",
    }

    # Draw pieces with clear text
    for (col, row), piece in pieces.items():
        x = col * square_size + square_size // 2
        y = row * square_size + square_size // 2

        # Get text size for centering
        bbox = draw.textbbox((0, 0), piece, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Color based on piece
        if piece.startswith("b"):  # Black pieces
            text_color = (0, 0, 0)  # Black
            bg_color = (255, 255, 255, 200)  # Semi-transparent white background
        else:  # White pieces
            text_color = (255, 255, 255)  # White
            bg_color = (0, 0, 0, 200)  # Semi-transparent black background

        # Draw background circle for better visibility
        circle_radius = 35
        draw.ellipse(
            [
                x - circle_radius,
                y - circle_radius,
                x + circle_radius,
                y + circle_radius,
            ],
            fill=bg_color[:3],
        )

        # Draw piece text
        draw.text(
            (x - text_width // 2, y - text_height // 2),
            piece,
            fill=text_color,
            font=font,
        )

    return img


def main():
    """Create and save the text-based test image"""
    print("Creating text-based chessboard for better AI recognition...")

    img = create_text_chessboard()

    # Save to current directory
    filename = "test_chessboard_text.png"
    img.save(filename, "PNG")

    print(f"✅ Text-based chessboard saved as: {filename}")
    print(f"📂 Location: {os.path.abspath(filename)}")
    print(f"📊 This image contains:")
    print(f"   • Clear text labels for all pieces")
    print(f"   • bK, bQ, bR, bB, bN, bP for black pieces")
    print(f"   • wK, wQ, wR, wB, wN, wP for white pieces")
    print(f"   • Coordinate labels (a1-h8)")
    print(f"\n🧪 This should be much easier for Gemini to analyze accurately!")


if __name__ == "__main__":
    main()
