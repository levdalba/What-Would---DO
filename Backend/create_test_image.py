#!/usr/bin/env python3
"""Create a realistic test chessboard image for frontend testing"""

from PIL import Image, ImageDraw, ImageFont
import os


def create_realistic_chessboard():
    """Create a more realistic looking chessboard with pieces"""
    # Create larger image for better detail
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

    # Draw some simple pieces as text (basic representation)
    try:
        # Try to load a decent font
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 60)
    except:
        # Fallback to default font
        font = ImageFont.load_default()

    # Simple starting position pieces
    pieces = {
        # White pieces (bottom)
        (0, 7): "♖",
        (1, 7): "♘",
        (2, 7): "♗",
        (3, 7): "♕",
        (4, 7): "♔",
        (5, 7): "♗",
        (6, 7): "♘",
        (7, 7): "♖",
        # White pawns
        (0, 6): "♙",
        (1, 6): "♙",
        (2, 6): "♙",
        (3, 6): "♙",
        (4, 6): "♙",
        (5, 6): "♙",
        (6, 6): "♙",
        (7, 6): "♙",
        # Black pieces (top)
        (0, 0): "♜",
        (1, 0): "♞",
        (2, 0): "♝",
        (3, 0): "♛",
        (4, 0): "♚",
        (5, 0): "♝",
        (6, 0): "♞",
        (7, 0): "♜",
        # Black pawns
        (0, 1): "♟",
        (1, 1): "♟",
        (2, 1): "♟",
        (3, 1): "♟",
        (4, 1): "♟",
        (5, 1): "♟",
        (6, 1): "♟",
        (7, 1): "♟",
    }

    # Draw pieces
    for (col, row), piece in pieces.items():
        x = col * square_size + square_size // 2
        y = row * square_size + square_size // 2

        # Get text size for centering
        bbox = draw.textbbox((0, 0), piece, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        draw.text(
            (x - text_width // 2, y - text_height // 2),
            piece,
            fill="black" if piece in "♛♜♝♞♚♟" else "white",
            font=font,
        )

    return img


def main():
    """Create and save the test image"""
    print("Creating realistic test chessboard...")

    img = create_realistic_chessboard()

    # Save to current directory
    filename = "test_chessboard.png"
    img.save(filename, "PNG")

    print(f"✅ Test chessboard saved as: {filename}")
    print(f"📂 Location: {os.path.abspath(filename)}")
    print("\n📋 Instructions:")
    print("1. Open your browser to http://localhost:3000")
    print("2. Click 'Choose File' in the Image Analysis section")
    print(f"3. Upload the file: {filename}")
    print("4. Click 'Analyze Position'")
    print("5. Check if you now see successful analysis instead of 'OCR Failed'")


if __name__ == "__main__":
    main()
