#!/usr/bin/env python3
"""Create a very clear test chessboard with better contrast"""

from PIL import Image, ImageDraw, ImageFont
import os


def create_clear_chessboard():
    """Create a crystal clear chessboard with all 32 pieces"""
    # Create larger image for better detail
    img = Image.new("RGB", (800, 800), "white")
    draw = ImageDraw.Draw(img)

    # Better board colors for contrast
    light_color = (255, 248, 220)  # Cornsilk - very light
    dark_color = (139, 69, 19)  # Saddle brown - dark but not black

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

    # Use larger, clearer font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 80)
    except:
        font = ImageFont.load_default()

    # Complete starting position with BOTH colors clearly differentiated
    pieces = {
        # Black pieces (top rows) - using black pieces
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
        # White pieces (bottom rows) - using white pieces
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
    }

    # Draw pieces with high contrast
    for (col, row), piece in pieces.items():
        x = col * square_size + square_size // 2
        y = row * square_size + square_size // 2

        # Get text size for centering
        bbox = draw.textbbox((0, 0), piece, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Use high contrast colors
        if piece in "♛♜♝♞♚♟":  # Black pieces
            text_color = (0, 0, 0)  # Black text
            outline_color = (255, 255, 255)  # White outline
        else:  # White pieces
            text_color = (255, 255, 255)  # White text
            outline_color = (0, 0, 0)  # Black outline

        # Draw with outline for better visibility
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                if dx != 0 or dy != 0:
                    draw.text(
                        (x - text_width // 2 + dx, y - text_height // 2 + dy),
                        piece,
                        fill=outline_color,
                        font=font,
                    )

        draw.text(
            (x - text_width // 2, y - text_height // 2),
            piece,
            fill=text_color,
            font=font,
        )

    return img


def main():
    """Create and save the improved test image"""
    print("Creating high-contrast test chessboard with all 32 pieces...")

    img = create_clear_chessboard()

    # Save to current directory
    filename = "test_chessboard_clear.png"
    img.save(filename, "PNG")

    print(f"✅ Clear test chessboard saved as: {filename}")
    print(f"📂 Location: {os.path.abspath(filename)}")
    print(f"📊 This image contains exactly 32 pieces:")
    print(f"   • 16 white pieces (bottom 2 rows)")
    print(f"   • 16 black pieces (top 2 rows)")
    print(f"   • High contrast for better OCR detection")
    print(f"\n🧪 Testing with this image should show 32 pieces detected!")


if __name__ == "__main__":
    main()
