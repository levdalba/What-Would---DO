#!/usr/bin/env python3
"""Test the improved OCR implementation"""

import requests
import json
import tempfile
import os

try:
    from PIL import Image, ImageDraw

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def test_improved_ocr():
    """Test the improved OCR implementation"""
    print("=== Testing Improved OCR Implementation ===")

    if PIL_AVAILABLE:
        # Create a simple test image
        img = Image.new("RGB", (400, 400), "white")
        draw = ImageDraw.Draw(img)

        # Draw a simple chessboard pattern
        for row in range(8):
            for col in range(8):
                x1, y1 = col * 50, row * 50
                x2, y2 = x1 + 50, y1 + 50
                if (row + col) % 2 == 1:
                    draw.rectangle([x1, y1, x2, y2], fill="brown")

        # Save temporarily
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img.save(tmp.name, "PNG")
            temp_path = tmp.name

        print(f"Created test image: {temp_path}")
    else:
        # Use existing test image if available
        temp_path = "test_chessboard.png"
        if not os.path.exists(temp_path):
            print("❌ No test image available and PIL not installed")
            return

    try:
        # Test the analyze-and-predict endpoint
        with open(temp_path, "rb") as f:
            files = {"file": ("test.png", f, "image/png")}
            data = {"api_choice": "opencv"}

            response = requests.post(
                "http://localhost:8000/analyze-and-predict", files=files, data=data
            )

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS!")
            print(json.dumps(result, indent=2))

            # Check if the FEN is valid
            if result.get("success") and result.get("ocrResult", {}).get("fen_valid"):
                print("\n✅ OCR generated valid FEN!")
                print(f"FEN: {result['ocrResult']['fen']}")
                print(f"Confidence: {result['ocrResult']['confidence']}")
                print(
                    f"Predicted Move: {result.get('predictionResult', {}).get('predicted_move', 'None')}"
                )
            else:
                print("\n❌ OCR failed or generated invalid FEN")

        else:
            print("❌ FAILED!")
            print(f"Response: {response.text}")

    except Exception as e:
        print(f"❌ Exception: {e}")
    finally:
        if PIL_AVAILABLE and os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    test_improved_ocr()
