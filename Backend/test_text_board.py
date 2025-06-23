#!/usr/bin/env python3
"""Test with the text-based chessboard image"""

import requests
import json
import os


def test_with_text_chessboard():
    """Test with the text-based chessboard image"""
    print("=== Testing with Text-Based Chessboard ===")

    test_image = "test_chessboard_text.png"
    if not os.path.exists(test_image):
        print(f"❌ {test_image} not found")
        return

    try:
        with open(test_image, "rb") as f:
            files = {"file": (test_image, f, "image/png")}
            data = {"api_choice": "gemini"}

            response = requests.post(
                "http://localhost:8000/analyze-and-predict", files=files, data=data
            )

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS!")

            ocr_result = result.get("ocrResult", {})
            detected_pieces = ocr_result.get("detected_pieces", 0)
            fen = ocr_result.get("fen", "")
            confidence = ocr_result.get("confidence", 0)

            print(f"\n🔍 Detailed OCR Analysis:")
            print(f"   • Detected pieces: {detected_pieces}/32")
            print(f"   • Confidence: {confidence}%")
            print(f"   • FEN: {fen}")

            # Analyze the FEN
            if fen:
                try:
                    import chess

                    board = chess.Board(fen)
                    fen_position = fen.split()[0]
                    actual_pieces = sum(1 for char in fen_position if char.isalpha())

                    print(f"   • Pieces in FEN: {actual_pieces}")
                    print(f"   • Board valid: {True}")

                    # Count piece types
                    white_pieces = sum(1 for char in fen_position if char.isupper())
                    black_pieces = sum(1 for char in fen_position if char.islower())
                    print(f"   • White pieces: {white_pieces}")
                    print(f"   • Black pieces: {black_pieces}")

                    if (
                        actual_pieces == 32
                        and white_pieces == 16
                        and black_pieces == 16
                    ):
                        print("   🎉 PERFECT! All pieces detected correctly!")
                    else:
                        print(
                            f"   ⚠️ Piece count mismatch - expected 32 total (16 each color)"
                        )

                except Exception as e:
                    print(f"   ❌ FEN analysis error: {e}")

            print(f"\n📝 Gemini's Notes:")
            print(f"   {ocr_result.get('notes', 'None')}")

            if result.get("success"):
                pred_result = result.get("predictionResult", {})
                print(f"\n🎯 Move Prediction:")
                print(f"   • Move: {pred_result.get('predicted_move', 'None')}")
                print(f"   • Confidence: {pred_result.get('confidence', 0):.1%}")

        else:
            print("❌ FAILED!")
            print(f"Response: {response.text}")

    except Exception as e:
        print(f"❌ Exception: {e}")


if __name__ == "__main__":
    test_with_text_chessboard()
