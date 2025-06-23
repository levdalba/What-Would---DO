#!/usr/bin/env python3
"""Test with the clear chessboard image"""

import requests
import json
import os


def test_with_clear_chessboard():
    """Test with the high-contrast chessboard image"""
    print("=== Testing with High-Contrast Chessboard ===")

    test_image = "test_chessboard_clear.png"
    if not os.path.exists(test_image):
        print(f"❌ {test_image} not found")
        return

    try:
        with open(test_image, "rb") as f:
            files = {"file": (test_image, f, "image/png")}
            data = {"api_choice": "gemini"}  # Use Gemini for best results

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

            print(f"\n🔍 OCR Analysis:")
            print(f"   • Detected pieces: {detected_pieces}")
            print(f"   • Expected pieces: 32 (starting position)")
            print(f"   • Confidence: {confidence}%")
            print(f"   • FEN: {fen}")

            # Count pieces in FEN to double-check
            if fen:
                fen_position = fen.split()[0]
                actual_pieces = sum(1 for char in fen_position if char.isalpha())
                print(f"   • Pieces in FEN: {actual_pieces}")

                if actual_pieces == 32:
                    print("   ✅ Perfect! All 32 pieces detected")
                elif actual_pieces == 16:
                    print("   ⚠️ Only 16 pieces detected - missing one color")
                else:
                    print(f"   ❓ Unexpected piece count: {actual_pieces}")

            print(f"\n📝 Notes: {ocr_result.get('notes', 'None')}")

            if result.get("success"):
                print("\n🎉 Frontend will show SUCCESS!")
                pred_result = result.get("predictionResult", {})
                print(f"🎯 Predicted Move: {pred_result.get('predicted_move', 'None')}")
                print(
                    f"🎲 Prediction Confidence: {pred_result.get('confidence', 0):.1%}"
                )
            else:
                print("\n❌ Frontend will show failure")

        else:
            print("❌ FAILED!")
            print(f"Response: {response.text}")

    except Exception as e:
        print(f"❌ Exception: {e}")


if __name__ == "__main__":
    test_with_clear_chessboard()
