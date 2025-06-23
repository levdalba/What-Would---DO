#!/usr/bin/env python3
"""Test with the realistic chess board image"""

import requests
import json
import os


def test_with_real_chessboard():
    """Test with the realistic chessboard image"""
    print("=== Testing with Realistic Chessboard ===")

    test_image = "test_chessboard.png"
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
            print(json.dumps(result, indent=2))

            # Check what the frontend will see
            if (
                result.get("success")
                and result.get("ocrResult", {}).get("success")
                and result.get("predictionResult")
            ):
                print("\n🎉 Frontend will show SUCCESS!")
                print(f"📋 FEN: {result['ocrResult']['fen']}")
                print(f"🎯 Move: {result['predictionResult']['predicted_move']}")
                print(f"🔍 OCR Confidence: {result['ocrResult']['confidence']}")
                print(
                    f"🎲 Prediction Confidence: {result['predictionResult']['confidence']}"
                )
            else:
                print("\n❌ Frontend will show failure")
                print(f"Error: {result.get('error', 'Unknown error')}")

        else:
            print("❌ FAILED!")
            print(f"Response: {response.text}")

    except Exception as e:
        print(f"❌ Exception: {e}")


if __name__ == "__main__":
    test_with_real_chessboard()
