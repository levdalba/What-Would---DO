#!/usr/bin/env python3
"""Comprehensive OCR debugging script"""

import requests
import json
import os
import tempfile
from PIL import Image, ImageDraw


def test_backend_health():
    """Test if backend is responding"""
    print("=== Testing Backend Health ===")
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"Health check status: {response.status_code}")
        if response.status_code == 200:
            print(f"Health response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Backend not responding: {e}")
        return False


def test_ocr_config():
    """Test OCR configuration"""
    print("\n=== Testing OCR Configuration ===")
    try:
        response = requests.get("http://localhost:8000/ocr-config")
        print(f"OCR config status: {response.status_code}")
        if response.status_code == 200:
            config = response.json()
            print(f"OCR config: {json.dumps(config, indent=2)}")
            return config
        return None
    except Exception as e:
        print(f"OCR config failed: {e}")
        return None


def create_simple_test_image():
    """Create a very simple test image"""
    img = Image.new("RGB", (400, 400), "white")
    draw = ImageDraw.Draw(img)

    # Simple checkerboard
    for row in range(8):
        for col in range(8):
            x1, y1 = col * 50, row * 50
            x2, y2 = x1 + 50, y1 + 50
            color = "gray" if (row + col) % 2 else "white"
            draw.rectangle([x1, y1, x2, y2], fill=color)

    return img


def test_analyze_board_image():
    """Test the analyze-board-image endpoint"""
    print("\n=== Testing analyze-board-image endpoint ===")

    img = create_simple_test_image()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp.name, "PNG")
        temp_path = tmp.name

    try:
        with open(temp_path, "rb") as f:
            files = {"file": ("test.png", f, "image/png")}
            data = {"api_choice": "opencv"}

            print("Sending request to analyze-board-image...")
            response = requests.post(
                "http://localhost:8000/analyze-board-image", files=files, data=data
            )

        print(f"Status: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")

        if response.status_code == 200:
            result = response.json()
            print("✅ analyze-board-image SUCCESS")
            print(json.dumps(result, indent=2))
            return result
        else:
            print("❌ analyze-board-image FAILED")
            print(f"Response: {response.text}")
            return None

    except Exception as e:
        print(f"Exception: {e}")
        return None
    finally:
        os.unlink(temp_path)


def test_analyze_and_predict():
    """Test the analyze-and-predict endpoint (what frontend uses)"""
    print("\n=== Testing analyze-and-predict endpoint ===")

    img = create_simple_test_image()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp.name, "PNG")
        temp_path = tmp.name

    try:
        with open(temp_path, "rb") as f:
            files = {"file": ("test.png", f, "image/png")}
            data = {"api_choice": "opencv"}

            print("Sending request to analyze-and-predict...")
            response = requests.post(
                "http://localhost:8000/analyze-and-predict", files=files, data=data
            )

        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("✅ analyze-and-predict SUCCESS")
            print(json.dumps(result, indent=2))

            # Check the structure that frontend expects
            has_ocr = "ocrResult" in result
            has_pred = "predictionResult" in result
            has_success = "success" in result

            print(f"\nStructure check:")
            print(f"  Has ocrResult: {has_ocr}")
            print(f"  Has predictionResult: {has_pred}")
            print(f"  Has success: {has_success}")

            if has_ocr and has_pred and has_success:
                print("✅ Response structure matches frontend expectations")

                if result["success"] and result["ocrResult"]["success"]:
                    print("✅ OCR succeeded")
                else:
                    print("❌ OCR failed in result")
            else:
                print("❌ Response structure doesn't match frontend")

            return result
        else:
            print("❌ analyze-and-predict FAILED")
            print(f"Response: {response.text}")
            return None

    except Exception as e:
        print(f"Exception: {e}")
        return None
    finally:
        os.unlink(temp_path)


def test_with_existing_image():
    """Test with the test image we created earlier"""
    print("\n=== Testing with existing test image ===")

    test_image_path = "test_chessboard.png"
    if not os.path.exists(test_image_path):
        print(f"Test image {test_image_path} not found, skipping...")
        return

    try:
        with open(test_image_path, "rb") as f:
            files = {"file": ("test_chessboard.png", f, "image/png")}
            data = {"api_choice": "opencv"}

            print(f"Testing with {test_image_path}...")
            response = requests.post(
                "http://localhost:8000/analyze-and-predict", files=files, data=data
            )

        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("✅ Real image test SUCCESS")
            print(json.dumps(result, indent=2))
        else:
            print("❌ Real image test FAILED")
            print(f"Response: {response.text}")

    except Exception as e:
        print(f"Exception with real image: {e}")


def main():
    print("🔍 Comprehensive OCR Debug Test")
    print("=" * 50)

    # Test 1: Backend health
    if not test_backend_health():
        print("❌ Backend is not running. Start it with: uvicorn main:app --reload")
        return

    # Test 2: OCR configuration
    config = test_ocr_config()
    if not config:
        print("❌ OCR configuration failed")
        return

    # Test 3: Simple image analysis
    test_analyze_board_image()

    # Test 4: Full workflow (what frontend uses)
    test_analyze_and_predict()

    # Test 5: With real test image
    test_with_existing_image()

    print("\n" + "=" * 50)
    print("Debug test complete. Check the results above.")


if __name__ == "__main__":
    main()
