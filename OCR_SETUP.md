# Chess Board OCR Setup Guide

## 🚀 Quick Start

Your chess AI now supports **image recognition**! You can upload photos of chess boards and the AI will:

1. **Extract the FEN position** from the image
2. **Predict the next move** using your trained model
3. **Return both results** with confidence scores

## 🔑 Getting a FREE API Key

### Option 1: Google Gemini (Recommended - Best Free Tier)

1. **Visit**: [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
2. **Sign in** with your Google account
3. **Click** "Create API Key"
4. **Copy** the generated key
5. **Set the environment variable**:
    ```bash
    export GEMINI_API_KEY="your-api-key-here"
    ```

**Free Tier**: 15 requests per minute, 1500 requests per day

### Option 2: OpenAI GPT-4 Vision (Trial Credits)

1. **Visit**: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. **Create account** and verify phone number
3. **Create API key**
4. **Set the environment variable**:
    ```bash
    export OPENAI_API_KEY="your-api-key-here"
    ```

**Free Trial**: $5 credit for new accounts

## 🧪 Testing the OCR

### 1. Check Configuration

```bash
curl http://localhost:8000/ocr-config
```

### 2. Test with Image

```bash
# Upload and analyze chess board image
curl -X POST 'http://localhost:8000/analyze-board-image' \
     -F 'file=@your_chess_board.jpg' \
     -F 'api_choice=gemini'
```

### 3. Full Workflow (OCR + Prediction)

```bash
# Get position from image AND predict next move
curl -X POST 'http://localhost:8000/analyze-and-predict' \
     -F 'file=@your_chess_board.jpg'
```

### 4. Test Script

```bash
cd Backend
python test_ocr.py your_chess_board.jpg
```

## 📱 How to Take Good Chess Board Photos

For best OCR results:

1. **Lighting**: Good, even lighting (avoid shadows)
2. **Angle**: Take photo from directly above or at slight angle
3. **Focus**: Make sure pieces are in focus
4. **Background**: Clear background, minimal distractions
5. **Pieces**: Standard Staunton chess pieces work best
6. **Board**: Clear contrast between squares

## 🔧 API Endpoints

### 1. `/analyze-board-image`

-   **Method**: POST
-   **Input**: Image file + API choice
-   **Output**: FEN position with confidence

### 2. `/analyze-and-predict`

-   **Method**: POST
-   **Input**: Image file
-   **Output**: FEN position + predicted move

### 3. `/ocr-config`

-   **Method**: GET
-   **Output**: API availability status

## 🎯 Usage Examples

### Python

```python
import requests

# Analyze image
with open('chess_board.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post(
        'http://localhost:8000/analyze-board-image',
        files=files
    )
result = response.json()
print(f"FEN: {result['fen']}")
print(f"Confidence: {result['confidence']:.1f}%")
```

### JavaScript (Frontend)

```javascript
const formData = new FormData()
formData.append('file', imageFile)

const response = await fetch('http://localhost:8000/analyze-board-image', {
    method: 'POST',
    body: formData,
})

const result = await response.json()
console.log('FEN:', result.fen)
console.log('Confidence:', result.confidence)
```

## 🚦 Status Check

Run this to verify everything is working:

```bash
cd Backend
python test_ocr.py
```

This will show:

-   ✅ API key status
-   ✅ Server availability
-   ✅ Configuration details
-   🧪 Sample API commands

## 🎮 Frontend Integration

Your React frontend now has these new functions:

-   `analyzeChessBoardImage(file)` - Extract FEN from image
-   `analyzeImageAndPredict(file)` - Full workflow
-   `getOCRConfig()` - Check API availability

## 🎉 What's Next?

1. **Get your free Gemini API key**
2. **Test with a chess board photo**
3. **Integrate into your frontend UI**
4. **Build amazing chess features!**

The OCR system understands:

-   All piece types and positions
-   Board orientation (can auto-detect)
-   Standard chess notation
-   Multiple image formats (JPG, PNG, WebP)

Happy coding! 🏆
