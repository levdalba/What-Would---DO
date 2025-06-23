from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel, ConfigDict
from typing import List
from data_processing.v2.lc0_inference import LC0ChessPredictor
from data_processing.v2.chess_ocr import ChessBoardOCR
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="Chess AI API with LC0",
    description="API for professional chess move prediction using Leela Chess Zero neural network",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, you can specify domains here
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Initialize the chess model and OCR globally
# Using Leela Chess Zero instead of GPT-2 for better chess analysis
chess_predictor = LC0ChessPredictor(
    lc0_path="/opt/homebrew/Cellar/lc0/0.31.2/libexec/lc0",
    weights_path="/opt/homebrew/Cellar/lc0/0.31.2/libexec/42850.pb.gz",
    time_limit=1.0,  # 1 second analysis time for responsive API
)
chess_ocr = ChessBoardOCR()
print(f"LC0 Chess Predictor initialized")
print(f"Chess OCR initialized")


class ModelMetadata(BaseModel):
    """Metadata about the ML model."""

    model_name: str
    version: str
    accuracy: float
    tags: List[str]


class PredictionRequest(BaseModel):
    """Input format for making a prediction."""

    features: List[float]

    model_config = ConfigDict(
        json_schema_extra={"example": {"features": [5.1, 3.5, 1.4, 0.2]}}
    )


class PredictionResponse(BaseModel):
    """Output format for prediction results."""

    predicted_move: str  # The predicted move in UCI format
    confidence: float  # Confidence value for the prediction


class BoardInput(BaseModel):
    board: str  # FEN string representing the current chess board


class OCRResponse(BaseModel):
    """Output format for OCR results."""

    success: bool
    fen: str = ""
    confidence: float = 0.0
    detected_pieces: int = 0
    notes: str = ""
    fen_valid: bool = False
    error: str = ""


@app.get("/")
def read_root() -> dict:
    """
    Root endpoint.

    Returns:
        dict: A welcome message.
    """

    return {"message": "Welcome to the ML Model API"}


@app.get("/model-info", response_model=ModelMetadata)
def get_model_metadata() -> ModelMetadata:
    """
    Returns metadata about the chess analysis model.

    Returns:
        ModelMetadata: Model name, version, accuracy, and tags.
    """

    return ModelMetadata(
        model_name="Leela Chess Zero (LC0)",
        version="0.31.2",
        accuracy=0.98,  # LC0 is extremely strong
        tags=["lc0", "neural-network", "chess-engine", "leela"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_move(request: BoardInput) -> PredictionResponse:
    """
    Predicts the next move based on the input FEN string (current board state).

    Args:
        request (BoardInput): The input FEN string for the current chess board.

    Returns:
        PredictionResponse: Predicted UCI move and confidence score.
    """
    try:
        # Validate FEN string
        if not request.board or not request.board.strip():
            raise HTTPException(
                status_code=400, detail="Board FEN string cannot be empty"
            )

        # Get the predicted move from the model
        predicted_move_uci, confidence = chess_predictor.predict_move_uci(request.board)

        if predicted_move_uci == "0000":
            raise HTTPException(
                status_code=400, detail="No valid moves found for the given position"
            )

        # Return the predicted move and the confidence
        return PredictionResponse(
            predicted_move=predicted_move_uci, confidence=confidence
        )

    except Exception as e:
        print(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": chess_predictor is not None}


@app.post("/test")
def test_prediction():
    """Test endpoint with a standard chess starting position"""
    test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    try:
        predicted_move_uci, confidence = chess_predictor.predict_move_uci(test_fen)
        return {
            "test_fen": test_fen,
            "predicted_move": predicted_move_uci,
            "confidence": confidence,
            "status": "success",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/analyze-board-image", response_model=OCRResponse)
async def analyze_board_image(
    file: UploadFile = File(...), api_choice: str = "opencv"
) -> OCRResponse:
    """
    Analyze chess board from uploaded image using OCR

    Args:
        file: Uploaded image file (jpg, png, webp)
        api_choice: "opencv", "gemini" or "openai" (default: opencv)

    Returns:
        OCRResponse: Analysis results with FEN notation
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Save uploaded file temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Analyze the image
        result = chess_ocr.analyze_chess_board(temp_file_path, preferred_api=api_choice)

        # Clean up temp file
        os.unlink(temp_file_path)

        # Convert result to response model
        return OCRResponse(
            success=result.get("success", False),
            fen=result.get("fen", ""),
            confidence=result.get("confidence", 0.0),
            detected_pieces=result.get("detected_pieces", 0),
            notes=result.get("notes", ""),
            fen_valid=result.get("fen_valid", False),
            error=result.get("error", ""),
        )

    except Exception as e:
        # Clean up temp file if it exists
        if "temp_file_path" in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass

        raise HTTPException(status_code=500, detail=f"OCR analysis failed: {str(e)}")


@app.post("/analyze-and-predict")
async def analyze_and_predict(file: UploadFile = File(...), api_choice: str = "opencv"):
    """
    Complete workflow: Analyze board image and predict next move

    Args:
        file: Uploaded chess board image
        api_choice: OCR API to use ("gemini" or "openai")

    Returns:
        Combined OCR and prediction results
    """
    # First, analyze the board
    ocr_result = await analyze_board_image(file, api_choice)

    if not ocr_result.success or not ocr_result.fen_valid:
        return {
            "ocrResult": ocr_result,
            "predictionResult": None,
            "success": False,
            "error": "Could not extract valid FEN from image",
        }

    # Then predict the next move
    try:
        predicted_move_uci, confidence = chess_predictor.predict_move_uci(
            ocr_result.fen
        )

        prediction_result = PredictionResponse(
            predicted_move=predicted_move_uci, confidence=confidence
        )

        return {
            "ocrResult": ocr_result,
            "predictionResult": prediction_result,
            "success": True,
        }

    except Exception as e:
        return {
            "ocrResult": ocr_result,
            "predictionResult": None,
            "success": False,
            "error": f"Prediction failed: {str(e)}",
        }


@app.get("/ocr-config")
def get_ocr_config():
    """Get OCR configuration status"""
    return {
        "gemini_available": bool(os.getenv("GEMINI_API_KEY")),
        "openai_available": bool(os.getenv("OPENAI_API_KEY")),
        "opencv_available": True,  # Always available as fallback
        "supported_formats": ["jpg", "jpeg", "png", "webp"],
        "recommended_api": (
            "gemini"  # Prefer Gemini as it's most reliable for chess boards
            if os.getenv("GEMINI_API_KEY")
            else (
                "openai" if os.getenv("OPENAI_API_KEY") else "opencv"
            )  # Fallback to starting position
        ),
    }
