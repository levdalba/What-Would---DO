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
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="Chess AI API with LC0 and Magnus Style",
    description="API for professional chess move prediction using Leela Chess Zero neural network and Magnus Carlsen style prediction",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, you can specify domains here
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Initialize the chess models and OCR globally
# Primary: Leela Chess Zero for strong analysis
chess_predictor = LC0ChessPredictor(
    lc0_path="/opt/homebrew/Cellar/lc0/0.31.2/libexec/lc0",
    weights_path="/opt/homebrew/Cellar/lc0/0.31.2/libexec/42850.pb.gz",
    time_limit=1.0,  # 1 second analysis time for responsive API
)

# Optional: Magnus Carlsen style predictor
magnus_predictor = None
try:
    from data_processing.v2.improved_magnus_training import MagnusStylePredictor

    magnus_model_path = Path("data_processing/v2/models/magnus_style_model")
    if magnus_model_path.exists():
        magnus_predictor = MagnusStylePredictor(str(magnus_model_path))
        print("Magnus Carlsen style predictor loaded successfully")
    else:
        print("Magnus model not found - train using improved_magnus_training.py")
except ImportError as e:
    print(f"Magnus predictor dependencies not available: {e}")
    print("To enable Magnus style prediction, install: pip install torch scikit-learn")

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


class MagnusStyleResponse(BaseModel):
    """Output format for Magnus Carlsen style predictions."""

    top_moves: List[dict]  # List of moves with probabilities
    magnus_choice: str  # Most likely Magnus move
    confidence: float  # Confidence in Magnus choice
    explanation: str  # Style explanation


class CombinedPredictionResponse(BaseModel):
    """Combined engine and style predictions."""

    lc0_analysis: PredictionResponse
    magnus_style: MagnusStyleResponse
    comparison: dict  # Comparison between engines


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
    Returns metadata about the chess analysis models.

    Returns:
        ModelMetadata: Model name, version, accuracy, and tags.
    """

    return ModelMetadata(
        model_name="LC0 + Magnus Style Hybrid",
        version="2.1.0",
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


@app.post("/predict-magnus-style", response_model=MagnusStyleResponse)
def predict_magnus_style(request: BoardInput) -> MagnusStyleResponse:
    """
    Predict moves in Magnus Carlsen's playing style.

    Args:
        request (BoardInput): The input FEN string for the current chess board.

    Returns:
        MagnusStyleResponse: Top moves with probabilities in Magnus's style.
    """
    if magnus_predictor is None:
        raise HTTPException(
            status_code=501,
            detail="Magnus style model not available. Train the model using improved_magnus_training.py",
        )

    try:
        import chess

        # Validate FEN string
        if not request.board or not request.board.strip():
            raise HTTPException(
                status_code=400, detail="Board FEN string cannot be empty"
            )

        # Create board from FEN
        board = chess.Board(request.board)

        # Get Magnus style predictions
        predictions = magnus_predictor.predict_move(board, top_k=5)

        if not predictions:
            raise HTTPException(
                status_code=400,
                detail="No valid Magnus-style moves found for the given position",
            )

        # Format response
        top_moves = [
            {
                "move": move_uci,
                "probability": prob,
                "san": board.san(chess.Move.from_uci(move_uci)),
            }
            for move_uci, prob in predictions
        ]

        magnus_choice = predictions[0][0]  # Top move
        confidence = predictions[0][1]  # Top move probability

        return MagnusStyleResponse(
            top_moves=top_moves,
            magnus_choice=magnus_choice,
            confidence=confidence,
            explanation=f"Moves predicted in Magnus Carlsen's playing style. Top choice: {board.san(chess.Move.from_uci(magnus_choice))} with {confidence:.1%} confidence.",
        )

    except Exception as e:
        print(f"Error in Magnus style prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/predict-combined", response_model=CombinedPredictionResponse)
def predict_combined(request: BoardInput) -> CombinedPredictionResponse:
    """
    Get both LC0 engine analysis and Magnus Carlsen style predictions.

    Args:
        request (BoardInput): The input FEN string for the current chess board.

    Returns:
        CombinedPredictionResponse: Combined predictions from both models.
    """
    try:
        import chess

        # Get LC0 analysis
        lc0_move, lc0_confidence = chess_predictor.predict_move_uci(request.board)
        lc0_analysis = PredictionResponse(
            predicted_move=lc0_move, confidence=lc0_confidence
        )

        # Get Magnus style analysis (if available)
        if magnus_predictor is not None:
            board = chess.Board(request.board)
            magnus_predictions = magnus_predictor.predict_move(board, top_k=3)

            top_moves = [
                {
                    "move": move_uci,
                    "probability": prob,
                    "san": board.san(chess.Move.from_uci(move_uci)),
                }
                for move_uci, prob in magnus_predictions
            ]

            magnus_style = MagnusStyleResponse(
                top_moves=top_moves,
                magnus_choice=magnus_predictions[0][0],
                confidence=magnus_predictions[0][1],
                explanation=f"Magnus style analysis with {len(magnus_predictions)} candidate moves",
            )

            # Compare predictions
            magnus_top_move = magnus_predictions[0][0]
            comparison = {
                "engines_agree": lc0_move == magnus_top_move,
                "lc0_move_san": board.san(chess.Move.from_uci(lc0_move)),
                "magnus_move_san": board.san(chess.Move.from_uci(magnus_top_move)),
                "analysis": (
                    "LC0 and Magnus agree on the best move"
                    if lc0_move == magnus_top_move
                    else "LC0 and Magnus suggest different moves - interesting position!"
                ),
            }
        else:
            magnus_style = MagnusStyleResponse(
                top_moves=[],
                magnus_choice="",
                confidence=0.0,
                explanation="Magnus style model not available",
            )
            comparison = {
                "engines_agree": False,
                "lc0_move_san": "",
                "magnus_move_san": "",
                "analysis": "Only LC0 analysis available",
            }

        return CombinedPredictionResponse(
            lc0_analysis=lc0_analysis, magnus_style=magnus_style, comparison=comparison
        )

    except Exception as e:
        print(f"Error in combined prediction: {e}")
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
