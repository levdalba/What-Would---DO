# Magnus Carlsen Style Chess Analyzer

A React-TypeScript application that analyzes chess positions and recommends moves based on Magnus Carlsen's playing style using a trained neural network.

## Features

### 🎯 Interactive Chess Board

-   Drag and drop pieces to set up positions
-   Real-time position analysis
-   Move history tracking
-   Board reset functionality

### 📷 Image Upload Analysis

-   Upload chess board images for analysis
-   Optical chess recognition (planned feature)
-   Position extraction from images

### 🧠 AI-Powered Analysis

-   **Magnus Carlsen Style Prediction**: Uses your trained neural network at `http://10.224.9.93:8000/predict`
-   **Confidence Scoring**: Shows prediction confidence percentage
-   **Move Explanations**: AI-generated explanations for recommended moves
-   **Style Analysis**: Understanding of Magnus's positional preferences

## API Integration

The application integrates with your trained Magnus Carlsen model:

**Endpoint**: `http://10.224.9.93:8000/predict`

**Request Format**:

```json
{
    "board": "r2q1rk1/2p1bppp/p1n1bn2/1p2p3/4P3/2P2N2/PPBN1PPP/R1BQR1K1"
}
```

**Response Format**:

```json
{
    "predicted_move": "h2h3",
    "confidence": 69.0
}
```

## Recent Improvements

### ✅ Fixed Issues

1. **Improved Button Spacing**: Increased gap between mode selector buttons for better UX
2. **Fixed Analysis Persistence Bug**: Analysis now clears when switching between modes
3. **Real API Integration**: Connected to your actual Magnus prediction model
4. **Enhanced Error Handling**: Better error messages and connection feedback

### 🎨 UI Enhancements

-   Better visual separation between sections
-   Confidence percentage display
-   Improved loading states
-   Modern card-based design
-   Responsive layout improvements

## Getting Started

1. **Install Dependencies**:

    ```bash
    cd Frontend
    npm install
    ```

2. **Start Development Server**:

    ```bash
    npm run dev
    ```

3. **Ensure AI Model is Running**:
    - Make sure your Magnus model server is running at `http://10.224.9.93:8000`
    - Test the `/predict` endpoint independently if needed

## Usage

### Interactive Board Analysis

1. Click "Interactive Board" mode
2. Set up your chess position by dragging pieces
3. Click "Analyze Position" to get Magnus-style move recommendations
4. View confidence scores and AI explanations

### Image Upload Analysis

1. Click "Upload Position" mode
2. Upload a chess board image
3. Click "Analyze Uploaded Position"
4. _Note: Image-to-FEN conversion is planned for future implementation_

## Future Enhancements

-   [ ] **Image Recognition**: Implement OCR for chess board images
-   [ ] **ChatGPT Integration**: Add detailed move explanations using GPT
-   [ ] **Multiple Player Styles**: Train models for other grandmasters
-   [ ] **Opening Database**: Integration with opening theory
-   [ ] **Game Analysis**: Full game analysis capabilities
-   [ ] **Position Evaluation**: Add position evaluation scores

## Technical Stack

-   **Frontend**: React + TypeScript + Vite
-   **Chess Logic**: chess.js library
-   **Board Component**: react-chessboard
-   **Styling**: Custom CSS with modern design
-   **AI Integration**: REST API calls to Python backend

## Model Training Notes

Your Magnus Carlsen model appears to be trained on FEN positions and returns moves in algebraic notation. The confidence scoring provides good feedback on prediction quality.

For optimal results:

-   Ensure training data covers various game phases (opening, middlegame, endgame)
-   Consider position evaluation alongside move prediction
-   Monitor prediction accuracy across different position types
