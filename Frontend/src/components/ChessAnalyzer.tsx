import { useState } from 'react'
import { Chess } from 'chess.js'
import { Chessboard } from 'react-chessboard'
import { Upload, Brain, MessageSquare, RotateCcw } from 'lucide-react'
import {
    predictMagnusMove,
    testApiConnection,
    analyzeChessBoardImage,
    analyzeImageAndPredict,
    getOCRConfig,
} from '../services/api'
import './ChessAnalyzer.css'

interface MoveRecommendation {
    move: string
    evaluation: number
    explanation: string
    confidence?: number
}

interface AnalysisResult {
    recommendation: MoveRecommendation
    aiExplanation: string
    loading: boolean
}

const ChessAnalyzer = () => {
    const [game, setGame] = useState(new Chess())
    const [gamePosition, setGamePosition] = useState(game.fen())
    const [analysisMode, setAnalysisMode] = useState<'board' | 'upload'>(
        'board'
    )
    const [uploadedImage, setUploadedImage] = useState<string | null>(null)
    const [uploadedFile, setUploadedFile] = useState<File | null>(null)
    const [analysis, setAnalysis] = useState<AnalysisResult | null>(null)
    const [gameHistory, setGameHistory] = useState<string[]>([])
    const [apiTestResult, setApiTestResult] = useState<string | null>(null)
    const [ocrConfig, setOcrConfig] = useState<{
        geminiAvailable: boolean
        openaiAvailable: boolean
        recommendedApi: string | null
    } | null>(null)

    const makeMove = (sourceSquare: string, targetSquare: string) => {
        const gameCopy = new Chess(game.fen())

        try {
            const move = gameCopy.move({
                from: sourceSquare,
                to: targetSquare,
                promotion: 'q', // Always promote to queen for simplicity
            })

            if (move) {
                setGame(gameCopy)
                setGamePosition(gameCopy.fen())
                setGameHistory((prev) => [...prev, move.san])
                return true
            }
        } catch (error) {
            console.log('Invalid move:', error)
        }

        return false
    }

    const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0]
        if (file) {
            setUploadedFile(file)
            const reader = new FileReader()
            reader.onload = (e) => {
                setUploadedImage(e.target?.result as string)
            }
            reader.readAsDataURL(file)
        }
    }

    // API call to predict next move using your trained model
    const analyzePosition = async () => {
        setAnalysis({ loading: true } as AnalysisResult)

        try {
            const prediction = await predictMagnusMove(game.fen())

            // Generate AI explanation (you can integrate ChatGPT here later)
            const aiExplanation = `This move has a ${prediction.confidence}% confidence based on Magnus Carlsen's playing style. The model analyzed the current position and determined this to be the most likely move Magnus would play in this situation.`

            const analysisResult: AnalysisResult = {
                recommendation: {
                    move: prediction.predicted_move,
                    evaluation: 0, // You might want to add evaluation to your API
                    explanation: `Predicted move based on Magnus Carlsen's style with ${prediction.confidence}% confidence.`,
                    confidence: prediction.confidence,
                },
                aiExplanation,
                loading: false,
            }

            setAnalysis(analysisResult)
        } catch (error) {
            console.error('Analysis failed:', error)
            setAnalysis({
                recommendation: {
                    move: 'Connection Error',
                    evaluation: 0,
                    explanation:
                        'Failed to connect to the Magnus AI model. Please check your connection and try again.',
                },
                aiExplanation:
                    'Make sure the AI model server is running at http://10.224.9.93:8000 and accessible from your network.',
                loading: false,
            })
        }
    }

    const resetGame = () => {
        const newGame = new Chess()
        setGame(newGame)
        setGamePosition(newGame.fen())
        setGameHistory([])
        setAnalysis(null)
    }

    const analyzeUploadedPosition = async () => {
        if (!uploadedFile) return

        setAnalysis({ loading: true } as AnalysisResult)

        try {
            // Check OCR configuration first
            const config = await getOCRConfig()
            setOcrConfig(config)

            if (!config.geminiAvailable && !config.openaiAvailable) {
                setAnalysis({
                    recommendation: {
                        move: 'OCR Not Configured',
                        evaluation: 0,
                        explanation:
                            'No OCR API keys are configured. Please set up a Gemini or OpenAI API key to analyze board images.',
                        confidence: 0,
                    },
                    aiExplanation:
                        'To use image analysis:\n1. Get a free Gemini API key at https://makersuite.google.com/app/apikey\n2. Set GEMINI_API_KEY environment variable\n3. Restart the backend server',
                    loading: false,
                })
                return
            }

            // Use the full workflow API (OCR + Prediction)
            const result = await analyzeImageAndPredict(
                uploadedFile,
                config.recommendedApi || 'gemini'
            )

            if (
                result.success &&
                result.ocrResult?.success &&
                result.predictionResult
            ) {
                // Update the chess board with the detected position
                const detectedGame = new Chess(result.ocrResult.fen)
                setGame(detectedGame)
                setGamePosition(detectedGame.fen())

                const analysisResult: AnalysisResult = {
                    recommendation: {
                        move: result.predictionResult.predicted_move,
                        evaluation: 0,
                        explanation: `Detected position from image with ${result.ocrResult.confidence?.toFixed(
                            1
                        )}% confidence. Predicted move: ${
                            result.predictionResult.predicted_move
                        } (${result.predictionResult.confidence?.toFixed(
                            1
                        )}% confidence)`,
                        confidence: result.predictionResult.confidence,
                    },
                    aiExplanation: `Successfully analyzed chess board image:\n• Detected ${
                        result.ocrResult.detected_pieces || 'unknown'
                    } pieces\n• FEN: ${result.ocrResult.fen}\n• Notes: ${
                        result.ocrResult.notes ||
                        'Position detected successfully'
                    }\n\nMagnus Carlsen style prediction based on this position.`,
                    loading: false,
                }

                setAnalysis(analysisResult)
            } else {
                // OCR failed, show error
                setAnalysis({
                    recommendation: {
                        move: 'OCR Failed',
                        evaluation: 0,
                        explanation:
                            result.error ||
                            result.ocrResult?.error ||
                            'Could not analyze the chess board image. Please try a clearer image.',
                        confidence: 0,
                    },
                    aiExplanation:
                        'Image analysis tips:\n• Ensure good lighting\n• Take photo from above\n• Make sure all pieces are visible\n• Use standard chess pieces\n• Avoid shadows and reflections',
                    loading: false,
                })
            }
        } catch (error) {
            console.error('OCR analysis failed:', error)
            setAnalysis({
                recommendation: {
                    move: 'Analysis Error',
                    evaluation: 0,
                    explanation: `Failed to analyze image: ${
                        error instanceof Error ? error.message : 'Unknown error'
                    }`,
                    confidence: 0,
                },
                aiExplanation:
                    'Make sure the backend server is running with OCR capabilities enabled.',
                loading: false,
            })
        }
    }

    const handleModeChange = (mode: 'board' | 'upload') => {
        setAnalysisMode(mode)
        // Clear analysis when switching modes to fix the bug
        setAnalysis(null)
        // Clear uploaded image and file when switching to board mode
        if (mode === 'board') {
            setUploadedImage(null)
            setUploadedFile(null)
        }
    }

    const testAPI = async () => {
        setApiTestResult('Testing API connection...')

        const result = await testApiConnection()

        if (result.success) {
            setApiTestResult(
                `✅ API Test Successful! Move: ${result.details?.predicted_move}, Confidence: ${result.details?.confidence}%`
            )
        } else {
            setApiTestResult(`❌ API Test Failed: ${result.message}`)
        }

        // Clear the test result after 10 seconds
        setTimeout(() => setApiTestResult(null), 10000)
    }

    return (
        <div className="chess-analyzer">
            <div className="mode-selector">
                <button
                    className={`mode-btn ${
                        analysisMode === 'board' ? 'active' : ''
                    }`}
                    onClick={() => handleModeChange('board')}
                >
                    <Brain size={20} />
                    Interactive Board
                </button>
                <button
                    className={`mode-btn ${
                        analysisMode === 'upload' ? 'active' : ''
                    }`}
                    onClick={() => handleModeChange('upload')}
                >
                    <Upload size={20} />
                    Upload Position
                </button>
            </div>

            <div className="analyzer-content">
                {analysisMode === 'board' ? (
                    <div className="board-section">
                        <div className="board-container">
                            <Chessboard
                                position={gamePosition}
                                onPieceDrop={makeMove}
                                boardWidth={400}
                                arePiecesDraggable={true}
                            />
                            <div className="board-controls">
                                <button
                                    onClick={analyzePosition}
                                    className="analyze-btn"
                                >
                                    <Brain size={16} />
                                    Analyze Position
                                </button>
                                <button
                                    onClick={resetGame}
                                    className="reset-btn"
                                >
                                    <RotateCcw size={16} />
                                    Reset Board
                                </button>
                                <button
                                    onClick={testAPI}
                                    className="test-api-btn"
                                >
                                    🔧 Test API
                                </button>
                            </div>
                        </div>

                        {apiTestResult && (
                            <div className="api-test-result">
                                <p>{apiTestResult}</p>
                            </div>
                        )}

                        {gameHistory.length > 0 && (
                            <div className="game-history">
                                <h3>Move History</h3>
                                <div className="moves">
                                    {gameHistory.map((move, index) => (
                                        <span key={index} className="move">
                                            {Math.floor(index / 2) + 1}
                                            {index % 2 === 0 ? '.' : '...'}{' '}
                                            {move}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                ) : (
                    <div className="upload-section">
                        <div className="upload-area">
                            <input
                                type="file"
                                accept="image/*"
                                onChange={handleImageUpload}
                                className="file-input"
                                id="board-upload"
                            />
                            <label
                                htmlFor="board-upload"
                                className="upload-label"
                            >
                                <Upload size={48} />
                                <p>Click to upload chess board image</p>
                                <span>Supports JPG, PNG, WebP formats</span>
                            </label>
                        </div>

                        {ocrConfig && (
                            <div className="ocr-status">
                                <h4>OCR Status</h4>
                                <p>
                                    Gemini API:{' '}
                                    {ocrConfig.geminiAvailable
                                        ? '✅ Available'
                                        : '❌ Not configured'}
                                </p>
                                <p>
                                    OpenAI API:{' '}
                                    {ocrConfig.openaiAvailable
                                        ? '✅ Available'
                                        : '❌ Not configured'}
                                </p>
                                {!ocrConfig.geminiAvailable &&
                                    !ocrConfig.openaiAvailable && (
                                        <p style={{ color: 'orange' }}>
                                            ⚠️ No OCR APIs configured. Get a
                                            free Gemini API key at{' '}
                                            <a
                                                href="https://makersuite.google.com/app/apikey"
                                                target="_blank"
                                                rel="noopener noreferrer"
                                            >
                                                Google AI Studio
                                            </a>
                                        </p>
                                    )}
                            </div>
                        )}

                        {uploadedImage && (
                            <div className="uploaded-image">
                                <img
                                    src={uploadedImage}
                                    alt="Uploaded chess position"
                                />
                                <button
                                    onClick={analyzeUploadedPosition}
                                    className="analyze-btn"
                                >
                                    <Brain size={16} />
                                    Analyze Uploaded Position
                                </button>
                            </div>
                        )}
                    </div>
                )}

                {analysis && (
                    <div className="analysis-panel">
                        <h3>Magnus Carlsen Style Analysis</h3>

                        {analysis.loading ? (
                            <div className="loading">
                                <div className="spinner"></div>
                                <p>Analyzing position...</p>
                            </div>
                        ) : (
                            <div className="analysis-results">
                                <div className="recommendation">
                                    <h4>Recommended Move</h4>
                                    <div className="move-info">
                                        <span className="move">
                                            {analysis.recommendation.move}
                                        </span>
                                        {analysis.recommendation.confidence && (
                                            <span className="confidence">
                                                {
                                                    analysis.recommendation
                                                        .confidence
                                                }
                                                % confidence
                                            </span>
                                        )}
                                        {analysis.recommendation.evaluation !==
                                            0 && (
                                            <span
                                                className={`evaluation ${
                                                    analysis.recommendation
                                                        .evaluation > 0
                                                        ? 'positive'
                                                        : 'negative'
                                                }`}
                                            >
                                                {analysis.recommendation
                                                    .evaluation > 0
                                                    ? '+'
                                                    : ''}
                                                {
                                                    analysis.recommendation
                                                        .evaluation
                                                }
                                            </span>
                                        )}
                                    </div>
                                    <p className="explanation">
                                        {analysis.recommendation.explanation}
                                    </p>
                                </div>

                                <div className="ai-explanation">
                                    <h4>
                                        <MessageSquare size={16} />
                                        AI Explanation
                                    </h4>
                                    <p>{analysis.aiExplanation}</p>
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {apiTestResult && (
                    <div className="api-test-result">
                        <h3>API Connection Test</h3>
                        <p>{apiTestResult}</p>
                    </div>
                )}
            </div>
        </div>
    )
}

export default ChessAnalyzer
