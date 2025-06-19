import { useState } from 'react'
import { Chess } from 'chess.js'
import { Chessboard } from 'react-chessboard'
import { Upload, Brain, MessageSquare, RotateCcw } from 'lucide-react'
import './ChessAnalyzer.css'

interface MoveRecommendation {
    move: string
    evaluation: number
    explanation: string
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
    const [analysis, setAnalysis] = useState<AnalysisResult | null>(null)
    const [gameHistory, setGameHistory] = useState<string[]>([])

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
            const reader = new FileReader()
            reader.onload = (e) => {
                setUploadedImage(e.target?.result as string)
            }
            reader.readAsDataURL(file)
        }
    }

    const analyzePosition = async () => {
        setAnalysis((prev) => ({ ...prev, loading: true } as AnalysisResult))

        // Simulate AI analysis - replace with actual API call
        setTimeout(() => {
            const mockRecommendation: MoveRecommendation = {
                move: 'Nf3',
                evaluation: 0.3,
                explanation:
                    "This move follows Magnus Carlsen's style of controlling the center early.",
            }

            const mockAnalysis: AnalysisResult = {
                recommendation: mockRecommendation,
                aiExplanation:
                    'Magnus typically prefers this type of positional play, focusing on piece development and center control rather than aggressive tactics in the opening.',
                loading: false,
            }

            setAnalysis(mockAnalysis)
        }, 2000)
    }

    const resetGame = () => {
        const newGame = new Chess()
        setGame(newGame)
        setGamePosition(newGame.fen())
        setGameHistory([])
        setAnalysis(null)
    }

    const analyzeUploadedPosition = async () => {
        if (!uploadedImage) return

        setAnalysis((prev) => ({ ...prev, loading: true } as AnalysisResult))

        // Simulate image analysis - replace with actual AI vision API
        setTimeout(() => {
            const mockAnalysis: AnalysisResult = {
                recommendation: {
                    move: 'Qd5+',
                    evaluation: 2.1,
                    explanation:
                        'Magnus would likely play this forcing move to gain tempo.',
                },
                aiExplanation:
                    "Based on the uploaded position, this move creates immediate threats and follows Magnus's aggressive style when ahead in development.",
                loading: false,
            }

            setAnalysis(mockAnalysis)
        }, 3000)
    }

    return (
        <div className="chess-analyzer">
            <div className="mode-selector">
                <button
                    className={`mode-btn ${
                        analysisMode === 'board' ? 'active' : ''
                    }`}
                    onClick={() => setAnalysisMode('board')}
                >
                    <Brain size={20} />
                    Interactive Board
                </button>
                <button
                    className={`mode-btn ${
                        analysisMode === 'upload' ? 'active' : ''
                    }`}
                    onClick={() => setAnalysisMode('upload')}
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
                            </div>
                        </div>

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
                                <span>Supports JPG, PNG formats</span>
                            </label>
                        </div>

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
                                            {analysis.recommendation.evaluation}
                                        </span>
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
            </div>
        </div>
    )
}

export default ChessAnalyzer
