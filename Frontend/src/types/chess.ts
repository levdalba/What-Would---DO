// Chess analysis related types
export interface MoveRecommendation {
    move: string
    evaluation: number
    explanation: string
    confidence: number
}

export interface AnalysisResult {
    recommendation: MoveRecommendation
    aiExplanation: string
    loading: boolean
    alternativeMoves?: MoveRecommendation[]
}

// Chess position types
export interface ChessPosition {
    fen: string
    pgn: string
    moveNumber: number
    turn: 'w' | 'b'
}

// Image upload types
export interface UploadedImage {
    file: File
    url: string
    analyzedPosition?: ChessPosition
}

// API response types
export interface MagnusAnalysisRequest {
    position: string // FEN string
    gamePhase: 'opening' | 'middlegame' | 'endgame'
    playerColor: 'white' | 'black'
}

export interface MagnusAnalysisResponse {
    bestMove: string
    evaluation: number
    explanation: string
    styleConfidence: number
    alternativeMoves: Array<{
        move: string
        evaluation: number
        probability: number
    }>
}

// Game history types
export interface GameMove {
    move: string
    san: string
    fen: string
    timestamp: Date
}

export interface GameHistory {
    moves: GameMove[]
    result?: '1-0' | '0-1' | '1/2-1/2' | '*'
    startPosition: string
}

// UI state types
export type AnalysisMode = 'board' | 'upload'

export interface AppState {
    currentPosition: ChessPosition
    analysisMode: AnalysisMode
    isAnalyzing: boolean
    error: string | null
}
