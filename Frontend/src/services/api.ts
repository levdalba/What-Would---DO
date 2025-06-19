import type {
    MagnusAnalysisRequest,
    MagnusAnalysisResponse,
} from '../types/chess'

// Configuration for different AI services
const API_CONFIG = {
    magnus: {
        endpoint: import.meta.env.VITE_MAGNUS_API_URL || '/api/magnus-analyze',
        apiKey: import.meta.env.VITE_MAGNUS_API_KEY || '',
    },
    chatgpt: {
        endpoint:
            import.meta.env.VITE_OPENAI_API_URL ||
            'https://api.openai.com/v1/chat/completions',
        apiKey: import.meta.env.VITE_OPENAI_API_KEY || '',
    },
    vision: {
        endpoint: import.meta.env.VITE_VISION_API_URL || '/api/chess-vision',
        apiKey: import.meta.env.VITE_VISION_API_KEY || '',
    },
}

/**
 * Analyzes a chess position using Magnus Carlsen style AI model
 */
export const analyzeMagnusStyle = async (
    _request: MagnusAnalysisRequest
): Promise<MagnusAnalysisResponse> => {
    // For now, return mock data
    // In production, this would call your trained AI model

    return new Promise((resolve) => {
        setTimeout(() => {
            const mockResponse: MagnusAnalysisResponse = {
                bestMove: 'Nf3',
                evaluation: 0.25,
                explanation:
                    "This move follows Magnus Carlsen's opening principles, controlling the center while maintaining flexibility.",
                styleConfidence: 0.85,
                alternativeMoves: [
                    { move: 'e4', evaluation: 0.2, probability: 0.75 },
                    { move: 'd4', evaluation: 0.18, probability: 0.65 },
                    { move: 'c4', evaluation: 0.15, probability: 0.55 },
                ],
            }
            resolve(mockResponse)
        }, 1500 + Math.random() * 1000) // Simulate API delay
    })
}

/**
 * Gets move explanation from ChatGPT
 */
export const explainMoveWithChatGPT = async (
    _position: string,
    move: string,
    _context: string
): Promise<string> => {
    // Mock implementation - replace with actual OpenAI API call
    return new Promise((resolve) => {
        setTimeout(() => {
            const explanations = [
                `The move ${move} is excellent because it improves piece coordination and follows Magnus's positional understanding. This type of move demonstrates the deep strategic thinking that makes Magnus so formidable.`,
                `${move} shows Magnus's characteristic style of finding the most practical move in the position. Rather than forcing tactics, this move improves the position while maintaining flexibility for future plans.`,
                `This move ${move} exemplifies Magnus's approach to chess - it's simple, strong, and creates long-term advantages. Magnus often chooses moves that are difficult for opponents to meet effectively.`,
                `${move} reflects Magnus's exceptional endgame technique and positional understanding. This type of move often looks simple but contains deep strategic ideas that become clear later in the game.`,
            ]

            const randomExplanation =
                explanations[Math.floor(Math.random() * explanations.length)]
            resolve(randomExplanation)
        }, 1000 + Math.random() * 500)
    })
}

/**
 * Analyzes uploaded chess board image
 */
export const analyzeChessBoardImage = async (
    _imageFile: File
): Promise<{
    fen: string
    confidence: number
    detectedPieces: number
}> => {
    // Mock implementation - replace with actual computer vision API
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            // Simulate occasional failures
            if (Math.random() < 0.1) {
                reject(
                    new Error(
                        'Could not detect chess board in image. Please ensure the board is clearly visible and well-lit.'
                    )
                )
                return
            }

            // Mock successful analysis
            const mockResult = {
                fen: 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1',
                confidence: 0.95,
                detectedPieces: 32,
            }

            resolve(mockResult)
        }, 2000 + Math.random() * 1000)
    })
}

/**
 * Generic API error handler
 */
export const handleApiError = (error: any): string => {
    if (error.response) {
        // Server responded with error status
        return `API Error: ${
            error.response.data?.message || 'Unknown server error'
        }`
    } else if (error.request) {
        // Request was made but no response received
        return 'Network Error: Could not connect to the analysis service'
    } else {
        // Something else happened
        return `Error: ${error.message || 'Unknown error occurred'}`
    }
}

/**
 * Validates API configuration
 */
export const validateApiConfig = (): {
    isValid: boolean
    missing: string[]
} => {
    const missing: string[] = []

    // Check which API keys are missing (optional for development)
    if (!API_CONFIG.magnus.apiKey) missing.push('Magnus API Key')
    if (!API_CONFIG.chatgpt.apiKey) missing.push('OpenAI API Key')
    if (!API_CONFIG.vision.apiKey) missing.push('Vision API Key')

    return {
        isValid: missing.length === 0,
        missing,
    }
}

/**
 * Test API connection
 */
export const testApiConnection = async (): Promise<boolean> => {
    try {
        // Simple test - try to analyze a basic position
        const testRequest: MagnusAnalysisRequest = {
            position:
                'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
            gamePhase: 'opening',
            playerColor: 'white',
        }

        await analyzeMagnusStyle(testRequest)
        return true
    } catch (error) {
        console.error('API connection test failed:', error)
        return false
    }
}
