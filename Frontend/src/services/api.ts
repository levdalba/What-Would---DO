import type {
    MagnusAnalysisRequest,
    MagnusAnalysisResponse,
} from '../types/chess'

// Configuration for different AI services
const API_CONFIG = {
    magnus: {
        endpoint: 'http://10.224.9.93:8000/predict',
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

interface PredictionResponse {
    predicted_move: string
    confidence: number
}

/**
 * Predicts the next move using the trained Magnus Carlsen model
 */
export const predictMagnusMove = async (
    boardFen: string
): Promise<PredictionResponse> => {
    console.log('Attempting to predict move for FEN:', boardFen)

    try {
        // The issue is CORS preflight - your server needs to handle OPTIONS requests
        // Let's try a different approach first
        const response = await fetch(API_CONFIG.magnus.endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            // Remove Accept header to avoid triggering CORS preflight
            body: JSON.stringify({
                board: boardFen,
            }),
        })

        console.log('Response status:', response.status)

        if (!response.ok) {
            throw new Error(
                `HTTP error! status: ${response.status} - ${response.statusText}. CORS Issue: Your server needs to handle OPTIONS requests.`
            )
        }

        const data: PredictionResponse = await response.json()
        console.log('Received prediction:', data)
        return data
    } catch (error) {
        console.error('Error predicting move:', error)

        // Provide specific CORS error information
        if (error instanceof TypeError && error.message.includes('fetch')) {
            throw new Error(
                'CORS Error: Your API server at http://10.224.9.93:8000 needs to:\n1. Handle OPTIONS requests\n2. Include CORS headers\n3. Accept POST requests to /predict'
            )
        }

        throw error
    }
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
export const testApiConnection = async (): Promise<{
    success: boolean
    message: string
    details?: any
}> => {
    try {
        console.log('Testing API connection to:', API_CONFIG.magnus.endpoint)

        // Test with a simple starting position
        const testFen =
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

        const result = await predictMagnusMove(testFen)

        return {
            success: true,
            message: 'API connection successful!',
            details: result,
        }
    } catch (error: any) {
        return {
            success: false,
            message: error.message || 'Unknown error',
            details: error,
        }
    }
}
