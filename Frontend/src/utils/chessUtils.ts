import { Chess } from 'chess.js'

/**
 * Determines the current phase of the game based on material and position
 */
export const getGamePhase = (
    game: Chess
): 'opening' | 'middlegame' | 'endgame' => {
    const history = game.history()
    const totalPieces = game
        .board()
        .flat()
        .filter((piece) => piece !== null).length

    // Opening phase: first 10-15 moves or if many pieces are still on board
    if (history.length < 20 || totalPieces > 24) {
        return 'opening'
    }

    // Endgame phase: very few pieces left
    if (totalPieces <= 12) {
        return 'endgame'
    }

    // Otherwise, it's middlegame
    return 'middlegame'
}

/**
 * Converts a move to standard algebraic notation with additional context
 */
export const formatMoveWithContext = (
    move: string,
    gamePhase: string
): string => {
    const phaseEmoji = {
        opening: '🌅',
        middlegame: '⚔️',
        endgame: '👑',
    }

    return `${phaseEmoji[gamePhase as keyof typeof phaseEmoji]} ${move}`
}

/**
 * Evaluates position strength (simplified evaluation)
 */
export const evaluatePosition = (game: Chess): number => {
    // This is a very simplified evaluation
    // In a real application, you'd use a proper chess engine

    const pieceValues = {
        p: 1,
        n: 3,
        b: 3,
        r: 5,
        q: 9,
        k: 0,
        P: 1,
        N: 3,
        B: 3,
        R: 5,
        Q: 9,
        K: 0,
    }

    let evaluation = 0
    const board = game.board()

    for (let row of board) {
        for (let square of row) {
            if (square) {
                const value =
                    pieceValues[square.type as keyof typeof pieceValues] || 0
                evaluation += square.color === 'w' ? value : -value
            }
        }
    }

    // Add small random factor to make it more interesting
    return evaluation + (Math.random() - 0.5) * 0.5
}

/**
 * Checks if a position is likely from Magnus Carlsen's style
 */
export const analyzeStyleSignature = (
    game: Chess
): {
    isTypicalMagnus: boolean
    confidence: number
    reasoning: string
} => {
    const gamePhase = getGamePhase(game)
    const history = game.history()

    // Simplified style analysis - in reality this would use ML models
    let confidence = 0.5
    let reasoning = ''
    let isTypicalMagnus = false

    if (gamePhase === 'opening') {
        // Magnus often plays solid openings
        const solidOpenings = ['Nf3', 'e4', 'd4', 'c4']
        const firstMove = history[0]

        if (solidOpenings.includes(firstMove)) {
            confidence += 0.2
            reasoning = 'Solid opening choice typical of Magnus'
            isTypicalMagnus = true
        }
    } else if (gamePhase === 'endgame') {
        // Magnus is known for excellent endgame technique
        confidence += 0.3
        reasoning = 'Endgame precision is a Magnus hallmark'
        isTypicalMagnus = true
    }

    return {
        isTypicalMagnus,
        confidence: Math.min(confidence, 1.0),
        reasoning,
    }
}

/**
 * Generates move suggestions based on position analysis
 */
export const generateMoveSuggestions = (
    game: Chess
): Array<{
    move: string
    evaluation: number
    reasoning: string
}> => {
    const legalMoves = game.moves({ verbose: true })
    const suggestions: Array<{
        move: string
        evaluation: number
        reasoning: string
    }> = []

    // Simple heuristics for move suggestions
    legalMoves.forEach((move) => {
        const gameCopy = new Chess(game.fen())
        gameCopy.move(move)

        const evaluation = evaluatePosition(gameCopy)
        let reasoning = 'Solid positional move'

        // Add specific reasoning based on move type
        if (move.flags.includes('c')) {
            reasoning = 'Captures material - typical Magnus efficiency'
        } else if (move.flags.includes('k') || move.flags.includes('q')) {
            reasoning = 'Castling for king safety - Magnus prioritizes safety'
        } else if (move.piece === 'n' || move.piece === 'b') {
            reasoning = 'Piece development - follows Magnus opening principles'
        }

        suggestions.push({
            move: move.san,
            evaluation,
            reasoning,
        })
    })

    // Return top 3 suggestions sorted by evaluation
    return suggestions.sort((a, b) => b.evaluation - a.evaluation).slice(0, 3)
}

/**
 * Validates if an uploaded image could be a chess position
 */
export const validateChessImage = (file: File): Promise<boolean> => {
    return new Promise((resolve) => {
        // Simple validation - check file type and size
        const validTypes = ['image/jpeg', 'image/png', 'image/webp']
        const maxSize = 10 * 1024 * 1024 // 10MB

        if (!validTypes.includes(file.type)) {
            resolve(false)
            return
        }

        if (file.size > maxSize) {
            resolve(false)
            return
        }

        // In a real app, you might do additional validation
        // like checking image dimensions, etc.
        resolve(true)
    })
}
