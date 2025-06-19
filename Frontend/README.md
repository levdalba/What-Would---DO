# Magnus Carlsen Chess Analyzer

A modern chess analysis application that provides move recommendations based on Magnus Carlsen's playing style. Built with React, TypeScript, and Vite.

## Features

### 🎯 Core Functionality

-   **Interactive Chess Board**: Drag and drop pieces to explore positions
-   **Image Upload Analysis**: Upload chess board images for position analysis
-   **AI-Powered Recommendations**: Get move suggestions based on Magnus Carlsen's style
-   **Move Explanations**: Detailed explanations for recommended moves
-   **Game History Tracking**: Keep track of moves and analysis

### 🧠 AI Integration Points

-   Magnus Carlsen style analysis API (ready for integration)
-   ChatGPT move explanation service
-   Computer vision for chess board recognition
-   Configurable confidence levels and alternative moves

### 🎨 User Experience

-   Modern, responsive design
-   Smooth animations and transitions
-   Accessible with keyboard navigation
-   Mobile-friendly interface

## Technology Stack

-   **Frontend**: React 18 + TypeScript
-   **Build Tool**: Vite
-   **Chess Logic**: chess.js
-   **Chess Board**: react-chessboard
-   **Icons**: Lucide React
-   **Styling**: CSS Modules

## Getting Started

### Prerequisites

-   Node.js (v18 or higher)
-   npm or yarn

### Installation

1. Install dependencies:

```bash
npm install
```

2. Set up environment variables:

```bash
cp .env.example .env.local
# Edit .env.local with your API keys
```

3. Start the development server:

```bash
npm run dev
```

4. Open your browser and navigate to `http://localhost:5173`

## Available Scripts

-   `npm run dev` - Start development server
-   `npm run build` - Build for production
-   `npm run preview` - Preview production build
-   `npm run lint` - Run ESLint

## Project Structure

```
src/
├── components/          # React components
│   ├── ChessAnalyzer.tsx
│   └── ChessAnalyzer.css
├── services/           # API services
│   └── api.ts
├── types/             # TypeScript type definitions
│   └── chess.ts
├── utils/             # Utility functions
│   └── chessUtils.ts
├── App.tsx            # Main app component
├── App.css           # App styles
├── main.tsx          # Entry point
└── index.css         # Global styles
```

## API Integration

The application is designed to integrate with various AI services:

### Magnus Style Analysis API

-   Endpoint: `/api/magnus-analyze`
-   Analyzes positions and returns moves in Magnus's style
-   Includes confidence levels and alternative moves

### Move Explanation Service

-   Uses ChatGPT or similar AI for move explanations
-   Provides educational context for recommended moves

### Chess Vision API

-   Analyzes uploaded chess board images
-   Converts images to FEN notation for analysis

## Configuration

### Environment Variables

Create a `.env.local` file with your API keys:

```env
VITE_MAGNUS_API_KEY=your_api_key
VITE_OPENAI_API_KEY=your_openai_key
VITE_VISION_API_KEY=your_vision_api_key
```

## Chess Features

### Supported Moves

-   All standard chess moves (including castling, en passant)
-   Automatic pawn promotion (defaults to Queen)
-   Move validation using chess.js

### Game Phases

-   Opening: First 20 moves or 24+ pieces on board
-   Middlegame: 13-24 pieces on board
-   Endgame: 12 or fewer pieces on board

### Analysis Features

-   Position evaluation
-   Move suggestions with explanations
-   Style signature analysis
-   Game phase detection

## Development

### Adding New Features

1. Create components in `src/components/`
2. Add types in `src/types/`
3. Implement utilities in `src/utils/`
4. Update API services in `src/services/`

### Code Style

-   Use TypeScript for all new code
-   Follow React functional component patterns
-   Use proper TypeScript interfaces
-   Include JSDoc comments for functions

## Future Enhancements

### Planned Features

-   [ ] Multiple player style models (Kasparov, Fischer, etc.)
-   [ ] Position database integration
-   [ ] Opening book analysis
-   [ ] Endgame tablebase lookup
-   [ ] Game annotation export
-   [ ] Social features and sharing

### AI Model Integration

-   [ ] Connect to trained Magnus Carlsen model
-   [ ] Implement real-time analysis
-   [ ] Add confidence metrics
-   [ ] Support for custom models
        })

````

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default tseslint.config({
  plugins: {
    // Add the react-x and react-dom plugins
    'react-x': reactX,
    'react-dom': reactDom,
  },
  rules: {
    // other rules...
    // Enable its recommended typescript rules
    ...reactX.configs['recommended-typescript'].rules,
    ...reactDom.configs.recommended.rules,
  },
})
````
