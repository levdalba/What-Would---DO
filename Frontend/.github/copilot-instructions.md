# Copilot Instructions

<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

## Project Overview

This is a chess frontend application built with Vite, React, and TypeScript. The application is designed to integrate with AI models trained on Magnus Carlsen's playing style to provide chess move recommendations.

## Key Features

-   Interactive chess board with drag-and-drop piece movement
-   Chess board position image upload and analysis
-   AI-powered move recommendations based on Magnus Carlsen's playing style
-   Move explanation integration with ChatGPT
-   Modern, responsive UI design

## Code Style Guidelines

-   Use TypeScript for all components and utilities
-   Follow React functional component patterns with hooks
-   Use CSS modules or styled-components for styling
-   Implement proper error handling and loading states
-   Use chess.js library for chess logic and validation
-   Ensure accessibility with proper ARIA labels and keyboard navigation

## Chess-Specific Considerations

-   Use standard chess notation (algebraic notation)
-   Implement proper chess piece movement validation
-   Handle special chess moves (castling, en passant, promotion)
-   Support both light and dark themes for the chess board
-   Ensure proper game state management

## AI Integration Points

-   Design components to accept move recommendation data
-   Create interfaces for AI model communication
-   Implement proper loading states for AI analysis
-   Handle API errors gracefully
