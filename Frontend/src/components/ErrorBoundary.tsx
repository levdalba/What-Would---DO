import { Component } from 'react'
import type { ErrorInfo, ReactNode } from 'react'

interface Props {
    children: ReactNode
}

interface State {
    hasError: boolean
    error?: Error
}

class ErrorBoundary extends Component<Props, State> {
    constructor(props: Props) {
        super(props)
        this.state = { hasError: false }
    }

    static getDerivedStateFromError(error: Error): State {
        return { hasError: true, error }
    }

    componentDidCatch(error: Error, errorInfo: ErrorInfo) {
        console.error('Chess Analyzer Error:', error, errorInfo)
    }

    render() {
        if (this.state.hasError) {
            return (
                <div className="error-boundary">
                    <div className="error-content">
                        <h2>Something went wrong</h2>
                        <p>
                            We apologize for the inconvenience. The chess
                            analyzer encountered an error.
                        </p>
                        <details>
                            <summary>Error Details</summary>
                            <pre>{this.state.error?.message}</pre>
                        </details>
                        <button
                            onClick={() =>
                                this.setState({
                                    hasError: false,
                                    error: undefined,
                                })
                            }
                            className="retry-btn"
                        >
                            Try Again
                        </button>
                        <button
                            onClick={() => window.location.reload()}
                            className="reload-btn"
                        >
                            Reload Page
                        </button>
                    </div>
                </div>
            )
        }

        return this.props.children
    }
}

export default ErrorBoundary
