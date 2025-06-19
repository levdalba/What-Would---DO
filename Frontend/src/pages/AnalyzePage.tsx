import { Link } from 'react-router-dom'
import { ArrowLeft } from 'lucide-react'
import Navigation from '../components/Navigation'
import ChessAnalyzer from '../components/ChessAnalyzer'
import './AnalyzePage.css'

const AnalyzePage = () => {
    return (
        <div>
            <Navigation />
            <div className="analyze-page">
                <div className="analyze-header">
                    <div className="container">
                        <Link to="/" className="back-button">
                            <ArrowLeft size={20} />
                            Back to Home
                        </Link>
                        <h1>Chess Analysis</h1>
                        <p>
                            Analyze positions with Magnus Carlsen's strategic
                            insights
                        </p>
                    </div>
                </div>

                <div className="analyze-content">
                    <ChessAnalyzer />
                </div>
            </div>
        </div>
    )
}

export default AnalyzePage
