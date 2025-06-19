import { Link } from 'react-router-dom'
import { Crown, Brain, Upload, Zap, Target, Trophy } from 'lucide-react'
import Navigation from '../components/Navigation'
import './WelcomePage.css'

const WelcomePage = () => {
    return (
        <div>
            <Navigation />
            <div className="welcome-page">
                <div className="hero-section">
                    <div className="hero-content">
                        <div className="hero-icon">
                            <Crown size={80} />
                        </div>
                        <h1 className="hero-title">
                            Magnus Carlsen Chess Analyzer
                        </h1>
                        <p className="hero-subtitle">
                            Master chess with AI-powered analysis based on the
                            World Champion's playing style
                        </p>
                        <Link to="/analyze" className="cta-button">
                            <Brain size={20} />
                            Start Analyzing
                        </Link>
                    </div>
                </div>

                <div className="features-section">
                    <div className="container">
                        <h2 className="section-title">
                            Unleash Your Chess Potential
                        </h2>
                        <div className="features-grid">
                            <div className="feature-card">
                                <div className="feature-icon">
                                    <Target size={40} />
                                </div>
                                <h3>Magnus-Style Analysis</h3>
                                <p>
                                    Get move recommendations powered by AI
                                    trained on Magnus Carlsen's games. Learn
                                    from the world's greatest chess mind.
                                </p>
                            </div>
                            <div className="feature-card">
                                <div className="feature-icon">
                                    <Crown size={40} />
                                </div>
                                <h3>Interactive Chess Board</h3>
                                <p>
                                    Play moves directly on our intuitive chess
                                    board. Analyze positions in real-time with
                                    drag-and-drop functionality.
                                </p>
                            </div>
                            <div className="feature-card">
                                <div className="feature-icon">
                                    <Upload size={40} />
                                </div>
                                <h3>Image Analysis</h3>
                                <p>
                                    Upload photos of chess positions from books,
                                    magazines, or screens. Our AI will recognize
                                    the position and provide analysis.
                                </p>
                            </div>
                            <div className="feature-card">
                                <div className="feature-icon">
                                    <Brain size={40} />
                                </div>
                                <h3>Deep Explanations</h3>
                                <p>
                                    Understand the 'why' behind each move with
                                    detailed explanations powered by advanced AI
                                    reasoning.
                                </p>
                            </div>
                            <div className="feature-card">
                                <div className="feature-icon">
                                    <Zap size={40} />
                                </div>
                                <h3>Instant Analysis</h3>
                                <p>
                                    Get lightning-fast position evaluation and
                                    move suggestions. Perfect for studying or
                                    improving your game.
                                </p>
                            </div>
                            <div className="feature-card">
                                <div className="feature-icon">
                                    <Trophy size={40} />
                                </div>
                                <h3>World Champion Style</h3>
                                <p>
                                    Learn the unique playing style that made
                                    Magnus Carlsen the strongest chess player in
                                    history.
                                </p>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="stats-section">
                    <div className="container">
                        <div className="stats-grid">
                            <div className="stat-card">
                                <div className="stat-number">16</div>
                                <div className="stat-label">
                                    World Championships
                                </div>
                            </div>
                            <div className="stat-card">
                                <div className="stat-number">2882</div>
                                <div className="stat-label">Peak Rating</div>
                            </div>
                            <div className="stat-card">
                                <div className="stat-number">10+</div>
                                <div className="stat-label">Years as #1</div>
                            </div>
                            <div className="stat-card">
                                <div className="stat-number">∞</div>
                                <div className="stat-label">
                                    Learning Potential
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="cta-section">
                    <div className="container">
                        <h2>Ready to Think Like Magnus?</h2>
                        <p>
                            Start analyzing positions and improve your chess
                            understanding today
                        </p>
                        <div className="cta-buttons">
                            <Link to="/analyze" className="primary-button">
                                <Brain size={20} />
                                Analyze Positions
                            </Link>
                            <Link to="/about" className="secondary-button">
                                Learn More
                            </Link>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default WelcomePage
