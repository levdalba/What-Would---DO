import { Crown, Brain, Award, Users, BookOpen, Zap } from 'lucide-react'
import Navigation from '../components/Navigation'
import './AboutPage.css'
const AboutPage = () => {
    return (
        <div>
            {' '}
            <Navigation />{' '}
            <div className="about-page">
                {' '}
                <div className="hero-section">
                    {' '}
                    <div className="hero-content">
                        {' '}
                        <div className="hero-icon">
                            {' '}
                            <Crown size={80} />{' '}
                        </div>{' '}
                        <h1 className="hero-title">
                            About Magnus Chess Analyzer
                        </h1>{' '}
                        <p className="hero-subtitle">
                            {' '}
                            Learn chess the way the world champion thinks and
                            plays{' '}
                        </p>{' '}
                    </div>{' '}
                </div>{' '}
                <div className="content-section">
                    {' '}
                    <div className="container">
                        {' '}
                        <div className="story-card">
                            {' '}
                            <h2>The Magnus Method</h2>{' '}
                            <p>
                                {' '}
                                Magnus Carlsen revolutionized chess with his
                                unique approach - combining deep calculation
                                with intuitive understanding. Our AI has been
                                trained on thousands of Magnus's games to
                                capture the essence of his playing style.{' '}
                            </p>{' '}
                            <p>
                                {' '}
                                From his breakthrough as the youngest World
                                Champion to maintaining dominance for over a
                                decade, Magnus's games provide a treasure trove
                                of strategic and tactical insights that our
                                analyzer brings to your fingertips.{' '}
                            </p>{' '}
                        </div>{' '}
                        <div className="features-grid">
                            {' '}
                            <div className="feature-card">
                                {' '}
                                <div className="feature-icon">
                                    {' '}
                                    <Brain size={48} />{' '}
                                </div>{' '}
                                <h3>AI-Powered Analysis</h3>{' '}
                                <p>
                                    {' '}
                                    Advanced machine learning algorithms trained
                                    specifically on Magnus Carlsen's playing
                                    patterns and decision-making processes.{' '}
                                </p>{' '}
                            </div>{' '}
                            <div className="feature-card">
                                {' '}
                                <div className="feature-icon">
                                    {' '}
                                    <Award size={48} />{' '}
                                </div>{' '}
                                <h3>World Champion Data</h3>{' '}
                                <p>
                                    {' '}
                                    Analysis based on games from the highest
                                    level of chess competition, including World
                                    Championships and elite tournaments.{' '}
                                </p>{' '}
                            </div>{' '}
                            <div className="feature-card">
                                {' '}
                                <div className="feature-icon">
                                    {' '}
                                    <Users size={48} />{' '}
                                </div>{' '}
                                <h3>For All Levels</h3>{' '}
                                <p>
                                    {' '}
                                    Whether you're a beginner or an expert,
                                    learn from moves and strategies that have
                                    proven successful at the highest level.{' '}
                                </p>{' '}
                            </div>{' '}
                            <div className="feature-card">
                                {' '}
                                <div className="feature-icon">
                                    {' '}
                                    <BookOpen size={48} />{' '}
                                </div>{' '}
                                <h3>Educational Focus</h3>{' '}
                                <p>
                                    {' '}
                                    Not just move suggestions, but explanations
                                    of the strategic and tactical concepts
                                    behind each recommendation.{' '}
                                </p>{' '}
                            </div>{' '}
                            <div className="feature-card">
                                {' '}
                                <div className="feature-icon">
                                    {' '}
                                    <Zap size={48} />{' '}
                                </div>{' '}
                                <h3>Real-Time Analysis</h3>{' '}
                                <p>
                                    {' '}
                                    Get instant feedback on positions with
                                    lightning-fast processing and comprehensive
                                    evaluation metrics.{' '}
                                </p>{' '}
                            </div>{' '}
                            <div className="feature-card">
                                {' '}
                                <div className="feature-icon">
                                    {' '}
                                    <Crown size={48} />{' '}
                                </div>{' '}
                                <h3>Champion Mindset</h3>{' '}
                                <p>
                                    {' '}
                                    Develop the same strategic thinking and
                                    positional understanding that made Magnus
                                    the strongest player in chess history.{' '}
                                </p>{' '}
                            </div>{' '}
                        </div>{' '}
                        <div className="stats-section">
                            {' '}
                            <h2>Magnus Carlsen: By the Numbers</h2>{' '}
                            <div className="stats-grid">
                                {' '}
                                <div className="stat-item">
                                    {' '}
                                    <div className="stat-number">2882</div>{' '}
                                    <div className="stat-label">
                                        Peak Rating (Highest Ever)
                                    </div>{' '}
                                </div>{' '}
                                <div className="stat-item">
                                    {' '}
                                    <div className="stat-number">5</div>{' '}
                                    <div className="stat-label">
                                        World Championship Titles
                                    </div>{' '}
                                </div>{' '}
                                <div className="stat-item">
                                    {' '}
                                    <div className="stat-number">125</div>{' '}
                                    <div className="stat-label">
                                        Months as World #1
                                    </div>{' '}
                                </div>{' '}
                                <div className="stat-item">
                                    {' '}
                                    <div className="stat-number">
                                        2013-2023
                                    </div>{' '}
                                    <div className="stat-label">
                                        World Champion Era
                                    </div>{' '}
                                </div>{' '}
                            </div>{' '}
                        </div>{' '}
                        <div className="mission-section">
                            {' '}
                            <h2>Our Mission</h2>{' '}
                            <p>
                                {' '}
                                We believe that everyone should have access to
                                world-class chess instruction. By analyzing
                                Magnus Carlsen's games and playing style through
                                advanced AI, we're democratizing access to the
                                strategic insights that have made him the
                                greatest chess player of our time.{' '}
                            </p>{' '}
                            <p>
                                {' '}
                                Our goal is not just to suggest moves, but to
                                help you understand the deeper principles of
                                chess - the way Magnus sees the game.{' '}
                            </p>{' '}
                        </div>{' '}
                    </div>{' '}
                </div>{' '}
            </div>{' '}
        </div>
    )
}
export default AboutPage
