import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import './App.css'
import ErrorBoundary from './components/ErrorBoundary'
import WelcomePage from './pages/WelcomePage'
import AboutPage from './pages/AboutPage'
import AnalyzePage from './pages/AnalyzePage'

function App() {
    return (
        <ErrorBoundary>
            <Router>
                <Routes>
                    <Route path="/" element={<WelcomePage />} />
                    <Route path="/about" element={<AboutPage />} />
                    <Route path="/analyze" element={<AnalyzePage />} />
                </Routes>
            </Router>
        </ErrorBoundary>
    )
}

export default App
