import { Link, useLocation } from 'react-router-dom'
import { Crown, Home, Brain, Info } from 'lucide-react'
import './Navigation.css'

const Navigation = () => {
    const location = useLocation()

    const navItems = [
        { path: '/', icon: Home, label: 'Home' },
        { path: '/analyze', icon: Brain, label: 'Analyze' },
        { path: '/about', icon: Info, label: 'About' },
    ]

    return (
        <nav className="navigation">
            <div className="nav-container">
                <Link to="/" className="nav-logo">
                    <Crown size={24} />
                    <span>Magnus AI</span>
                </Link>

                <div className="nav-links">
                    {navItems.map(({ path, icon: Icon, label }) => (
                        <Link
                            key={path}
                            to={path}
                            className={`nav-link ${
                                location.pathname === path ? 'active' : ''
                            }`}
                        >
                            <Icon size={18} />
                            <span>{label}</span>
                        </Link>
                    ))}
                </div>
            </div>
        </nav>
    )
}

export default Navigation
