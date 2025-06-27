import { useState } from 'react'
import { useNavigate, useLocation, Outlet } from 'react-router-dom'
import RoboGymLogo from '../../../../../../RoboGym/src/Assets/RogoGymLogo.png'
import './MainLayout.css'

const MenuItem = ({ icon, text, onClick, active }: { icon: string, text: string, onClick: () => void, active: boolean }) => (
  <div className={`menu-item ${active ? 'active' : ''}`} onClick={onClick}>
    <span className="material-icons">{icon}</span>
    <span>{text}</span>
  </div>
)

interface MainLayoutProps {
  userProfile: any;
  setUserProfile: (profile: any) => void;
}

const MainLayout: React.FC<MainLayoutProps> = ({ userProfile, setUserProfile }) => {
  const navigate = useNavigate()
  const location = useLocation()

  const logOut = () => {
    setUserProfile({})
    navigate("/")
  }

  return (
    <div className="layout-container">
      {/* Sidebar */}
      <div className="sidebar glass">
        <div className="logo-section">
          <img src={RoboGymLogo} alt="RoboGym Logo" />
          <h2>RoboGym</h2>
        </div>
        
        <div className="menu-items">
          <MenuItem 
            icon="home" 
            text="Dashboard" 
            onClick={() => navigate('/')} 
            active={location.pathname === '/'}
          />
          <MenuItem 
            icon="science" 
            text="Train Model" 
            onClick={() => navigate('/Train')} 
            active={location.pathname === '/Train'}
          />
          <MenuItem 
            icon="speed" 
            text="Test Model" 
            onClick={() => navigate('/Test')} 
            active={location.pathname === '/Test'}
          />
          <MenuItem 
            icon="analytics" 
            text="Analytics" 
            onClick={() => navigate('/Analytics')} 
            active={location.pathname === '/Analytics'}
          />
          <MenuItem 
            icon="list" 
            text="All Models" 
            onClick={() => navigate('/AllModels')} 
            active={location.pathname === '/AllModels'}
          />
        </div>

        <div className="user-section glass">
          <div className="user-info">
            <span className="material-icons">account_circle</span>
            <span>{userProfile.username}</span>
          </div>
          <button className="logout-btn" onClick={logOut}>
            <span className="material-icons">logout</span>
            Log Out
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="main-content">
        <Outlet />
      </div>
    </div>
  )
}

export default MainLayout 