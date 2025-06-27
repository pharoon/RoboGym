import { MemoryRouter as Router, Routes, Route } from 'react-router-dom'
import HomePage from './components/HomePage/HomePage'
import Train from './components/Trainpage/Train'
import "./app.css"
import Test from './components/TestPage/Test'
import AllModel from './components/ListModelsPage/AllModel'
import AnalyticsComponent from './components/AnalyticsPage/Analytics'
import Login from './components/LoginPage/LoginPage'
import MainLayout from './components/Layout/MainLayout'
import { useState } from 'react'
import { Profile } from './utils/interfaces'

function App(): React.JSX.Element {
  const [userProfile, setUserProfile] = useState<Partial<Profile>>({})

  return (
    <Router>
      <Routes>
        <Route path="/" element={<Login userProfile={userProfile} setUserProfile={setUserProfile} />} />
        
        {/* Protected Routes with MainLayout */}
        <Route element={<MainLayout userProfile={userProfile} setUserProfile={setUserProfile} />}>
          <Route path="/HomePage" element={<HomePage userProfile={userProfile} setUserProfile={setUserProfile} />} />
          <Route path="/Train" element={<Train userProfile={userProfile} />} />
          <Route path="/Test" element={<Test userProfile={userProfile} />} />
          <Route path="/AllModels" element={<AllModel userProfile={userProfile}/>} />
          <Route path="/Analytics" element={<AnalyticsComponent userProfile={userProfile} />} />
        </Route>
      </Routes>
    </Router>
  )
}

export default App
