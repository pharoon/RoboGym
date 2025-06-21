import { MemoryRouter as Router, Routes, Route } from 'react-router-dom'
import HomePage from './components/HomePage/HomePage'
import Train from './components/Trainpage/Train'
import "./app.css"
import Test from './components/TestPage/Test'
import AllModel from './components/ListModelsPage/AllModel'
import AnalyticsComponent from './components/AnalyticsPage/Analytics'
function App(): React.JSX.Element {
  
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/Train" element={<Train />} />
        <Route path="/Test" element={<Test />} />
        <Route path="/AllModels" element={<AllModel />} />
        <Route path="/Analytics" element={<AnalyticsComponent />} />
      </Routes>
    </Router>
  )
}

export default App
