import { MemoryRouter as Router, Routes, Route } from 'react-router-dom'
import HomePage from './components/HomePage/HomePage'
import { useEffect } from 'react'
import Train from './components/Trainpage/Train'
import "./app.css"
import Test from './components/TestPage/Test'
function App(): React.JSX.Element {
  // starts the iniitialization of the app
  // useEffect(() => {
  //   fetch('http://localhost:5000/initialize', {
  //     method: 'post'
  //   }).finally(()=>{
  //     console.warn("Sccessfully initalized ")
  //   })
  // }, [])
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/Train" element={<Train />} />
        <Route path="/Test" element={<Test />} />
      </Routes>
    </Router>
  )
}

export default App
