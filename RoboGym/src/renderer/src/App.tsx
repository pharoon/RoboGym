import { MemoryRouter as Router, Routes, Route} from "react-router-dom"
import HomePage from "./components/HomePage/HomePage"

function App(): React.JSX.Element {
  // const ipcHandle = (): void => window.electron.ipcRenderer.send('ping')

  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage/>} />
      </Routes>
    </Router>
  )
}

export default App
