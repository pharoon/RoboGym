import { useState, useEffect } from 'react'
import './HomePage.css'
import DeleteModel from '../Modals/DeleteModel'
import NameInputModal from '../Modals/NameInputModal'
import { toast, ToastContainer } from 'react-toastify'
import { HomePageProps } from '@renderer/utils/interfaces'
import { useNavigate } from 'react-router-dom'
import LoadingScreen from '@renderer/utils/LoadingScreen'
import { GetModels } from '@renderer/utils/FetchData'

const HomePage: React.FC<HomePageProps> = (props) => {
  const { userProfile } = props
  const [showDeleteModel, setShowDeleteModel] = useState<boolean>(false)
  const [activeModels, setActiveModels] = useState<number>(0)
  const [modalOpen, setModalOpen] = useState(false)
  const [pendingFilePath, setPendingFilePath] = useState<string | null>(null)
  const [uploading, setUploading] = useState<boolean>(false)
  const navigate = useNavigate()
  const handleFileUpload = async () => {
    // @ts-ignore
    const filePath = await window.electronAPI.openFileDialog()
    if (filePath) {
      setPendingFilePath(filePath)
      setModalOpen(true)
    }
  }

  const handleModelNameSubmit = (modelName: string) => {
    console.log("We are here")
    setUploading(true)
    if (pendingFilePath) {
      fetch('http://localhost:5000/upload', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          FilePath: pendingFilePath,
          ModelName: modelName,
          currUserID: userProfile.user_id
        })
      })
        .then((res) => res.json())
        .then(() => toast.success("Model has been uploaded successfully"))
        .catch((err) => console.error('Upload error:', err))
        .finally(() => setUploading(false))
    }
    setModalOpen(false)
    setPendingFilePath(null)
  }
  
  useEffect(() => {
    const getActiveModels = async () => {
      const models = await GetModels(userProfile.user_id ?? '-1')
      setActiveModels(models.length)
    }
    getActiveModels()
  }, [userProfile.user_id])

  return (
    <div className="home-page">
      <div className="welcome-section glass">
        <h1>Welcome back, {userProfile.username}!</h1>
        <p>Train, validate, and test your reinforcement learning models using our realistic robotic arm simulation.</p>
      </div>

      <div className="stats-section">
        <div className="stat-card glass">
          <span className="material-icons">model_training</span>
          <h3>Active Models</h3>
          <p className="stat-number">{activeModels}</p>
        </div>
        <div className="stat-card glass">
          <span className="material-icons">psychology</span>
          <h3>Training Sessions</h3>
          <p className="stat-number">12</p>
        </div>
        <div className="stat-card glass">
          <span className="material-icons">speed</span>
          <h3>Tests Run</h3>
          <p className="stat-number">24</p>
        </div>
      </div>

      <div className="quick-actions">
        <div className="action-card glass">
          <div className="action-icon">
            <span className="material-icons">upload_file</span>
          </div>
          <div className="action-content">
            <h3>Upload Model</h3>
            <p>Import an existing model to continue working on it</p>
            <button onClick={handleFileUpload} className="action-button">
              <span className="material-icons">add</span>
              Upload Model
            </button>
          </div>
        </div>

        <div className="action-card glass">
          <div className="action-icon">
            <span className="material-icons">play_arrow</span>
          </div>
          <div className="action-content">
            <h3>Quick Train</h3>
            <p>Start training a new model with default settings</p>
            <button onClick={() => navigate('/Train')} className="action-button">
              <span className="material-icons">play_arrow</span>
              Start Training
            </button>
          </div>
        </div>
      </div>

      <ToastContainer style={{zIndex: 1000}} position='bottom-left' />
      <DeleteModel 
        userProfile={userProfile} 
        showDeleteModal={showDeleteModel} 
        setShowdeleteModal={setShowDeleteModel} 
      />
      <NameInputModal
        open={modalOpen}
        onClose={() => setModalOpen(false)}
        onSubmit={handleModelNameSubmit}
      />
      <LoadingScreen loading={uploading} text="Uploading model..." />
    </div>
  )
}

export default HomePage
