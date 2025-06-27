import { useState } from 'react'
import './HomePage.css'
import DeleteModel from '../Modals/DeleteModel'
import NameInputModal from '../Modals/NameInputModal'
import { toast, ToastContainer } from 'react-toastify'
import { HomePageProps } from '@renderer/utils/interfaces'

const HomePage: React.FC<HomePageProps> = (props) => {
  const { userProfile } = props
  const [showDeleteModel, setShowDeleteModel] = useState<boolean>(false)
  const [modalOpen, setModalOpen] = useState(false)
  const [pendingFilePath, setPendingFilePath] = useState<string | null>(null)

  const handleFileUpload = async () => {
    // @ts-ignore
    const filePath = await window.electronAPI.openFileDialog()
    if (filePath) {
      setPendingFilePath(filePath)
      setModalOpen(true)
    }
  }

  const handleModelNameSubmit = (modelName: string) => {
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
        .then(() => toast.success("File is uploaded Successfully"))
        .catch((err) => console.error('Upload error:', err))
    }
    setModalOpen(false)
    setPendingFilePath(null)
  }

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
          <p className="stat-number">3</p>
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
            <span className="material-icons">delete</span>
          </div>
          <div className="action-content">
            <h3>Delete Model</h3>
            <p>Remove models you no longer need</p>
            <button onClick={() => setShowDeleteModel(true)} className="action-button danger">
              <span className="material-icons">delete_outline</span>
              Delete Model
            </button>
          </div>
        </div>
      </div>

      <ToastContainer position='bottom-left' />
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
    </div>
  )
}

export default HomePage
