import { useEffect, useState } from 'react'
import { toast, ToastContainer } from 'react-toastify'
import { AllDataProps, Model } from '@renderer/utils/interfaces'
import { GetModels } from '@renderer/utils/FetchData'
import './AllModel.css'
import LoadingScreen from '@renderer/utils/LoadingScreen'
import Modal from '../Modals/Modal'
import ModelCard from './ModelCard'
import EmptyState from './EmptyState'
import { useNavigate } from 'react-router-dom'

const AllModel: React.FC<AllDataProps> = ({ userProfile }) => {
  const [models, setModels] = useState<Model[]>([])
  const [loading, setLoading] = useState(true)
  const [loadingText, setLoadingText] = useState('Loading models ...')
  const [renameDialogOpen, setRenameDialogOpen] = useState(false)
  const [renameTarget, setRenameTarget] = useState<string | null>(null)
  const [newName, setNewName] = useState('')
  const [sessions, setSessions] = useState<any[]>([])
  const [sessionsModalOpen, setSessionsModalOpen] = useState(false)
  const [selectedSessionIdx, setSelectedSessionIdx] = useState(0)
  const navigate = useNavigate()

  const fetchModels = async () => {
    const models: Model[] = await GetModels(userProfile.user_id ?? '-1')
    setModels(models)
  }

  useEffect(() => {
    fetchModels().finally(() => setLoading(false))
  }, [])

  console.log('Sessions are ', sessions)

  const handleDelete = (modelName: string) => {
    setLoading(true)
    setLoadingText('Deleting model ...')
    fetch('http://localhost:5000/delete', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ model_name: modelName, currUserID: userProfile.user_id })
    })
      .then((res) => {
        if (!res.ok) throw new Error('Failed to delete model')
        return res.json()
      })
      .then(() => {
        fetchModels()
        toast.success('Model has been deleted')
      })
      .catch((err) => {
        console.error('Error deleting model:', err)
        toast.error('Failed to delete model')
      })
      .finally(() => setLoading(false))
  }

  const handleDownload = (modelName: string) => {
    // Placeholder for download logic
    toast.info(`Download for ${modelName} not implemented yet.`)
  }

  const handleRenameClick = (modelName: string) => {
    setRenameTarget(modelName)
    setNewName('')
    setRenameDialogOpen(true)
  }

  const handleRename = () => {
    if (!renameTarget || !newName.trim()) return
    setLoading(true)
    setLoadingText('Renaming model ...')
    fetch('http://localhost:5000/rename', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model_name: renameTarget,
        new_name: newName,
        currUserID: userProfile.user_id
      })
    })
      .then((res) => {
        if (!res.ok) throw new Error('Failed to rename model')
        return res.json()
      })
      .then(() => {
        fetchModels()
        toast.success('Model has been renamed')
        setRenameDialogOpen(false)
      })
      .catch((err) => {
        console.error('Error renaming model:', err)
        toast.error('Failed to rename model')
      })
      .finally(() => setLoading(false))
  }

  const handleDialogClose = () => {
    setRenameDialogOpen(false)
    setRenameTarget(null)
    setNewName('')
  }

  // Add a placeholder for continue training
  const handleContinueTraining = (modelName: string) => {
    navigate(`/Train?modelName=${modelName}`)
  }

  // Add a placeholder for view sessions
  const handleViewSessions = (modelID: number) => {
    setLoading(true)
    fetch('http://localhost:5000/getModelSessions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ modelID: modelID, currUserID: userProfile.user_id })
    })
      .then((res) => {
        if (!res.ok) throw new Error('Failed to get model sessions')
        return res.json()
      })
      .then((data) => {
        setSessions(data.sessions || [])
        setSessionsModalOpen(true)
      })
      .catch((err) => {
        toast.error('Failed to fetch sessions')
        console.error('Error getting model sessions:', err)
      })
      .finally(() => setLoading(false))
  }

  return (
    <div className="models-container">
      <div className="models-header">
        <span className="material-icons">model_training</span>
        <h1>Available AI Models</h1>
        <p>Here's a list of all your trained machine learning models</p>
      </div>

      <div className="models-grid">
        {models.length === 0 ? (
          <EmptyState
            title="No Models Found"
            message="You haven't trained or uploaded any models yet."
          />
        ) : (
          models.map((model, index) => (
            <ModelCard
              key={index}
              model={model}
              onDownload={handleDownload}
              onRename={handleRenameClick}
              onDelete={handleDelete}
              onContinueTraining={handleContinueTraining}
              onViewSessions={handleViewSessions}
            />
          ))
        )}
      </div>

      <Modal isOpen={renameDialogOpen} onClose={handleDialogClose}>
        <div className="dialog-header">
          <span
            className="material-icons"
            style={{ color: '#f1c40f', fontSize: '2rem', marginRight: '8px' }}
          >
            warning
          </span>
          <h2>Rename Model</h2>
        </div>
        <div className="dialog-body">
          <p>Enter a new name for the model "{renameTarget}".</p>
          <input
            type="text"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            placeholder="New model name"
            className="dialog-input"
            autoFocus
            onFocus={(e) => (e.target.style.outline = 'none')}
          />
          <p style={{ color: '#f39c12', marginTop: '8px' }}>This action cannot be undone.</p>
        </div>
        <div className="dialog-actions">
          <button className="dialog-cancel" onClick={handleDialogClose}>
            Cancel
          </button>
          <button className="dialog-confirm" onClick={handleRename} disabled={!newName.trim()}>
            Rename
          </button>
        </div>
      </Modal>

      <Modal isOpen={sessionsModalOpen} onClose={() => setSessionsModalOpen(false)}>
        <div className="dialog-header">
          <span
            className="material-icons"
            style={{ color: '#3b82f6', fontSize: '2rem', marginRight: '8px' }}
          >
            timeline
          </span>
          <h2>Training Sessions</h2>
        </div>
        <div className="dialog-body sessions-modal-body">
          {sessions.length === 0 ? (
            <div style={{ textAlign: 'center', color: '#94a3b8', padding: '2rem' }}>
              <span className="material-icons" style={{ fontSize: '3rem', marginBottom: '1rem' }}>
                folder_open
              </span>
              <p>No training sessions found for this model.</p>
            </div>
          ) : (
            <div className="sessions-modal-content">
              {sessions.map((session, idx) => (
                <div 
                  key={idx}
                  className={`session-card ${selectedSessionIdx === idx ? 'selected' : ''}`}
                  onClick={() => setSelectedSessionIdx(idx)}
                >
                  <div className="session-header">
                    <span className="material-icons">science</span>
                    <h4>Session {idx + 1}</h4>
                  </div>
                  
                  <div className="session-info">
                    <div className="info-item">
                      <span className="material-icons">calendar_today</span>
                      <span>{new Date(session.started_at).toLocaleDateString()}</span>
                    </div>
                    <div className="info-item">
                      <span className="material-icons">timer</span>
                      <span>{Number(session.total_time).toFixed(2)}s</span>
                    </div>
                    <div className="info-item">
                      <span className="material-icons">trending_up</span>
                      <span>{session.timesteps} steps</span>
                    </div>
                    <div className="info-item">
                      <span className="material-icons">star</span>
                      <span>Reward: {session.mean_reward.toFixed(2)}</span>
                    </div>
                  </div>

                  <button className="view-details-button" title="View Details">
                    <span className="material-icons">visibility</span>
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="dialog-actions">
          <button className="dialog-cancel" onClick={() => setSessionsModalOpen(false)}>
            Close
          </button>
        </div>
      </Modal>

      <ToastContainer position="bottom-left" />
      <LoadingScreen loading={loading} text={loadingText} />
    </div>
  )
}

export default AllModel
