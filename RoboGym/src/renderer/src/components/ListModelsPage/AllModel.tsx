import React, { useEffect, useState } from 'react'
import { toast, ToastContainer } from 'react-toastify'
import { AllDataProps, Model } from '@renderer/utils/interfaces'
import { GetModels } from '@renderer/utils/FetchData'
import './AllModel.css'
import LoadingScreen from '@renderer/utils/LoadingScreen'
import Modal from '../Modals/Modal'
import ModelCard from './ModelCard'
import EmptyState from './EmptyState'
import { useNavigate } from 'react-router-dom'
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Legend } from 'recharts'

interface TrainingSession {
  completed_at: string;
  final_timesteps: number;
  is_completed: boolean;
  model_id: number;
  model_name: string;
  progress: Array<{
    mean_reward: number;
    timestep: number;
  }>;
  session_id: number;
  started_at: string;
  total_time: number;
  user_id: number;
}

interface ChartData {
  timestep: number;
  mean_reward: number;
  session_id: number;
  session_start: string;
  session_duration: string;
}

const AllModel: React.FC<AllDataProps> = ({ userProfile }) => {
  const [models, setModels] = useState<Model[]>([])
  const [loading, setLoading] = useState(true)
  const [loadingText, setLoadingText] = useState('Loading models ...')
  const [renameDialogOpen, setRenameDialogOpen] = useState(false)
  const [renameTarget, setRenameTarget] = useState<string | null>(null)
  const [newName, setNewName] = useState('')
  const [sessions, setSessions] = useState<any[]>([])
  const [sessionsModalOpen, setSessionsModalOpen] = useState(false)
  const [selectedSessionIdx, setSelectedSessionIdx] = useState<number | null>(null)
  const [sessionDetailsModalOpen, setSessionDetailsModalOpen] = useState(false)
  const [sessionDetails, setSessionDetails] = useState<ChartData[]>([])
  const [sessionDetailsLoading, setSessionDetailsLoading] = useState(false)
  const [selectedSession, setSelectedSession] = useState<any>(null)
  const navigate = useNavigate()

  const fetchModels = async () => {
    const modelsData: Model[] = await GetModels(userProfile.user_id ?? '-1')
    setModels(modelsData)
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

  const handleDownload = async (modelName: string) => {
    try {
      // Open save dialog to let user choose download location
      const savePath = await window.electronAPI.saveFileDialog(`${modelName}.zip`)
      
      if (!savePath) {
        // User cancelled the dialog
        return
      }

      setLoading(true)
      setLoadingText('Downloading model ...')

      // Call the download API with required parameters
      const params = new URLSearchParams({
        model_name: modelName,
        local_model_path: savePath,
        curr_user_id: userProfile.user_id?.toString() || '-1'
      })

      const response = await fetch(`http://localhost:5000/download?${params}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      })

      if (!response.ok) {
        throw new Error('Failed to download model')
      }

      const data = await response.json()
      
      if (data.status === 'ok') {
        toast.success('Model downloaded successfully')
      } else {
        throw new Error(data.message || 'Download failed')
      }
    } catch (err) {
      console.error('Error downloading model:', err)
      toast.error('Failed to download model')
    } finally {
      setLoading(false)
    }
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
  const handleContinueTraining = (modelName: string, modelID: number) => {
    navigate(`/Train?modelName=${modelName}&continueTraining=true&modelID=${modelID}`)
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

  const handleViewSessionDetails = async (session: any) => {
    setSelectedSession(session)
    setSessionDetailsLoading(true)
    setSessionDetailsModalOpen(true)
    
    try {
      // Parse the train_log string from the session data
      if (session.train_log) {
        const trainLogData = JSON.parse(session.train_log);
        
        // Transform the parsed data to chart format
        const transformedData = trainLogData.map((point: any, index: number) => ({
          timestep: point.timestep || index * 10, // Use timestep from data or calculate from index
          mean_reward: point.mean_reward,
          session_id: session.session_id,
          session_start: new Date(session.started_at).toLocaleDateString(),
          session_duration: session.total_time ? Number(session.total_time).toFixed(2) : '0.00'
        }));
        
        setSessionDetails(transformedData);
      } else {
        // Fallback: try to fetch from API if train_log is not available
        const response = await fetch('http://localhost:5000/getRewards', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            "Model_Name": session.model_name,
            "currUserID": userProfile.user_id
          })
        });

        const data = await response.json()
        if (data.status === 'ok' && Array.isArray(data.data)) {
          // Find the specific session and transform its data
          const sessionData = data.data.find((s: TrainingSession) => s.session_id === session.session_id);
          if (sessionData) {
            const transformedData = sessionData.progress.map((point) => ({
              ...point,
              session_id: sessionData.session_id,
              session_start: new Date(sessionData.started_at).toLocaleDateString(),
              session_duration: sessionData.total_time.toFixed(2)
            }));
            setSessionDetails(transformedData);
          } else {
            setSessionDetails([]);
          }
        } else {
          setSessionDetails([]);
        }
      }
    } catch (error) {
      console.error('Error parsing session details:', error);
      setSessionDetails([]);
    } finally {
      setSessionDetailsLoading(false);
    }
  }

  const colors = ['#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6', '#EC4899', '#06B6D4', '#84CC16'];

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
                      <span>Reward: {session.mean_reward ? session.mean_reward.toFixed(2) : 'N/A'}</span>
                    </div>
                  </div>

                  <button 
                    className="view-details-button" 
                    title="View Details"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleViewSessionDetails(session);
                    }}
                  >
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

      {/* Session Details Modal */}
      <Modal isOpen={sessionDetailsModalOpen} onClose={() => setSessionDetailsModalOpen(false)} className="session-details-modal">
        <div className="dialog-header">
          <span
            className="material-icons"
            style={{ color: '#10b981', fontSize: '2rem', marginRight: '8px' }}
          >
            show_chart
          </span>
          <h2>Session Training Progress</h2>
        </div>
        <div className="dialog-body session-details-modal-body">
          {sessionDetailsLoading ? (
            <div style={{ textAlign: 'center', padding: '2rem' }}>
              <span className="material-icons" style={{ fontSize: '2rem', animation: 'spin 1s linear infinite' }}>
                hourglass_empty
              </span>
              <p>Loading session data...</p>
            </div>
          ) : sessionDetails.length > 0 ? (
            <div className="session-chart-container">
              <div className="session-info-summary">
                <div className="summary-item">
                  <span className="material-icons">calendar_today</span>
                  <span>Started: {selectedSession?.started_at ? new Date(selectedSession.started_at).toLocaleDateString() : 'N/A'}</span>
                </div>
                <div className="summary-item">
                  <span className="material-icons">timer</span>
                  <span>Duration: {selectedSession?.total_time ? Number(selectedSession.total_time).toFixed(2) + 's' : 'N/A'}</span>
                </div>
                <div className="summary-item">
                  <span className="material-icons">trending_up</span>
                  <span>Steps: {selectedSession?.timesteps || 'N/A'}</span>
                </div>
              </div>
              
              <div className="chart-wrapper" style={{ height: '300px', marginTop: '1rem' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={sessionDetails} margin={{ top: 20, right: 30, left: 20, bottom: 40 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="timestep" stroke="#9CA3AF" label={{ value: 'Timesteps', position: 'insideBottom', offset: -10 }} />
                    <YAxis stroke="#9CA3AF" label={{ value: 'Mean Reward', angle: -90, position: 'insideLeft' }} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px', color: '#F9FAFB' }}
                      labelFormatter={(value) => `Timestep: ${value}`}
                      formatter={(value) => [value, 'Mean Reward']}
                    />
                    <Legend verticalAlign="top" height={36} wrapperStyle={{ paddingBottom: '10px' }} />
                    <Line 
                      type="monotone" 
                      dataKey="mean_reward" 
                      stroke={colors[0]} 
                      strokeWidth={2}
                      dot={{ fill: colors[0], strokeWidth: 2, r: 3 }}
                      activeDot={{ r: 5, stroke: colors[0], strokeWidth: 2 }}
                      name="Training Progress"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              
              <div className="chart-stats" style={{ display: 'flex', justifyContent: 'space-around', marginTop: '1rem', padding: '1rem', backgroundColor: '#1f2937', borderRadius: '8px' }}>
                <div className="stat-item">
                  <span className="stat-label">Data Points:</span>
                  <span className="stat-value">{sessionDetails.length}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Max Reward:</span>
                  <span className="stat-value">{sessionDetails.length > 0 ? Math.max(...sessionDetails.map(d => d.mean_reward || 0)).toFixed(2) : 'N/A'}</span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Min Reward:</span>
                  <span className="stat-value">{sessionDetails.length > 0 ? Math.min(...sessionDetails.map(d => d.mean_reward || 0)).toFixed(2) : 'N/A'}</span>
                </div>
              </div>
            </div>
          ) : (
            <div style={{ textAlign: 'center', color: '#94a3b8', padding: '2rem' }}>
              <span className="material-icons" style={{ fontSize: '3rem', marginBottom: '1rem' }}>
                error_outline
              </span>
              <p>No training data found for this session.</p>
            </div>
          )}
        </div>
        <div className="dialog-actions">
          <button className="dialog-cancel" onClick={() => setSessionDetailsModalOpen(false)}>
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
