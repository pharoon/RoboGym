import { useEffect, useState } from 'react'
import { toast, ToastContainer } from 'react-toastify'
import { AllDataProps, Model } from '@renderer/utils/interfaces'
import { GetModels } from '@renderer/utils/FetchData'
import './AllModel.css'

const AllModel: React.FC<AllDataProps> = ({ userProfile }) => {
  const [models, setModels] = useState<Model[]>([])
  
  const fetchModels = async () => {
    const models: Model[] = await GetModels(userProfile.user_id ?? '-1')
    setModels(models)
  }

  useEffect(() => {
    fetchModels()
  }, [])

  const handleDelete = (modelName: string) => {
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
          <div className="empty-state">
            <span className="material-icons">science</span>
            <h2>No Models Found</h2>
            <p>You haven't trained or uploaded any models yet.</p>
          </div>
        ) : (
          models.map((model, index) => (
            <div key={index} className="model-card glass">
              <div className="model-header">
                <span className="material-icons">smart_toy</span>
                <h3>{model.name}</h3>
              </div>
              
              <div className="model-content">
                <div className="info-row">
                  <span className="material-icons">calendar_today</span>
                  <p>Created: {model.created_at}</p>
                </div>
                
                <div className="info-row">
                  <span className="material-icons">psychology</span>
                  <p>Algorithm: {model.algorithm}</p>
                </div>
                
                <div className="info-row">
                  <span className="material-icons">precision_manufacturing</span>
                  <p>Robotic Arm: {model.robotic_arm}</p>
                </div>
                
                <div className="info-row">
                  <span className="material-icons">folder</span>
                  <p>Path: {model.model_path}</p>
                </div>
              </div>

              <button 
                className="delete-button"
                onClick={() => handleDelete(model.name)}
              >
                <span className="material-icons">delete</span>
                Delete Model
              </button>
            </div>
          ))
        )}
      </div>

      <ToastContainer position="bottom-left" />
    </div>
  )
}

export default AllModel
