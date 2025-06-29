import React from 'react'
import { Model } from '@renderer/utils/interfaces'

interface ModelCardProps {
  model: Model
  onDownload: (modelName: string) => void
  onRename: (modelName: string) => void
  onDelete: (modelName: string) => void
  onContinueTraining: (modelName: string, modelID: number) => void
  onViewSessions: (modelID: number) => void
}

// Utility function to format seconds into appropriate time units
const formatTrainingTime = (seconds: number): string => {
  if (seconds < 60) {
    return `${Math.round(seconds)} seconds`
  } else if (seconds < 3600) {
    const minutes = Math.round(seconds / 60)
    return `${minutes} minute${minutes !== 1 ? 's' : ''}`
  } else if (seconds < 86400) {
    const hours = Math.round(seconds / 3600)
    return `${hours} hour${hours !== 1 ? 's' : ''}`
  } else {
    const days = Math.round(seconds / 86400)
    return `${days} day${days !== 1 ? 's' : ''}`
  }
}

// Utility function to format reward without decimals
const formatReward = (reward: number): string => {
  return Math.round(reward).toString()
}

const ModelCard: React.FC<ModelCardProps> = ({
  model,
  onDownload,
  onRename,
  onDelete,
  onContinueTraining,
  onViewSessions
}) => (
  <div className="model-card glass">
    <div className="model-header">
      <span className="material-icons">smart_toy</span>
      <h3>{model.name}</h3>
      <div className="model-actions">
        <span
          className="material-icons action-icon"
          title="Rename Model"
          onClick={() => onRename(model.name)}
        >
          edit
        </span>
        <span
          className="material-icons action-icon"
          title="Download Model"
          onClick={() => onDownload(model.name)}
        >
          download
        </span>
        <span
          className="material-icons delete-icon action-icon"
          title="Delete Model"
          onClick={() => onDelete(model.name)}
        >
          delete
        </span>
      </div>
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
        <span className="material-icons">timer</span>
        <p>Total Trained Time: {formatTrainingTime(model.total_training_time)}</p>
      </div>
      <div className="info-row">
        <span className="material-icons">timeline</span>
        <p>Total Time Steps: {model.total_timesteps}</p>
      </div>
      <div className="info-row">
        <span className="material-icons">star</span>
        <p>Max Reward: {formatReward(model.final_mean_reward)}</p>
      </div>
    </div>
    <div className="model-card-actions">
      <button className="continue-training-button" onClick={() => onContinueTraining(model.name, model.id ?? 0)}>
        <span className="material-icons">play_arrow</span>
        Continue Training
      </button>
      <button className="view-sessions-button" onClick={() => onViewSessions(model.id)}>
        <span className="material-icons">visibility</span>
        View Sessions
      </button>
    </div>
  </div>
)

export default ModelCard
