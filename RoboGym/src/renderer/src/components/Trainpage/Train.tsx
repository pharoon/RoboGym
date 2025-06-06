import React, { useEffect, useState } from 'react'
import './Train.css'
import { useNavigate } from 'react-router-dom'
const Train = () => {
  const [modelName, setModelName] = useState<string>()
  const [TimeSteps, setTimeSteps] = useState<string>()
  const [Task, setTask] = useState<number>()
  const [trainingLogs, setTrainigLogs] = useState<string>()

  const [isTraining, setIsTraining] = useState<boolean>(false)
  const navigate = useNavigate()
  const startTraining = () => {
    if (!modelName || !TimeSteps || !Task) return
    const eventSource = new EventSource(
      `http://localhost:5000/train?model_name=${modelName}&timesteps=${TimeSteps}&task_number=${Task}`
    )

    eventSource.onmessage = (event) => {
      setTrainigLogs((prev) => prev + event.data + '\n')
    }

    eventSource.addEventListener('end', () => {
      eventSource.close()
      setIsTraining(false)
    })

    eventSource.onerror = (err) => {
      console.error('Training error:', err)
      setTrainigLogs((prev) => prev + '\n🚨 Training connection error.')
      eventSource.close()
      setIsTraining(false)
    }
  }
  return (
    <div className="RoboGym-Train">
      <div className="TrainForm">
        <div className="Field">
          <label htmlFor="ModelName">Model Name</label>
          <input
            id="ModelName"
            type="text"
            value={modelName}
            onChange={(e) => {
              setModelName(e.target.value)
            }}
          />
        </div>
        <div className="Field">
          <label htmlFor="ModelName">TimeSteps</label>
          <input
            id="ModelName"
            type="text"
            value={TimeSteps}
            onChange={(e) => {
              setTimeSteps(e.target.value)
            }}
          />
        </div>
        <div className="Field">
          <label htmlFor="ModelName">Pick Task</label>
          <select
            onChange={(e) => {
              setTask(Number(e.target.value))
            }}
          >
            <option hidden>Tasks</option>
            <option value={1}> Pick And Place</option>
          </select>
        </div>
      </div>
      <div className="TrainResults">
        <textarea
          value={trainingLogs}
          readOnly
          placeholder="Logs of the training process appears here"
        />
        <div className="Buttons">
          <button onClick={startTraining}>StartTrainig</button>
          <button
            onClick={() => {
              navigate('/')
            }}
          >
            Return Back
          </button>
        </div>
        {isTraining && (
          <div className="spinner-container">
            <div className="spinner"></div>
            <p>Training in progress...</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default Train
