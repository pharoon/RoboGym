import React, { useEffect, useState } from 'react'
import './Train.css'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { TrainProps } from '@renderer/utils/interfaces'
import { toast, ToastContainer } from 'react-toastify'


const Train:React.FC<TrainProps> = (props) => {
  const { userProfile } = props

  const searchParams = useSearchParams()
  const [modelName, setModelName] = useState<string | undefined>(searchParams[0].get('modelName') ?? undefined  )
  const [continueTraining,_] = useState<boolean>(searchParams[0].get('continueTraining') === 'true')
  const [modelID, __] = useState<number>(Number(searchParams[0].get('modelID') ?? 0))
  const [TimeSteps, setTimeSteps] = useState<string>()
  const [learningRate, setLearningRate] = useState<string>()
  const [n_steps, setN_Steps] = useState<Number>()
  const [batchSize, setBatchSize] = useState<Number>()
  const [Task, setTask] = useState<number>()
  const [trainingLogs, setTrainigLogs] = useState<string>()
  const [isTraining, setIsTraining] = useState<boolean>(false)

  const navigate = useNavigate()
  const startTraining = () => {
    if (!modelName || !TimeSteps || !Task || !learningRate || !batchSize || !n_steps) {
      toast.warn("Please make sure all fields are entered before training")
      return
    }
    setIsTraining(true)
    const eventSource = new EventSource(
      `http://localhost:5000/${continueTraining ? 'continue_train' : 'train'}?model_name=${modelName}&timesteps=${TimeSteps}&task_number=${Task}&curr_user_id=${userProfile.user_id}&model_id=${modelID}&learning_rate=${Number(learningRate)}&n_steps=${n_steps}&batch_size=${batchSize}`
    )

    eventSource.onmessage = (event) => {
      console.log(event.data)
      setTrainigLogs((prev) => prev + event.data + '\n')
    }

    eventSource.addEventListener('end', () => {
      eventSource.close()
      setIsTraining(false)
    })

    eventSource.onerror = (err) => {
    if (isTraining) {
      console.error('Training error:', err)
      setTrainigLogs((prev) => prev + '\nTraining connection error.')
      setIsTraining(false)
    }
    eventSource.close()
  }
  }
  return (
    <div className="RoboGym-Train">
      <ToastContainer position='bottom-left'/>
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
          <label htmlFor="ModelName">Learning Rate</label>
          <input
            id="ModelName"
            type="text"
            value={learningRate === undefined ? "" : String(learningRate)}
            onChange={(e) => {
              setLearningRate((e.target.value))
            }}
          />
        </div>
        <div className="Field">
          <label htmlFor="ModelName">Batch Size</label>
          <input
            id="ModelName"
            type="text"
            value={batchSize === undefined ? "" : String(batchSize)}
            onChange={(e) => {
              setBatchSize(Number(e.target.value))
            }}
          />
        </div>
        <div className="Field">
          <label htmlFor="ModelName">n_steps</label>
          <input
            id="ModelName"
            type="text"
            value={n_steps === undefined ? "" : String(n_steps)}
            onChange={(e) => {
              setN_Steps(Number(e.target.value))
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
              navigate('/HomePage')
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
