import React, { SetStateAction, useEffect, useState } from 'react'
import './DeleteModel.css'
import { Modal } from '@mui/material'
interface DeleteModelProps {
  showDeleteModal: boolean
  setShowdeleteModal: React.Dispatch<SetStateAction<boolean>>
}
const DeleteModel: React.FC<DeleteModelProps> = (props) => {
  const { showDeleteModal, setShowdeleteModal } = props
  const [models, setModels] = useState<{ name: string }[]>([])
  const [selectedModel, setSelectedModel] = useState<string>()

  useEffect(() => {
    fetch('http://localhost:5000/models')
      .then((res) => res.json())
      .then((data) => {
        setModels(data)
      })
  }, [])

  const deleteModel = () => {
    if (selectedModel) {
      fetch('http://localhost:5000/delete', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ model_name: selectedModel })
      })
      setShowdeleteModal(false)
    }
  }
  if (!showDeleteModal) return null

  return (
    <Modal
      open={showDeleteModal}
      onClose={() => {
        setShowdeleteModal(false)
      }}
    >
      <div className="centered-modal">
        <h4> Select a model to delete</h4>

        <select
          className="Select"
          onChange={(e) => {
            setSelectedModel(e.target.value)
          }}
        >
          <option>Choose Model</option>
          {models.map(({ name }) => (
            <option value={name}>{name}</option>
          ))}
        </select>
        <div className="deleteButtonWrapper">
          <button className="DeleteModel__Btn" onClick={deleteModel}>Delete</button>
        </div>
      </div>
    </Modal>
  )
}

export default DeleteModel
