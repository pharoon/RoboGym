import React, { SetStateAction, useEffect, useState } from 'react'
import './DeleteModel.css'
import { Modal } from '@mui/material'
import { DeleteModelProps, Model } from '@renderer/utils/interfaces'
import { GetModels } from '@renderer/utils/FetchData'
import { toast } from 'react-toastify'

const DeleteModel: React.FC<DeleteModelProps> = (props) => {
  const { showDeleteModal, setShowdeleteModal, userProfile } = props
  const [models, setModels] = useState<{ name: string }[]>([])
  const [selectedModel, setSelectedModel] = useState<string>()

  useEffect(() => {
      const fetchModels = async () => {
        const models: Model[] = await GetModels(userProfile.user_id ?? "-1");
        setModels(models.map(({ name }) => ({ name })));
      };
    
      fetchModels();
  }, [])

  const deleteModel = () => {
    if (selectedModel) {
      fetch('http://localhost:5000/delete', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ model_name: selectedModel, currUserID:userProfile.user_id })
      }).then(()=>{
        toast.success("Model has been deleted successfully")
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
          <option hidden>Choose Model</option>
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
