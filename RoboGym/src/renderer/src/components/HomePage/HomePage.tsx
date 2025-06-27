import { SetStateAction, useEffect, useState } from 'react'
import './HomePage.css'
import RoboGymLogo from '../../../../../../RoboGym/src/Assets/RogoGymLogo.png'
import { useNavigate } from 'react-router-dom'
import DeleteModel from '../Modals/DeleteModel'
import UploadModel from '../Modals/DeleteModel'
import NameInputModal from '../Modals/NameInputModal'
import { toast, ToastContainer } from 'react-toastify'
import { HomePageProps } from '@renderer/utils/interfaces'



const HomePage:React.FC<HomePageProps> = (props) => {
  const { userProfile, setUserProfile  } = props

  const navigate = useNavigate()
  const [showDeleteModel, setShowDeleteModel] = useState<boolean>(false)
  const [modalOpen, setModalOpen] = useState(false)
  const [pendingFilePath, setPendingFilePath] = useState<string | null>(null)


  // useEffect(()=>{
  //   toast.success(`Logging successfully ${userProfile.username}`)
  // }, [userProfile])

  const handleFileUpload = async () => {
    // @ts-ignore
    const filePath = await window.electronAPI.openFileDialog()
    if (filePath) {
      setPendingFilePath(filePath)
      setModalOpen(true) // show name input modal
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
          currUserID:userProfile.user_id
        })
      })
        .then((res) => res.json())
        .then(() =>toast.success("File is uploaded Successfully"))
        .catch((err) => console.error('Upload error:', err))
    }
    setModalOpen(false)
    setPendingFilePath(null)
  }

  const logOut = () => {
    setUserProfile({})
    navigate("/")
  }

  return (
    <div className="HomePage">
      <ToastContainer position='bottom-left'/>
      <div className="HomrPage_Header">
        <img src={RoboGymLogo} className="RoboGymLogo" />
        <h2>RoboGym</h2>
      </div>
      <p className="WelcomeText">
        Welcome, {userProfile.username}
        <br />
        This Platfrom enables you to train, validate, and test your reinforcment learning modules
        using a realistic simulation of a robotic arm
      </p>

      <div className="HomePage_Buttons">
        <button
          onClick={() => {
            navigate('/Train')
          }}
        >
          Train Model
        </button>
        <button
          onClick={() => {
            navigate('/Test')
          }}
        >
          Test a model
        </button>
        <button
          onClick={() => {
            setShowDeleteModel(true)
          }}
        >
          Delete Model
        </button>
        <button onClick={handleFileUpload}>Upload Exisiting Model</button>
        <button onClick={()=>{navigate("/Analytics")}}>Analytics Results</button>
        <button
          onClick={() => {
            navigate('/AllModels')
          }}
        >
          List All Models
        </button>

        <button onClick={logOut}>Log Out</button>

      </div>
      <DeleteModel userProfile={userProfile} showDeleteModal={showDeleteModel} setShowdeleteModal={setShowDeleteModel} />
      <NameInputModal
        open={modalOpen}
        onClose={() => setModalOpen(false)}
        onSubmit={handleModelNameSubmit}
      />
    </div>
  )
}

export default HomePage
