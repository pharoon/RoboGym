import { useState } from 'react'
import './HomePage.css'
import RoboGymLogo from './RoboticARm.png'
import { useNavigate } from 'react-router-dom'
import DeleteModel from '../Modals/DeleteModel'
const HomePage = () => {
  const navigate = useNavigate()
  const [showDeleteModel, setShowDeleteModel] = useState<boolean>(false)
  
  return (
    <div className="HomePage">
      <div className="HomrPage_Header">
        <img src={RoboGymLogo} className="RoboGymLogo" />
        <h2>RoboGym</h2>
      </div>
      <p className="WelcomeText">
        Welcome,
        <br />
        This Platfrom enables you to train, validate, and test your reinforcment learning modules
        using a realistic simulationof a robotic arm
      </p>

      <div className="HomePage_Buttons">
        <button onClick={()=>{ navigate("/Train")}}>
          Train Model
        </button>
        <button onClick={()=>{ navigate("/Test")}}>
          Test a model
        </button>
         <button onClick={()=>{ setShowDeleteModel(true)}}>
          Delete Model
        </button>

        <button onClick={() => {navigate("/AllModels")}}>List All Models</button>
      </div>
      <DeleteModel showDeleteModal={showDeleteModel} setShowdeleteModal={setShowDeleteModel}/>
    </div>
  )
}

export default HomePage
