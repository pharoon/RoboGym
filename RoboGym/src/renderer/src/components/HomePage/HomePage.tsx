import React from 'react'
import "./HomePage.css"
import RoboGymLogo from "./RoboticARm.png"
import { FileUploader } from 'react-drag-drop-files'
const HomePage = () => {
  return (
    <div className='HomePage'>
        <div className="HomrPage_Header">
            <img src={RoboGymLogo} className='RoboGymLogo'/>
            <h2>RoboGym</h2>
        </div>
        <p className='WelcomeText'>
            Welcome,<br/>
            This Platfrom enables you to train, validate, and test your reinforcment learning modules using a realistic simulationof a robotic arm
        </p>

        <div className="fileUploader">
            <FileUploader >
            </FileUploader>
        </div>

        <div className="HomePage_Buttons">
            <button onClick={()=>{}}>
                Start New Project
            </button>
        
            <button onClick={()=>{}}>
                Select Existing Project
            </button>
        </div>

    </div>
  )
}

export default HomePage