import React from 'react'
import './HomePage.css'
import RoboGymLogo from './RoboticARm.png'
import { useNavigate } from 'react-router-dom'
const HomePage = () => {
  const navigate = useNavigate()
  // const initalizeServer = ()=>{
  //   fetch('http://localhost:5000/initialize', {
  //     method: 'post'
  //   }).catch((e) =>{console.warn("Failed to initalize ", e)}).then((e) => {console.log(e)})
  // }
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
         <button onClick={()=>{}}>
          initalize
        </button>

        <button onClick={() => {}}>Select Existing Project</button>
      </div>
    </div>
  )
}

export default HomePage
