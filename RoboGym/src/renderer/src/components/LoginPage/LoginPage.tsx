import React, { SetStateAction, useEffect, useState } from 'react'
import { FaUser, FaLock, FaEnvelope, FaRobot, FaIcons } from 'react-icons/fa'
import './LoginPage.css'
import { useNavigate } from 'react-router-dom'
import { toast, ToastContainer } from 'react-toastify'
import LoadingScreen from '@renderer/utils/LoadingScreen'

interface LoginProps{
  setUserProfile:React.Dispatch<SetStateAction<{}>>
  userProfile:{}
}

const Login:React.FC<LoginProps> = ( props) => {

  const { setUserProfile } = props

  const [view, setView] = useState('login')
  const [loading, setLoading] = useState(false)

  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: ''
  })
  const navigate = useNavigate()

  const [errors, setErrors] = useState({})

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>  ) => {
    setFormData({ ...formData, [e.target.name]: e.target.value })
    if (e.target.name === 'email') {
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
      setErrors({ ...errors, email: emailRegex.test(e.target.value) ? null : 'Invalid email' })
    }
    if (e.target.name === 'password') {
      setErrors({ ...errors, password: e.target.value.length >= 6 ? null : 'Password too short' })
    }
  }

  const register = async () => {
    setLoading(true)
    fetch('http://localhost:5000/register', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ ...formData })
    })
      .then( async (response) => {
        const status = response.status
        console.log("status is now ", status)
        if(status === 201){
          toast.success("Account has been created Successfully")
          setLoading(false)
          setView("login")
        }
        else{
          const result : any = await response.json()
          console.log("Result is now ", result)
          toast.error(`Error: ${result.message}`)
          setLoading(false)
        }
      })
      .catch((error) => console.log('Error occured while trying to sign user ', error))
  }

  const logUser = async () => {
    setLoading(true)
    fetch('http://localhost:5000/login', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ username: formData.username, password: formData.password })
    })
      .then(async (response) => {
        const result:any = await response.json()
        console.log("Result is now ",result)

        if(response.status === 401 || response.status === 400){
          toast.error(`Error: ${result.message}`)    
        }
        else{
          setUserProfile(result)
          navigate("/HomePage")
        }
        setLoading(false)
      })
      .catch((error) => {
        console.log('Error occured while trying to log user ', error)

      })
  }

  const renderLogin = () => (
    <div className="form-container">
      <h2 style={{ textAlign: 'center', marginBottom: 20 }}> Login</h2>
      <div className="input-group">
        <FaUser />
        <input
          type="text"
          name="username"
          placeholder="Username"
          value={formData.username}
          onChange={handleChange}
        />
      </div>
      <div className="input-group">
        <FaLock />
        <input
          type="password"
          name="password"
          placeholder="Password"
          value={formData.password}
          onChange={handleChange}
        />
      </div>
      <button className="primary-btn" onClick={logUser}>
        Login
      </button>
      <p onClick={() => setView('signup')} className="switch-link">
        Don't have an account? Sign up
      </p>
    </div>
  )

  const renderSignup = () => (
    <div className="form-container">
      <h2 style={{ textAlign: 'center', marginBottom:20 }}> Sign Up</h2>

      <div className="input-group">
        <FaUser />
        <input
          type="text"
          name="username"
          placeholder="Username"
          value={formData.username}
          onChange={handleChange}
        />
      </div>
      <div className="input-group">
        <FaEnvelope />
        <input
          type="email"
          name="email"
          placeholder="Email"
          value={formData.email}
          onChange={handleChange}
        />
      </div>
      <div className="input-group">
        <FaLock />
        <input
          type="password"
          name="password"
          placeholder="Password"
          value={formData.password}
          onChange={handleChange}
        />
      </div>
      <button className="primary-btn" onClick={register}>
        Signup
      </button>
      <p onClick={() => setView('login')} className="switch-link">
        Already have an account? Login
      </p>
    </div>
  )

  return (
    <div className="app-bg">
      <ToastContainer position='bottom-left'/>
        <div className="welcome-box">
          <h1>Welcome to RoboGym</h1>
          <p>Train your robotic arm with the power of Reinforcement Learning</p>
        </div>

      <div className="cardWrapper">
        <div className="card fade-in no-select">
          {view === 'login' ? renderLogin() : renderSignup()}
        </div>
      </div>
      <LoadingScreen loading={loading} text={view === 'login' ? "Logging in..." : "Signing up..."} />
    </div>
  )
}

export default Login