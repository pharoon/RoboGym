interface LoadingScreenProps {
  loading: boolean
  text: string
}

const LoadingScreen: React.FC<LoadingScreenProps> = ({ loading, text }) => {
  if (!loading) return null
  return (
    <div className="loading-overlay">
      <div className="spinner"></div>
      <p>{text}</p>
    </div>
  )
}

export default LoadingScreen
