import { useEffect, useState } from 'react'
import { Card, CardContent, Typography, Grid, Box, Button } from '@mui/material'
import { useNavigate } from 'react-router-dom'
import DeleteIcon from '@mui/icons-material/Delete'
import { toast, ToastContainer } from 'react-toastify'
import { AllDataProps, Model } from '@renderer/utils/interfaces'
import { GetModels } from '@renderer/utils/FetchData'

const AllModel: React.FC<AllDataProps> = (props) => {
  const { userProfile } = props
  const [Models, setModels] = useState<Model[]>([])
  const navigate = useNavigate()
  const fetchModels = async () => {
    const models: Model[] = await GetModels(userProfile.user_id ?? '-1')
    setModels(models)
    console.log('Models are now ', models)
  }
  useEffect(() => {
    fetchModels()
  }, [])

  // Handle delete model
  const handleDelete = (modelName: string) => {
    fetch('http://localhost:5000/delete', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ model_name: modelName, currUserID:userProfile.user_id })
    })
      .then((res) => {
        if (!res.ok) throw new Error('Failed to delete model')
        return res.json()
      })
      .then(() => {
        fetchModels()
        toast.success('Model has been deleted')
      })
      .catch((err) => console.error('Error deleting model:', err))
  }

  return (
    <div style={{ background: '#1d1d2e', height: '100vh', width: '100vw', overflow: 'auto' }}>
      <Box textAlign="center" my={4}>
        <Typography variant="h4" fontWeight="bold" gutterBottom>
          📦 Available AI Models
        </Typography>
        <Typography variant="subtitle1" color="#fffff5db">
          Here’s a list of all your trained machine learning models.
        </Typography>
      </Box>

      <Grid container spacing={3} sx={{ display: 'flex', justifyContent: 'center' }}>
        {Models.length === 0 ? (
          // @ts-ignore
          <Grid item xs={12}>
            <Box
              display="flex"
              flexDirection="column"
              alignItems="center"
              justifyContent="center"
              height="60vh"
              textAlign="center"
              color="#ccc"
            >
              <Typography variant="h5" fontWeight="bold" gutterBottom>
                No Models Found
              </Typography>
              <Typography variant="body1">
                You haven't trained or uploaded any models yet.
              </Typography>
            </Box>
          </Grid>
        ) : (
          Models.map((model, index) => (
            // @ts-ignore
            <Grid item xs={12} sm={6} md={4} key={index}>
              <Card
                sx={{
                  borderRadius: 3,
                  boxShadow: 4,
                  transition: 'transform 0.2s',
                  '&:hover': {
                    transform: 'scale(1.03)'
                  }
                }}
              >
                <CardContent>
                  <Typography variant="h6" color="primary" gutterBottom>
                    {model.name}
                  </Typography>
                  <Typography color="text.secondary">📅 Created: {model.created_at}</Typography>
                  <Typography color="text.secondary">🧠 Algorithm: {model.algorithm}</Typography>
                  <Typography color="text.secondary">
                    🤖 Robotic Arm: {model.robotic_arm}
                  </Typography>
                  <Typography color="text.secondary">📁 Model Path: {model.model_path}</Typography>
                </CardContent>
                <div
                  className="DeleteBtn__Wrapper"
                  style={{ display: 'flex', justifyContent: 'end', padding: 5 }}
                >
                  <button
                    className="DeleteModel__Btn"
                    onClick={() => {
                      handleDelete(model.name)
                    }}
                  >
                    <DeleteIcon />
                  </button>
                </div>
              </Card>
            </Grid>
          ))
        )}
      </Grid>

      <Box textAlign="center" mt={5}>
        <button onClick={() => navigate('/HomePage')}>Return Back</button>
      </Box>
      <ToastContainer position="bottom-left" />
    </div>
  )
}

export default AllModel
