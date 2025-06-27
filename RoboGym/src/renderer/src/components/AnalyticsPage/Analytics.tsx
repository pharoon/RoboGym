import React, { useEffect, useState } from 'react';
import {
  Box,
  Button,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  SelectChangeEvent,
  Typography,
  Paper,
  Grid
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { AnalyticsProps, Model } from '@renderer/utils/interfaces';
import { GetModels } from '@renderer/utils/FetchData';

const AnalyticsComponent: React.FC<AnalyticsProps> = ( props ) => {

  const { userProfile } = props
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [plotModel, setPlotModel] = useState('');
  const [compareModel1, setCompareModel1] = useState('');
  const [compareModel2, setCompareModel2] = useState('');
  const navigate = useNavigate(); // <-- 👈 Hook to navigate

  useEffect(() => {
      const fetchModels = async () => {
         const models: Model[] = await GetModels(userProfile.user_id ?? '-1')
         setAvailableModels(models.map(({name})=>(name)))
         console.log('Models are now ', models)
       }
      fetchModels()
    }, [])

  const handlePlotReward = () => {
    if (!plotModel) return;
    console.log('Plotting rewards for:', plotModel);
    fetch('http://localhost:5000/getRewards', {
        method:"POST",
         headers: {
          'Content-Type': 'application/json'
        },
        body:JSON.stringify({"Model_Name": plotModel, "currUserID":userProfile.user_id})
    })

  };

  const handleCompareModels = () => {
    if (!compareModel1 || !compareModel2 || compareModel1 === compareModel2) return;
    fetch('http://localhost:5000/compareModels', {
        method:"POST",
         headers: {
          'Content-Type': 'application/json'
        },
        body:JSON.stringify({
            "first_model": compareModel1 , 
            "second_model": compareModel2 , 

        })
    })
  };

  return (
    <Box
      sx={{
        height: '100vh',
        width: '100vw',
        backgroundColor: '#1d1d2e',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        padding: 4
      }}
    >
      <Grid container spacing={4} maxWidth="md">
            {/* @ts-ignore */}
        <Grid item xs={12} md={6}>
          <Paper
            elevation={5}
            sx={{
              p: 4,
              borderRadius: 3,
              backgroundColor: '#2a2a40',
              color: 'white'
            }}
          >
            <Typography variant="h6" fontWeight="bold" gutterBottom>
              📈 Plot Model Rewards
            </Typography>

            <FormControl fullWidth sx={{ mt: 2 }}>
              <InputLabel sx={{ color: '#ccc' }}>Select Model</InputLabel>
              <Select
                value={plotModel}
                onChange={(e: SelectChangeEvent) => setPlotModel(e.target.value)}
                label="Select Model"
                sx={{ color: 'white', backgroundColor: '#3a3a52' }}
              >
                {availableModels.map((model) => (
                  <MenuItem key={model} value={model}>
                    {model}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <Box mt={3}>
              <Button
                variant="contained"
                fullWidth
                disabled={!plotModel}
                onClick={handlePlotReward}
              >
                Show Rewards
              </Button>
            </Box>
          </Paper>
        </Grid>
        {/* @ts-ignore */}
        <Grid item xs={12} md={6}>
          <Paper
            elevation={5}
            sx={{
              p: 4,
              borderRadius: 3,
              backgroundColor: '#2a2633',
              color: 'white'
            }}
          >
            <Typography variant="h6" fontWeight="bold" gutterBottom>
              🔍 Compare Models
            </Typography>

            <FormControl fullWidth sx={{ mt: 2 }}>
              <InputLabel sx={{ color: '#ccc' }}>First Model</InputLabel>
              <Select
                value={compareModel1}
                onChange={(e: SelectChangeEvent) => setCompareModel1(e.target.value)}
                label="First Model"
                sx={{ color: 'white', backgroundColor: '#3a3a52' }}
              >
                {availableModels.map((model) => (
                  <MenuItem key={model} value={model}>
                    {model}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <FormControl fullWidth sx={{ mt: 2 }}>
              <InputLabel sx={{ color: '#ccc' }}>Second Model</InputLabel>
              <Select
                value={compareModel2}
                onChange={(e: SelectChangeEvent) => setCompareModel2(e.target.value)}
                label="Second Model"
                sx={{ color: 'white', backgroundColor: '#3a3a52' }}
              >
                {availableModels.map((model) => (
                  <MenuItem key={model} value={model}>
                    {model}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <Box mt={3}>
              <Button
                variant="contained"
                fullWidth
                disabled={!compareModel1 || !compareModel2 || compareModel2 === compareModel1}
                onClick={handleCompareModels}
              >
                Compare Models
              </Button>
            </Box>
          </Paper>
        </Grid>
      </Grid>

      {/* ⬅ Back Button */}
      <Box mt={5}>
        <Button variant="outlined" color="inherit" onClick={() => navigate('/HomePage')}>
          Return back
        </Button>
      </Box>
    </Box>
  );
};

export default AnalyticsComponent;
