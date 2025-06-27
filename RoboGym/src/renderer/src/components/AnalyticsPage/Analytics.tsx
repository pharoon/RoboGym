import React, { useEffect, useState } from 'react';
import { AnalyticsProps, Model } from '@renderer/utils/interfaces';
import { GetModels } from '@renderer/utils/FetchData';
import './Analytics.css';

const AnalyticsComponent: React.FC<AnalyticsProps> = ({ userProfile }) => {
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [plotModel, setPlotModel] = useState('');
  const [compareModel1, setCompareModel1] = useState('');
  const [compareModel2, setCompareModel2] = useState('');

  useEffect(() => {
    const fetchModels = async () => {
      const models: Model[] = await GetModels(userProfile.user_id ?? '-1')
      setAvailableModels(models.map(({name}) => name))
    }
    fetchModels()
  }, [])

  const handlePlotReward = () => {
    if (!plotModel) return;
    fetch('http://localhost:5000/getRewards', {
      method: "POST",
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        "Model_Name": plotModel,
        "currUserID": userProfile.user_id
      })
    })
  };

  const handleCompareModels = () => {
    if (!compareModel1 || !compareModel2 || compareModel1 === compareModel2) return;
    fetch('http://localhost:5000/compareModels', {
      method: "POST",
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        "first_model": compareModel1,
        "second_model": compareModel2,
      })
    })
  };

  return (
    <div className="analytics-container">
      <div className="analytics-grid">
        <div className="analytics-card glass">
          <div className="card-header">
            <span className="material-icons">insights</span>
            <h2>Plot Model Rewards</h2>
          </div>
          
          <div className="form-group">
            <label>Select Model</label>
            <select
              value={plotModel}
              onChange={(e) => setPlotModel(e.target.value)}
              className="select-input"
            >
              <option value="" disabled>Choose a model</option>
              {availableModels.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
          </div>

          <button
            className="action-button"
            disabled={!plotModel}
            onClick={handlePlotReward}
          >
            <span className="material-icons">show_chart</span>
            Show Rewards
          </button>
        </div>

        <div className="analytics-card glass">
          <div className="card-header">
            <span className="material-icons">compare</span>
            <h2>Compare Models</h2>
          </div>

          <div className="form-group">
            <label>First Model</label>
            <select
              value={compareModel1}
              onChange={(e) => setCompareModel1(e.target.value)}
              className="select-input"
            >
              <option value="" disabled>Choose first model</option>
              {availableModels.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Second Model</label>
            <select
              value={compareModel2}
              onChange={(e) => setCompareModel2(e.target.value)}
              className="select-input"
            >
              <option value="" disabled>Choose second model</option>
              {availableModels.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
          </div>

          <button
            className="action-button"
            disabled={!compareModel1 || !compareModel2 || compareModel2 === compareModel1}
            onClick={handleCompareModels}
          >
            <span className="material-icons">compare_arrows</span>
            Compare Models
          </button>
        </div>
      </div>
    </div>
  );
};

export default AnalyticsComponent;
