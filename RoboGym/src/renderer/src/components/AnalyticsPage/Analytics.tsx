import React, { useEffect, useState } from 'react';
import { AnalyticsProps, Model } from '@renderer/utils/interfaces';
import { GetModels } from '@renderer/utils/FetchData';
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Legend } from 'recharts';
import './Analytics.css';

interface TrainingSession {
  completed_at: string;
  final_timesteps: number;
  is_completed: boolean;
  model_id: number;
  model_name: string;
  progress: Array<{
    mean_reward: number;
    timestep: number;
  }>;
  session_id: number;
  started_at: string;
  total_time: number;
  user_id: number;
}

interface ChartData {
  timestep: number;
  mean_reward: number;
  session_id: number;
  session_start: string;
  session_duration: string;
  model_name?: string;
}

// Utility functions
const transformSessionData = (sessions: TrainingSession[], modelName?: string): ChartData[] => {
  return sessions.flatMap((session) => 
    session.progress.map((point) => ({
      ...point,
      model_name: modelName,
      session_id: session.session_id,
      session_start: new Date(session.started_at).toLocaleDateString(),
      session_duration: session.total_time.toFixed(2)
    }))
  );
};

const fetchModelRewards = async (modelName: string, userId: string): Promise<ChartData[]> => {
  const response = await fetch('http://localhost:5000/getRewards', {
    method: "POST",
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ "Model_Name": modelName, "currUserID": userId })
  });

  const data = await response.json();
  if (data.status === 'ok' && Array.isArray(data.data)) {
    return transformSessionData(data.data);
  }
  throw new Error(data.message || 'Failed to fetch rewards');
};

const fetchComparisonData = async (model1: string, model2: string, userId: string): Promise<ChartData[]> => {
  const response = await fetch('http://localhost:5000/compareModels', {
    method: "POST",
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ "first_model": model1, "second_model": model2, "currUserID": userId })
  });

  const data = await response.json();
  if (data.status === 'ok' && data.data) {
    const firstModelData = transformSessionData(data.data.first_model, model1);
    const secondModelData = transformSessionData(data.data.second_model, model2);
    return [...firstModelData, ...secondModelData];
  }
  throw new Error(data.message || 'Failed to fetch comparison data');
};

// Chart components
const RewardChart: React.FC<{
  data: ChartData[];
  title: string;
  colors: string[];
}> = ({ data, title, colors }) => {
  const uniqueSessions = [...new Set(data.map(d => d.model_name ? `${d.model_name}_${d.session_id}` : String(d.session_id)))];
  
  return (
    <div className="analytics-card glass chart-container">
      <div className="card-header">
        <span className="material-icons">show_chart</span>
        <h2>{title}</h2>
      </div>
      
      <div className="chart-wrapper">
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 40 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="timestep" stroke="#9CA3AF" label={{ value: 'Timesteps', position: 'insideBottom', offset: -10 }} />
            <YAxis stroke="#9CA3AF" label={{ value: 'Mean Reward', angle: -90, position: 'insideLeft' }} />
            <Tooltip 
              contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px', color: '#F9FAFB' }}
              labelFormatter={(value) => `Timestep: ${value}`}
              formatter={(value, name, props) => [
                value,
                props.payload.model_name 
                  ? `${props.payload.model_name} - Session ${props.payload.session_id}`
                  : `Session ${props.payload.session_id} (${props.payload.session_start})`
              ]}
            />
            <Legend verticalAlign="top" height={36} wrapperStyle={{ paddingBottom: '10px' }} />
            {uniqueSessions.map((session, index) => {
              const sessionData = data.filter(d => 
                d.model_name 
                  ? `${d.model_name}_${d.session_id}` === session
                  : String(d.session_id) === session
              );
              return (
                <Line 
                  key={session}
                  type="monotone" 
                  dataKey="mean_reward" 
                  data={sessionData}
                  stroke={colors[index % colors.length]} 
                  strokeWidth={2}
                  dot={{ fill: colors[index % colors.length], strokeWidth: 2, r: 3 }}
                  activeDot={{ r: 5, stroke: colors[index % colors.length], strokeWidth: 2 }}
                  name={session}
                />
              );
            })}
          </LineChart>
        </ResponsiveContainer>
      </div>
      
      <div className="chart-stats">
        <div className="stat-item">
          <span className="stat-label">Total Data Points:</span>
          <span className="stat-value">{data.length}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Training Sessions:</span>
          <span className="stat-value">{uniqueSessions.length}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Max Reward:</span>
          <span className="stat-value">{Math.max(...data.map(d => d.mean_reward)).toFixed(2)}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Min Reward:</span>
          <span className="stat-value">{Math.min(...data.map(d => d.mean_reward)).toFixed(2)}</span>
        </div>
      </div>
    </div>
  );
};

const AnalyticsComponent: React.FC<AnalyticsProps> = ({ userProfile }) => {
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [plotModel, setPlotModel] = useState('');
  const [compareModel1, setCompareModel1] = useState('');
  const [compareModel2, setCompareModel2] = useState('');
  const [rewardData, setRewardData] = useState<ChartData[]>([]);
  const [comparisonData, setComparisonData] = useState<ChartData[]>([]);
  const [loading, setLoading] = useState(false);
  const [compareLoading, setCompareLoading] = useState(false);
  const [showChart, setShowChart] = useState(false);
  const [showComparison, setShowComparison] = useState(false);

  const colors = ['#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6', '#EC4899', '#06B6D4', '#84CC16'];

  useEffect(() => {
    const fetchModels = async () => {
      const models: Model[] = await GetModels(userProfile.user_id ?? '-1');
      setAvailableModels(models.map(({name}) => name));
    };
    fetchModels();
  }, []);

  const handlePlotReward = async () => {
    if (!plotModel) return;
    
    setLoading(true);
    setShowChart(false);
    
    try {
      const data = await fetchModelRewards(plotModel, userProfile.user_id ?? '-1');
      setRewardData(data);
      setShowChart(true);
    } catch (error) {
      console.error('Error fetching rewards:', error);
      setRewardData([]);
      setShowChart(true);
    } finally {
      setLoading(false);
    }
  };

  const handleCompareModels = async () => {
    if (!compareModel1 || !compareModel2 || compareModel1 === compareModel2) return;
    
    setCompareLoading(true);
    setShowComparison(false);
    
    try {
      const data = await fetchComparisonData(compareModel1, compareModel2, userProfile.user_id ?? '-1');
      setComparisonData(data);
      setShowComparison(true);
    } catch (error) {
      console.error('Error comparing models:', error);
      setComparisonData([]);
      setShowComparison(true);
    } finally {
      setCompareLoading(false);
    }
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
            <select value={plotModel} onChange={(e) => setPlotModel(e.target.value)} className="select-input">
              <option value="" disabled>Choose a model</option>
              {availableModels.map((model) => <option key={model} value={model}>{model}</option>)}
            </select>
          </div>

          <button className="action-button" disabled={!plotModel || loading} onClick={handlePlotReward}>
            <span className="material-icons">{loading ? 'hourglass_empty' : 'show_chart'}</span>
            {loading ? 'Loading...' : 'Show Rewards'}
          </button>
        </div>

        <div className="analytics-card glass">
          <div className="card-header">
            <span className="material-icons">compare</span>
            <h2>Compare Models</h2>
          </div>

          <div className="form-group">
            <label>First Model</label>
            <select value={compareModel1} onChange={(e) => setCompareModel1(e.target.value)} className="select-input">
              <option value="" disabled>Choose first model</option>
              {availableModels.map((model) => <option key={model} value={model}>{model}</option>)}
            </select>
          </div>

          <div className="form-group">
            <label>Second Model</label>
            <select value={compareModel2} onChange={(e) => setCompareModel2(e.target.value)} className="select-input">
              <option value="" disabled>Choose second model</option>
              {availableModels.map((model) => <option key={model} value={model}>{model}</option>)}
            </select>
          </div>

          <button className="action-button" disabled={!compareModel1 || !compareModel2 || compareModel2 === compareModel1 || compareLoading} onClick={handleCompareModels}>
            <span className="material-icons">{compareLoading ? 'hourglass_empty' : 'compare_arrows'}</span>
            {compareLoading ? 'Loading...' : 'Compare Models'}
          </button>
        </div>
      </div>

      {showChart && rewardData.length > 0 && <RewardChart data={rewardData} title={`Training Rewards: ${plotModel}`} colors={colors} />}
      {showChart && rewardData.length === 0 && (
        <div className="analytics-card glass">
          <div className="card-header">
            <span className="material-icons">info</span>
            <h2>No Data Available</h2>
          </div>
          <p>No training reward data found for the selected model.</p>
        </div>
      )}

      {showComparison && comparisonData.length > 0 && <RewardChart data={comparisonData} title={`Model Comparison: ${compareModel1} vs ${compareModel2}`} colors={colors} />}
      {showComparison && comparisonData.length === 0 && (
        <div className="analytics-card glass">
          <div className="card-header">
            <span className="material-icons">info</span>
            <h2>No Comparison Data Available</h2>
          </div>
          <p>No training data found for one or both selected models.</p>
        </div>
      )}
    </div>
  );
};

export default AnalyticsComponent;
