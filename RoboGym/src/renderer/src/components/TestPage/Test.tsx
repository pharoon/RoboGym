import { GetModels } from '@renderer/utils/FetchData';
import { Model, TestPageProps } from '@renderer/utils/interfaces';
import React, { useEffect, useState } from 'react';
import "./Test.css"

const TestComponent: React.FC<TestPageProps> = ({ userProfile }) => {
  const [models, setModels] = useState<{ name: string }[]>([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [selectedTask, setSelectedTask] = useState('1');
  const [episodes, setEpisodes] = useState(10);
  const [logs, setLogs] = useState('');
  const [isTesting, setIsTesting] = useState(false);
  
  useEffect(() => {
    const fetchModels = async () => {
      const models: Model[] = await GetModels(userProfile.user_id ?? "-1");
      setModels(models.map(({ name }) => ({ name })));
    };

    fetchModels();
  }, []);

  const handleTest = () => {
    if (!selectedModel) return;
    setLogs('');
    setIsTesting(true);

    const eventSource = new EventSource(
      `http://localhost:5000/test?model=${selectedModel}&task=${selectedTask}&episodes=${episodes}&currUserID=${userProfile.user_id}`
    );

    eventSource.onmessage = (event) => {
      setLogs(prev => prev + event.data + '\n');
    };

    eventSource.addEventListener('end', () => {
      eventSource.close();
      setIsTesting(false);
    });

    eventSource.onerror = (err) => {
      console.error('EventSource failed:', err);
      setLogs(prev => prev + '\n🚨 Test connection error.');
      eventSource.close();
      setIsTesting(false);
    };
  };

  return (
    <div className="test-container">
      <div className="test-card glass">
        <div className="card-header">
          <span className="material-icons">science</span>
          <h2>Test Trained Model</h2>
        </div>

        <div className="form-content">
          <div className="form-group">
            <label>Model</label>
            <select 
              className="select-input"
              value={selectedModel} 
              onChange={e => setSelectedModel(e.target.value)}
            >
              <option value="" disabled>Select Model</option>
              {models.map((model) => (
                <option key={model.name} value={model.name}>{model.name}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Task</label>
            <select 
              className="select-input"
              value={selectedTask} 
              onChange={e => setSelectedTask(e.target.value)}
            >
              <option value="1">Pick And Place</option>
            </select>
          </div>

          <div className="form-group">
            <label>Episodes</label>
            <input
              className="number-input"
              type="number"
              min={1}
              value={episodes}
              onChange={e => setEpisodes(Number(e.target.value))}
            />
          </div>

          <button 
            className={`action-button ${isTesting ? 'loading' : ''}`}
            onClick={handleTest} 
            disabled={isTesting || !selectedModel}
          >
            <span className="material-icons">
              {isTesting ? 'hourglass_empty' : 'play_arrow'}
            </span>
            {isTesting ? 'Testing...' : 'Start Test'}
          </button>
        </div>
      </div>

      <div className="logs-card glass">
        <div className="card-header">
          <span className="material-icons">terminal</span>
          <h2>Test Logs</h2>
        </div>
        
        <div className="logs-content">
          {logs ? (
            <pre>{logs}</pre>
          ) : (
            <div className="empty-logs">
              <span className="material-icons">description</span>
              <p>Test logs will appear here</p>
            </div>
          )}
        </div>
      </div>

      {isTesting && (
        <div className="loading-overlay">
          <div className="spinner"></div>
          <p>Running test...</p>
        </div>
      )}
    </div>
  );
};

export default TestComponent;
