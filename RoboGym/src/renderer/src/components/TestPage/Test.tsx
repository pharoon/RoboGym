import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

const TestComponent = () => {
  const [models, setModels] = useState<{ name: string }[]>([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [selectedTask, setSelectedTask] = useState('1'); // default to 1 (Pick and Place)
  const [episodes, setEpisodes] = useState(10);
  const [logs, setLogs] = useState('');
  const [isTesting, setIsTesting] = useState(false);
  const navigate = useNavigate()

  useEffect(() => {
    fetch('http://localhost:5000/models')
      .then(res => res.json())
      .then(data => {
        setModels(data);
        if (data.length > 0) setSelectedModel(data[0].name);
      });
  }, []);

 const handleTest = () => {
  if (!selectedModel) return alert('Please select a model.');
  setLogs('');
  setIsTesting(true);

  const eventSource = new EventSource(`http://localhost:5000/test?model=${selectedModel}&task=${selectedTask}&episodes=${episodes}`);

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
    <div style={{ padding: '1rem', backgroundColor: '#1d1d2e', color: 'white', fontFamily: 'monospace', height:"100vh", width:"100vw" }}>
      <h2 style={{textAlign:"center"}}> Test Trained Model</h2>

      <div style={{ marginBottom: '1rem' }}>
        <label>Model: </label>
        <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)}>
          {models.map((model) => (
            <option key={model.name} value={model.name}>{model.name}</option>
          ))}
        </select>
      </div>

      <div style={{ marginBottom: '1rem' }}>
        <label>Task: </label>
        <select value={selectedTask} onChange={e => setSelectedTask(e.target.value)}>
          <option value="1">Pick And Place</option>
          {/* Add more tasks like <option value="2">Button Pressing</option> if needed */}
        </select>
      </div>

      <div style={{ marginBottom: '1rem' }}>
        <label>Episodes: </label>
        <input
          type="number"
          min={1}
          value={episodes}
          onChange={e => setEpisodes(Number(e.target.value))}
        />
      </div>

      <button onClick={handleTest} disabled={isTesting} style={{ padding: '0.5rem 1rem' }}>
        {isTesting ? 'Testing...' : 'Start Test'}
      </button>

      <div style={{
        marginTop: '1rem',
        backgroundColor: 'white',
        color: 'black',
        padding: '1rem',
        height: '300px',
        overflowY: 'scroll',
        borderRadius: '8px',
        fontFamily: 'monospace',
        whiteSpace: 'pre-wrap',
        scrollbarColor: '#1d1d2e white'
      }}>
        {logs}
      </div>
      <div className="ReturnBack" style={{display:"flex", justifyContent:"center", marginTop:20}}>
        <button onClick={()=>{ navigate("/")}}>Return Back</button>
      </div>
    </div>
  );
};

export default TestComponent;
