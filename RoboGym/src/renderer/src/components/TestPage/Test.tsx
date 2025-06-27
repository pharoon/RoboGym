import { GetModels } from '@renderer/utils/FetchData';
import { Model, TestPageProps } from '@renderer/utils/interfaces';
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import "./Test.css"

const TestComponent:React.FC<TestPageProps> = (props) => {
  const { userProfile } = props
  const [models, setModels] = useState<{ name: string }[]>([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [selectedTask, setSelectedTask] = useState('1'); // default to 1 (Pick and Place)
  const [episodes, setEpisodes] = useState(10);
  const [logs, setLogs] = useState('');
  const [isTesting, setIsTesting] = useState(false);
  const navigate = useNavigate()
  
useEffect(() => {
  const fetchModels = async () => {
    const models: Model[] = await GetModels(userProfile.user_id ?? "-1");
    setModels(models.map(({ name }) => ({ name })));
  };

  fetchModels();
}, []);

 const handleTest = () => {
  if (!selectedModel) return alert('Please select a model.');
  setLogs('');
  setIsTesting(true);

  const eventSource = new EventSource(`http://localhost:5000/test?model=${selectedModel}&task=${selectedTask}&episodes=${episodes}&currUserID=${userProfile.user_id}`);

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
        <select className='ModelTest' value={selectedModel} onChange={e => setSelectedModel(e.target.value)}>
          <option hidden>Select Model</option>
          {models.map((model) => (
            <option key={model.name} value={model.name}>{model.name}</option>
          ))}
        </select>
      </div>

      <div style={{ marginBottom: '1rem' }}>
        <label>Task: </label>
        <select className='TaskTest' value={selectedTask} onChange={e => setSelectedTask(e.target.value)}>
          <option value="1">Pick And Place</option>
        </select>
      </div>

      <div style={{ marginBottom: '1rem' }}>
        <label>Episodes: </label>
        <input
          className='EpisodeTest'
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
        overflowY: 'auto',
        borderRadius: '8px',
        fontFamily: 'monospace',
        whiteSpace: 'pre-wrap',
        scrollbarColor: '#1d1d2e white'
      }}>
        {logs}
      </div>
      <div className="ReturnBack" style={{display:"flex", justifyContent:"center", marginTop:20}}>
        <button onClick={()=>{ navigate("/HomePage")}}>Return Back</button>
      </div>
    </div>
  );
};

export default TestComponent;
