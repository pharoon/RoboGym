import React, { useState } from 'react';
import Modal from './Modal';

interface NameInputModalProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (name: string) => void;
}

const NameInputModal: React.FC<NameInputModalProps> = ({ open, onClose, onSubmit }) => {
  const [modelName, setModelName] = useState('');

  const handleSubmit = () => {
    if (modelName.trim()) {
      onSubmit(modelName);
      setModelName('');
    }
  };

  return (
    <Modal isOpen={open} onClose={onClose}>
      <div className="dialog-header">
        <span className="material-icons" style={{ color: '#3b82f6', fontSize: '2rem', marginRight: '8px' }}>edit</span>
        <h2>Enter Model Name</h2>
      </div>
      <div className="dialog-body">
        <input
          type="text"
          value={modelName}
          onChange={e => setModelName(e.target.value)}
          placeholder="Model Name"
          className="dialog-input"
          autoFocus
        />
      </div>
      <div className="dialog-actions">
        <button className="dialog-cancel" onClick={onClose}>Cancel</button>
        <button className="dialog-confirm" onClick={handleSubmit} disabled={!modelName.trim()}>Submit</button>
      </div>
    </Modal>
  );
};

export default NameInputModal;
