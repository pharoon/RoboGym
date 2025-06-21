import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
} from '@mui/material';

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
    <Dialog open={open} onClose={onClose} sx={{padding:10}}>
      <DialogTitle>Enter Model Name</DialogTitle>
        <TextField
          autoFocus
          fullWidth
          label="Model Name"
          value={modelName}
          onChange={(e) => setModelName(e.target.value)}
        />
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button onClick={handleSubmit} variant="contained">
          Submit
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default NameInputModal;
