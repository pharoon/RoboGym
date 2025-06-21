import React, { SetStateAction, useCallback } from 'react';
import { Modal, Typography, Box, Button } from '@mui/material';
import { useDropzone } from 'react-dropzone';
import './UploadModel.css';

interface UploadModelProps {
  showUploadModal: boolean;
  setShowUploadModal: React.Dispatch<SetStateAction<boolean>>;
}

const UploadModel: React.FC<UploadModelProps> = ({ showUploadModal, setShowUploadModal }) => {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const zipFile = acceptedFiles[0];

    if (zipFile && zipFile.type === 'application/zip') {
      console.log('Uploading:', zipFile);
      const formData = new FormData();
      formData.append('file', zipFile);

      fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
      })
        .then((res) => res.json())
        .then((data) => {
          console.log('Upload response:', data);
          setShowUploadModal(false);
        })
        .catch((err) => {
          console.error('Upload error:', err);
        });
    } else {
      alert('Please upload a .zip file only');
    }
  }, [setShowUploadModal]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/zip': ['.zip'] },
    multiple: false,
  });

  return (
    <Modal open={showUploadModal} onClose={() => setShowUploadModal(false)}>
      <Box className="centered-modal">
        <Typography variant="h6" sx={{color:"black"}} gutterBottom>
          📁 Upload a .zip Model File
        </Typography>

        <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
          <input {...getInputProps()} />
          {isDragActive ? (
            <p>Drop the ZIP file here...</p>
          ) : (
            <p>Drag & drop a .zip file here, or click to select one</p>
          )}
        </div>

        <div className="uploadButtonWrapper">
          <Button onClick={() => setShowUploadModal(false)} variant="outlined" color="error">
            Cancel
          </Button>
        </div>
      </Box>
    </Modal>
  );
};

export default UploadModel;
