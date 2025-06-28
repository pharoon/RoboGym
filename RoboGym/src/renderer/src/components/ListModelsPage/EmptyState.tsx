import React from 'react';

interface EmptyStateProps {
  title: string;
  message: string;
}

const EmptyState: React.FC<EmptyStateProps> = ({ title, message }) => (
  <div className="empty-state">
    <span className="material-icons">science</span>
    <h2>{title}</h2>
    <p>{message}</p>
  </div>
);

export default EmptyState; 