import React, { useCallback } from 'react';

const ImageUpload = ({ onUpload, isLoading }) => {
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      onUpload(e.target.files[0]);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onUpload(e.dataTransfer.files[0]);
    }
  };

  return (
    <div 
      className="card"
      style={{ 
        border: '2px dashed var(--text-secondary)', 
        textAlign: 'center',
        cursor: isLoading ? 'wait' : 'pointer',
        opacity: isLoading ? 0.7 : 1
      }}
      onDrop={handleDrop}
      onDragOver={(e) => e.preventDefault()}
      onClick={() => document.getElementById('file-input').click()}
    >
      <input 
        type="file" 
        id="file-input" 
        hidden 
        accept="image/*" 
        onChange={handleFileChange}
        disabled={isLoading}
      />
      <div style={{ padding: '3rem' }}>
        <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--accent)', marginBottom: '1rem' }}>
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
          <polyline points="17 8 12 3 7 8"></polyline>
          <line x1="12" y1="3" x2="12" y2="15"></line>
        </svg>
        <h3 style={{ margin: '0 0 0.5rem 0' }}>Upload Food Image</h3>
        <p style={{ color: 'var(--text-secondary)', margin: 0 }}>
          Drag & drop or click to select
        </p>
      </div>
    </div>
  );
};

export default ImageUpload;
