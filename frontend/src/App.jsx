import React, { useState } from 'react';
import ImageUpload from './components/ImageUpload';
import ResultDisplay from './components/ResultDisplay';

function App() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleUpload = async (file) => {
    setImage(file);
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to analyze image');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setError('Something went wrong. Please try again.');
      setImage(null);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setImage(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="container">
      <header style={{ marginBottom: '3rem', textAlign: 'center', paddingTop: '2rem' }}>
        <h1 className="title">AI Calorie Estimator</h1>
        <p className="subtitle">Powered by SAM2 & EfficientNet v2</p>
      </header>

      <main>
        {error && (
          <div style={{
            background: 'rgba(239, 68, 68, 0.1)',
            color: 'var(--error)',
            padding: '1rem',
            borderRadius: '0.5rem',
            marginBottom: '2rem',
            textAlign: 'center'
          }}>
            {error}
          </div>
        )}

        {!image ? (
          <ImageUpload onUpload={handleUpload} isLoading={loading} />
        ) : (
          loading ? (
            <div className="card" style={{ textAlign: 'center', padding: '4rem' }}>
              <div style={{
                width: '40px',
                height: '40px',
                border: '3px solid var(--accent)',
                borderTopColor: 'transparent',
                borderRadius: '50%',
                margin: '0 auto 1rem',
                animation: 'spin 1s linear infinite'
              }} />
              <p>Analyzing image with SAM2...</p>
              <style>{`
                @keyframes spin { to { transform: rotate(360deg); } }
              `}</style>
            </div>
          ) : (
            <ResultDisplay image={image} result={result} onReset={handleReset} />
          )
        )}
      </main>
    </div>
  );
}

export default App;
