import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import './App.css';

// ⚠️ IMPORTANT: Replace this with YOUR HuggingFace Space URL
// const API_URL = 'https://huggingface.co/spaces/HimadriBiswas/emotion-caption-api/api/caption';
const API_URL = 'https://HimadriBiswas-emotion-caption-api.hf.space/api/caption';

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Handle file drop
  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedImage(file);
      setImagePreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.bmp']
    },
    multiple: false
  });

  // Generate caption
  const handleGenerateCaption = async () => {
    if (!selectedImage) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedImage);

      const response = await axios.post(API_URL, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResult(response.data);
    } catch (err) {
      console.error('Error:', err);
      setError(
        err.response?.data?.detail || 
        'Failed to generate caption. Please try again.'
      );
    } finally {
      setLoading(false);
    }
  };

  // Reset
  const handleReset = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="App">
      <div className="container">
        <div className="header">
          <h1>🎨 Emotion Caption</h1>
          <p>Generate emotion-aware captions for your images</p>
        </div>

        <div className="upload-section">
          {!imagePreview ? (
            <div
              {...getRootProps()}
              className={`dropzone ${isDragActive ? 'active' : ''}`}
            >
              <input {...getInputProps()} />
              <div className="dropzone-content">
                <div className="upload-icon">📸</div>
                <div className="dropzone-text">
                  <h3>Drop your image here</h3>
                  <p>or click to browse</p>
                </div>
              </div>
            </div>
          ) : (
            <div className="image-preview">
              <img src={imagePreview} alt="Preview" className="preview-image" />
              <div className="button-group">
                <button
                  onClick={handleGenerateCaption}
                  disabled={loading}
                  className="btn btn-primary"
                >
                  {loading ? (
                    <>
                      <span>⏳</span>
                      <span>Generating...</span>
                    </>
                  ) : (
                    <>
                      <span>✨</span>
                      <span>Generate Caption</span>
                    </>
                  )}
                </button>
                <button onClick={handleReset} className="btn btn-secondary">
                  🔄 Reset
                </button>
              </div>
            </div>
          )}
        </div>

        {loading && (
          <div className="loading-section">
            <div className="spinner"></div>
            <p className="loading-text">
              Analyzing image and generating caption...
            </p>
          </div>
        )}

        {error && (
          <div className="error-message">
            ⚠️ {error}
          </div>
        )}

        {result && (
          <div className="result-section">
            <h2>📝 Results</h2>
            
            <div className="result-item">
              <div className="result-label">Base Caption</div>
              <div className="result-value">{result.base_caption}</div>
            </div>

            <div className="result-item">
              <div className="result-label">Detected Emotion</div>
              <div className="result-value">
                <span className={`emotion-badge emotion-${result.detected_emotion}`}>
                  {result.detected_emotion}
                </span>
              </div>
            </div>

            <div className="result-item">
              <div className="result-label">Emotion-Aware Caption</div>
              <div className="result-value">{result.emotion_aware_caption}</div>
            </div>
          </div>
        )}
      </div>

      <div className="footer">
        <p>
          Made with ❤️ using{' '}
          <a
            href="https://huggingface.co/"
            target="_blank"
            rel="noopener noreferrer"
          >
            HuggingFace
          </a>
          {' '}and{' '}
          <a
            href="https://vercel.com/"
            target="_blank"
            rel="noopener noreferrer"
          >
            Vercel
          </a>
        </p>
      </div>
    </div>
  );
}

export default App;