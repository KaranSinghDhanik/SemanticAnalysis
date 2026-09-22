import { useState } from 'react'

const EMOTION_EMOJIS = {
  joy: '😊',
  sadness: '😢',
  anger: '😡',
  love: '❤️',
  fear: '😨',
  surprise: '😲'
}

export default function App() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handlePredict = async (e) => {
    e.preventDefault()

    if (!text.trim()) {
      setError('Please enter a sentence to classify.')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text })
      })

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}))
        throw new Error(errData.error || `Server responded with status ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message || 'Failed to communicate with the emotion backend.')
    } finally {
      setLoading(false)
    }
  }

  const emotion = result?.emotion?.toLowerCase() || ''
  const emoji = EMOTION_EMOJIS[emotion] || '✨'
  const confidencePct = result ? Math.round(result.confidence * 100) : 0

  return (
    <div className="container">
      <header className="header">
        <h1 className="title">Emotion Classifier</h1>
        <p className="subtitle">
          Enter a sentence and let the ML model predict the emotion.
        </p>
      </header>

      <form onSubmit={handlePredict} className="form-group">
        <textarea
          className="textarea"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Write your text here..."
          rows={4}
        />

        <button
          type="submit"
          className="submit-btn"
          disabled={loading || !text.trim()}
        >
          {loading ? (
            <>
              <div className="spinner" />
              <span>Analyzing...</span>
            </>
          ) : (
            <span>Predict Emotion</span>
          )}
        </button>
      </form>

      {error && (
        <div className="error-banner">
          ⚠️ {error}
        </div>
      )}

      {result && (
        <div className="result-card">
          <div className="result-header">Prediction Result</div>

          <div className="emotion-display">
            <span className="emotion-emoji" role="img" aria-label={emotion}>
              {emoji}
            </span>
            <span className="emotion-label">{result.emotion}</span>
          </div>

          <div className="confidence-section">
            <div className="confidence-header">
              <span>Confidence</span>
              <span className="confidence-percent">{confidencePct}%</span>
            </div>
            <div className="progress-bar-bg">
              <div
                className="progress-bar-fill"
                style={{ width: `${confidencePct}%` }}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
