import React, { useState, useEffect, useRef } from 'react';
import './App.css';

// This line tells the app: "Use the Vercel variable if it exists, otherwise use Localhost"
const API_URL = process.env.REACT_APP_BACKEND_URL || "http://127.0.0.1:5000";

function App() {
  const [question, setQuestion] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isIndexing, setIsIndexing] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [files, setFiles] = useState([]);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory, isThinking]);

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setIsIndexing(true);
    const formData = new FormData();
    formData.append('file', file);
    try {
      // UPDATED: Used ${API_URL}
      const res = await fetch(`${API_URL}/upload`, { method: 'POST', body: formData });
      if (res.ok) {
        setFiles(prev => [...prev, { name: file.name, type: file.name.split('.').pop() }]);
      }
    } catch (err) { 
      console.error("Upload error", err); 
    } finally { 
      setIsIndexing(false); 
    }
  };

  const handleDelete = async (fileName) => {
    try {
      // UPDATED: Used ${API_URL}
      const res = await fetch(`${API_URL}/delete_file`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: fileName })
      });
      if (res.ok) {
        setFiles(prev => prev.filter(f => f.name !== fileName));
        setChatHistory([]); 
      }
    } catch (err) { 
      console.error("Delete error", err); 
    }
  };

  const handleAsk = async () => {
    if (!question.trim()) return;
    setChatHistory(prev => [...prev, { role: 'user', content: question }]);
    const currentQ = question;
    setQuestion('');
    setIsThinking(true);

    try {
      // UPDATED: Used ${API_URL}
      const res = await fetch(`${API_URL}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: currentQ })
      });
      const data = await res.json();
      setChatHistory(prev => [...prev, { role: 'ai', content: data.answer || data.error }]);
    } catch (err) {
      setChatHistory(prev => [...prev, { role: 'ai', content: "Connection Error. Make sure the backend is awake!" }]);
    } finally { 
      setIsThinking(false); 
    }
  };

  return (
    <div className="app-container">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="logo-container">
            <img 
              src="/askme_logo.png" 
              alt="AskMe AI Logo" 
              className="sidebar-logo-centered"
              onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'flex'; }}
            />
            <div className="logo-fallback" style={{ display: 'none' }}>A</div>
          </div>
          <h1 className="sidebar-title-centered">AskMe AI</h1>
        </div>

        <label className="upload-btn">
          <span>{isIndexing ? "Indexing..." : "+ Upload Document"}</span>
          <input type="file" onChange={handleUpload} hidden accept=".pdf,.docx,.txt" disabled={isIndexing} />
        </label>
        
        <div className="file-list">
          <h3>Knowledge Base</h3>
          {files.map((f, i) => (
            <div key={i} className="file-item">
              <span className="file-name">{f.name}</span>
              <button className="delete-file-btn" onClick={() => handleDelete(f.name)}>×</button>
            </div>
          ))}
        </div>
      </aside>

      <main className="chat-area">
        <div className="messages-container">
          {chatHistory.length === 0 && (
            <div className="welcome-text">Upload documents to start chatting.</div>
          )}
          {chatHistory.map((msg, i) => (
            <div key={i} className={`message-bubble ${msg.role}`}>{msg.content}</div>
          ))}
          {isThinking && <div className="message-bubble ai thinking">...</div>}
          <div ref={chatEndRef} />
        </div>

        <div className="input-section">
          <div className="gemini-pill">
            <input 
              value={question} 
              onChange={e => setQuestion(e.target.value)} 
              onKeyPress={e => e.key === 'Enter' && handleAsk()} 
              placeholder="Ask about your documents..." 
            />
            <button className="send-btn" onClick={handleAsk} disabled={isThinking || !question.trim()}>
              <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
              </svg>
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;