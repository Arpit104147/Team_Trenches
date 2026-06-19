import React, { useState, useEffect, useRef } from "react";
import "./index.css";

/* ═══════════════════════════════════════════════════
   PLOTLY 3D CHART COMPONENT
   ═══════════════════════════════════════════════════ */
const PlotlyChart = ({ jsonStr }) => {
  const chartRef = useRef(null);
  const plotted = useRef(false);

  useEffect(() => {
    if (!chartRef.current || plotted.current) return;
    try {
      const fig = JSON.parse(jsonStr);
      const layout = {
        ...fig.layout,
        template: "plotly_dark",
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: "#e0e0e0" },
        margin: { l: 0, r: 0, t: 40, b: 0 },
      };
      window.Plotly.newPlot(chartRef.current, fig.data, layout, {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
      });
      plotted.current = true;
    } catch (err) {
      console.error("Plotly render error:", err);
    }
  }, [jsonStr]);

  return (
    <div
      ref={chartRef}
      className="plotly-chart-container"
      style={{
        width: "100%",
        height: "450px",
        borderRadius: "12px",
        overflow: "hidden",
        margin: "12px 0",
        border: "1px solid rgba(255,255,255,0.08)",
        background: "rgba(0,0,0,0.3)",
      }}
    />
  );
};

/* ═══════════════════════════════════════════════════
   ARTIFACT SANDBOX (Claude-style iframe renderer)
   Renders AI-generated HTML/JS in a secure iframe
   ═══════════════════════════════════════════════════ */
const ArtifactSandbox = ({ htmlCode }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [hasError, setHasError] = useState(false);
  const iframeRef = useRef(null);

  // Build a blob URL from the HTML code for secure sandboxed rendering
  const blobUrl = useRef(null);
  useEffect(() => {
    if (!htmlCode) return;
    try {
      // Bulletproof dark mode injection
      let injectedHtml = htmlCode;
      if (!injectedHtml.includes("background-color")) {
        injectedHtml = injectedHtml.replace("</head>", `
          <style>
            body { background-color: #0d0d0d !important; color: #e0e0e0 !important; margin: 0; padding: 0; }
            #chart, .js-plotly-plot { background-color: transparent !important; }
            .bg { fill: transparent !important; }
          </style>
        </head>`);
      }
      
      const blob = new Blob([injectedHtml], { type: "text/html" });
      blobUrl.current = URL.createObjectURL(blob);
      setHasError(false);
    } catch (err) {
      console.error("Artifact blob error:", err);
      setHasError(true);
    }
    return () => {
      if (blobUrl.current) URL.revokeObjectURL(blobUrl.current);
    };
  }, [htmlCode]);

  if (hasError || !htmlCode) {
    return (
      <div className="artifact-error">
        <span>⚠️</span> Failed to render artifact
      </div>
    );
  }

  return (
    <div className={`artifact-container ${isExpanded ? "expanded" : ""}`}>
      <div className="artifact-header">
        <div className="artifact-header-left">
          <div className="artifact-dot" />
          <span className="artifact-label">Live Artifact</span>
          <span className="artifact-badge">HTML/JS</span>
        </div>
        <div className="artifact-header-right">
          <button
            className="artifact-btn"
            onClick={() => {
              const w = window.open("", "_blank");
              if (w) { w.document.write(htmlCode); w.document.close(); }
            }}
            title="Open in new tab"
          >
            ↗
          </button>
          <button
            className="artifact-btn"
            onClick={() => setIsExpanded(!isExpanded)}
            title={isExpanded ? "Collapse" : "Expand"}
          >
            {isExpanded ? "⊖" : "⊕"}
          </button>
        </div>
      </div>
      <div className="artifact-iframe-wrap">
        {blobUrl.current && (
          <iframe
            ref={iframeRef}
            src={blobUrl.current}
            sandbox="allow-scripts"
            title="AI Artifact"
            className="artifact-iframe"
          />
        )}
      </div>
    </div>
  );
};

/* ═══════════════════════════════════════════════════
   MARKDOWN RENDERER (with Plotly + Artifact support)
   ═══════════════════════════════════════════════════ */
const MessageRenderer = ({ text }) => {
  if (!text) return null;

  // Split on both Plotly JSON blocks and HTML Artifact blocks
  const specialParts = text.split(/(<!--PLOTLY_JSON-->[\s\S]*?<!--\/PLOTLY_JSON-->|<!--ARTIFACT_HTML-->[\s\S]*?<!--\/ARTIFACT_HTML-->)/g);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
      {specialParts.map((segment, si) => {
        // Render Plotly chart if this segment is a PLOTLY_JSON block
        if (segment.startsWith("<!--PLOTLY_JSON-->")) {
          const jsonStr = segment
            .replace("<!--PLOTLY_JSON-->", "")
            .replace("<!--/PLOTLY_JSON-->", "")
            .trim();
          return <PlotlyChart key={`plotly-${si}`} jsonStr={jsonStr} />;
        }

        // Render HTML Artifact in secure iframe sandbox
        if (segment.startsWith("<!--ARTIFACT_HTML-->")) {
          const htmlCode = segment
            .replace("<!--ARTIFACT_HTML-->", "")
            .replace("<!--/ARTIFACT_HTML-->", "")
            .trim();
          return <ArtifactSandbox key={`artifact-${si}`} htmlCode={htmlCode} />;
        }

        // Otherwise render as markdown with code blocks
        const parts = segment.split(/(```[\s\S]*?```)/g);
        return (
          <React.Fragment key={`seg-${si}`}>
            {parts.map((part, i) => {
              if (part.startsWith("```") && part.endsWith("```")) {
                const lines = part.slice(3, -3).split("\n");
                const lang = lines[0].trim().split(" ")[0];
                const code = lines.slice(1).join("\n");
                return (
                  <div key={i} className="code-block-wrapper">
                    <div className="code-block-header">
                      <span>{lang || "code"}</span>
                      <button className="copy-btn" onClick={() => navigator.clipboard.writeText(code)}>
                        Copy
                      </button>
                    </div>
                    <pre><code>{code}</code></pre>
                  </div>
                );
              }
              return (
                <div key={i}>
                  {part.split("\n").map((line, j) => {
                    const formatted = line.split(/(\*\*.*?\*\*|`.*?`)/g).map((chunk, k) => {
                      if (chunk.startsWith("**") && chunk.endsWith("**"))
                        return <strong key={k}>{chunk.slice(2, -2)}</strong>;
                      if (chunk.startsWith("`") && chunk.endsWith("`"))
                        return <code key={k}>{chunk.slice(1, -1)}</code>;
                      return chunk;
                    });
                    return <React.Fragment key={j}>{formatted}<br /></React.Fragment>;
                  })}
                </div>
              );
            })}
          </React.Fragment>
        );
      })}
    </div>
  );
};

/* ═══════════════════════════════════════════════════
   THINKING BLOCK (DeepSeek-style)
   ═══════════════════════════════════════════════════ */
const ThinkingBlock = ({ logs, isActive }) => {
  const [isOpen, setIsOpen] = useState(true);

  return (
    <div className="thinking-block">
      <div className="thinking-header" onClick={() => setIsOpen(!isOpen)}>
        {isActive ? <div className="thinking-spinner" /> : <span style={{ fontSize: "0.85rem" }}>✅</span>}
        <span className="thinking-label">
          {isActive ? "Thinking..." : `Thought for ${logs.length} steps`}
        </span>
        <span className={`thinking-chevron ${isOpen ? "open" : ""}`}>▼</span>
      </div>
      {isOpen && (
        <div className="thinking-content">
          {logs.map((log, i) => (
            <div key={i} className="thinking-step">
              <span className="thinking-step-icon">▸</span>
              <span>{log}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

/* ═══════════════════════════════════════════════════
   MAIN APP
   ═══════════════════════════════════════════════════ */
export default function App() {
  const [serverUrl, setServerUrl] = useState(() =>
    localStorage.getItem("server_url") || "http://127.0.0.1:8000"
  );

  // Session management
  const [sessions, setSessions] = useState(() => {
    try { return JSON.parse(localStorage.getItem("chat_sessions") || "[]"); }
    catch { return []; }
  });
  const [currentSessionId, setCurrentSessionId] = useState(Date.now());
  const [history, setHistory] = useState([]);

  // UI state
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [menuOpen, setMenuOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);

  // Chat state
  const [prompt, setPrompt] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentLogs, setCurrentLogs] = useState([]);
  const [currentStream, setCurrentStream] = useState("");
  const [attachedImage, setAttachedImage] = useState(null);
  const [abortController, setAbortController] = useState(null);

  // Settings
  const [enableWebSearch, setEnableWebSearch] = useState(false);
  const [contextLength, setContextLength] = useState(0);
  const [maxTokens, setMaxTokens] = useState(2048);
  const [temperature, setTemperature] = useState(0.7);
  const [deviceMode, setDeviceMode] = useState("gpu");

  // Typing animation
  const [displayText, setDisplayText] = useState("");

  const bottomRef = useRef(null);
  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history, currentStream, currentLogs]);

  // Typing effect
  useEffect(() => {
    const target = "What's on your mind today?";
    let i = 0;
    setDisplayText("");
    const iv = setInterval(() => {
      setDisplayText(target.substring(0, i + 1));
      i++;
      if (i >= target.length) clearInterval(iv);
    }, 50);
    return () => clearInterval(iv);
  }, []);

  // Persist sessions
  useEffect(() => {
    if (history.length === 0 && sessions.length === 0) return;
    setSessions(prev => {
      const existing = prev.find(s => s.id === currentSessionId);
      let title = "New Chat";
      const first = history.find(m => m.type === "user");
      if (first) title = first.text.substring(0, 35) + (first.text.length > 35 ? "..." : "");

      let next;
      if (existing) {
        next = prev.map(s => s.id === currentSessionId ? { ...s, history, title } : s);
      } else {
        if (history.length === 0) return prev;
        next = [{ id: currentSessionId, title, history }, ...prev];
      }
      localStorage.setItem("chat_sessions", JSON.stringify(next));
      return next;
    });
  }, [history, currentSessionId]);

  // Auto-resize textarea
  const handleTextareaInput = (e) => {
    setPrompt(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 180) + "px";
  };

  const createNewChat = () => {
    setCurrentSessionId(Date.now());
    setHistory([]);
  };

  const loadSession = (id) => {
    const s = sessions.find(x => x.id === id);
    if (s) { setCurrentSessionId(id); setHistory(s.history); setSidebarOpen(false); }
  };

  const deleteSession = (id, e) => {
    e.stopPropagation();
    setSessions(prev => {
      const next = prev.filter(s => s.id !== id);
      localStorage.setItem("chat_sessions", JSON.stringify(next));
      return next;
    });
    if (id === currentSessionId) createNewChat();
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onloadend = () => setAttachedImage(reader.result);
    reader.readAsDataURL(file);
  };

  const handleStop = async () => {
    // Tell the backend to stop generation (models stay in RAM)
    try { await fetch(`${serverUrl}/api/cancel`, { method: "POST" }); } catch {}
    if (abortController) {
      abortController.abort();
      setIsGenerating(false);
      setAbortController(null);
    }
  };

  const handleOffload = async () => {
    try {
      await fetch(`${serverUrl}/api/offload`, { method: "POST" });
      alert("All models offloaded from VRAM!");
    } catch { alert("Failed to offload."); }
  };

  /* ─── SEND MESSAGE ─── */
  const handleSend = async (e) => {
    if (e?.key === "Enter" && !e.shiftKey) e.preventDefault();
    else if (e?.type !== "click" && e?.key !== "Enter") return;

    if (!prompt.trim() && !attachedImage) return;

    const userText = prompt.trim();
    const img = attachedImage;
    setPrompt("");
    setAttachedImage(null);
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    setHistory(prev => [...prev, { type: "user", text: userText || "📎 Image attached" }]);
    setIsGenerating(true);
    setCurrentStream("");
    setCurrentLogs([]);
    setMenuOpen(false);

    const controller = new AbortController();
    setAbortController(controller);

    try {
      // Show a cold-start hint — model loading takes 2-4 minutes on first run
      setCurrentLogs(["🧊 Cold start: loading AI models into GPU memory (2-4 min on first run, instant after)..."]);

      const res = await fetch(`${serverUrl}/api/chat`, {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "Bypass-Tunnel-Reminder": "true"
        },
        signal: controller.signal,
        body: JSON.stringify({
          prompt: userText,
          image: img,
          mode: "auto",
          context_length: contextLength,
          max_tokens: maxTokens,
          temperature,
          device_mode: deviceMode,
          gpu_layers: -1,
          enable_web_search: enableWebSearch,
        }),
      });

      if (!res.ok) {
        let msg = `Server error (${res.status})`;
        try { const d = await res.json(); if (d.detail) msg = d.detail; } catch {}
        throw new Error(msg);
      }

      // Check content-type — if localtunnel returns HTML, catch it immediately
      const contentType = res.headers.get("content-type") || "";
      if (contentType.includes("text/html")) {
        const htmlBody = await res.text();
        console.error("Tunnel returned HTML instead of JSON:", htmlBody.substring(0, 500));
        throw new Error("Tunnel is blocking the request (returned HTML). Open the tunnel URL directly in your browser first, click 'Continue', then retry.");
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let fullText = "";
      let firstChunkLogged = false;
      let lineBuffer = ""; // Buffer for incomplete JSON lines split across chunks

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        
        // Log the very first chunk to console for debugging
        if (!firstChunkLogged) {
          console.log("First chunk from server:", chunk.substring(0, 300));
          firstChunkLogged = true;
        }

        // Prepend any leftover from the previous chunk
        const combined = lineBuffer + chunk;
        const lines = combined.split("\n");
        
        // The last element might be incomplete — save it for the next chunk
        lineBuffer = lines.pop() || "";

        for (let line of lines) {
          line = line.trim();
          if (line.startsWith("data: ")) line = line.substring(6);
          if (!line) continue;
          // Skip keep_alive pings
          if (line.includes('"keep_alive"')) continue;
          try {
            const data = JSON.parse(line);
            if (data.type === "status") {
              setCurrentLogs(prev => [...prev, data.message]);
            } else if (data.type === "chunk") {
              fullText += data.content || data.text || "";
              setCurrentStream(fullText);
            } else if (data.type === "final_response") {
              fullText = data.text;
              setCurrentStream(fullText);
              setHistory(prev => [...prev, { type: "ai", text: fullText, logs: [] }]);
              setIsGenerating(false);
            } else if (data.type === "error") {
              setHistory(prev => [...prev, { type: "ai", text: "Error: " + data.message }]);
              setIsGenerating(false);
            }
          } catch (parseErr) {
            console.warn("Failed to parse line:", line.substring(0, 200), parseErr);
          }
        }
      }

      // Process any remaining buffered content after stream ends
      if (lineBuffer.trim()) {
        try {
          const data = JSON.parse(lineBuffer.trim());
          if (data.type === "final_response") {
            fullText = data.text;
            setCurrentStream(fullText);
            setHistory(prev => [...prev, { type: "ai", text: fullText, logs: [] }]);
            setIsGenerating(false);
          } else if (data.type === "status") {
            setCurrentLogs(prev => [...prev, data.message]);
          }
        } catch {}
      }

      // Fallback: if stream ended with text but no final_response event
      if (fullText) {
        setHistory(prev => {
          const last = prev[prev.length - 1];
          // Only add if the last message isn't already this text
          if (!last || last.type !== "ai" || last.text !== fullText) {
            return [...prev, { type: "ai", text: fullText }];
          }
          return prev;
        });
      } else {
        // Stream closed with NO text at all — backend likely crashed (GPU OOM)
        setHistory(prev => {
          const last = prev[prev.length - 1];
          if (!last || last.type !== "ai") {
            return [...prev, { type: "ai", text: "⚠️ **Backend crashed during generation** (likely GPU out-of-memory).\n\nThe server process died before it could send a response. Please:\n1. Check the Kaggle notebook for error logs\n2. Restart Cell 2 to relaunch the server\n3. Try a simpler prompt first to warm up the models" }];
          }
          return prev;
        });
      }

    } catch (err) {
      if (err.name === "AbortError") {
        setHistory(prev => [...prev, { type: "ai", text: fullText || "Cancelled." }]);
      } else if (err.message && (err.message.toLowerCase().includes("networkerror") || err.message.toLowerCase().includes("failed to fetch"))) {
        setHistory(prev => [...prev, { type: "ai", text: `❌ **Cannot reach backend.**\n\n**Backend not started?** Open a terminal and run:\n\`\`\`\nsource venv/bin/activate\npython backend/app.py\n\`\`\`\nWait for: \`Uvicorn running on http://127.0.0.1:8000\`\n\n**First prompt?** If the backend IS running, the models are still loading into GPU memory — this takes **2-4 minutes on first run**. Please wait and try again.` }]);
      } else {
        setHistory(prev => [...prev, { type: "ai", text: `Error: ${err.message}` }]);
      }
    } finally {
      setIsGenerating(false);
      setAbortController(null);
      setCurrentStream("");
      // Save final logs into last AI message
      setHistory(prev => {
        const copy = [...prev];
        const lastAi = [...copy].reverse().find(m => m.type === "ai");
        if (lastAi) lastAi.logs = currentLogs;
        return copy;
      });
      setCurrentLogs([]);
    }
  };

  /* ═══════════════════════════════════════════════════
     RENDER
     ═══════════════════════════════════════════════════ */
  return (
    <div className="app">
      {/* ── SIDEBAR ── */}
      <div className={`sidebar ${!sidebarOpen ? "closed" : ""}`}>
        <div className="sidebar-top">
          <button className="sidebar-toggle" onClick={() => setSidebarOpen(false)}>☰</button>
        </div>

        <button className="new-chat-btn" onClick={createNewChat}>
          <span>＋</span> New chat
        </button>

        <div className="sidebar-nav">
          <button className="nav-item" onClick={() => setSettingsOpen(true)}>
            <span className="nav-icon">⚙️</span> Settings
          </button>
          <button className="nav-item" onClick={handleOffload}>
            <span className="nav-icon">🧹</span> Offload Memory
          </button>
        </div>

        <div className="sidebar-section-title">Recents</div>
        <div className="history-list">
          {sessions.map(s => (
            <div
              key={s.id}
              className={`history-item ${s.id === currentSessionId ? "active" : ""}`}
              onClick={() => loadSession(s.id)}
            >
              <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1 }}>
                💬 {s.title}
              </span>
              <button className="delete-btn" onClick={(e) => deleteSession(s.id, e)}>✕</button>
            </div>
          ))}
          {sessions.length === 0 && (
            <div style={{ color: "var(--text-tertiary)", fontSize: "0.8rem", padding: "12px" }}>
              No recent chats.
            </div>
          )}
        </div>

        <div className="sidebar-footer">
          <div className="user-row">
            <div className="user-avatar">A</div>
            <span className="user-name">ARPIT BEHERA</span>
          </div>
        </div>
      </div>

      {/* ── MAIN CONTENT ── */}
      <div className="main">
        {!sidebarOpen && (
          <button className="floating-open-btn" onClick={() => setSidebarOpen(true)}>☰</button>
        )}

        <div className="chat-area">
          {history.length === 0 ? (
            <div className="empty-state">
              <h1>{displayText}<span className="cursor-blink">|</span></h1>
            </div>
          ) : (
            <div style={{ width: "100%", display: "flex", flexDirection: "column", alignItems: "center" }}>
              {history.map((msg, i) => (
                <div key={i} className="msg-row">
                  <div className={`msg-avatar ${msg.type}`}>
                    {msg.type === "user" ? "A" : "✦"}
                  </div>
                  <div className="msg-body">
                    {msg.type === "ai" && msg.logs && msg.logs.length > 0 && (
                      <ThinkingBlock logs={msg.logs} isActive={false} />
                    )}
                    <MessageRenderer text={msg.text} />
                  </div>
                </div>
              ))}

              {/* Active generation */}
              {isGenerating && (
                <div className="msg-row">
                  <div className="msg-avatar ai">✦</div>
                  <div className="msg-body">
                    <ThinkingBlock logs={currentLogs} isActive={true} />
                    {currentStream && <MessageRenderer text={currentStream} />}
                  </div>
                </div>
              )}

              <div ref={bottomRef} />
            </div>
          )}
        </div>

        {/* ── INPUT AREA ── */}
        <div className="input-area">
          <div className="input-wrapper">
            {attachedImage && <span className="image-badge">📎 Image attached</span>}

            {/* Popup menu */}
            {menuOpen && (
              <div className="popup-menu">
                <input type="file" accept="image/*" ref={fileInputRef} style={{ display: "none" }} onChange={handleFileUpload} />
                <button className="popup-item" onClick={() => { fileInputRef.current?.click(); setMenuOpen(false); }}>
                  <span className="popup-icon">📷</span> Upload photo or file
                </button>
                <div className="popup-divider" />
                <div className="popup-item" style={{ cursor: "default" }}>
                  <span className="popup-icon">🌐</span> Web search
                  <div
                    className={`toggle-switch ${enableWebSearch ? "on" : ""}`}
                    onClick={(e) => { e.stopPropagation(); setEnableWebSearch(!enableWebSearch); }}
                  />
                </div>
                <div className="popup-divider" />
                <button className="popup-item" onClick={() => { setSettingsOpen(true); setMenuOpen(false); }}>
                  <span className="popup-icon">⚙️</span> Settings
                </button>
              </div>
            )}

            <button
              className={`input-plus-btn ${menuOpen ? "active" : ""}`}
              onClick={() => setMenuOpen(!menuOpen)}
            >
              ＋
            </button>

            <textarea
              ref={textareaRef}
              className="input-box"
              rows={1}
              placeholder="Ask anything"
              value={prompt}
              onChange={handleTextareaInput}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) handleSend(e); }}
              disabled={isGenerating}
            />

            {isGenerating ? (
              <button className="send-btn stop" onClick={handleStop} title="Stop">■</button>
            ) : (
              <button
                className="send-btn"
                onClick={handleSend}
                disabled={!prompt.trim() && !attachedImage}
                title="Send"
              >
                ↑
              </button>
            )}
          </div>
        </div>
      </div>

      {/* ── SETTINGS MODAL ── */}
      {settingsOpen && (
        <div className="modal-overlay" onClick={() => setSettingsOpen(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h2>Settings</h2>

            <div className="modal-field">
              <label>Device Mode</label>
              <select value={deviceMode} onChange={e => setDeviceMode(e.target.value)}>
                <option value="gpu">GPU (CUDA / Vulkan)</option>
                <option value="cpu">CPU Only</option>
                <option value="hybrid">Hybrid (CPU + GPU)</option>
              </select>
            </div>
            <div className="modal-field">
              <label>Server URL</label>
              <input
                type="text"
                value={serverUrl}
                onChange={e => {
                  let val = e.target.value;
                  setServerUrl(val);
                }}
                onBlur={e => {
                  let val = e.target.value.trim();
                  if (val && !val.startsWith("http")) val = "http://" + val;
                  if (val.endsWith("/")) val = val.slice(0, -1);
                  val = val.replace("localhost", "127.0.0.1").replace("0.0.0.0", "127.0.0.1");
                  localStorage.setItem("server_url", val);
                  setServerUrl(val);
                }}
              />
            </div>
            <div className="modal-actions">
              <button onClick={() => setSettingsOpen(false)}>Close</button>
              <button className="primary-btn" onClick={() => {
                // Sanitize and persist the URL first (always succeeds locally)
                let finalUrl = serverUrl.trim();
                if (finalUrl && !finalUrl.startsWith("http")) finalUrl = "http://" + finalUrl;
                if (finalUrl.endsWith("/")) finalUrl = finalUrl.slice(0, -1);
                finalUrl = finalUrl.replace("localhost", "127.0.0.1").replace("0.0.0.0", "127.0.0.1");
                localStorage.setItem("server_url", finalUrl);
                setServerUrl(finalUrl);

                // Close the modal immediately — don't block the user
                setSettingsOpen(false);

                // Fire the backend POST silently in the background
                fetch(`${finalUrl}/api/settings`, {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ context_length: contextLength, max_tokens: maxTokens, temperature, device_mode: deviceMode, gpu_layers: -1, enable_web_search: enableWebSearch })
                }).catch(() => console.warn("Settings sync to backend deferred — will apply on next request."));
              }}>Save</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
