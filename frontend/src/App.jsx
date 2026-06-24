import React, { useState, useEffect, useRef } from "react";
import "./index.css";

/* ═══════════════════════════════════════════════════
   KATEX DEFERRED LOADING HOOK
   KaTeX loads with defer for faster page startup.
   This hook triggers a re-render when KaTeX finishes
   loading, so math formulas auto-render (Gemini-style).
   ═══════════════════════════════════════════════════ */
const useKatexReady = () => {
  const [ready, setReady] = useState(!!window.katex);
  useEffect(() => {
    if (window.katex) { setReady(true); return; }
    const onReady = () => setReady(true);
    window.addEventListener('katex-ready', onReady);
    return () => window.removeEventListener('katex-ready', onReady);
  }, []);
  return ready;
};

/* ═══════════════════════════════════════════════════
   PLOTLY 3D CHART COMPONENT
   ═══════════════════════════════════════════════════ */
const PlotlyChart = ({ jsonStr }) => {
  const chartRef = useRef(null);
  const [errorMsg, setErrorMsg] = useState(null);

  useEffect(() => {
    if (!chartRef.current) return;
    setErrorMsg(null);
    try {
      const fig = JSON.parse(jsonStr);
      
      // Clear fixed width and height from backend to let CSS control size
      if (fig.layout) {
        delete fig.layout.width;
        delete fig.layout.height;
      }

      const layout = {
        ...fig.layout,
        autosize: true,
        template: "plotly_dark",
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: "#e0e0e0" },
        margin: { l: 20, r: 20, t: 40, b: 20 },
      };

      const data = Array.isArray(fig.data) ? fig.data : [fig.data];

      if (!window.Plotly) {
        throw new Error("Plotly.js library failed to load from CDN.");
      }

      // Plotly.react handles initial render or updates automatically
      window.Plotly.react(chartRef.current, data, layout, {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
      }).catch(err => {
        setErrorMsg(`Plotly drawing error: ${err.message}`);
      });
    } catch (err) {
      console.error("Plotly render error:", err);
      setErrorMsg(err.message || String(err));
    }
  }, [jsonStr]);

  if (errorMsg) {
    return (
      <div className="plotly-chart-container error-state" style={{
        width: "100%", height: "450px", borderRadius: "12px", margin: "12px 0",
        border: "1px solid #ff4444", background: "rgba(255,0,0,0.1)",
        display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "20px"
      }}>
        <span style={{ fontSize: "2rem", marginBottom: "10px" }}>⚠️</span>
        <h3 style={{ color: "#ff8888", margin: "0 0 10px 0" }}>Visualization Render Error</h3>
        <p style={{ color: "#ffaaaa", textAlign: "center", wordBreak: "break-all" }}>{errorMsg}</p>
        <details style={{ marginTop: "15px", color: "#888", width: "100%" }}>
          <summary style={{ cursor: "pointer", outline: "none" }}>Show Raw JSON Data</summary>
          <pre style={{ 
            marginTop: "10px", maxHeight: "150px", overflowY: "auto", 
            background: "rgba(0,0,0,0.3)", padding: "10px", borderRadius: "5px", fontSize: "0.75rem" 
          }}>
            {jsonStr}
          </pre>
        </details>
      </div>
    );
  }

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
        background: "rgba(0,0,0,0.2)",
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
  const [consoleLogs, setConsoleLogs] = useState([]);
  const [showConsole, setShowConsole] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);
  const iframeRef = useRef(null);

  // Message receiver for inside-sandbox logs and errors
  useEffect(() => {
    const handleMessage = (e) => {
      if (e.data && e.data.type) {
        if (e.data.type === 'CONSOLE_LOG') {
          setConsoleLogs(prev => [...prev, { type: 'log', text: e.data.text }]);
        } else if (e.data.type === 'CONSOLE_ERROR') {
          setConsoleLogs(prev => [...prev, { type: 'error', text: e.data.text }]);
        } else if (e.data.type === 'CONSOLE_WARN') {
          setConsoleLogs(prev => [...prev, { type: 'warn', text: e.data.text }]);
        }
      }
    };
    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, []);

  useEffect(() => {
    if (!htmlCode || !iframeRef.current) return;
    setConsoleLogs([]); // Reset logs on reload or new code

    try {
      let doc = htmlCode;
      
      // Inject dark style, error listener, and console log capture script
      const injection = `
        <style>
          html, body { background-color: #0d0d0d !important; color: #e0e0e0 !important; margin: 0; padding: 0; font-family: system-ui, -apple-system, sans-serif; height: 100%; overflow: hidden; }
          #chart, .js-plotly-plot { background-color: transparent !important; }
          .bg { fill: transparent !important; }
          .error-box { margin: 20px; border: 1px solid #ff4444; padding: 15px; background: #2a0000; border-radius: 5px; color: #ff8888; }
        </style>
        <script>
          (function() {
            const _log = console.log;
            const _error = console.error;
            const _warn = console.warn;
            
            console.log = function(...args) {
              _log.apply(console, args);
              window.parent.postMessage({ type: 'CONSOLE_LOG', text: args.map(x => typeof x === 'object' ? JSON.stringify(x) : String(x)).join(' ') }, '*');
            };
            console.error = function(...args) {
              _error.apply(console, args);
              window.parent.postMessage({ type: 'CONSOLE_ERROR', text: args.map(x => typeof x === 'object' ? JSON.stringify(x) : String(x)).join(' ') }, '*');
            };
            console.warn = function(...args) {
              _warn.apply(console, args);
              window.parent.postMessage({ type: 'CONSOLE_WARN', text: args.map(x => typeof x === 'object' ? JSON.stringify(x) : String(x)).join(' ') }, '*');
            };
          })();

          window.onerror = function(msg, url, line, col, error) {
            window.parent.postMessage({ type: 'CONSOLE_ERROR', text: msg + ' (Line ' + line + ')' }, '*');
            return false;
          };
        </script>
      `;

      const lowerHtml = doc.toLowerCase();
      if (lowerHtml.includes("</head>")) {
        const index = lowerHtml.indexOf("</head>");
        doc = doc.substring(0, index) + injection + doc.substring(index);
      } else if (lowerHtml.includes("<body>")) {
        const index = lowerHtml.indexOf("<body>");
        doc = doc.substring(0, index + 6) + injection + doc.substring(index + 6);
      } else {
        doc = injection + doc;
      }

      // Write directly into the iframe's document
      const iframe = iframeRef.current;
      const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
      iframeDoc.open();
      iframeDoc.write(doc);
      iframeDoc.close();
      setHasError(false);
    } catch (err) {
      console.error("Artifact write error:", err);
      setHasError(true);
    }
  }, [htmlCode, reloadKey]);

  if (hasError || !htmlCode) {
    return (
      <div className="artifact-error">
        <span>⚠️</span> Failed to render sandbox
      </div>
    );
  }

  return (
    <div className={`artifact-container ${isExpanded ? "expanded" : ""}`} style={{ display: "flex", flexDirection: "column" }}>
      <div className="artifact-header">
        <div className="artifact-header-left">
          <div className="artifact-dot" />
          <span className="artifact-label">Live Sandbox Simulation</span>
          <span className="artifact-badge">HTML/JS</span>
        </div>
        <div className="artifact-header-right">
          <button
            className="artifact-btn"
            onClick={() => setReloadKey(prev => prev + 1)}
            title="Reload simulation"
          >
            ↻
          </button>
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
      <div className="artifact-iframe-wrap" style={{ flexGrow: 1, position: "relative", display: "flex", flexDirection: "column" }}>
        <iframe
          ref={iframeRef}
          title="AI Artifact"
          className="artifact-iframe"
          style={{ background: "#0d0d0d", flexGrow: 1, border: "none" }}
        />
        
        {/* Real-time Sandbox Console Overlay */}
        <div 
          className="sandbox-console-drawer"
          style={{
            position: "absolute",
            bottom: 0,
            left: 0,
            right: 0,
            borderTop: "1px solid rgba(139, 92, 246, 0.25)",
            background: "rgba(10, 10, 15, 0.95)",
            color: "#e0e0e0",
            fontFamily: "monospace",
            fontSize: "0.78rem",
            zIndex: 10,
            display: "flex",
            flexDirection: "column",
            backdropFilter: "blur(8px)"
          }}
        >
          <div 
            className="sandbox-console-header"
            onClick={() => setShowConsole(!showConsole)}
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              padding: "6px 14px",
              cursor: "pointer",
              background: "rgba(255,255,255,0.03)",
              userSelect: "none"
            }}
          >
            <span style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <span>🛠️</span> Console Output {consoleLogs.length > 0 && `(${consoleLogs.length})`}
            </span>
            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              {consoleLogs.length > 0 && (
                <button 
                  onClick={(e) => { e.stopPropagation(); setConsoleLogs([]); }} 
                  style={{
                    background: "none", border: "none", color: "#888", cursor: "pointer", fontSize: "0.75rem"
                  }}
                  title="Clear Console"
                >
                  Clear
                </button>
              )}
              <span>{showConsole ? "▼" : "▲"}</span>
            </div>
          </div>
          {showConsole && (
            <div 
              className="sandbox-console-body"
              style={{
                maxHeight: "150px",
                overflowY: "auto",
                padding: "8px 14px",
                display: "flex",
                flexDirection: "column",
                gap: "4px",
                borderTop: "1px solid rgba(255,255,255,0.05)"
              }}
            >
              {consoleLogs.length === 0 ? (
                <span style={{ color: "#777", fontStyle: "italic" }}>No console output captured. Simulation running cleanly.</span>
              ) : (
                consoleLogs.map((log, idx) => (
                  <div 
                    key={idx} 
                    style={{
                      color: log.type === 'error' ? '#ff6666' : log.type === 'warn' ? '#ffcc44' : '#a0e0a0',
                      borderLeft: `3px solid ${log.type === 'error' ? '#ff4444' : log.type === 'warn' ? '#ffaa00' : '#44bb44'}`,
                      paddingLeft: "8px",
                      whiteSpace: "pre-wrap",
                      lineHeight: "1.4"
                    }}
                  >
                    {log.text}
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

/* ═══════════════════════════════════════════════════
   MARKDOWN RENDERER (with Plotly + Artifact support)
   ═══════════════════════════════════════════════════ */
const renderMath = (tex, isBlock) => {
  if (window.katex) {
    try {
      return (
        <span 
          dangerouslySetInnerHTML={{ 
            __html: window.katex.renderToString(tex, { 
              displayMode: isBlock,
              throwOnError: false
            }) 
          }} 
        />
      );
    } catch (e) {
      console.error(e);
    }
  }
  // Fallback if KaTeX is not loaded
  return isBlock ? (
    <div className="math-block-fallback">{tex}</div>
  ) : (
    <span className="math-inline-fallback">{tex}</span>
  );
};

const renderInlineElements = (text) => {
  // Split inline elements: inline math \( ... \) or $ ... $, bold **...**, inline code `...`
  const inlineParts = text.split(/(\\\([\s\S]*?\\\)|\$.*?\$|\*\*.*?\*\*|`.*?`)/g);
  return inlineParts.map((chunk, index) => {
    if (chunk.startsWith("\\(") && chunk.endsWith("\\)")) {
      return <React.Fragment key={index}>{renderMath(chunk.slice(2, -2).trim(), false)}</React.Fragment>;
    }
    if (chunk.startsWith("$") && chunk.endsWith("$")) {
      const content = chunk.slice(1, -1).trim();
      // Smart check: if content contains common english words or is a simple currency amount, don't treat as math
      const commonWords = /\b(and|or|the|a|an|of|to|in|is|that|it|costs|buy|price|each|for|with|at|from|by|on|this|that|these|those)\b/i;
      const isCurrency = /^\d+(\.\d{2})?$/; // e.g. 10 or 9.99
      if (commonWords.test(content) || isCurrency.test(content)) {
        return chunk; // Return original text including dollar signs
      }
      return <React.Fragment key={index}>{renderMath(content, false)}</React.Fragment>;
    }
    if (chunk.startsWith("**") && chunk.endsWith("**")) {
      return <strong key={index}>{chunk.slice(2, -2)}</strong>;
    }
    if (chunk.startsWith("`") && chunk.endsWith("`")) {
      return <code key={index}>{chunk.slice(1, -1)}</code>;
    }
    return chunk;
  });
};

const parseAndRenderSegment = (segment) => {
  // Extract block math: \[ ... \] or $$ ... $$
  const parts = segment.split(/(\\\[[\s\S]*?\\\]|\$\$[\s\S]*?\$\$)/g);
  return parts.map((part, index) => {
    if (part.startsWith("\\[") && part.endsWith("\\]")) {
      const tex = part.slice(2, -2).trim();
      return <div key={index} className="math-block">{renderMath(tex, true)}</div>;
    }
    if (part.startsWith("$$") && part.endsWith("$$")) {
      const tex = part.slice(2, -2).trim();
      return <div key={index} className="math-block">{renderMath(tex, true)}</div>;
    }

    const lines = part.split("\n");
    return (
      <React.Fragment key={index}>
        {lines.map((line, j) => {
          const trimmed = line.trim();
          
          if (trimmed === "") {
            return <div key={j} style={{ height: "6px" }} />;
          }

          if (trimmed === "---") {
            return <hr key={j} className="md-hr" />;
          }

          if (line.startsWith("### ")) {
            return <h3 key={j} className="md-h3">{renderInlineElements(line.slice(4))}</h3>;
          }
          if (line.startsWith("## ")) {
            return <h2 key={j} className="md-h2">{renderInlineElements(line.slice(3))}</h2>;
          }
          if (line.startsWith("# ")) {
            return <h1 key={j} className="md-h1">{renderInlineElements(line.slice(2))}</h1>;
          }

          const listMatch = trimmed.match(/^([\-\*]|\d+\.)\s+(.*)/);
          if (listMatch) {
            const indent = line.length - line.trimStart().length;
            const marker = listMatch[1];
            const content = listMatch[2];
            const isNumbered = /^\d+\.$/.test(marker);
            return (
              <div 
                key={j} 
                className={`md-list-item ${isNumbered ? "numbered" : "bullet"}`}
                style={{ paddingLeft: `${indent * 8 + 12}px` }}
              >
                {isNumbered ? (
                  <span className="num-prefix">{marker}</span>
                ) : (
                  <span className="bullet-dot">•</span>
                )}
                <span className="bullet-content">
                  {renderInlineElements(content)}
                </span>
              </div>
            );
          }

          return (
            <p key={j} className="md-p">
              {renderInlineElements(line)}
            </p>
          );
        })}
      </React.Fragment>
    );
  });
};

const MessageRenderer = ({ text }) => {
  // Subscribe to KaTeX load event — triggers re-render when KaTeX finishes loading
  // eslint-disable-next-line no-unused-vars
  const katexReady = useKatexReady();

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
                <div key={i} className="md-content">
                  {parseAndRenderSegment(part)}
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
   USER MESSAGE COMPONENT (Collapsible after 4 lines)
   ═══════════════════════════════════════════════════ */
const UserMessage = ({ text }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isTruncated, setIsTruncated] = useState(false);
  const [copied, setCopied] = useState(false);
  const textRef = useRef(null);

  useEffect(() => {
    const checkTruncation = () => {
      if (textRef.current) {
        setIsTruncated(textRef.current.scrollHeight > 108);
      }
    };
    checkTruncation();
    const timer = setTimeout(checkTruncation, 50);
    window.addEventListener("resize", checkTruncation);
    return () => {
      clearTimeout(timer);
      window.removeEventListener("resize", checkTruncation);
    };
  }, [text]);

  const handleCopy = () => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="user-message-container">
      <div
        ref={textRef}
        className={`user-message-content ${isExpanded ? "expanded" : "collapsed"}`}
      >
        <MessageRenderer text={text} />
      </div>
      <div className="user-message-actions">
        {isTruncated && (
          <button
            className="user-message-toggle"
            onClick={() => setIsExpanded(!isExpanded)}
          >
            {isExpanded ? "Show Less ▲" : "Show More ▼"}
          </button>
        )}
        <button
          className="user-copy-btn"
          onClick={handleCopy}
          title="Copy prompt"
        >
          {copied ? (
            <>Copied! ✓</>
          ) : (
            <>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
              </svg>
              Copy
            </>
          )}
        </button>
      </div>
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
  const currentLogsRef = useRef([]);

  // Settings
  const [enableWebSearch, setEnableWebSearch] = useState(false);
  const [contextLength, setContextLength] = useState(0);
  const [maxTokens, setMaxTokens] = useState(2048);
  const [temperature, setTemperature] = useState(0.7);
  const [deviceMode, setDeviceMode] = useState("gpu");
  const [routingMode, setRoutingMode] = useState(() => localStorage.getItem("routing_mode") || "auto");

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

  // Auto-focus on "ask anything" input when session changes or component mounts
  useEffect(() => {
    textareaRef.current?.focus();
  }, [currentSessionId]);

  // Dynamically adjust textarea height when prompt changes
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = textareaRef.current.scrollHeight + "px";
    }
  }, [prompt]);


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
    e.target.style.height = e.target.scrollHeight + "px";
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
      setCurrentStream("");
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
    currentLogsRef.current = [];
    setMenuOpen(false);

    const controller = new AbortController();
    setAbortController(controller);

    let fullText = "";

    try {
      // Show a cold-start hint — model loading takes 2-4 minutes on first run
      const initLog = "🧊 Cold start: loading AI models into GPU memory (2-4 min on first run, instant after)...";
      setCurrentLogs([initLog]);
      currentLogsRef.current = [initLog];

      const res = await fetch(`${serverUrl}/api/chat`, {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "Bypass-Tunnel-Reminder": "true",
          "bypass-tunnel-reminder": "true"
        },
        signal: controller.signal,
        body: JSON.stringify({
          prompt: userText,
          image: img,
          mode: routingMode,
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
              currentLogsRef.current = [...currentLogsRef.current, data.message];
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
            currentLogsRef.current = [...currentLogsRef.current, data.message];
            setCurrentLogs(prev => [...prev, data.message]);
          } else if (data.type === "error") {
            setHistory(prev => [...prev, { type: "ai", text: "Error: " + data.message }]);
            setIsGenerating(false);
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
        setHistory(prev => [...prev, { type: "ai", text: fullText || "Cancelled.", logs: currentLogsRef.current }]);
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
        if (lastAi && (!lastAi.logs || lastAi.logs.length === 0)) lastAi.logs = currentLogsRef.current;
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
            <button className="sidebar-settings-btn" onClick={(e) => { e.stopPropagation(); setSettingsOpen(true); }} title="Settings">
              ⚙️
            </button>
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
                <div key={i} className={`msg-row ${msg.type}`}>
                  <div className={`msg-avatar ${msg.type}`}>
                    {msg.type === "user" ? "A" : "✦"}
                  </div>
                  <div className="msg-body">
                    {msg.type === "ai" && msg.logs && msg.logs.length > 0 && (
                      <ThinkingBlock logs={msg.logs} isActive={false} />
                    )}
                    {msg.type === "user" ? (
                      <UserMessage text={msg.text} />
                    ) : (
                      <MessageRenderer text={msg.text} />
                    )}
                  </div>
                </div>
              ))}

              {/* Active generation */}
              {isGenerating && (
                <div className="msg-row ai">
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
            {attachedImage && (
              <span className="image-badge">
                📎 Image attached
                <button 
                  onClick={() => setAttachedImage(null)}
                  style={{
                    background: "none",
                    border: "none",
                    color: "white",
                    marginLeft: "6px",
                    cursor: "pointer",
                    fontWeight: "bold",
                    fontSize: "0.8rem",
                    padding: "0 2px"
                  }}
                  title="Remove image"
                >
                  ✕
                </button>
              </span>
            )}

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
              <label>Routing Mode</label>
              <select value={routingMode} onChange={e => setRoutingMode(e.target.value)}>
                <option value="auto">Auto (Smart Router)</option>
                <option value="reasoning">Reasoning (DeepSeek Math/Theory)</option>
                <option value="coding">Coding (Actor-Critic Sandbox)</option>
                <option value="simple">Simple (Direct Response)</option>
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
                localStorage.setItem("routing_mode", routingMode);
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
