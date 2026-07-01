import React, { useState, useEffect, useRef } from "react";
import "./index.css";

import Sidebar from "./components/layout/Sidebar"
import SettingsModal from "./components/layout/SettingsModal"
import InputArea from "./components/input/InputArea
import ThinkingBlock from "./components/chat/ThinkingBlock"
import UserMessage from "./components/chat/UserMessage"
import MessageRenderer from "./components/chat/MessageRenderer"
export default function App() {
  // Server connection
  const [serverUrl, setServerUrl] = useState(() =>
    localStorage.getItem("server_url") || "http://127.0.0.1:8000"
  );
  const [isConnected, setIsConnected] = useState(false);
  const [isEvmActive, setIsEvmActive] = useState(false);
  const [isPreloading, setIsPreloading] = useState(false);

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await fetch(`${serverUrl}/api/status`);
        if (res.ok) {
          const data = await res.json();
          setIsConnected(true);
          setIsEvmActive(!!data.system?.evm_active);
        } else {
          setIsConnected(false);
          setIsEvmActive(false);
        }
      } catch {
        setIsConnected(false);
        setIsEvmActive(false);
      }
    };
    checkStatus();
    const interval = setInterval(checkStatus, 5000);
    return () => clearInterval(interval);
  }, [serverUrl]);

  // Session management
  const [sessions, setSessions] = useState(() => {
    try { return JSON.parse(localStorage.getItem("chat_sessions") || "[]"); }
    catch { return []; }
  });
  const [currentSessionId, setCurrentSessionId] = useState(Date.now())
  const [history, setHistory] = useState([]);

  // UI state
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [menuOpen, setMenuOpen] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)

  // Chat state
  const [prompt, setPrompt] = useState("")
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentLogs, setCurrentLogs] = useState([null]); //added null 
  const [currentStream, setCurrentStream] = useState("");
  const [attachedImage, setAttachedImage] = useState(null);
  const [abortController, setAbortController] = useState(null);
  const currentLogsRef = useRef([]);

  // Settings
  const [searchMode, setSearchMode] = useState("off");
  const [contextLength, setContextLength] = useState(0);
  const [maxTokens, setMaxTokens] = useState(2048);
  const [temperature, setTemperature] = useState(0.7);
  const [deviceMode, setDeviceMode] = useState("gpu");
  const [routingMode, setRoutingMode] = useState(() =>
    localStorage.getItem("routing_mode") || "auto"
  );

  // Typing animation for empty state
  const [displayText, setDisplayText] = useState("")
  const bottomRef = useRef(null);
  const textareaRef = useRef(null);

  // Auto-scroll on new content
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history, currentStream, currentLogs]);

  // Empty-state typing effect
  useEffect(() => {
    const target = "What is on your mind today?";
    let i = 0;
    setDisplayText("");
    const iv = setInterval(() => {
      setDisplayText(target.substring(0, i + 1));
      i++;
      if (i >= target.length) clearInterval(iv);
    }, 50);
    return () => clearInterval(iv);
  }, []);

  // Auto-focus textarea on session switch
  useEffect(() => {
    textareaRef.current?.focus();
  }, [currentSessionId]);

  // Dynamically adjust textarea height
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = textareaRef.current.scrollHeight + "px";
    }
  }, [prompt]);

  // Persist sessions to localStorage
  useEffect(() => {
    if (history.length === 0 && sessions.length === 0) return;
    setSessions((prev) => {
      const existing = prev.find((s) => s.id === currentSessionId);
      let title = "New Chat";
      const first = history.find((m) => m.type === "user");
      if (first) title = first.text.substring(0, 35) + (first.text.length > 35 ? "..." : "");

      let next;
      if (existing) {
        next = prev.map((s) => s.id === currentSessionId ? { ...s, history, title } : s);
      } else {
        if (history.length === 0) return prev;
        next = [{ id: currentSessionId, title, history }, ...prev];
      }
      localStorage.setItem("chat_sessions", JSON.stringify(next));
      return next;
    });
  }, [history, currentSessionId]);

  // Session actions
  const createNewChat = () => { setCurrentSessionId(Date.now()); setHistory([]); };

  const loadSession = (id) => {
    const s = sessions.find((x) => x.id === id);
    if (s) { setCurrentSessionId(id); setHistory(s.history); setSidebarOpen(false); }
  };

  const deleteSession = (id, e) => {
    e.stopPropagation();
    setSessions((prev) => {
      const next = prev.filter((s) => s.id !== id);
      localStorage.setItem("chat_sessions", JSON.stringify(next));
      return next;
    });
    if (id === currentSessionId) createNewChat();
  };

  // Model management actions
  const handleOffload = async () => {
    try {
      await fetch(`${serverUrl}/api/offload`, { method: "POST" });
      alert("All models offloaded from VRAM!");
    } catch { alert("Failed to offload."); }
  };

  const handleLoadAll = async () => {
    if (!isConnected || !isEvmActive || isPreloading) return;
    setIsPreloading(true);
    try {
      const res = await fetch(`${serverUrl}/api/load_all`, { method: "POST" });
      alert(res.ok ? "All models successfully loaded into System RAM!" : "Failed to load all models.");
    } catch {
      alert("Failed to load all models.");
    } finally {
      setIsPreloading(false);
    }
  };

  const handleStop = async () => {
    try { await fetch(`${serverUrl}/api/cancel`, { method: "POST" }); } catch {}
    if (abortController) {
      abortController.abort();
      setIsGenerating(false);
      setAbortController(null);
      setCurrentStream("");
    }
  };

  // Core chat send handler
  const handleSend = async (e) => {
    if (e?.key === "Enter" && !e.shiftKey) e.preventDefault();
    else if (e?.type !== "click" && e?.key !== "Enter") return;
    if (!prompt.trim() && !attachedImage) return;

    const userText = prompt.trim();
    const img = attachedImage;
    setPrompt("");
    setAttachedImage(null);
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    setHistory((prev) => [...prev, { type: "user", text: userText || "📎 Image attached" }]);
    setIsGenerating(true);
    setCurrentStream("");
    setCurrentLogs([]);
    currentLogsRef.current = [];
    setMenuOpen(false);

    const controller = new AbortController();
    setAbortController(controller);
    let fullText = "";

    try {
      // Cold-start hint
      const initLog = "🧊 Cold start: loading AI models into GPU memory (2-4 min on first run, instant after)...";
      setCurrentLogs([initLog]);
      currentLogsRef.current = [initLog];

      const res = await fetch(`${serverUrl}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Bypass-Tunnel-Reminder": "true",
          "bypass-tunnel-reminder": "true",
        },
        signal: controller.signal,
        body: JSON.stringify({
          prompt: userText, image: img, mode: routingMode,
          context_length: contextLength, max_tokens: maxTokens, temperature,
          device_mode: deviceMode, gpu_layers: -1, search_mode: searchMode,
        }),
      });

      if (!res.ok) {
        let msg = `Server error (${res.status})`;
        try { const d = await res.json(); if (d.detail) msg = d.detail; } catch {}
        throw new Error(msg);
      }

      const contentType = res.headers.get("content-type") || "";
      if (contentType.includes("text/html")) {
        const htmlBody = await res.text();
        console.error("Tunnel returned HTML instead of JSON:", htmlBody.substring(0, 500));
        throw new Error("Tunnel is blocking the request (returned HTML). Open the tunnel URL directly in your browser first, click 'Continue', then retry.");
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let firstChunkLogged = false;
      let lineBuffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });

        if (!firstChunkLogged) {
          console.log("First chunk from server:", chunk.substring(0, 300));
          firstChunkLogged = true;
        }

        const combined = lineBuffer + chunk;
        const lines = combined.split("\n");
        lineBuffer = lines.pop() || "";

        for (let line of lines) {
          line = line.trim();
          if (line.startsWith("data: ")) line = line.substring(6);
          if (!line || line.includes('"keep_alive"')) continue;
          try {
            const data = JSON.parse(line);
            if (data.type === "status") {
              currentLogsRef.current = [...currentLogsRef.current, data.message];
              setCurrentLogs((prev) => [...prev, data.message]);
            } else if (data.type === "chunk") {
              fullText += data.content || data.text || "";
              setCurrentStream(fullText);
            } else if (data.type === "final_response") {
              fullText = data.text;
              setCurrentStream(fullText);
              setHistory((prev) => [...prev, { type: "ai", text: fullText, logs: [] }]);
              setIsGenerating(false);
            } else if (data.type === "error") {
              setHistory((prev) => [...prev, { type: "ai", text: "Error: " + data.message }]);
              setIsGenerating(false);
            }
          } catch (parseErr) {
            console.warn("Failed to parse line:", line.substring(0, 200), parseErr);
          }
        }
      }

      // Process any remaining buffer
      if (lineBuffer.trim()) {
        try {
          const data = JSON.parse(lineBuffer.trim());
          if (data.type === "final_response") {
            fullText = data.text;
            setCurrentStream(fullText);
            setHistory((prev) => [...prev, { type: "ai", text: fullText, logs: [] }]);
            setIsGenerating(false);
          } else if (data.type === "status") {
            currentLogsRef.current = [...currentLogsRef.current, data.message];
            setCurrentLogs((prev) => [...prev, data.message]);
          } else if (data.type === "error") {
            setHistory((prev) => [...prev, { type: "ai", text: "Error: " + data.message }]);
            setIsGenerating(false);
          }
        } catch {}
      }

      // Fallback: stream ended with text but no final_response event
      if (fullText) {
        setHistory((prev) => {
          const last = prev[prev.length - 1];
          if (!last || last.type !== "ai" || last.text !== fullText) {
            return [...prev, { type: "ai", text: fullText }];
          }
          return prev;
        });
      } else {
        setHistory((prev) => {
          const last = prev[prev.length - 1];
          if (!last || last.type !== "ai") {
            return [...prev, { type: "ai", text: "⚠️ **Backend crashed during generation** (likely GPU out-of-memory).\n\nThe server process died before it could send a response. Please:\n1. Check the Kaggle notebook for error logs\n2. Restart Cell 2 to relaunch the server\n3. Try a simpler prompt first to warm up the models" }];
          }
          return prev;
        });
      }
    } catch (err) {
      if (err.name === "AbortError") {
        setHistory((prev) => [...prev, { type: "ai", text: fullText || "Cancelled.", logs: currentLogsRef.current }]);
      } else if (err.message && (err.message.toLowerCase().includes("networkerror") || err.message.toLowerCase().includes("failed to fetch"))) {
        setHistory((prev) => [...prev, { type: "ai", text: `❌ **Cannot reach backend.**\n\n**Backend not started?** Open a terminal and run:\n\`\`\`\nsource venv/bin/activate\npython backend/app.py\n\`\`\`\nWait for: \`Uvicorn running on http://127.0.0.1:8000\`\n\n**First prompt?** If the backend IS running, the models are still loading into GPU memory — this takes **2-4 minutes on first run**. Please wait and try again.` }]);
      } else {
        setHistory((prev) => [...prev, { type: "ai", text: `Error: ${err.message}` }]);
      }
    } finally {
      setIsGenerating(false);
      setAbortController(null);
      setCurrentStream("");
      setHistory((prev) => {
        const copy = [...prev];
        const lastAi = [...copy].reverse().find((m) => m.type === "ai");
        if (lastAi && (!lastAi.logs || lastAi.logs.length === 0)) lastAi.logs = currentLogsRef.current;
        return copy;
      });
      setCurrentLogs([]);
    }
  };

  return (
    <div className="app">
      <Sidebar
        sidebarOpen={sidebarOpen}
        setSidebarOpen={setSidebarOpen}
        sessions={sessions}
        currentSessionId={currentSessionId}
        createNewChat={createNewChat}
        loadSession={loadSession}
        deleteSession={deleteSession}
        handleOffload={handleOffload}
        handleLoadAll={handleLoadAll}
        isConnected={isConnected}
        isEvmActive={isEvmActive}
        isPreloading={isPreloading}
        setSettingsOpen={setSettingsOpen}
      />

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
            <div className="chat-messages">
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
                      <MessageRenderer text={msg.text} animate={i === history.length - 1 && !isGenerating} />
                    )}
                  </div>
                </div>
              ))}

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

        <InputArea
          prompt={prompt}
          setPrompt={setPrompt}
          isGenerating={isGenerating}
          attachedImage={attachedImage}
          setAttachedImage={setAttachedImage}
          menuOpen={menuOpen}
          setMenuOpen={setMenuOpen}
          searchMode={searchMode}
          setSearchMode={setSearchMode}
          handleSend={handleSend}
          handleStop={handleStop}
          setSettingsOpen={setSettingsOpen}
          textareaRef={textareaRef}
        />
      </div>

      <SettingsModal
        settingsOpen={settingsOpen}
        setSettingsOpen={setSettingsOpen}
        serverUrl={serverUrl}
        setServerUrl={setServerUrl}
        deviceMode={deviceMode}
        setDeviceMode={setDeviceMode}
        routingMode={routingMode}
        setRoutingMode={setRoutingMode}
        contextLength={contextLength}
        maxTokens={maxTokens}
        temperature={temperature}
        searchMode={searchMode}
      />
    </div>
  );
}
