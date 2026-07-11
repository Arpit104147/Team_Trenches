import React, { useState } from "react";

/**
 * @component ThinkingBlock
 * Displays a collapsible DeepSeek-style thinking/status log panel.
 * Each log entry is color-coded based on the active pipeline stage.
 */
const ThinkingBlock = ({ logs, isActive }) => {
  const [isOpen, setIsOpen] = useState(true);

  const getLogStyle = (log) => {
    const l = log.toLowerCase();
    if (l.includes("prediction") || l.includes("🔮")) return { color: "#ab47bc", icon: "🔮" };
    if (l.includes("extreme") || l.includes("🔬")) return { color: "#ef5350", icon: "🔬" };
    if (l.includes("qwen") || l.includes("vision") || l.includes("transcri")) return { color: "#26c6da", icon: "👁️" };
    if (l.includes("vibethinker")) return { color: "#7c4dff", icon: "🧠" };
    if (l.includes("deepseek") || l.includes("reasoning") || l.includes("logic plan")) return { color: "#ffa726", icon: "⚡" };
    if (l.includes("ornith") || l.includes("writing code") || l.includes("sandbox") || l.includes("html artifact")) return { color: "#66bb6a", icon: "💻" };
    if (l.includes("phi-3.5") || l.includes("router") || l.includes("checking intent") || l.includes("classified as")) return { color: "#42a5f5", icon: "🔀" };
    if (l.includes("web search") || l.includes("scraping") || l.includes("search query")) return { color: "#4fc3f7", icon: "🌐" };
    if (l.includes("verified") || l.includes("complete") || l.includes("done") || l.includes("success")) return { color: "#66bb6a", icon: "✅" };
    if (l.includes("error") || l.includes("fail") || l.includes("fixing")) return { color: "#ef5350", icon: "⚠️" };
    return { color: "#999", icon: "▸" };
  };

  return (
    <div className="thinking-block">
      <div className="thinking-header" onClick={() => setIsOpen(!isOpen)}>
        {isActive ? <div className="thinking-spinner" /> : <span className="thinking-done-icon">✅</span>}
        <span className="thinking-label">
          {isActive ? "Thinking..." : `Thought for ${logs.length} steps`}
        </span>
        <span className={`thinking-chevron ${isOpen ? "open" : ""}`}>▼</span>
      </div>
      {isOpen && (
        <div className="thinking-content">
          {logs.map((log, i) => {
            const style = getLogStyle(log);
            const isClassification = log.toLowerCase().includes("classified as");
            return (
              <div key={i} className={`thinking-step ${isClassification ? "classification" : ""}`}>
                <span className="thinking-step-icon" style={{ color: style.color }}>
                  {style.icon}
                </span>
                <span style={{ color: style.color }}>{log}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default ThinkingBlock;
