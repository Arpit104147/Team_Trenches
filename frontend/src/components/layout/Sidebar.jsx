import React from "react";

/**
 * @component Sidebar
 * Session management sidebar with chat history, memory controls,
 * connection status indicator, and user profile footer.
 */
const Sidebar = ({
  sidebarOpen,
  setSidebarOpen,
  sessions,
  currentSessionId,
  createNewChat,
  loadSession,
  deleteSession,
  handleOffload,
  handleLoadAll,
  isConnected,
  isEvmActive,
  isPreloading,
  setSettingsOpen,
}) => {
  return (
    <div className={`sidebar ${!sidebarOpen ? "closed" : ""}`}>
      <div className="sidebar-top">
        <button className="sidebar-toggle" onClick={() => setSidebarOpen(false)}>☰</button>
        {/* Connection Status Badge */}
        <div className={`connection-badge ${isConnected ? "connected" : "disconnected"}`}>
          <span className="connection-dot"></span>
          <span className="connection-text">{isConnected ? "Online" : "Offline"}</span>
        </div>
      </div>

      <button className="new-chat-btn" onClick={createNewChat}>
        <span>＋</span> New chat
      </button>

      <div className="sidebar-nav">
        <button className="nav-item" onClick={handleOffload}>
          <span className="nav-icon">🧹</span> Offload Memory
        </button>
        <button
          className="nav-item"
          onClick={handleLoadAll}
          disabled={!isConnected || !isEvmActive || isPreloading}
          title={
            !isConnected
              ? "Backend disconnected"
              : !isEvmActive
                ? "EVM mode not active (requires EVM to pre-load)"
                : "Load all models into System RAM"
          }
        >
          <span className="nav-icon">{isPreloading ? "⏳" : "⚡"}</span>
          {isPreloading ? "Loading Swarm..." : "Load All Models"}
        </button>
      </div>

      <div className="sidebar-section-title">Recents</div>
      <div className="history-list">
        {sessions.map((s) => (
          <div
            key={s.id}
            className={`history-item ${s.id === currentSessionId ? "active" : ""}`}
            onClick={() => loadSession(s.id)}
          >
            <span className="history-item-title">💬 {s.title}</span>
            <button className="delete-btn" onClick={(e) => deleteSession(s.id, e)}>✕</button>
          </div>
        ))}
        {sessions.length === 0 && (
          <div className="history-empty">No recent chats.</div>
        )}
      </div>

      <div className="sidebar-footer">
        <div className="user-row">
          <div className="user-avatar">A</div>
          <span className="user-name">ARPIT BEHERA</span>
          <button
            className="sidebar-settings-btn"
            onClick={(e) => { e.stopPropagation(); setSettingsOpen(true); }}
            title="Settings"
          >
            ⚙️
          </button>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
