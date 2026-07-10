import React, { useState } from "react";

/**
 * @component SettingsModal
 * Enhanced glassmorphic overlay modal with tabbed navigation for
 * configuring device mode, routing mode, server URL, security,
 * and workspace settings.
 */
const SettingsModal = ({
  settingsOpen,
  setSettingsOpen,
  serverUrl,
  setServerUrl,
  deviceMode,
  setDeviceMode,
  routingMode,
  setRoutingMode,
  contextLength,
  maxTokens,
  temperature,
  searchMode,
}) => {
  const [activeTab, setActiveTab] = useState("general");
  const [githubToken, setGithubToken] = useState(() =>
    localStorage.getItem("github_token") || ""
  );

  if (!settingsOpen) return null;

  const handleSave = () => {
    let finalUrl = serverUrl.trim();
    if (finalUrl && !finalUrl.startsWith("http")) finalUrl = "http://" + finalUrl;
    if (finalUrl.endsWith("/")) finalUrl = finalUrl.slice(0, -1);
    finalUrl = finalUrl.replace("localhost", "127.0.0.1").replace("0.0.0.0", "127.0.0.1");
    localStorage.setItem("server_url", finalUrl);
    localStorage.setItem("routing_mode", routingMode);
    if (githubToken) localStorage.setItem("github_token", githubToken);
    setServerUrl(finalUrl);
    setSettingsOpen(false);

    // Sync settings to backend silently
    fetch(`${finalUrl}/api/settings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        context_length: contextLength,
        max_tokens: maxTokens,
        temperature,
        device_mode: deviceMode,
        gpu_layers: -1,
        search_mode: searchMode,
      }),
    }).catch(() => console.warn("Settings sync to backend deferred — will apply on next request."));
  };

  const handleUrlBlur = (e) => {
    let val = e.target.value.trim();
    if (val && !val.startsWith("http")) val = "http://" + val;
    if (val.endsWith("/")) val = val.slice(0, -1);
    val = val.replace("localhost", "127.0.0.1").replace("0.0.0.0", "127.0.0.1");
    localStorage.setItem("server_url", val);
    setServerUrl(val);
  };

  const tabs = [
    { id: "general", label: "General", icon: "⚙️" },
    { id: "security", label: "Security", icon: "🛡️" },
    { id: "workspace", label: "Workspace", icon: "📁" },
  ];

  return (
    <div className="modal-overlay" onClick={() => setSettingsOpen(false)}>
      <div className="modal settings-modal-enhanced" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header-enhanced">
          <h2>Settings</h2>
          <button className="modal-close-btn" onClick={() => setSettingsOpen(false)}>✕</button>
        </div>

        {/* Tab Navigation */}
        <div className="settings-tabs">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              className={`settings-tab ${activeTab === tab.id ? "active" : ""}`}
              onClick={() => setActiveTab(tab.id)}
            >
              <span className="tab-icon">{tab.icon}</span>
              <span>{tab.label}</span>
            </button>
          ))}
        </div>

        {/* Tab Content */}
        <div className="settings-tab-content">
          {activeTab === "general" && (
            <div className="tab-panel">
              <div className="modal-field">
                <label>Device Mode</label>
                <select value={deviceMode} onChange={(e) => setDeviceMode(e.target.value)}>
                  <option value="gpu">GPU (CUDA / Vulkan)</option>
                  <option value="cpu">CPU Only</option>
                  <option value="hybrid">Hybrid (CPU + GPU)</option>
                </select>
              </div>

              <div className="modal-field">
                <label>Routing Mode</label>
                <select value={routingMode} onChange={(e) => setRoutingMode(e.target.value)}>
                  <option value="auto">Auto (Smart Router)</option>
                  <option value="reasoning">Reasoning (DeepSeek Math/Theory)</option>
                  <option value="coding">Coding (Actor-Critic Sandbox)</option>
                  <option value="simple">Simple (Direct Response)</option>
                  <option value="chip_design">Chip Design (EDA Sandbox)</option>
                </select>
              </div>

              <div className="modal-field">
                <label>Server URL</label>
                <input
                  type="text"
                  value={serverUrl}
                  onChange={(e) => setServerUrl(e.target.value)}
                  onBlur={handleUrlBlur}
                  placeholder="http://127.0.0.1:8000"
                />
              </div>
            </div>
          )}

          {activeTab === "security" && (
            <div className="tab-panel">
              <div className="security-status-card">
                <div className="security-item">
                  <span className="security-icon">🛡️</span>
                  <span className="security-label">SAST Code Scanning</span>
                  <span className="security-badge active">Active</span>
                </div>
                <div className="security-item">
                  <span className="security-icon">🔒</span>
                  <span className="security-label">Sandbox Isolation</span>
                  <span className="security-badge active">Active</span>
                </div>
                <div className="security-item">
                  <span className="security-icon">🌐</span>
                  <span className="security-label">Air-Gap Mode</span>
                  <span className="security-badge">Set via ENV</span>
                </div>
              </div>
              <p className="settings-hint">
                Security features are automatically enabled. Air-gap mode can be activated
                by setting <code>AIOS_AIR_GAP=1</code> environment variable before starting the server.
              </p>
            </div>
          )}

          {activeTab === "workspace" && (
            <div className="tab-panel">
              <div className="modal-field">
                <label>GitHub Token (optional)</label>
                <input
                  type="password"
                  value={githubToken}
                  onChange={(e) => setGithubToken(e.target.value)}
                  placeholder="ghp_xxxxxxxxxxxx"
                />
                <span className="field-hint">Used for automated PR creation. Stored locally only.</span>
              </div>
              <p className="settings-hint">
                Git workspace operations allow the AIOS to clone repositories, create branches,
                and commit generated code (Verilog, testbenches, layouts) directly.
              </p>
            </div>
          )}
        </div>

        <div className="modal-actions">
          <button onClick={() => setSettingsOpen(false)}>Close</button>
          <button className="primary-btn" onClick={handleSave}>Save</button>
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;
