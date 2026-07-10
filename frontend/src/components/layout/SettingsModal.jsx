import React from "react";

/**
 * @component SettingsModal
 * Glassmorphic overlay modal for configuring device mode,
 * routing mode, and server URL.
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
  if (!settingsOpen) return null;

  const handleSave = () => {
    let finalUrl = serverUrl.trim();
    if (finalUrl && !finalUrl.startsWith("http")) finalUrl = "http://" + finalUrl;
    if (finalUrl.endsWith("/")) finalUrl = finalUrl.slice(0, -1);
    finalUrl = finalUrl.replace("localhost", "127.0.0.1").replace("0.0.0.0", "127.0.0.1");
    localStorage.setItem("server_url", finalUrl);
    localStorage.setItem("routing_mode", routingMode);
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

  return (
    <div className="modal-overlay" onClick={() => setSettingsOpen(false)}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h2>Settings</h2>

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
          />
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
