// SentinelNet Background Service Worker v5.0
// Handles: API proxy for popup + email scanning from content.js

const BASE = 'http://localhost:8000';

async function apiFetch(path, params = {}) {
  const url = new URL(BASE + path);
  Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, v));
  const r = await fetch(url.toString());
  return r.json();
}

async function apiPost(path, body, key) {
  const r = await fetch(BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-SentinelNet-Key': key || '' },
    body: JSON.stringify(body)
  });
  return r.json();
}

// ── Message listener ──────────────────────────────────────
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {

  // Popup API proxy
  if (msg.type === 'API_CALL') {
    apiFetch(msg.path, msg.query || {})
      .then(data => sendResponse({ success: true, data }))
      .catch(err  => sendResponse({ success: false, error: err.message }));
    return true;
  }

  // Save key from popup
  if (msg.type === 'SAVE_KEY') {
    chrome.storage.local.set({ sn_key: msg.key });
    apiFetch('/api/ext/verify', { key: msg.key })
      .then(d => {
        updateBadge(d.valid ? 'connected' : 'error');
        sendResponse({ success: d.valid });
      })
      .catch(() => { updateBadge('disconnected'); sendResponse({ success: false }); });
    return true;
  }

  // Clear key from popup
  if (msg.type === 'CLEAR_KEY') {
    chrome.storage.local.remove(['sn_key']);
    updateBadge('disconnected');
    sendResponse({ success: true });
    return false;
  }

  // ── Email scan from content.js ────────────────────────
  if (msg.type === 'SCAN_EMAIL') {
    chrome.storage.local.get(['sn_key'], async data => {
      const key = data.sn_key;
      if (!key) {
        sendResponse({ severity: 'UNKNOWN', verdict: 'Not connected', confidence: 0, agents: {} });
        return;
      }
      try {
        const result = await apiPost('/api/extension/scan-email', msg.payload, key);
        sendResponse(result);
      } catch (e) {
        sendResponse({ severity: 'UNKNOWN', verdict: 'Scan failed', confidence: 0, agents: {} });
      }
    });
    return true; // async
  }
});

// ── Badge ─────────────────────────────────────────────────
function updateBadge(status) {
  const map = {
    connected:    ['ON',  '#00cc66'],
    disconnected: ['OFF', '#555555'],
    error:        ['ERR', '#ff4444'],
  };
  const [text, color] = map[status] || map.disconnected;
  chrome.action.setBadgeText({ text });
  chrome.action.setBadgeBackgroundColor({ color });
}

// ── Startup ───────────────────────────────────────────────
chrome.runtime.onInstalled.addListener(() => updateBadge('disconnected'));

chrome.storage.local.get(['sn_key'], data => {
  if (!data.sn_key) return updateBadge('disconnected');
  apiFetch('/api/ext/verify', { key: data.sn_key })
    .then(d => updateBadge(d.valid ? 'connected' : 'disconnected'))
    .catch(() => updateBadge('disconnected'));
});
