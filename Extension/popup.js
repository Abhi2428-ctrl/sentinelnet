// SentinelNet Popup v5.0 - direct fetch() to localhost
// Works because: CORS=allow_origins["*"] in backend + host_permissions in manifest

const $ = id => document.getElementById(id);
const BASE = 'http://localhost:8000';

document.addEventListener('DOMContentLoaded', () => {
  // Wire buttons
  $('btn-connect').addEventListener('click', connectKey);
  $('btn-disconnect').addEventListener('click', disconnect);
  $('btn-dashboard').addEventListener('click', () => chrome.tabs.create({ url: BASE }));
  $('key-input').addEventListener('keydown', e => { if (e.key === 'Enter') connectKey(); });

  // Auto-format key
  $('key-input').addEventListener('input', e => {
    let v = e.target.value.replace(/[^A-Za-z0-9]/g, '').toUpperCase();
    const p = [];
    if (v.length > 0)  p.push(v.slice(0, 3));
    if (v.length > 3)  p.push(v.slice(3, 7));
    if (v.length > 7)  p.push(v.slice(7, 11));
    if (v.length > 11) p.push(v.slice(11, 15));
    if (v.length > 15) p.push(v.slice(15, 19));
    e.target.value = p.join('-');
  });

  // Restore saved key
  chrome.storage.local.get(['sn_key'], d => {
    if (d.sn_key) { $('key-input').value = d.sn_key; checkStatus(d.sn_key); }
    else setUI('disconnected');
  });
});

async function checkStatus(key) {
  setUI('checking');
  try {
    const r = await fetch(`${BASE}/api/ext/verify?key=${encodeURIComponent(key)}`);
    const d = await r.json();
    if (d.valid) { setUI('connected', maskKey(key)); loadStats(key); }
    else setUI('invalid_key');
  } catch { setUI('error'); }
}

async function connectKey() {
  const key = $('key-input').value.trim();
  if (!key || key.length < 6) { showMsg('err', 'Enter your Config Key first'); return; }

  $('btn-connect').textContent = 'CONNECTING...';
  $('btn-connect').disabled = true;

  try {
    // Test server alive
    const ping = await fetch(`${BASE}/api/ext/ping`);
    await ping.json();
  } catch {
    showMsg('err', 'Cannot reach SentinelNet\nMake sure it is running');
    setUI('error');
    resetBtn(); return;
  }

  try {
    const r = await fetch(`${BASE}/api/ext/verify?key=${encodeURIComponent(key)}`);
    const d = await r.json();
    if (d.valid) {
      chrome.storage.local.set({ sn_key: key });
      chrome.runtime.sendMessage({ type: 'SAVE_KEY', key }, () => {});
      setUI('connected', maskKey(key));
      showMsg('ok', 'Connected!');
      loadStats(key);
    } else {
      showMsg('err', 'Key rejected - generate new key in dashboard');
      setUI('error');
    }
  } catch {
    showMsg('err', 'Connection failed');
    setUI('error');
  }
  resetBtn();
}

function resetBtn() {
  $('btn-connect').textContent = 'CONNECT TO SENTINELNET';
  $('btn-connect').disabled = false;
}

function disconnect() {
  chrome.storage.local.remove(['sn_key']);
  chrome.runtime.sendMessage({ type: 'CLEAR_KEY' }, () => {});
  $('key-input').value = '';
  setUI('disconnected');
}

async function loadStats(key) {
  try {
    const r = await fetch(`${BASE}/api/ext/stats-jsonp?key=${encodeURIComponent(key)}`);
    const d = await r.json();
    if (d.scanned !== undefined) {
      $('s-scanned').textContent = d.scanned || 0;
      $('s-threats').textContent = d.threats || 0;
      $('s-blocked').textContent = d.blocked || 0;
      $('s-safe').textContent    = d.safe    || 0;
      $('section-stats').classList.remove('hidden');
    }
  } catch {}
}

function setUI(status, maskedKey) {
  const S = {
    connected:    { cls:'connected',    color:'#00ff88', text:'● CONNECTED',      sub:'Scanning emails in real time',        conn:true  },
    checking:     { cls:'checking',     color:'#ffcc00', text:'◌ CHECKING...',    sub:'Verifying key...',                    conn:false },
    invalid_key:  { cls:'error',        color:'#ff2d55', text:'✗ INVALID KEY',    sub:'Generate new key in dashboard',       conn:false },
    error:        { cls:'error',        color:'#ff6600', text:'✗ CANNOT CONNECT', sub:'Is SentinelNet running on port 8000?',conn:false },
    disconnected: { cls:'disconnected', color:'#888',    text:'NOT CONNECTED',    sub:'Enter your Config Key to activate',   conn:false },
  };
  const s = S[status] || S.disconnected;
  $('dot').className = 'dot ' + s.cls;
  $('status-main').textContent = s.text;
  $('status-main').style.color = s.color;
  $('status-sub').textContent  = s.sub;
  $('view-connected').classList.toggle('hidden', !s.conn);
  $('view-disconnected').classList.toggle('hidden', s.conn);
  if (!s.conn && status !== 'checking') $('section-stats').classList.add('hidden');
  if (s.conn && maskedKey) $('key-masked').textContent = maskedKey;
}

function maskKey(k) {
  const p = k.split('-');
  return p.length >= 4 ? `${p[0]}-****-****-${p[p.length-1]}` : k.slice(0,4)+'****'+k.slice(-4);
}

function showMsg(type, text) {
  const el = $('result');
  el.className = 'result ' + type;
  el.textContent = text;
  el.style.display = 'block';
  if (type === 'ok') setTimeout(() => el.style.display = 'none', 2000);
}
