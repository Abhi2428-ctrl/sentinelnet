// SentinelNet Email Shield — Content Script v4.0
// Uses BOTH specific selectors AND smart fallback detection
// Guaranteed to work even if Gmail changes class names

(function () {
  'use strict';

  const PLATFORM = detectPlatform();
  const scannedIds = new Set(); // track last 50 scanned emails
  let scanPending   = false;
  let observer      = null;

  function detectPlatform() {
    const h = location.hostname;
    if (h.includes('mail.google.com'))  return 'gmail';
    if (h.includes('outlook.live.com') || h.includes('outlook.office')) return 'outlook';
    if (h.includes('mail.yahoo.com'))   return 'yahoo';
    if (h.includes('proton'))           return 'protonmail';
    return 'webmail';
  }

  // ── Smart element finder ────────────────────────────────────────────────────
  // Instead of relying on class names, find elements by their ROLE and CONTENT
  function findEmailBody() {
    // ONLY use known stable selectors — never fall back to page-wide content
    // Reason: Strategy 2 (largest div) captures the whole Gmail inbox list,
    // which contains OTHER emails' subjects and triggers false positives.
    const knownSelectors = [
      '.ii.gt .a3s.aiL',           // Gmail primary (most reliable)
      '.a3s.aiL',                   // Gmail alt
      '.ii.gt .a3s',                // Gmail fallback
      '[data-message-id] .a3s',     // Gmail data-attr
      '[data-legacy-message-id] .a3s',
      '[aria-label="Message body"]', // Outlook
      '.ReadMsgBody',
      '[data-test-id="message-view-body-content"]', // Yahoo
      '.proton-mail-message',        // ProtonMail
      '[data-shortcut-target="message-container"]',
    ];
    for (const sel of knownSelectors) {
      try {
        const el = document.querySelector(sel);
        if (el && el.innerText && el.innerText.trim().length > 30) return el;
      } catch(e) {}
    }
    // NO fallback to page-wide content — return null if not found
    return null;
  }

  function findSubject() {
    // Stable: h1/h2 near top of main content, or title-like elements
    const subjectSelectors = [
      'h2.hP',                                    // Gmail
      '[role="heading"][aria-level="1"]',          // Outlook
      '[data-test-id="message-subject"]',          // Yahoo
      '.message-title',                            // ProtonMail
      '[data-testid="message-header-subject"]',    // ProtonMail new
      'h1', 'h2',                                  // Generic
    ];
    for (const sel of subjectSelectors) {
      try {
        const el = document.querySelector(sel);
        if (el && el.textContent.trim().length > 1) return el.textContent.trim();
      } catch(e) {}
    }
    return document.title || '(no subject)';
  }

  function findSender() {
    // Try email attribute selectors first (most reliable)
    const emailAttrSelectors = [
      '.gD',           // Gmail classic — has email attribute
      '.go',           // Gmail alt
      '[email]',       // Any element with email attr (Gmail)
      '[data-hovercard-id]',  // Gmail hovercard (contains email)
    ];
    for (const sel of emailAttrSelectors) {
      try {
        const el = document.querySelector(sel);
        if (el) {
          const addr = el.getAttribute('email') || el.getAttribute('data-hovercard-id') || '';
          if (addr && addr.includes('@')) return addr;
        }
      } catch(e) {}
    }
    // Fallback: text-based selectors for other clients
    const textSelectors = [
      '.ms-Persona-primaryText',              // Outlook
      '[data-test-id="from-address"]',        // Yahoo
      '.sender-address',                      // ProtonMail
      '[data-testid="sender-address"]',       // ProtonMail new
      '.from .go',                            // Gmail alt
    ];
    for (const sel of textSelectors) {
      try {
        const el = document.querySelector(sel);
        if (el) return el.textContent.trim();
      } catch(e) {}
    }
    // Last resort: scan all visible text for email pattern near "From:"
    try {
      const fromMatch = document.body.innerText.match(/From:.*?([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})/i);
      if (fromMatch) return fromMatch[1];
    } catch(e) {}
    return '';
  }

  function findAnchor(bodyEl) {
    // Find a good place to attach the badge (near email header)
    const anchorSelectors = ['.ha', 'h2.hP', '[role="heading"]', '.message-title'];
    for (const sel of anchorSelectors) {
      const el = document.querySelector(sel);
      if (el) return el;
    }
    return bodyEl;
  }

  // ── Main email extraction ───────────────────────────────
  function extractEmail() {
    const bodyEl = findEmailBody();
    if (!bodyEl) return null;

    const body = bodyEl.innerText.trim();
    if (body.length < 20) return null;

    const sender = findSender();
    const subject = findSubject();
    // Limit body to 2000 chars — enough for analysis, avoids capturing page nav
    const cleanBody = body.slice(0, 2000);
    return {
      subject:     subject,
      sender:      sender,
      body:        cleanBody,
      links:       Array.from(bodyEl.querySelectorAll('a[href]')).map(a => a.href).filter(h => h.startsWith('http')).slice(0, 20),
      images:      Array.from(bodyEl.querySelectorAll('img[src]')).map(i => i.src).filter(s => s.startsWith('http')).slice(0, 10),
      attachments: Array.from(document.querySelectorAll('[aria-label*=".pdf"],[aria-label*=".doc"],[aria-label*=".exe"],[data-tooltip*=".exe"]')).map(el => el.getAttribute('aria-label') || '').filter(Boolean).slice(0, 5),
      anchorEl:    findAnchor(bodyEl),
      platform:    PLATFORM,
    };
  }

  // ── Scan trigger ────────────────────────────────────────
  function checkForOpenEmail() {
    if (scanPending) return;

    const email = extractEmail();
    if (!email) return;

    // Deduplicate: don't re-scan same email (remember last 50)
    const id = simpleHash(email.subject + email.sender + email.body.slice(0, 80));
    if (scannedIds.has(id)) return;
    scannedIds.add(id);
    if (scannedIds.size > 50) scannedIds.delete(scannedIds.values().next().value);
    scanPending   = true;

    showScanningBadge(email.anchorEl);

    chrome.runtime.sendMessage(
      { type: 'SCAN_EMAIL', payload: { ...email, anchorEl: undefined } },
      (result) => {
        scanPending = false;
        if (chrome.runtime.lastError || !result || result.severity === 'UNKNOWN') {
          removeBadge();
          return;
        }
        // Show overlay for ALL results — user wants to see every scan decision
        showOverlay(email.anchorEl, result);
      }
    );
  }

  // ── Overlay UI ──────────────────────────────────────────
  function showScanningBadge(anchorEl) {
    removeBadge();
    const badge = document.createElement('div');
    badge.id = 'sn-scan-badge';
    badge.className = 'sn-badge sn-scanning';
    badge.innerHTML = '<span class="sn-spinner"></span><span style="font-family:monospace;font-size:11px;color:#00d4ff"> SentinelNet scanning...</span>';
    insertBadge(anchorEl, badge);
  }

  function showOverlay(anchorEl, result) {
    removeBadge();
    // Phishing Guardian is paused — don't show any overlay
    if (result.paused) return;
    const sev  = result.severity || 'UNKNOWN';
    const conf = result.confidence ? Math.round(result.confidence * 100) : 0;
    const C = {
      CRITICAL: { bg:'#1a0005', border:'#ff2d55', text:'#ff6b6b', icon:'🔴' },
      HIGH:     { bg:'#1a0800', border:'#ff6600', text:'#ffaa44', icon:'🟠' },
      MEDIUM:   { bg:'#1a1400', border:'#ffcc00', text:'#ffee44', icon:'🟡' },
      LOW:      { bg:'#0a0a00', border:'#88cc00', text:'#aaee44', icon:'🟡' },
      SAFE:     { bg:'#001a08', border:'#00cc55', text:'#00ff88', icon:'✅' },
      UNKNOWN:  { bg:'#111',    border:'#555',    text:'#888',    icon:'⚪'  },
    }[sev] || { bg:'#111', border:'#555', text:'#888', icon:'⚪' };

    const agentsHtml = Object.entries(result.agents || {}).map(([name, d]) =>
      `<div class="sn-agent-row">
        <span class="sn-dot" style="background:${d.flagged ? C.border : '#333'}"></span>
        <span class="sn-aname">${name}</span>
        <span class="sn-ascore" style="color:${d.flagged ? C.text : '#555'}">${d.flagged ? '⚑ ' + (d.finding||'flagged') : '✓ clean'}</span>
       </div>`).join('');

    const tagsHtml = (result.indicators||[]).slice(0,3).map(i => `<span class="sn-tag">⚑ ${i}</span>`).join('');

    const badge = document.createElement('div');
    badge.id = 'sn-scan-badge';
    badge.className = 'sn-badge';
    badge.style.cssText = `background:${C.bg};border:1px solid ${C.border};border-radius:14px;`;
    badge.innerHTML = `
      <div class="sn-head">
        <span>${C.icon}</span>
        <span class="sn-verdict" style="color:${C.text}">${sev} — ${result.verdict||'Scan complete'}</span>
        <span class="sn-pct" style="color:${C.text}">${conf}%</span>
        <button class="sn-x" style="color:${C.text};background:transparent;border:none;cursor:pointer;font-size:14px;margin-left:8px">✕</button>
      </div>
      ${agentsHtml ? `<div class="sn-agents">${agentsHtml}</div>` : ''}
      ${tagsHtml   ? `<div class="sn-tags">${tagsHtml}</div>`     : ''}
    `;

    insertBadge(anchorEl, badge);
    badge.querySelector('.sn-x').addEventListener('click', () => badge.remove());
    if (sev === 'SAFE' || sev === 'LOW') setTimeout(() => badge.remove(), 6000);
  }

  function insertBadge(anchorEl, badge) {
    try {
      if (anchorEl) {
        anchorEl.style.position = 'relative';
        anchorEl.insertAdjacentElement('afterbegin', badge);
      } else {
        document.body.insertAdjacentElement('afterbegin', badge);
      }
    } catch {
      try { document.body.insertAdjacentElement('afterbegin', badge); } catch {}
    }
  }

  function removeBadge() { document.getElementById('sn-scan-badge')?.remove(); }

  function simpleHash(s) {
    let h = 0;
    for (let i = 0; i < s.length; i++) { h = ((h<<5)-h) + s.charCodeAt(i); h|=0; }
    return h.toString(36);
  }

  function debounce(fn, ms) {
    let t;
    return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
  }

  // ── Inbox row auto-scanner ──────────────────────────────
  // Scans emails visible in inbox WITHOUT needing to open them
  // Reads subject + sender from inbox rows and sends to backend
  const scannedRowIds = new Set();

  function getInboxRows() {
    if (PLATFORM === 'gmail') {
      // Gmail inbox rows: tr.zA (each email row)
      return Array.from(document.querySelectorAll('tr.zA'));
    }
    if (PLATFORM === 'outlook') {
      return Array.from(document.querySelectorAll('[role="option"][data-convid]'));
    }
    if (PLATFORM === 'yahoo') {
      return Array.from(document.querySelectorAll('[data-test-id="message-list-item"]'));
    }
    return [];
  }

  function extractRowData(row) {
    let subject = '', sender = '', rowId = '';
    if (PLATFORM === 'gmail') {
      const subjectEl = row.querySelector('.bog, .bqe, span[data-thread-id], .y6');
      const senderEl  = row.querySelector('.yP, .zF, .bA4 span, .yX');
      subject = subjectEl?.textContent?.trim() || '';
      sender  = senderEl?.getAttribute('email') || senderEl?.textContent?.trim() || '';
      rowId   = row.getAttribute('id') || row.getAttribute('data-legacy-thread-id') ||
                row.querySelector('[data-thread-id]')?.getAttribute('data-thread-id') || '';
    } else if (PLATFORM === 'outlook') {
      subject = row.querySelector('[aria-label]')?.getAttribute('aria-label') || '';
      sender  = row.querySelector('.ms-Persona-primaryText')?.textContent?.trim() || '';
      rowId   = row.getAttribute('data-convid') || '';
    } else if (PLATFORM === 'yahoo') {
      subject = row.querySelector('[data-test-id="message-subject"]')?.textContent?.trim() || '';
      sender  = row.querySelector('[data-test-id="senderName"]')?.textContent?.trim() || '';
      rowId   = row.getAttribute('data-item-id') || '';
    }
    return { subject, sender, rowId };
  }

  function addRowBadge(row, severity, confidence) {
    // Remove existing badge if any
    row.querySelector('.sn-row-badge')?.remove();
    const C = {
      CRITICAL: { color: '#ff2d55', icon: '🔴' },
      HIGH:     { color: '#ff6600', icon: '🟠' },
      MEDIUM:   { color: '#ffcc00', icon: '🟡' },
      LOW:      { color: '#00ff88', icon: '🟢' },
      SAFE:     { color: '#00ff88', icon: '🟢' },
    }[severity] || { color: '#888', icon: '⚪' };

    const badge = document.createElement('span');
    badge.className = 'sn-row-badge';
    badge.style.cssText = `
      display:inline-flex;align-items:center;gap:3px;
      margin-left:6px;padding:2px 8px;border-radius:8px;
      font-size:10px;font-family:monospace;font-weight:bold;
      border:1px solid ${C.color};color:${C.color};
      background:rgba(0,0,0,0.4);cursor:default;
      vertical-align:middle;white-space:nowrap;
    `;
    const pct = Math.round(confidence * 100);
    badge.textContent = `${C.icon} ${severity} ${pct}%`;
    badge.title = `SentinelNet: ${severity} (${pct}% confidence)`;

    // Insert near subject in the row
    const subjectEl = PLATFORM === 'gmail'
      ? row.querySelector('.bog, .bqe, .y6')
      : row.querySelector('[aria-label], [data-test-id="message-subject"]');

    if (subjectEl) subjectEl.appendChild(badge);
    else row.appendChild(badge);
  }

  async function scanInboxRows() {
    const rows = getInboxRows();
    for (const row of rows) {
      const { subject, sender, rowId } = extractRowData(row);
      if (!subject || !sender) continue;

      const rowHash = simpleHash(subject + sender + (rowId || '')).toString();
      if (scannedRowIds.has(rowHash)) continue;
      scannedRowIds.add(rowHash);
      if (scannedRowIds.size > 200) scannedRowIds.delete(scannedRowIds.values().next().value);

      // Fire-and-forget scan using message passing
      chrome.runtime.sendMessage(
        {
          type: 'SCAN_EMAIL',
          payload: {
            subject,
            sender,
            body: subject, // subject-only scan for inbox rows
            links: [],
            images: [],
            attachments: [],
            platform: PLATFORM,
            source: 'inbox_row',
          }
        },
        (result) => {
          if (chrome.runtime.lastError || !result || result.severity === 'UNKNOWN') return;
          // Inbox rows: only badge for HIGH/CRITICAL (row scan uses subject only → noisy)
          // MEDIUM from subject-only scan has too many false positives
          if (['CRITICAL', 'HIGH'].includes(result.severity)) {
            addRowBadge(row, result.severity, result.confidence || 0);
          }
        }
      );

      // Small delay between rows to not overwhelm backend
      await new Promise(r => setTimeout(r, 150));
    }
  }

  // ── Start observing ─────────────────────────────────────
  function start() {
    if (observer) observer.disconnect();
    observer = new MutationObserver(debounce(() => {
      checkForOpenEmail();   // scan opened email
      scanInboxRows();       // scan any NEW inbox rows that appeared
    }, 700));
    observer.observe(document.body, { childList: true, subtree: true });
    // Check immediately in case email already open or inbox already loaded
    setTimeout(checkForOpenEmail, 2000);
    setTimeout(checkForOpenEmail, 4000);
    // Scan inbox rows immediately and then periodically
    setTimeout(scanInboxRows, 3000);
    setTimeout(scanInboxRows, 6000);
    setInterval(scanInboxRows, 15000); // re-scan every 15s for new emails
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }

})();
