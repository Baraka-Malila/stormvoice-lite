/* stormVoice Lite — frontend */

const RECORD_MS = 4000;
let isRecording = false;

// ── Loader ────────────────────────────────────────────────────────────────────
document.getElementById("loader").addEventListener("click", () => {
  const loader = document.getElementById("loader");
  loader.classList.add("fade-out");
  setTimeout(() => {
    loader.style.display = "none";
    document.getElementById("app").classList.remove("app-hidden");
    init();
  }, 620);
});

// ── Init ──────────────────────────────────────────────────────────────────────
async function init() {
  try {
    const data = await fetchJSON("/api/status");
    const loaded = data.svm_loaded || data.cnn_loaded;

    if (!loaded) {
      setBadge("no-model", "NO MODEL");
      showView("no-model");
      openPanel();
    } else {
      const parts = [];
      if (data.svm_loaded) parts.push(`SVM`);
      if (data.cnn_loaded) parts.push(`CNN`);
      setBadge("ready", parts.join(" + ") + " READY");

      const chips = document.getElementById("trained-speakers");
      const classes = [...new Set([...data.svm_classes, ...data.cnn_classes])];
      chips.innerHTML = classes.map(c =>
        `<span class="speaker-chip">${c}</span>`
      ).join("");

      showView("idle");
    }
  } catch (e) {
    setBadge("no-model", "SERVER ERROR");
    showView("error");
    document.getElementById("error-text").textContent = e.message;
    openPanel();
  }
  loadHistory();
}

// ── Fulcrum state ─────────────────────────────────────────────────────────────
function setFulcrum(state) {
  document.getElementById("fulcrum-svg").dataset.state = state;
}

// ── Badge ─────────────────────────────────────────────────────────────────────
function setBadge(state, text) {
  const el = document.getElementById("mode-badge");
  el.dataset.state = state;
  el.textContent = text;
}

// ── Panel ─────────────────────────────────────────────────────────────────────
function openPanel()  { document.body.classList.add("panel-pinned"); }
function closePanel() { document.body.classList.remove("panel-pinned"); }
function togglePanel() { document.body.classList.toggle("panel-pinned"); }

document.getElementById("panel-btn").addEventListener("click", togglePanel);
document.getElementById("panel-close").addEventListener("click", closePanel);

// ── Views (inside panel) ──────────────────────────────────────────────────────
const VIEWS = ["no-model", "idle", "results", "error"];

function showView(name) {
  VIEWS.forEach(v => {
    const el = document.getElementById(`view-${v}`);
    if (el) el.style.display = v === name ? "block" : "none";
  });
}

// ── Status line ───────────────────────────────────────────────────────────────
function setStatus(msg) {
  document.getElementById("status-line").textContent = msg;
}

// ── MediaRecorder ─────────────────────────────────────────────────────────────
function recordClip(durationMs) {
  return navigator.mediaDevices.getUserMedia({ audio: true }).then(stream =>
    new Promise((resolve, reject) => {
      const chunks = [];
      const mr = new MediaRecorder(stream, { mimeType: "audio/webm" });
      mr.ondataavailable = e => chunks.push(e.data);
      mr.onstop = () => {
        stream.getTracks().forEach(t => t.stop());
        resolve(new Blob(chunks, { type: "audio/webm" }));
      };
      mr.onerror = reject;
      mr.start();
      setTimeout(() => mr.stop(), durationMs);
    })
  );
}

// ── Mic button ────────────────────────────────────────────────────────────────
document.getElementById("mic-btn").addEventListener("click", async () => {
  if (isRecording) return;
  const badgeEl = document.getElementById("mode-badge");
  if (badgeEl.dataset.state === "no-model") {
    openPanel();
    return;
  }

  isRecording = true;
  document.getElementById("mic-btn").classList.add("recording");
  setFulcrum("listening");
  setStatus(`Recording ${RECORD_MS / 1000} s…`);

  try {
    const blob = await recordClip(RECORD_MS);
    document.getElementById("mic-btn").classList.remove("recording");
    document.getElementById("mic-btn").classList.add("processing");
    setFulcrum("processing");
    setStatus("Analyzing…");

    const fd = new FormData();
    fd.append("audio", blob, "clip.webm");
    const data = await fetchJSON("/api/analyze", { method: "POST", body: fd });

    renderResult(data);
    setFulcrum("idle");
    setStatus("");
    loadHistory();
  } catch (err) {
    showView("error");
    document.getElementById("error-text").textContent = err.message;
    openPanel();
    setFulcrum("idle");
    setStatus("");
  } finally {
    isRecording = false;
    document.getElementById("mic-btn").classList.remove("recording", "processing");
  }
});

// ── Render result ─────────────────────────────────────────────────────────────
function badgeCls(level) {
  return ({
    Low:      "badge-low",
    Medium:   "badge-medium",
    High:     "badge-high",
    Critical: "badge-critical",
  })[level] || "badge-low";
}

function renderResult(data) {
  setText("r-svm-name", data.svm_speaker || "—");
  setText("r-svm-conf", data.svm_confidence != null
    ? `${(data.svm_confidence * 100).toFixed(1)}%` : "—");
  setText("r-cnn-name", data.cnn_speaker || "—");
  setText("r-cnn-conf", data.cnn_confidence != null
    ? `${(data.cnn_confidence * 100).toFixed(1)}%` : "—");
  setText("r-transcript", data.transcript || "(no speech detected)");
  setText("r-category",   data.fraud_category);
  setText("r-action",     data.recommended_action);

  document.getElementById("r-risk").innerHTML =
    `<span class="badge ${badgeCls(data.risk_level)}">${data.risk_level}</span>` +
    `<span class="badge-score">score ${data.risk_score}</span>`;

  const signalsEl = document.getElementById("r-signals");
  const signalsBlock = document.getElementById("r-signals-block");
  if (data.detected_signals && data.detected_signals.length) {
    signalsEl.innerHTML = data.detected_signals
      .map(s => `<span class="signal-tag">${s.keyword} +${s.points}</span>`).join("");
    signalsBlock.style.display = "block";
  } else {
    signalsBlock.style.display = "none";
  }

  const vizBlock = document.getElementById("viz-block");
  if (data.waveform_png_b64) {
    document.getElementById("waveform-img").src = "data:image/png;base64," + data.waveform_png_b64;
    document.getElementById("spectrogram-img").src = "data:image/png;base64," + data.spectrogram_png_b64;
    vizBlock.style.display = "block";
  } else {
    vizBlock.style.display = "none";
  }

  showView("results");
  openPanel();
}

// ── History ───────────────────────────────────────────────────────────────────
async function loadHistory() {
  const tbody = document.getElementById("history-body");
  try {
    const rows = await fetchJSON("/api/sessions");
    if (!rows.length) {
      tbody.innerHTML = `<tr><td colspan="4" class="h-empty">no sessions yet</td></tr>`;
      return;
    }
    tbody.innerHTML = rows.slice(0, 10).map(s => `
      <tr>
        <td>${s.svm_speaker}</td>
        <td>${s.cnn_speaker}</td>
        <td><span class="badge ${badgeCls(s.risk_level)}" style="font-size:9px">${s.risk_level}</span></td>
        <td>${new Date(s.created_at).toLocaleTimeString()}</td>
      </tr>`).join("");
  } catch {
    tbody.innerHTML = `<tr><td colspan="4" class="h-empty">error</td></tr>`;
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

async function fetchJSON(url, opts = {}) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    const msg = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status}: ${msg}`);
  }
  return res.json();
}
