/* stormVoice Lite */

const ENROLL_MS        = 4000;
const MAX_ENROLL_CLIPS = 10;
const CHUNK_MS         = 4000; // continuous analysis window size

let modelReady   = false;
let enrollBlobs  = [];

// ── Continuous recording state ────────────────────────────────────────────────
let contRecorder = null;
let contStream   = null;
let contChunk    = 0;
let contActive   = false;

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
  document.getElementById("mic-btn").disabled = true;
  try {
    const data = await fetchJSON("/api/status");
    modelReady = data.svm_loaded || data.cnn_loaded;

    if (!modelReady) {
      setBadge("no-model", "NO MODEL");
      switchView("no-model");
      openPanel();
    } else {
      const parts = [];
      if (data.svm_loaded) parts.push("SVM");
      if (data.cnn_loaded) parts.push("CNN");
      setBadge("ready", parts.join("+") + " READY");
      const classes = [...new Set([...data.svm_classes, ...data.cnn_classes])];
      renderSpeakerChips(classes);
      switchView("idle");
    }
  } catch (e) {
    setBadge("error", "SERVER ERROR");
    switchView("error");
    document.getElementById("error-text").textContent =
      "Cannot reach server: " + e.message;
    openPanel();
  } finally {
    document.getElementById("mic-btn").disabled = false;
  }
  refreshEnrolledList();
  loadHistory();
}

// ── Fulcrum ───────────────────────────────────────────────────────────────────
function setFulcrum(state) {
  document.getElementById("fulcrum-svg").dataset.state = state;
}

// ── Status badge ──────────────────────────────────────────────────────────────
function setBadge(state, text) {
  const el = document.getElementById("mode-badge");
  el.dataset.state = state;
  el.textContent   = text;
}

// ── Panel ─────────────────────────────────────────────────────────────────────
function openPanel()  { document.body.classList.add("panel-pinned"); }
function closePanel() { document.body.classList.remove("panel-pinned"); }

document.getElementById("panel-btn").addEventListener("click", () =>
  document.body.classList.toggle("panel-pinned")
);
document.getElementById("panel-close").addEventListener("click", closePanel);

// ── Tabs ──────────────────────────────────────────────────────────────────────
document.querySelectorAll(".panel-tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".panel-tab").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
    openPanel();
    if (btn.dataset.tab === "history") loadHistory();
    if (btn.dataset.tab === "enroll")  refreshEnrolledList();
  });
});

function switchTab(name) {
  document.querySelectorAll(".panel-tab").forEach(b => b.classList.remove("active"));
  document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
  document.querySelector(`.panel-tab[data-tab="${name}"]`).classList.add("active");
  document.getElementById(`tab-${name}`).classList.add("active");
}

// ── Analyze views ─────────────────────────────────────────────────────────────
const VIEWS = ["no-model", "idle", "results", "error"];

function switchView(name) {
  VIEWS.forEach(v => {
    const el = document.getElementById(`view-${v}`);
    if (el) el.style.display = v === name ? "block" : "none";
  });
}

function renderSpeakerChips(classes) {
  document.getElementById("trained-speakers").innerHTML =
    classes.length
      ? classes.map(c => `<span class="speaker-chip">${c}</span>`).join("")
      : `<span class="view-body">none yet</span>`;
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

// ── Continuous recording ──────────────────────────────────────────────────────
const micBtn = document.getElementById("mic-btn");

micBtn.addEventListener("click", () => {
  if (!modelReady) { switchTab("enroll"); openPanel(); return; }
  contActive ? stopContinuous() : startContinuous();
});

async function startContinuous() {
  try {
    contStream   = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    switchView("error");
    document.getElementById("error-text").textContent = "Mic access denied: " + err.message;
    switchTab("analyze"); openPanel();
    return;
  }

  contActive = true;
  contChunk  = 0;
  micBtn.classList.add("recording");
  setFulcrum("listening");
  setStatus("Listening — click mic to stop");

  contRecorder = new MediaRecorder(contStream, { mimeType: "audio/webm" });

  contRecorder.ondataavailable = async (e) => {
    if (!e.data || e.data.size < 500) return; // skip empty/silent tail chunks
    contChunk++;
    const n = contChunk;

    // Brief processing flash — don't change if already stopped
    if (contActive) { micBtn.classList.replace("recording", "processing"); setFulcrum("processing"); }
    setStatus(`Chunk ${n} — analyzing…`);

    const fd = new FormData();
    fd.append("audio", e.data, `chunk${n}.webm`);
    try {
      const data = await fetchJSON("/api/analyze", { method: "POST", body: fd });
      renderResult(data);
      loadHistory();
    } catch (err) {
      // Show error but keep recording — one bad chunk shouldn't kill the session
      document.getElementById("error-text").textContent = `Chunk ${n}: ${err.message}`;
      switchView("error"); switchTab("analyze"); openPanel();
    }

    if (contActive) {
      micBtn.classList.replace("processing", "recording");
      setFulcrum("listening");
      setStatus(`Chunk ${n} done — listening (click mic to stop)`);
    }
  };

  contRecorder.onstop = () => {
    contStream.getTracks().forEach(t => t.stop());
    contStream   = null;
    contRecorder = null;
    contActive   = false;
    micBtn.classList.remove("recording", "processing");
    setFulcrum("idle");
    setStatus("");
  };

  // Fire ondataavailable every CHUNK_MS while still recording
  contRecorder.start(CHUNK_MS);
}

function stopContinuous() {
  contActive = false;
  setStatus("Stopping…");
  if (contRecorder && contRecorder.state === "recording") contRecorder.stop();
}

// ── Render analysis result ────────────────────────────────────────────────────
function badgeCls(level) {
  return ({ Low: "badge-low", Medium: "badge-medium",
            High: "badge-high", Critical: "badge-critical" })[level] || "badge-low";
}

function renderResult(data) {
  setText("r-svm-name", data.svm_speaker || "—");
  setText("r-svm-conf", data.svm_confidence != null
    ? `${(data.svm_confidence * 100).toFixed(1)}%` : "—");
  setText("r-cnn-name", data.cnn_speaker || "—");
  setText("r-cnn-conf", data.cnn_confidence != null
    ? `${(data.cnn_confidence * 100).toFixed(1)}%` : "—");
  setText("r-transcript", data.transcript || "(no speech detected)");
  setText("r-category", data.fraud_category);
  setText("r-action",   data.recommended_action);

  document.getElementById("r-risk").innerHTML =
    `<span class="badge ${badgeCls(data.risk_level)}">${data.risk_level}</span>` +
    `<span class="badge-score">score ${data.risk_score}</span>`;

  const signalsBlock = document.getElementById("r-signals-block");
  if (data.detected_signals && data.detected_signals.length) {
    document.getElementById("r-signals").innerHTML = data.detected_signals
      .map(s => `<span class="signal-tag">${s.keyword} +${s.points}</span>`).join("");
    signalsBlock.style.display = "block";
  } else {
    signalsBlock.style.display = "none";
  }

  const vizBlock = document.getElementById("viz-block");
  if (data.waveform_png_b64) {
    document.getElementById("waveform-img").src    = "data:image/png;base64," + data.waveform_png_b64;
    document.getElementById("spectrogram-img").src = "data:image/png;base64," + data.spectrogram_png_b64;
    vizBlock.style.display = "block";
  } else {
    vizBlock.style.display = "none";
  }

  switchView("results");
  switchTab("analyze");
  openPanel();
}

// ── Enroll tab ────────────────────────────────────────────────────────────────
const enrollRecordBtn = document.getElementById("enroll-record-btn");
const enrollSubmitBtn = document.getElementById("enroll-submit-btn");
const enrollClearBtn  = document.getElementById("enroll-clear-btn");
const enrollStatus    = document.getElementById("enroll-status");
const clipsList       = document.getElementById("clips-list");
const clipCounter     = document.getElementById("clip-counter");

enrollRecordBtn.addEventListener("click", async () => {
  if (enrollRecordBtn.classList.contains("recording")) return;
  if (enrollBlobs.length >= MAX_ENROLL_CLIPS) {
    setEnrollStatus(`Max ${MAX_ENROLL_CLIPS} clips reached.`, "error");
    return;
  }

  enrollRecordBtn.classList.add("recording");
  setEnrollStatus(`Recording clip ${enrollBlobs.length + 1}…`);
  try {
    const blob = await recordClip(ENROLL_MS);
    enrollBlobs.push(blob);
    updateClipList();
    setEnrollStatus(
      enrollBlobs.length < 3
        ? `${enrollBlobs.length} clip(s) — record at least 3 for good accuracy`
        : `${enrollBlobs.length} clip(s) — ready to enroll`
    );
    enrollSubmitBtn.disabled = false;
  } catch (err) {
    setEnrollStatus("Mic error: " + err.message, "error");
  } finally {
    enrollRecordBtn.classList.remove("recording");
  }
});

function updateClipList() {
  clipsList.innerHTML = enrollBlobs.map((_, i) =>
    `<span class="clip-tag">clip ${i + 1}</span>`
  ).join("");
  clipCounter.textContent = `(${enrollBlobs.length} recorded)`;
}

enrollClearBtn.addEventListener("click", () => {
  enrollBlobs = [];
  updateClipList();
  enrollSubmitBtn.disabled = true;
  setEnrollStatus("");
});

enrollSubmitBtn.addEventListener("click", async () => {
  const name = document.getElementById("enroll-name").value.trim();
  if (!name) { setEnrollStatus("Enter a speaker name first.", "error"); return; }
  if (!enrollBlobs.length) { setEnrollStatus("Record at least one clip.", "error"); return; }

  enrollSubmitBtn.disabled = true;
  setEnrollStatus("Enrolling and retraining SVM…");
  setFulcrum("processing");

  const fd = new FormData();
  fd.append("name", name);
  enrollBlobs.forEach((b, i) => fd.append("clips", b, `clip${i}.webm`));

  try {
    const data = await fetchJSON("/api/speakers/enroll", { method: "POST", body: fd });
    setEnrollStatus(`✓ Enrolled "${data.speaker}" (${data.clips_saved} clips). ${data.message}`, "ok");

    // Update model state
    if (data.svm_classes && data.svm_classes.length >= 2) {
      modelReady = true;
      setBadge("ready", "SVM READY");
      renderSpeakerChips(data.svm_classes);
      switchView("idle");
    }

    // Reset form
    enrollBlobs = [];
    updateClipList();
    document.getElementById("enroll-name").value = "";
    refreshEnrolledList();
  } catch (err) {
    setEnrollStatus("Error: " + err.message, "error");
    enrollSubmitBtn.disabled = false;
  } finally {
    setFulcrum("idle");
  }
});

function setEnrollStatus(msg, cls = "") {
  enrollStatus.textContent = msg;
  enrollStatus.className   = "enroll-status" + (cls ? ` ${cls}` : "");
}

async function refreshEnrolledList() {
  const el = document.getElementById("enrolled-list");
  try {
    const names = await fetchJSON("/api/speakers");
    el.innerHTML = names.length
      ? names.map(n => `<span class="speaker-chip">${n}</span>`).join("")
      : `<span class="view-body">none yet</span>`;
  } catch { /* ignore */ }
}

// ── History tab ───────────────────────────────────────────────────────────────
async function loadHistory() {
  const tbody = document.getElementById("history-body");
  try {
    const rows = await fetchJSON("/api/sessions");
    if (!rows.length) {
      tbody.innerHTML = `<tr><td colspan="5" class="h-empty">no sessions yet</td></tr>`;
      return;
    }
    tbody.innerHTML = rows.slice(0, 20).map(s => `
      <tr>
        <td title="${s.svm_speaker}">${s.svm_speaker}</td>
        <td title="${s.cnn_speaker}">${s.cnn_speaker}</td>
        <td><span class="badge ${badgeCls(s.risk_level)}" style="font-size:9px">${s.risk_level}</span></td>
        <td title="${s.transcript}">${(s.transcript || "").slice(0, 28) || "—"}</td>
        <td>${new Date(s.created_at).toLocaleTimeString()}</td>
      </tr>`).join("");
  } catch {
    tbody.innerHTML = `<tr><td colspan="5" class="h-empty">error</td></tr>`;
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
    throw new Error(msg || `HTTP ${res.status}`);
  }
  return res.json();
}
