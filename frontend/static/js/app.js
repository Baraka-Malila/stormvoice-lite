/* stormVoice Lite — vanilla JS frontend */

const RECORD_MS = 4000;

// ── Tab routing ──────────────────────────────────────────────────────────────
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
    if (btn.dataset.tab === "history") loadHistory();
    if (btn.dataset.tab === "enroll") loadSpeakers();
  });
});

// ── MediaRecorder helper ─────────────────────────────────────────────────────
async function recordClip(durationMs) {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  return new Promise((resolve, reject) => {
    const chunks = [];
    const mr = new MediaRecorder(stream, { mimeType: "audio/webm" });
    mr.ondataavailable = (e) => chunks.push(e.data);
    mr.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      resolve(new Blob(chunks, { type: "audio/webm" }));
    };
    mr.onerror = reject;
    mr.start();
    setTimeout(() => mr.stop(), durationMs);
  });
}

// ── Analyze tab ──────────────────────────────────────────────────────────────
const analyzeBtn    = document.getElementById("analyze-btn");
const analyzeStatus = document.getElementById("analyze-status");

analyzeBtn.addEventListener("click", async () => {
  if (analyzeBtn.classList.contains("recording")) return;
  analyzeBtn.classList.add("recording");
  analyzeStatus.textContent = `Recording ${RECORD_MS / 1000}s…`;
  analyzeStatus.className = "status";
  try {
    const blob = await recordClip(RECORD_MS);
    analyzeStatus.textContent = "Analyzing…";
    const fd = new FormData();
    fd.append("audio", blob, "clip.webm");
    const res = await fetch("/api/analyze", { method: "POST", body: fd });
    if (!res.ok) throw new Error(await res.text());
    renderResult(await res.json());
    analyzeStatus.textContent = "";
  } catch (err) {
    analyzeStatus.textContent = "Error: " + err.message;
    analyzeStatus.className = "status error";
  } finally {
    analyzeBtn.classList.remove("recording");
  }
});

function badgeClass(level) {
  return { Low: "badge-low", Medium: "badge-medium", High: "badge-high", Critical: "badge-critical" }[level] || "badge-low";
}

function renderResult(data) {
  document.getElementById("waveform-img").src   = "data:image/png;base64," + data.waveform_png_b64;
  document.getElementById("spectrogram-img").src = "data:image/png;base64," + data.spectrogram_png_b64;
  document.getElementById("viz-row").style.display = "grid";

  document.getElementById("r-speaker").textContent    = data.speaker;
  document.getElementById("r-confidence").textContent = `${(data.speaker_confidence * 100).toFixed(1)}%`;
  document.getElementById("r-transcript").textContent = data.transcript || "(no speech detected)";
  document.getElementById("r-risk").innerHTML =
    `<span class="badge ${badgeClass(data.risk_level)}">${data.risk_level}</span>&nbsp; score: ${data.risk_score}`;
  document.getElementById("r-category").textContent = data.fraud_category;
  document.getElementById("r-signals").innerHTML = data.detected_signals.length
    ? data.detected_signals.map((s) => `<span class="signal-tag">${s.keyword} +${s.points}</span>`).join("")
    : `<span style="color:var(--fg-dim)">none</span>`;
  document.getElementById("r-action").textContent = data.recommended_action;
  document.getElementById("result-panel").style.display = "block";
}

// ── Enroll tab ───────────────────────────────────────────────────────────────
const enrollRecordBtn = document.getElementById("enroll-record-btn");
const enrollSubmitBtn = document.getElementById("enroll-submit-btn");
const enrollStatus    = document.getElementById("enroll-status");
const clipsList       = document.getElementById("clips-list");
const enrolledBlobs   = [];

enrollRecordBtn.addEventListener("click", async () => {
  if (enrollRecordBtn.classList.contains("recording")) return;
  if (enrolledBlobs.length >= 3) { enrollStatus.textContent = "3 clips already recorded."; return; }
  enrollRecordBtn.classList.add("recording");
  enrollStatus.textContent = `Recording clip ${enrolledBlobs.length + 1}/3…`;
  try {
    const blob = await recordClip(RECORD_MS);
    enrolledBlobs.push(blob);
    const tag = document.createElement("span");
    tag.className = "clip-item";
    tag.textContent = `clip ${enrolledBlobs.length}`;
    clipsList.appendChild(tag);
    enrollStatus.textContent = enrolledBlobs.length < 3 ? `${enrolledBlobs.length}/3 recorded` : "Ready to enroll";
    enrollSubmitBtn.disabled = false;
  } catch (err) {
    enrollStatus.textContent = "Mic error: " + err.message;
    enrollStatus.className = "status error";
  } finally {
    enrollRecordBtn.classList.remove("recording");
  }
});

enrollSubmitBtn.addEventListener("click", async () => {
  const name = document.getElementById("enroll-name").value.trim();
  if (!name) { enrollStatus.textContent = "Enter a name first."; return; }
  if (!enrolledBlobs.length) { enrollStatus.textContent = "Record at least one clip."; return; }
  enrollStatus.textContent = "Enrolling…";
  enrollSubmitBtn.disabled = true;
  const fd = new FormData();
  fd.append("name", name);
  enrolledBlobs.forEach((b, i) => fd.append("clips", b, `clip${i}.webm`));
  try {
    const res = await fetch("/api/speakers/enroll", { method: "POST", body: fd });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    enrollStatus.textContent = `✓ Enrolled "${data.name}" (${data.sample_count} clip${data.sample_count > 1 ? "s" : ""})`;
    enrollStatus.className = "status";
    enrolledBlobs.length = 0;
    clipsList.innerHTML = "";
    document.getElementById("enroll-name").value = "";
    await loadSpeakers();
  } catch (err) {
    enrollStatus.textContent = "Error: " + err.message;
    enrollStatus.className = "status error";
    enrollSubmitBtn.disabled = false;
  }
});

async function loadSpeakers() {
  const el = document.getElementById("speakers-list");
  try {
    const data = await (await fetch("/api/speakers")).json();
    el.innerHTML = data.length
      ? data.map((s) => `<span class="clip-item" style="margin:0.15rem">${s.name}</span>`).join("")
      : `<span style="color:var(--fg-dim)">none enrolled yet</span>`;
  } catch { el.textContent = "error loading speakers"; }
}

// ── History tab ──────────────────────────────────────────────────────────────
async function loadHistory() {
  const body = document.getElementById("history-body");
  try {
    const rows = await (await fetch("/api/sessions")).json();
    if (!rows.length) {
      body.innerHTML = `<tr><td colspan="5" style="color:var(--fg-dim)">no sessions yet</td></tr>`;
      return;
    }
    body.innerHTML = rows.map((s) => `
      <tr>
        <td>${s.speaker}</td>
        <td><span class="badge ${badgeClass(s.risk_level)}">${s.risk_level}</span></td>
        <td style="font-size:0.78rem;color:var(--fg-dim)">${s.fraud_category}</td>
        <td class="transcript-cell">${s.transcript || "—"}</td>
        <td style="font-size:0.72rem;color:var(--fg-dim)">${new Date(s.created_at).toLocaleTimeString()}</td>
      </tr>`).join("");
  } catch {
    body.innerHTML = `<tr><td colspan="5" style="color:var(--fg-dim)">error loading history</td></tr>`;
  }
}
