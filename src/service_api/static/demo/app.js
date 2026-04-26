const state = {
  cvText: "",
  cvSkills: [],
  jds: [],
  selectedJd: null,
  currentRunId: null,
  statusPollTimer: null,
};

function parseSkills(text) {
  if (!text) return [];
  const parts = text
    .split(/[,;\n]/g)
    .map((x) => x.trim().toLowerCase())
    .filter(Boolean);
  return [...new Set(parts)];
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setTab(tab) {
  document.getElementById("tab-user").classList.toggle("active", tab === "user");
  document.getElementById("tab-admin").classList.toggle("active", tab === "admin");
  document.getElementById("panel-user").classList.toggle("active", tab === "user");
  document.getElementById("panel-admin").classList.toggle("active", tab === "admin");
}

async function loadJds() {
  const res = await fetch("/api/v1/jds/list");
  if (!res.ok) throw new Error("Failed to load JDs");
  const data = await res.json();
  state.jds = data.jds || [];

  const select = document.getElementById("jd-select");
  select.innerHTML = "";
  state.jds.forEach((jd, idx) => {
    const opt = document.createElement("option");
    opt.value = jd.jd_id;
    opt.textContent = `${jd.title} (${(jd.jd_skills || []).length} skills)`;
    select.appendChild(opt);
    if (idx === 0) state.selectedJd = jd;
  });

  select.addEventListener("change", () => {
    state.selectedJd = state.jds.find((x) => x.jd_id === select.value) || null;
  });
}

function renderCvSkills() {
  const root = document.getElementById("cv-skills");
  if (!state.cvSkills.length) {
    root.innerHTML = '<span class="hint">No skills detected yet</span>';
    return;
  }
  root.innerHTML = state.cvSkills.map((s) => `<span class="chip">${escapeHtml(s)}</span>`).join("");
}

function renderRecommendations(payload) {
  const root = document.getElementById("recommendations");
  const courses = payload?.recommended_courses || [];
  if (!courses.length) {
    root.innerHTML = '<div class="item"><p>No recommended course found.</p></div>';
    return;
  }

  root.innerHTML = courses
    .map(
      (c) => `
      <div class="item">
        <h4>${escapeHtml(c.course_title)}</h4>
        <p>Category: ${escapeHtml(c.category || "N/A")}</p>
        <p>Coverage: ${escapeHtml(c.coverage_count || 0)} skills</p>
      </div>
    `
    )
    .join("");
}

async function onCvFileChange(event) {
  const file = event.target.files?.[0];
  if (!file) return;
  const text = await file.text();
  state.cvText = text;
  state.cvSkills = parseSkills(text);
  renderCvSkills();
}

async function onRecommend() {
  const btn = document.getElementById("btn-recommend");
  const extraKeywords = document.getElementById("extra-keywords").value;
  const extra = parseSkills(extraKeywords);

  if (!state.selectedJd) {
    alert("Please select a JD");
    return;
  }
  if (!state.cvSkills.length) {
    alert("Please upload a .txt CV first");
    return;
  }

  const payload = {
    jd_skills: state.selectedJd.jd_skills || [],
    cv_skills: state.cvSkills,
    jd_title: state.selectedJd.title,
    jd_keywords: extra,
    max_courses: 10,
  };

  btn.disabled = true;
  btn.textContent = "Recommending...";

  try {
    const res = await fetch("/api/v1/recommendations/gap", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Recommendation failed");
    renderRecommendations(data);
  } catch (err) {
    alert(err.message || "Recommendation failed");
  } finally {
    btn.disabled = false;
    btn.textContent = "Recommend";
  }
}

async function refreshQueue() {
  const res = await fetch("/api/v1/admin/courses/queue");
  const data = await res.json();

  const summary = document.getElementById("queue-summary");
  summary.textContent = `Files: ${data.total_pending_files || 0} | Courses: ${data.total_pending_courses || 0}`;

  const list = document.getElementById("queue-list");
  const files = data.pending_files || [];
  if (!files.length) {
    list.innerHTML = '<div class="item"><p>Queue is empty</p></div>';
    return;
  }

  list.innerHTML = files
    .map(
      (f) => `
      <div class="item">
        <h4>${escapeHtml(f.filename)}</h4>
        <p>Courses: ${escapeHtml(f.courses_count)} | Size: ${escapeHtml(f.size_bytes)} bytes</p>
      </div>
    `
    )
    .join("");
}

async function uploadCourseFiles() {
  const input = document.getElementById("course-files");
  const files = [...(input.files || [])];
  if (!files.length) {
    alert("Please choose JSON files");
    return;
  }

  const fd = new FormData();
  files.forEach((f) => fd.append("files", f));

  const btn = document.getElementById("btn-upload");
  btn.disabled = true;
  btn.textContent = "Uploading...";

  try {
    const res = await fetch("/api/v1/admin/courses/upload", {
      method: "POST",
      body: fd,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Upload failed");
    alert(`Uploaded ${data.saved_count} file(s)`);
    await refreshQueue();
    input.value = "";
  } catch (err) {
    alert(err.message || "Upload failed");
  } finally {
    btn.disabled = false;
    btn.textContent = "Upload to Queue";
  }
}

function renderStatus(status) {
  const root = document.getElementById("pipeline-status");
  const log = document.getElementById("pipeline-log");

  root.textContent = `Run: ${status.run_id || "-"} | Status: ${status.status} | Phase: ${status.phase || "-"} | Progress: ${status.progress_percent || 0}% | ${status.message || ""}`;
  log.textContent = status.log_tail || "";
}

async function refreshStatus() {
  const q = state.currentRunId ? `?run_id=${encodeURIComponent(state.currentRunId)}` : "";
  const res = await fetch(`/api/v1/admin/pipeline/status${q}`);
  const data = await res.json();
  renderStatus(data);

  if (data.run_id) {
    state.currentRunId = data.run_id;
  }

  if (data.status === "completed" || data.status === "failed") {
    stopPolling();
    await refreshQueue();
  }
}

function startPolling() {
  stopPolling();
  state.statusPollTimer = setInterval(() => {
    refreshStatus().catch(() => {});
  }, 2000);
}

function stopPolling() {
  if (state.statusPollTimer) {
    clearInterval(state.statusPollTimer);
    state.statusPollTimer = null;
  }
}

async function runPipeline() {
  const clearExisting = document.getElementById("clear-existing").checked;
  const btn = document.getElementById("btn-run");

  btn.disabled = true;
  btn.textContent = "Starting...";

  try {
    const res = await fetch(`/api/v1/admin/pipeline/run?clear_existing=${clearExisting ? "true" : "false"}`, {
      method: "POST",
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Run failed");

    if (data.run_id) {
      state.currentRunId = data.run_id;
      await refreshStatus();
      startPolling();
    }
  } catch (err) {
    alert(err.message || "Cannot start pipeline");
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Ingest";
  }
}

async function init() {
  document.getElementById("tab-user").addEventListener("click", () => setTab("user"));
  document.getElementById("tab-admin").addEventListener("click", () => setTab("admin"));

  document.getElementById("cv-file").addEventListener("change", onCvFileChange);
  document.getElementById("btn-recommend").addEventListener("click", onRecommend);

  document.getElementById("btn-upload").addEventListener("click", uploadCourseFiles);
  document.getElementById("btn-refresh-queue").addEventListener("click", refreshQueue);
  document.getElementById("btn-run").addEventListener("click", runPipeline);
  document.getElementById("btn-refresh-status").addEventListener("click", refreshStatus);

  await loadJds();
  renderCvSkills();
  renderRecommendations(null);
  await refreshQueue();
  await refreshStatus();
}

init().catch((err) => {
  alert(err.message || "Failed to initialize demo page");
});
