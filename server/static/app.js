const $ = (sel) => document.querySelector(sel);

const LABELS = {
  phone_call: "Phone call",
  texting: "Using phone",
  drinking: "Drinking",
  reaching: "Reaching / turned away",
  looking_away: "Looking away",
  hand_to_face: "Hand at face",
};
const COLORS = {
  phone_call: "#eb4646",
  texting: "#c846c8",
  drinking: "#ff9600",
  reaching: "#ebcd00",
  looking_away: "#ff6e8c",
  hand_to_face: "#3caacd",
};
const DRIVER_COLORS = { 1: "#3ca0e6", 2: "#f0c850", 3: "#f06ebe", 4: "#78dc78", 5: "#f0783c" };
const GRADE_COLORS = { A: "#3ecf7a", B: "#9bd34a", C: "#ffb020", D: "#ff7a3d", E: "#ff5a5f" };

let pollTimer = null;
let report = null;
let runId = null;

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch { /* keep status text */ }
    throw new Error(msg);
  }
  return res.status === 204 ? null : res.json();
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null) node.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

function fmtTime(s) {
  s = Math.max(0, s);
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
}
function fmtDuration(s) {
  if (s == null) return "–";
  return s >= 60 ? `${Math.floor(s / 60)}m ${Math.round(s % 60)}s` : `${s.toFixed(1)}s`;
}

/* ---------- routing ---------- */

function route() {
  clearTimeout(pollTimer);
  const m = location.hash.match(/^#\/runs\/([\w-]+)/);
  $("#view-home").hidden = !!m;
  $("#view-run").hidden = !m;
  document.querySelectorAll("[data-nav]").forEach((a) => a.classList.toggle("active", !m));
  if (m) showRun(m[1]);
  else showHome();
}
window.addEventListener("hashchange", route);

/* ---------- home ---------- */

async function showHome() {
  const [videos, runs] = await Promise.all([api("/api/videos"), api("/api/runs")]);
  const grid = $("#video-grid");
  grid.replaceChildren(...videos.map(videoCard));
  if (!videos.length) grid.append(el("p", { class: "muted" }, "No videos yet. Upload one or run scripts/download_samples.py."));
  renderRuns(runs);
  if (runs.some((r) => r.status === "running" || r.status === "queued")) {
    pollTimer = setTimeout(showHome, 3000);
  }
}

function videoCard(v) {
  return el("div", { class: "video-card" },
    el("img", { src: `/api/videos/${encodeURIComponent(v.id)}/thumb.jpg`, alt: "", loading: "lazy" }),
    el("div", { class: "body" },
      el("div", {},
        el("div", { class: "name", title: v.name }, v.name),
        el("div", { class: "muted small" }, `${fmtDuration(v.duration_s)} · ${v.width}×${v.height}${v.sample ? " · sample" : ""}`)),
      el("button", { class: "btn primary", onclick: () => startRun(v.id) }, "Analyse")));
}

async function startRun(videoId) {
  try {
    const { id } = await api("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id: videoId, options: {} }),
    });
    location.hash = `#/runs/${id}`;
  } catch (e) {
    alert(`Could not start: ${e.message}`);
  }
}

function renderRuns(runs) {
  const table = $("#run-table");
  $("#no-runs").hidden = runs.length > 0;
  table.hidden = !runs.length;
  if (!runs.length) return;
  const head = el("tr", {}, ["Video", "Started", "Status", "Drivers", "Incidents", "Scores", ""].map((h) => el("th", {}, h)));
  const rows = runs.map((r) => el("tr", { class: "clickable", onclick: () => (location.hash = `#/runs/${r.id}`) },
    el("td", {}, r.video_name),
    el("td", { class: "muted" }, new Date(r.created * 1000).toLocaleString()),
    el("td", {}, el("span", { class: `status ${r.status}` },
      r.status)),
    el("td", {}, r.drivers ?? "–"),
    el("td", {}, r.events ?? "–"),
    el("td", {}, (r.scores || []).map((s) => gradePill(s.grade, `D${s.driver} ${s.score}`))),
    el("td", {}, el("button", {
      class: "icon-btn", title: "Delete report",
      onclick: async (ev) => {
        ev.stopPropagation();
        if (!confirm("Delete this report?")) return;
        await api(`/api/runs/${r.id}`, { method: "DELETE" });
        showHome();
      },
    }, "×"))));
  table.replaceChildren(el("thead", {}, head), el("tbody", {}, rows));
}

function gradePill(grade, text) {
  const c = GRADE_COLORS[grade] || "#888";
  return el("span", { class: "pill", style: `background:${c}22;color:${c};margin-right:6px` }, text);
}

$("#upload").addEventListener("change", async (ev) => {
  const file = ev.target.files[0];
  if (!file) return;
  const status = $("#upload-status");
  status.textContent = `Uploading ${file.name}…`;
  const form = new FormData();
  form.append("file", file);
  try {
    await api("/api/videos", { method: "POST", body: form });
    status.textContent = "";
    showHome();
  } catch (e) {
    status.textContent = `Upload failed: ${e.message}`;
  }
  ev.target.value = "";
});

/* ---------- run page ---------- */

async function showRun(id) {
  let run;
  try {
    run = await api(`/api/runs/${id}`);
  } catch {
    $("#run-title").textContent = "Report not found";
    return;
  }
  $("#run-title").textContent = run.video_name;
  $("#run-sub").textContent = `Started ${new Date(run.created * 1000).toLocaleString()}`;
  const actions = $("#run-actions");
  actions.replaceChildren();
  $("#run-failed").hidden = run.status !== "failed" && run.status !== "cancelled";
  $("#run-failed").textContent = run.status === "cancelled" ? "Cancelled." : `Failed: ${run.error || "unknown error"}`;

  if (run.status === "queued" || run.status === "running") {
    $("#run-progress").hidden = false;
    $("#run-results").hidden = true;
    const live = run.live || {};
    const p = live.progress || 0;
    $("#progress-bar").style.width = `${Math.round(p * 100)}%`;
    const stage = live.stage === "rendering" ? "Rendering annotated video" : run.status === "queued" ? "Waiting in queue" : "Detecting driver and objects";
    $("#progress-info").textContent = `${stage} · ${Math.round(p * 100)}%${live.fps ? ` · ${live.fps} frames/s` : ""}`;
    actions.append(el("button", {
      class: "btn", onclick: async () => { await api(`/api/runs/${id}/cancel`, { method: "POST" }); showRun(id); },
    }, "Cancel"));
    pollTimer = setTimeout(() => showRun(id), 2000);
    return;
  }
  $("#run-progress").hidden = true;
  if (run.status !== "done") return;

  report = run.report;
  runId = id;
  if (!report) return;
  $("#run-results").hidden = false;
  $("#run-sub").textContent =
    `${fmtDuration(report.duration_s)} trip · ${report.drivers.length} driver${report.drivers.length === 1 ? "" : "s"} · ` +
    `${report.events.length} incident${report.events.length === 1 ? "" : "s"} · analysed in ${fmtDuration(report.processing_s)}`;
  actions.append(el("a", { class: "btn", href: `/files/${id}/events.csv`, download: "" }, "Export CSV"));
  renderCards();
  const player = $("#player");
  const src = `/files/${id}/annotated.mp4`;
  if (!player.src.endsWith(src)) {
    player.src = src;
    if (report.events.length) player.poster = `/files/${id}/${report.events[0].snapshot}`;
  }
  renderTimeline();
  renderFilters();
  renderEvents();
  renderDownloads(id);
}

function renderCards() {
  $("#cards").replaceChildren(...report.drivers.map((d) => {
    const color = DRIVER_COLORS[d.driver] || "#888";
    const gc = GRADE_COLORS[d.grade] || "#888";
    const deg = Math.round(3.6 * d.score);
    const maxSecs = Math.max(1, ...Object.values(d.by_behaviour).map((b) => b.seconds));
    return el("div", { class: "card", style: `border-top-color:${color}` },
      el("div", { class: "card-top" },
        el("div", { class: "gauge", style: `background:conic-gradient(${gc} ${deg}deg, #263241 0)` },
          el("div", { class: "gauge-inner" },
            el("div", {}, el("div", { class: "gauge-score" }, Math.round(d.score)),
              el("div", { class: "gauge-grade" }, `grade ${d.grade}`)))),
        el("div", {},
          el("div", { class: "card-title" }, `Driver ${d.driver}`),
          el("div", { class: "card-meta" },
            `${fmtDuration(d.driving_s)} at the wheel`, el("br"),
            `${d.events} incident${d.events === 1 ? "" : "s"} · distracted ${d.distracted_pct.toFixed(0)}% of the time`))),
      el("div", { class: "bars" }, report.behaviours.map((b) => {
        const s = d.by_behaviour[b] || { events: 0, seconds: 0 };
        return el("div", { class: "bar-row" },
          el("span", { class: s.events ? "" : "muted" }, LABELS[b]),
          el("div", { class: "bar" }, el("i", { style: `width:${(100 * s.seconds) / maxSecs}%;background:${COLORS[b]}` })),
          el("span", { class: "n" }, s.events ? `${s.events}× ${s.seconds.toFixed(0)}s` : "–"));
      })));
  }));
}

function renderTimeline() {
  const W = 1200, left = 150, right = 10, laneH = 16, top = 26;
  const lanes = report.behaviours;
  const H = top + laneH * lanes.length + 22;
  const dur = report.duration_s;
  const x = (t) => left + ((W - left - right) * t) / dur;
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const add = (tag, attrs, text) => {
    const n = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (text) n.textContent = text;
    svg.append(n);
    return n;
  };
  add("text", { x: 4, y: 16, class: "lane-label" }, "Driver");
  for (const s of report.driver_spans) {
    add("rect", { x: x(s.start), y: 6, width: Math.max(1, x(s.end) - x(s.start)), height: 12, rx: 2,
                  fill: DRIVER_COLORS[s.driver] || "#888" });
    if (x(s.end) - x(s.start) > 60) add("text", { x: x(s.start) + 5, y: 16, fill: "#0e1319", "font-size": 10, "font-weight": 600 }, `Driver ${s.driver}`);
  }
  lanes.forEach((b, i) => {
    const y = top + i * laneH;
    add("line", { x1: left, x2: W - right, y1: y + laneH / 2, y2: y + laneH / 2, class: "grid" });
    add("text", { x: 4, y: y + 12, class: "lane-label" }, LABELS[b]);
  });
  const step = dur > 600 ? 120 : dur > 240 ? 60 : 30;
  for (let t = 0; t <= dur; t += step) {
    add("text", { x: x(t), y: H - 6, class: "axis", "text-anchor": "middle" }, fmtTime(t));
  }
  for (const e of report.events) {
    const y = top + lanes.indexOf(e.behaviour) * laneH + 3;
    const r = add("rect", { x: x(e.start), y, width: Math.max(2, x(e.end) - x(e.start)), height: laneH - 6, rx: 2,
                            fill: COLORS[e.behaviour] });
    const title = document.createElementNS(ns, "title");
    title.textContent = `${LABELS[e.behaviour]} · driver ${e.driver} · ${fmtTime(e.start)}–${fmtTime(e.end)}`;
    r.append(title);
  }
  const head = add("line", { x1: left, x2: left, y1: 2, y2: H - 18, class: "playhead" });
  const tl = $("#timeline");
  tl.replaceChildren(svg);
  tl.onclick = (ev) => {
    const box = svg.getBoundingClientRect();
    const px = ((ev.clientX - box.left) / box.width) * W;
    if (px < left) return;
    seek(((px - left) / (W - left - right)) * dur);
  };
  $("#player").ontimeupdate = () => {
    const px = x($("#player").currentTime);
    head.setAttribute("x1", px);
    head.setAttribute("x2", px);
  };
  $("#legend").replaceChildren(...lanes.map((b) => el("span", {}, el("i", { style: `background:${COLORS[b]}` }), LABELS[b])));
}

function seek(t) {
  const p = $("#player");
  p.currentTime = Math.max(0, t);
  p.play().catch(() => {});
  p.scrollIntoView({ behavior: "smooth", block: "center" });
}

function renderFilters() {
  const fd = $("#filter-driver");
  const fb = $("#filter-behaviour");
  fd.replaceChildren(el("option", { value: "" }, "All drivers"),
    ...report.drivers.map((d) => el("option", { value: d.driver }, `Driver ${d.driver}`)));
  fb.replaceChildren(el("option", { value: "" }, "All behaviours"),
    ...report.behaviours.map((b) => el("option", { value: b }, LABELS[b])));
  fd.onchange = fb.onchange = renderEvents;
}

function renderEvents() {
  const d = $("#filter-driver").value;
  const b = $("#filter-behaviour").value;
  const evs = report.events.filter((e) => (!d || String(e.driver) === d) && (!b || e.behaviour === b));
  $("#event-count").textContent = evs.length;
  const head = el("tr", {}, ["", "Behaviour", "Driver", "Time", "Duration", "Evidence", "Clip"].map((h) => el("th", {}, h)));
  const rows = evs.map((e) => el("tr", { class: "clickable", onclick: () => seek(e.start - 1) },
    el("td", {}, el("img", { src: `/files/${runId}/${e.snapshot}`, alt: "", loading: "lazy" })),
    el("td", {}, el("span", { class: "pill", style: `background:${COLORS[e.behaviour]}26;color:${COLORS[e.behaviour]}` }, e.label)),
    el("td", {}, `Driver ${e.driver}`),
    el("td", {}, `${fmtTime(e.start)} – ${fmtTime(e.end)}`),
    el("td", {}, fmtDuration(e.duration)),
    el("td", { class: "muted" }, e.evidence),
    el("td", {}, e.clip ? el("a", { href: `/files/${runId}/${e.clip}`, target: "_blank", onclick: (ev) => ev.stopPropagation() }, "mp4") : "–")));
  $("#events").replaceChildren(el("thead", {}, head), el("tbody", {}, rows));
}

function renderDownloads(id) {
  const files = [["annotated.mp4", "Annotated video"], ["report.json", "Full report (JSON)"], ["events.csv", "Incidents (CSV)"]];
  $("#downloads").replaceChildren(...files.map(([f, label]) => el("li", {}, el("a", { href: `/files/${id}/${f}`, download: "" }, label))));
}

route();
