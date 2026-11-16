// litmapr Studio: forms for every command, and a console that follows them.
// Every request carries the Studio header: the server refuses requests without
// it, so another website can't make your computer run commands.
"use strict";

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
};
const listItem = (child) => { const li = el("li"); li.append(child); return li; };
const HEADERS = { "X-Litmapr-Studio": "1" };
const KIND_COLORS = {
  character: "#c8ff2e", device: "#2fd9ff", concept: "#9b86ff",
  location: "#ffb23e", faction: "#ff3355", event: "#ffffff",
};
const PICTURE_KINDS = ["character", "device", "location", "faction", "event"];
const store = {
  get(key, fallback) { try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch { /* private window: fine */ } },
};

async function api(path, { method = "GET", json, body } = {}) {
  const options = { method, headers: { ...HEADERS } };
  if (json !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(json);
  } else if (body !== undefined) {
    options.body = body;
  }
  const response = await fetch(`/api/studio${path}`, options);
  if (!response.ok) {
    const answer = await response.json().catch(() => ({}));
    const detail = Array.isArray(answer.detail)
      ? answer.detail.map((d) => `${d.loc?.slice(-1)[0]}: ${d.msg}`).join("; ")
      : answer.detail;
    throw new Error(detail || `The server answered ${response.status}.`);
  }
  return response;
}
const getJSON = async (path, options) => (await api(path, options)).json();

let toastTimer;
function toast(message, bad = false) {
  const box = $("toast");
  box.textContent = message;
  box.classList.toggle("bad", bad);
  box.classList.add("is-on");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => box.classList.remove("is-on"), bad ? 6000 : 3500);
}

const state = { config: null, works: [], selectedJob: null, logLength: 0, after: new Map() };

// ---------------------------------------------------------------- sections
function showSection(name) {
  const known = [...document.querySelectorAll("#sections button")].map((b) => b.dataset.section);
  if (!known.includes(name)) name = "library";
  for (const button of document.querySelectorAll("#sections button")) {
    button.setAttribute("aria-current", String(button.dataset.section === name));
  }
  for (const section of document.querySelectorAll(".section")) section.hidden = section.id !== name;
  if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
}

// ---------------------------------------------------------------- library
async function loadLibrary() {
  try {
    state.works = await getJSON("/library");
  } catch (error) {
    toast(error.message, true);
    return;
  }
  const works = state.works;
  $("library-empty").hidden = works.length > 0;
  $("library-list").hidden = works.length === 0;
  $("stats").hidden = works.length === 0;

  const series = new Set(works.map((w) => w.series));
  let entities = 0;
  try { entities = (await (await fetch("/api/graph")).json()).nodes.length; } catch { /* map not ready */ }
  $("stats").replaceChildren(
    ...[[works.length, "Issues"], [series.size, "Series"], [entities, "Entries"]].map(([n, label]) => {
      const box = el("div", "stat");
      box.append(el("b", null, String(n)), el("span", null, label));
      return box;
    }),
  );

  const table = el("table");
  const head = el("thead");
  const headRow = el("tr");
  for (const title of ["#", "Issue", "Date", "Entries", "File", ""]) headRow.append(el("th", null, title));
  head.append(headRow);
  const rows = el("tbody");
  works.forEach((work, i) => {
    const row = el("tr");
    const date = work.year ? `${work.year}${work.month ? "-" + String(work.month).padStart(2, "0") : ""}` : "-";
    row.append(
      el("td", "num", String(i + 1)),
      el("td", "issue", work.label),
      el("td", "num", date),
      el("td", "num", String(work.entities)),
      Object.assign(el("td", "file", work.file || "-"), { title: work.file || "" }),
    );
    const act = el("td", "act");
    const remove = el("button", "danger", "Remove");
    remove.onclick = () => confirmRemove(act, work);
    act.append(remove);
    row.append(act);
    rows.append(row);
  });
  table.append(head, rows);
  $("library-list").replaceChildren(table);

  // the issue picker on the Pictures page
  $("pic-issue").replaceChildren(
    new Option("Pick an issue", ""),
    ...works.map((w) => new Option(w.label, w.issue)),
  );
}

function confirmRemove(cell, work) {
  const box = el("span", "confirm", `Remove ${work.label}? `);
  const yes = el("button", "danger", "Yes, remove");
  const no = el("button", "danger", "Keep");
  no.style.color = "var(--ink-soft)";
  yes.onclick = async () => {
    try {
      const answer = await getJSON(`/works/${work.id}`, { method: "DELETE" });
      toast(`Removed ${work.label}` + (answer.orphans ? ` and ${answer.orphans} entries only found there` : ""));
      loadLibrary();
      loadNames();
    } catch (error) {
      toast(error.message, true);
    }
  };
  no.onclick = () => loadLibrary();
  box.append(yes, no);
  cell.replaceChildren(box);
  yes.focus();
}

async function loadNames() {
  try {
    const graph = await (await fetch("/api/graph")).json();
    const names = [...new Set(graph.nodes.map((n) => n.name))].sort((a, b) => a.localeCompare(b));
    $("names").replaceChildren(...names.map((n) => new Option(n)));
  } catch { /* nothing yet */ }
}

// ---------------------------------------------------------------- file browser
// Lists folders on this computer (through the server) so you can pick a comic
// without typing its path.
async function openBrowser(targetId, boxId, folderPicks) {
  const box = $(boxId);
  if (!box.hidden && box.dataset.target === targetId) { box.hidden = true; return; }
  box.dataset.target = targetId;
  box.hidden = false;
  const typed = $(targetId).value.trim().replace(/^"|"$/g, "");
  const start = typed ? typed.replace(/[\\/][^\\/]*\.(cbz|pdf)$/i, "") : store.get("litmapr.folder", "");
  await showFolder(box, targetId, start, folderPicks);
}

async function showFolder(box, targetId, path, folderPicks) {
  let listing;
  try {
    listing = await getJSON(`/browse?path=${encodeURIComponent(path)}`);
  } catch (error) {
    if (path) return showFolder(box, targetId, "", folderPicks); // a typo'd path: start from the top
    box.replaceChildren(el("p", "browser-empty", error.message));
    return;
  }
  if (listing.path) store.set("litmapr.folder", listing.path);
  const head = el("div", "browser-head");
  const up = el("button", "ghost small-btn");
  up.innerHTML = '<svg class="icon"><use href="#i-up"/></svg>';
  up.title = "Up one folder";
  up.disabled = listing.parent === null;
  up.onclick = () => showFolder(box, targetId, listing.parent, folderPicks);
  head.append(up, el("span", "browser-path", listing.path || "This computer"));
  if (folderPicks && listing.path && listing.files.length) {
    const all = el("button", "ghost small-btn", `Add all ${listing.files.length}`);
    all.title = "Add every comic in this folder, one issue at a time";
    all.onclick = () => pick(box, targetId, listing.path);
    head.append(all);
  }
  const list = el("ul", "browser-list");
  for (const folder of listing.folders) {
    const button = el("button");
    button.innerHTML = '<svg class="icon"><use href="#i-folder"/></svg>';
    button.append(el("span", null, folder.name));
    button.onclick = () => showFolder(box, targetId, folder.path, folderPicks);
    list.append(listItem(button));
  }
  for (const file of listing.files) {
    const button = el("button");
    button.innerHTML = '<svg class="icon"><use href="#i-file"/></svg>';
    button.append(el("span", null, file.name));
    if (file.contents) button.append(el("span", "badge", "contents"));
    button.append(el("span", "size", formatSize(file.size)));
    button.onclick = () => pick(box, targetId, file.path);
    list.append(listItem(button));
  }
  if (!listing.folders.length && !listing.files.length) {
    list.append(el("li", "browser-empty", "No folders or .cbz / .pdf files here."));
  }
  box.replaceChildren(head, list);
}

function pick(box, targetId, path) {
  $(targetId).value = path;
  box.hidden = true;
  $(targetId).dispatchEvent(new Event("change"));
}

function formatSize(bytes) {
  if (bytes > 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes > 1e6) return `${Math.round(bytes / 1e6)} MB`;
  return `${Math.max(1, Math.round(bytes / 1e3))} KB`;
}

// ---------------------------------------------------------------- dropping files
function upload(file) {
  const row = el("div", "upload");
  const label = el("span", null, file.name);
  const pct = el("span", "mono", "0%");
  const bar = el("div", "bar");
  const fill = el("i");
  bar.append(fill);
  row.append(label, pct, bar);
  $("uploads").append(row);
  return new Promise((resolve) => {
    // XMLHttpRequest, not fetch: it reports upload progress.
    const request = new XMLHttpRequest();
    request.open("PUT", `/api/studio/upload?name=${encodeURIComponent(file.name)}`);
    request.setRequestHeader("X-Litmapr-Studio", "1");
    request.upload.onprogress = (e) => {
      if (!e.lengthComputable) return;
      const share = Math.round((e.loaded / e.total) * 100);
      pct.textContent = `${share}%`;
      fill.style.width = `${share}%`;
    };
    request.onload = () => {
      let answer = {};
      try { answer = JSON.parse(request.responseText); } catch { /* not JSON */ }
      if (request.status === 200) {
        pct.textContent = "ready";
        fill.style.width = "100%";
        resolve(answer.path);
      } else {
        pct.textContent = answer.detail || "failed";
        pct.style.color = "var(--alarm)";
        resolve(null);
      }
    };
    request.onerror = () => { pct.textContent = "failed"; resolve(null); };
    request.send(file);
  });
}

async function uploadAll(files) {
  const wanted = [...files].filter((f) => /\.(cbz|pdf)$/i.test(f.name));
  if (!wanted.length) { toast("litmapr reads .cbz and .pdf files", true); return; }
  $("uploads").replaceChildren();
  const paths = [];
  for (const file of wanted) {
    const path = await upload(file);
    if (path) paths.push(path);
  }
  if (!paths.length) return;
  // One file: select it. Several: select the library folder, which adds them all.
  $("add-path").value = paths.length === 1 ? paths[0] : state.config.library;
  $("add-path").dispatchEvent(new Event("change"));
  toast(paths.length === 1 ? "Ready to add" : `${paths.length} files ready: Add to map reads the whole folder`);
}

// When the chosen file has a contents file next to it, say so: it's an omnibus.
// A folder takes each file's own date, so the date boxes are switched off for it.
async function checkContents() {
  const path = $("add-path").value.trim();
  const note = $("contents-note");
  note.hidden = true;
  const folder = Boolean(path) && !/\.(cbz|pdf)"?$/i.test(path);
  for (const id of ["add-year", "add-month"]) $(id).disabled = folder;
  $("date-hint").textContent = folder
    ? "A folder: each issue takes its own date from its file (ComicInfo.xml)."
    : "Year and month put each issue in reading order across series. An omnibus takes its dates from its contents file.";
  if (folder || !path) return;
  try {
    const found = await getJSON(`/contents?path=${encodeURIComponent(path)}`);
    note.replaceChildren();
    if (found.exists) {
      const issues = found.text.split("\n").filter((l) => l.trim() && !l.trim().startsWith("#") && !/skip/i.test(l)).length;
      note.className = "note ok";
      note.append(`Omnibus contents file found: ${issues} issues, each with its own place in the reading order. `);
    } else {
      note.className = "note";
      note.append("A single issue? Add it as it is. An omnibus? ");
    }
    const link = el("button", "danger", found.exists ? "Check it" : "Set up its contents first");
    link.style.color = "var(--cobalt)";
    link.onclick = () => { $("omni-path").value = path; showSection("omnibus"); loadContents(); };
    note.append(link);
    note.hidden = false;
  } catch { /* a missing file is reported when adding */ }
}

// ---------------------------------------------------------------- models
function models() {
  const pick = (id) => $(id).value.trim() || null;
  return { vision: pick("m-vision"), text: pick("m-text"), embed: pick("m-embed") };
}
function loadModels() {
  const defaults = state.config.models;
  for (const [id, key] of [["m-vision", "vision"], ["m-text", "text"], ["m-embed", "embed"]]) {
    $(id).placeholder = defaults[key];
    $(id).value = store.get(`litmapr.model.${key}`, "");
    $(id).onchange = () => {
      store.set(`litmapr.model.${key}`, $(id).value.trim());
      $("models-status").textContent = "Saved.";
      $("models-status").className = "status good";
    };
  }
  $("pull-hint").textContent = `ollama pull ${defaults.vision}`;
  $("models-reset").onclick = () => {
    for (const key of ["vision", "text", "embed"]) {
      store.set(`litmapr.model.${key}`, "");
      $(`m-${key}`).value = "";
    }
    $("models-status").textContent = "Using the defaults.";
    $("models-status").className = "status good";
  };
}

// ---------------------------------------------------------------- running commands
async function run(command, form, button, after) {
  if (button) button.disabled = true;
  try {
    const job = await getJSON(`/run/${command}`, { method: "POST", json: form });
    if (after) state.after.set(job.id, after);
    selectJob(job.id);
    refreshJobs();
    toast(job.status === "queued" && state.jobs?.some((j) => j.status === "running")
      ? "Queued: it starts when the current command finishes"
      : `Started: ${job.title}`);
  } catch (error) {
    toast(error.message, true);
  } finally {
    if (button) setTimeout(() => { button.disabled = false; }, 600);
  }
}

function wireForms() {
  $("add-run").onclick = () => {
    const path = $("add-path").value.trim();
    if (!path) { toast("Pick a comic first: Browse, or drop one in", true); $("add-path").focus(); return; }
    const year = $("add-year").disabled ? NaN : parseInt($("add-year").value, 10);
    const month = $("add-month").disabled ? NaN : parseInt($("add-month").value, 10);
    run("add", {
      path,
      series: $("add-series").value.trim() || null,
      year: Number.isFinite(year) ? year : null,
      month: Number.isFinite(month) ? month : null,
      models: models(),
    }, $("add-run"));
  };
  $("add-path").addEventListener("change", checkContents);
  $("add-path").addEventListener("input", () => {
    clearTimeout(state.typing);
    state.typing = setTimeout(checkContents, 400);
  });

  $("merge-preview").onclick = () => run("merge", { dry_run: true, models: models() }, $("merge-preview"));
  $("merge-run").onclick = () => run("merge", { dry_run: false, models: models() }, $("merge-run"));
  $("merge-two").onclick = () => {
    const keep = $("keep").value.trim();
    const drop = $("drop-name").value.trim();
    if (!keep || !drop) { toast("Fill in both names", true); return; }
    if (keep.toLowerCase() === drop.toLowerCase()) { toast("Those are the same name", true); return; }
    run("merge-entities", { keep, drop }, $("merge-two"));
  };
  $("link-run").onclick = () => run("link", { models: models() }, $("link-run"));

  $("kinds").replaceChildren(...PICTURE_KINDS.map((kind) => {
    const label = el("label", "kind");
    const box = Object.assign(el("input"), { type: "checkbox", value: kind, checked: true });
    const swatch = el("span", "swatch");
    swatch.style.setProperty("--c", KIND_COLORS[kind]);
    label.append(box, swatch, kind === "location" ? "places" : `${kind}s`);
    return label;
  }));
  $("pic-run").onclick = () => {
    const kinds = [...document.querySelectorAll("#kinds input:checked")].map((b) => b.value);
    if (!kinds.length) { toast("Pick at least one kind", true); return; }
    run("portraits", {
      kinds,
      top: parseInt($("pic-top").value, 10) || 30,
      force: $("pic-force").checked,
      models: models(),
    }, $("pic-run"));
  };
  $("pic-one").onclick = () => {
    const name = $("pic-name").value.trim();
    const page = parseInt($("pic-page").value, 10);
    if (!name) { toast("Who should it look for?", true); $("pic-name").focus(); return; }
    if (!page) { toast("Which page?", true); $("pic-page").focus(); return; }
    const issue = $("pic-issue").value || null;
    if (!issue && state.works.length > 1) { toast("Pick the issue that page is in", true); return; }
    run("portraits", { name, page, issue, force: true, models: models() }, $("pic-one"));
  };

  // omnibus
  $("omni-load").onclick = loadContents;
  $("omni-path").addEventListener("change", loadContents);
  $("omni-draft").onclick = () => {
    const path = $("omni-path").value.trim();
    if (!path) { toast("Pick the omnibus first", true); return; }
    run("contents", { path, force: $("omni-force").checked }, $("omni-draft"), loadContents);
  };
  $("omni-save").onclick = async () => {
    const status = $("omni-status");
    try {
      const saved = await getJSON("/contents", {
        method: "PUT",
        json: { path: $("omni-path").value.trim(), text: $("omni-text").value },
      });
      status.textContent = `Saved: ${saved.issues} issues.`;
      status.className = "status good";
    } catch (error) {
      status.textContent = error.message;
      status.className = "status bad";
    }
  };

  // share
  $("export").onclick = async () => {
    try {
      const blob = await (await api("/export")).blob();
      const link = Object.assign(el("a"), { href: URL.createObjectURL(blob), download: "litmapr-map.json" });
      document.body.append(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(link.href), 2000);
    } catch (error) {
      toast(error.message, true);
    }
  };
  $("import-pick").onclick = () => $("import-file").click();
  $("import-file").onchange = async () => {
    const file = $("import-file").files[0];
    $("import-file").value = "";
    if (!file) return;
    const status = $("import-status");
    status.textContent = "Importing...";
    status.className = "status";
    try {
      const answer = await getJSON("/import", { method: "POST", body: file });
      status.textContent = `Imported ${answer.works} issues and ${answer.entities} entries.`;
      status.className = "status good";
      loadLibrary();
      loadNames();
    } catch (error) {
      status.textContent = error.message;
      status.className = "status bad";
    }
  };

  // browsing and dropping
  for (const button of document.querySelectorAll("[data-browse]")) {
    const box = button.dataset.browse === "add-path" ? "browser" : "omni-browser";
    button.onclick = () => openBrowser(button.dataset.browse, box, button.dataset.folders === "yes");
  }
  const drop = $("drop");
  drop.onclick = () => $("drop-input").click();
  drop.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("drop-input").click(); } };
  $("drop-input").onchange = () => { uploadAll($("drop-input").files); $("drop-input").value = ""; };
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("is-over"); };
  drop.ondragleave = () => drop.classList.remove("is-over");
  drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("is-over"); uploadAll(e.dataTransfer.files); };
  // A file dropped anywhere else shouldn't make the browser open it and leave the page.
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  for (const button of document.querySelectorAll("[data-goto]")) button.onclick = () => showSection(button.dataset.goto);
  for (const button of document.querySelectorAll("#sections button")) button.onclick = () => showSection(button.dataset.section);
}

async function loadContents() {
  const path = $("omni-path").value.trim();
  const status = $("omni-status");
  if (!path) return;
  try {
    const found = await getJSON(`/contents?path=${encodeURIComponent(path)}`);
    $("omni-text").value = found.text;
    $("omni-file").textContent = found.path;
    status.textContent = found.exists ? "" : "No contents file yet: draft one from the bookmarks, or type it in.";
    status.className = "status";
  } catch (error) {
    status.textContent = error.message;
    status.className = "status bad";
  }
}

// ---------------------------------------------------------------- the console
async function refreshJobs() {
  let jobs;
  try { jobs = await getJSON("/jobs"); } catch { return; }
  const before = new Map((state.jobs || []).map((j) => [j.id, j.status]));
  state.jobs = jobs;
  const live = jobs.some((j) => j.status === "running" || j.status === "queued");
  $("pulse").classList.toggle("is-live", live);

  for (const job of jobs) {
    const was = before.get(job.id);
    if (was && was !== job.status && ["done", "failed", "stopped"].includes(job.status)) finished(job);
  }
  if (!state.selectedJob && jobs.length) selectJob(jobs[0].id);

  $("history-title").hidden = jobs.length < 2;
  $("history").replaceChildren(...jobs.slice(0, 15).map((job) => {
    const button = el("button");
    button.setAttribute("aria-current", String(job.id === state.selectedJob));
    button.append(el("span", `dot ${job.status}`), el("span", "title", job.title), el("span", "when", clock(job.queued_at)));
    button.title = `${job.status}: ${job.command}`;
    button.onclick = () => selectJob(job.id);
    return listItem(button);
  }));
  setTimeout(refreshJobs, live ? 1500 : 5000);
}

function finished(job) {
  if (job.status === "done") toast(`Done: ${job.title}`);
  else if (job.status === "failed") toast(`${job.title} stopped with an error: see the console`, true);
  loadLibrary();
  loadNames();
  const after = state.after.get(job.id);
  if (after && job.status === "done") after();
  state.after.delete(job.id);
}

function clock(seconds) {
  return new Date(seconds * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}
function elapsed(job) {
  if (!job.started_at) return "waiting for its turn";
  const end = job.ended_at || Date.now() / 1000;
  const s = Math.max(0, Math.round(end - job.started_at));
  const text = s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
  return job.ended_at ? `took ${text}` : `running ${text}`;
}

function selectJob(id) {
  state.selectedJob = id;
  state.logLength = 0;
  $("log").replaceChildren();
  delete $("log").dataset.waiting;
  $("job").hidden = false;
  $("console-empty").hidden = true;
  followLog(id);
}

// Lines worth a colour: problems in red, good news in green.
function lineClass(line) {
  if (/error|failed|not found|couldn't|can't|unclear|traceback|exception/i.test(line)) return "bad";
  if (/^\s*(skipped|warning)|timed out|try /i.test(line)) return "warn";
  if (/^(done|saved|merged|added|removed|imported|exported|wrote)|^\s+saved/i.test(line)) return "good";
  return null;
}

async function followLog(id) {
  if (state.selectedJob !== id) return;
  let job;
  try {
    job = await getJSON(`/jobs/${id}?since=${state.logLength}`);
  } catch {
    return;
  }
  if (state.selectedJob !== id) return;
  const log = $("log");
  const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
  if (job.log.length && log.dataset.waiting) { log.replaceChildren(); delete log.dataset.waiting; }
  for (const line of job.log) {
    const kind = lineClass(line);
    const node = kind ? el("span", kind, line) : document.createTextNode(line);
    log.append(node, "\n");
  }
  state.logLength = job.lines;
  if (atBottom) log.scrollTop = log.scrollHeight;
  if (!job.log.length && !state.logLength && !log.dataset.waiting) {
    log.dataset.waiting = "yes";
    log.textContent = job.status === "queued" ? "Waiting for the command before it...\n" : "Starting...\n";
  }

  $("job-title").textContent = job.title;
  $("job-status").textContent = job.status;
  $("job-status").className = `pill ${job.status}`;
  $("job-time").textContent = elapsed(job);
  $("job-command").textContent = job.command;
  $("job-stop").hidden = !["running", "queued"].includes(job.status);
  $("job-stop").onclick = async () => {
    try { await getJSON(`/jobs/${id}/stop`, { method: "POST" }); } catch (error) { toast(error.message, true); }
  };
  if (["running", "queued"].includes(job.status)) setTimeout(() => followLog(id), 700);
}

// ---------------------------------------------------------------- start
async function start() {
  try {
    state.config = await getJSON("/config");
  } catch (error) {
    document.querySelector(".sheet").replaceChildren(el("h1", null, "The Studio isn't available"), el("p", "lede", error.message));
    return;
  }
  $("db-path").textContent = state.config.db;
  $("db-path").title = "The map's database";
  $("library-folder").textContent = state.config.library;
  wireForms();
  loadModels();
  await loadLibrary();
  loadNames();
  showSection(location.hash.slice(1) || (state.works.length ? "library" : "add"));
  window.addEventListener("hashchange", () => showSection(location.hash.slice(1)));
  refreshJobs();
}

start();
