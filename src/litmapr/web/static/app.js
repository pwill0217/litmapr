// litmapr explorer. Talks to the FastAPI server in web/api.py.
// Text from the map is always inserted with textContent, never innerHTML,
// so a character's name can never be treated as code.

const KINDS = {
  character: { label: "Characters", color: "#c8ff2e" },
  device: { label: "Devices", color: "#2fd9ff" },
  concept: { label: "Concepts", color: "#9b86ff" },
  location: { label: "Places", color: "#ffb23e" },
  faction: { label: "Factions", color: "#ff3355" },
  event: { label: "Events", color: "#ffffff" },
};
const SINGULAR = { character: "Character", device: "Device", concept: "Concept", location: "Place", faction: "Faction", event: "Event" };
const LINK_COLORS = {
  ally: "#c8ff2e", loves: "#ff7ad9", family: "#ffd34d", created: "#2fd9ff", member_of: "#8a93ff",
  rival: "#ff3355", opposes: "#ff3355", fears: "#9b86ff", other: "#4a55c8", callback: "#ffffff",
};
const TIMELINE_SPACING = 120;
const NAMED_MARKERS = 40; // names shown for the most-mentioned; the rest show on hover
const MAX_MARKERS = 160;
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

const state = {
  works: [], upto: null, hidden: new Set(), data: { nodes: [], links: [] },
  selected: null, neighbors: new Set(), config: { narrator: false, portraits: false },
};
const $ = (id) => document.getElementById(id);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}
function icon(kind, className = "icon") {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", className);
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#i-${kind}`);
  svg.append(use);
  return svg;
}
async function getJSON(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `The server answered ${response.status}.`);
  }
  return response.json();
}
const uptoParam = () => (state.upto === null ? "" : `?upto=${state.upto}`);
const idOf = (end) => (typeof end === "object" ? end.id : end);

// ---------------------------------------------------------------- 3D graph
const graph = ForceGraph3D({ controlType: "orbit" })($("graph"))
  .backgroundColor("rgba(0,0,0,0)")
  .showNavInfo(false)
  .nodeRelSize(2.2)
  .nodeVal((n) => 1 + Math.sqrt(n.mentions))
  .nodeColor((n) => KINDS[n.kind]?.color || "#ffffff")
  .nodeOpacity(0.55)
  .nodeLabel(() => "")
  .linkColor((l) => linkColor(l))
  .linkWidth((l) => (isFocusLink(l) ? 0.9 : Math.min(0.2 + l.count * 0.15, 1)))
  .linkOpacity(0.55)
  .linkCurvature((l) => (l.kinds.includes("callback") ? 0.35 : 0))
  .linkDirectionalParticles((l) => (l.kinds.includes("callback") || isFocusLink(l) ? 2 : 0))
  .linkDirectionalParticleWidth(1.6)
  .linkLabel((l) => l.labels.slice(0, 3).join(" / "))
  .onNodeClick((n) => select(n.id))
  .onBackgroundClick(() => closeDossier());

// Spread nodes out so markers and names don't pile up, then frame the whole map once.
graph.d3Force("charge").strength(-220);
graph.d3Force("link").distance(70);
let framed = false;
graph.onEngineStop(() => {
  if (!framed) { framed = true; graph.zoomToFit(reducedMotion ? 0 : 600, 140); }
});

function isFocusLink(l) {
  return state.selected && (idOf(l.source) === state.selected || idOf(l.target) === state.selected);
}
function linkColor(l) {
  const base = LINK_COLORS[l.kinds[0]] || "#4a55c8";
  if (!state.selected) return base;
  return isFocusLink(l) ? base : "rgba(74, 85, 200, 0.18)";
}

// ---------------------------------------------------------------- markers
const markerLayer = $("markers");
let markers = [];

function buildMarkers(nodes) {
  const ranked = [...nodes].sort((a, b) => b.mentions - a.mentions).slice(0, MAX_MARKERS);
  const named = new Set(ranked.slice(0, NAMED_MARKERS).map((n) => n.id));
  markers = ranked.map((node) => {
    const wrap = el("div", "marker");
    const dot = el("button", "marker-dot");
    dot.style.setProperty("--c", KINDS[node.kind]?.color || "#fff");
    dot.setAttribute("aria-label", `${node.name}, ${SINGULAR[node.kind] || node.kind}`);
    paintDot(dot, node);
    dot.onclick = () => select(node.id);
    dot.onmouseenter = () => wrap.classList.remove("is-quiet");
    dot.onmouseleave = () => named.has(node.id) || node.id === state.selected || wrap.classList.add("is-quiet");
    const name = el("span", "marker-name", node.name);
    wrap.append(dot, name);
    if (!named.has(node.id)) wrap.classList.add("is-quiet");
    return { node, wrap, named: named.has(node.id) };
  });
  markerLayer.replaceChildren(...markers.map((m) => m.wrap));
  paintMarkerStates();
}

// Places and moments are whole panels: centre them. People: favour the face.
const SCENE_KINDS = new Set(["location", "event", "faction"]);

function paintDot(dot, node) {
  dot.classList.toggle("has-picture", Boolean(node.portrait));
  dot.classList.toggle("is-scene", SCENE_KINDS.has(node.kind));
  if (node.portrait) {
    const img = el("img");
    img.src = node.portrait;
    img.alt = "";
    img.onerror = () => dot.replaceChildren(icon(node.kind)); // picture removed meanwhile
    dot.replaceChildren(img);
  } else {
    dot.replaceChildren(icon(node.kind));
  }
}

function paintMarkerStates() {
  for (const m of markers) {
    const selected = m.node.id === state.selected;
    m.wrap.classList.toggle("is-selected", selected);
    m.wrap.classList.toggle("is-dim", Boolean(state.selected) && !selected && !state.neighbors.has(m.node.id));
    if (selected) m.wrap.classList.remove("is-quiet");
    else if (!m.named) m.wrap.classList.add("is-quiet");
  }
}

function moveMarkers() {
  for (const { node, wrap } of markers) {
    if (node.x === undefined) continue;
    const { x, y } = graph.graph2ScreenCoords(node.x, node.y, node.z);
    const visible = x > -40 && y > -40 && x < innerWidth + 40 && y < innerHeight + 40;
    wrap.style.display = visible ? "" : "none";
    wrap.style.transform = `translate(${x}px, ${y}px) translate(-50%, -17px)`;
  }
  requestAnimationFrame(moveMarkers);
}
requestAnimationFrame(moveMarkers);

// ---------------------------------------------------------------- data
async function loadWorks() {
  state.works = await getJSON("/api/works");
  const input = $("upto");
  input.max = String(state.works.length);
  input.value = String(state.works.length); // the end of the track = everything
  const ticks = $("ticks");
  ticks.replaceChildren(
    ...state.works.map((work, i) => {
      const tick = el("span");
      tick.style.left = `calc(7px + (100% - 14px) * ${state.works.length ? i / state.works.length : 0})`;
      tick.title = work.label;
      return tick;
    })
  );
  input.oninput = () => showReadingPoint();
  input.onchange = async () => {
    await loadGraph();
    if (state.selected) openDossier(state.selected, { fly: false, animate: false });
  };
  showReadingPoint();
}

function showReadingPoint() {
  const value = Number($("upto").value);
  const everything = value >= state.works.length;
  state.upto = everything ? null : value;
  $("upto-value").textContent = everything ? "Everything" : state.works[value].label;
  $("upto").setAttribute("aria-valuetext", $("upto-value").textContent);
  [...$("ticks").children].forEach((tick, i) => tick.classList.toggle("is-read", everything || i <= value));
}

async function loadGraph() {
  state.data = await getJSON(`/api/graph${uptoParam()}`);
  const empty = state.data.nodes.length === 0;
  $("empty").hidden = !empty;
  if (empty && state.works.length) {
    $("empty-title").textContent = "Nothing yet at this reading point";
    $("empty-text").textContent = "Move the reading point further along to see the story so far.";
  }
  buildRail();
  draw();
  $("names").replaceChildren(...state.data.nodes.map((n) => new Option(n.name)));
}

function draw() {
  const nodes = state.data.nodes.filter((n) => !state.hidden.has(n.kind));
  const keep = new Set(nodes.map((n) => n.id));
  const links = state.data.links.filter((l) => keep.has(idOf(l.source)) && keep.has(idOf(l.target)));
  const depth = $("timeline").checked;
  for (const n of nodes) n.fz = depth ? n.first * TIMELINE_SPACING : undefined;
  graph.graphData({ nodes, links });
  framed = false;
  buildMarkers(nodes);
  const point = state.upto === null ? "the end" : state.works[state.upto].label;
  $("status-text").textContent = `${nodes.length} entries, ${links.length} connections, read up to ${point}`;
}

function buildRail() {
  const counts = {};
  for (const n of state.data.nodes) counts[n.kind] = (counts[n.kind] || 0) + 1;
  $("rail").replaceChildren(
    ...Object.entries(KINDS).map(([kind, { label, color }]) => {
      const button = el("button");
      button.style.setProperty("--c", color);
      button.setAttribute("aria-pressed", String(!state.hidden.has(kind)));
      button.append(icon(kind), el("span", "label", label), el("span", "count", String(counts[kind] || 0)));
      button.onclick = () => {
        state.hidden.has(kind) ? state.hidden.delete(kind) : state.hidden.add(kind);
        button.setAttribute("aria-pressed", String(!state.hidden.has(kind)));
        draw();
      };
      return button;
    })
  );
}

// ---------------------------------------------------------------- selection
function focusCamera(id) {
  const node = graph.graphData().nodes.find((n) => n.id === id);
  if (!node || node.x === undefined) return;
  const distance = 140;
  const length = Math.hypot(node.x, node.y, node.z || 1) || 1;
  const ratio = 1 + distance / length;
  graph.cameraPosition(
    { x: node.x * ratio, y: node.y * ratio, z: (node.z || 1) * ratio },
    node,
    reducedMotion ? 0 : 1100
  );
}

function select(id) {
  openDossier(id, { fly: true, animate: true });
}

function setFocus(id) {
  state.selected = id;
  state.neighbors = new Set();
  if (id) {
    for (const l of state.data.links) {
      if (idOf(l.source) === id) state.neighbors.add(idOf(l.target));
      if (idOf(l.target) === id) state.neighbors.add(idOf(l.source));
    }
  }
  graph.linkColor(graph.linkColor()).linkWidth(graph.linkWidth()).linkDirectionalParticles(graph.linkDirectionalParticles());
  paintMarkerStates();
}

function closeDossier() {
  stopNarration();
  $("dossier").hidden = true;
  setFocus(null);
}

async function openDossier(id, { fly, animate }) {
  let detail;
  try {
    detail = await getJSON(`/api/entity/${id}${uptoParam()}`);
  } catch {
    closeDossier(); // not reached yet at this reading point
    return;
  }
  setFocus(id);
  if (fly) focusCamera(id);
  fillDossier(detail);
  const dossier = $("dossier");
  dossier.hidden = false;
  if (animate) {
    dossier.classList.remove("is-opening");
    void dossier.offsetWidth; // restart the opening animation
    dossier.classList.add("is-opening");
  }
}

function fillDossier(d) {
  $("d-kind").textContent = SINGULAR[d.kind] || d.kind;
  $("d-name").textContent = d.name;
  $("d-aliases").textContent = d.aliases.length ? `Also called ${d.aliases.join(", ")}` : "";
  $("d-file").textContent = `file ${d.slug}`;
  $("d-description").textContent = d.description;
  $("d-description").hidden = !d.description;

  showPicture(d.portrait, d.name, d.kind);
  $("pic-tools").hidden = !state.config.portraits;
  $("pic-status").textContent = "";

  const stats = [
    ["Appearances", d.stats.appearances], ["Issues", d.stats.issues],
    ["Connections", d.stats.relations], ["Callbacks", d.stats.callbacks],
  ];
  $("d-stats").replaceChildren(...stats.map(([label, value]) => {
    const box = el("div");
    box.append(el("dt", null, label), el("dd", null, String(value)));
    return box;
  }));

  const first = d.first_appearance;
  $("d-first").textContent = first ? `First appears in ${first.work}, page ${first.page}` : "No appearances yet";
  const peak = Math.max(1, ...d.presence.map((p) => p.count));
  let seenFirst = false;
  $("d-presence").replaceChildren(...d.presence.map((p) => {
    const bar = el("span");
    if (p.count) {
      bar.className = seenFirst ? "on" : "first";
      seenFirst = true;
      bar.style.height = `${6 + (20 * p.count) / peak}px`;
    }
    bar.title = `${p.label}: ${p.count} appearance${p.count === 1 ? "" : "s"}`;
    return bar;
  }));
  $("d-presence").setAttribute("aria-label", `Appears in ${d.stats.issues} of ${d.presence.length} issues read so far`);

  fillPlot(d.appearances);
  fillLinks(d.relations);
  fillCallbacks(d.callbacks);
  stopNarration();
  $("narration").textContent = "";
  $("narration").classList.remove("is-error");
  $("tab-narrator").hidden = !state.config.narrator;
  if (!state.config.narrator && currentTab === "narrator") showTab("plot");
  else showTab(currentTab);
}

function showPicture(url, name, kind) {
  const img = $("d-img");
  img.hidden = !url;
  img.classList.toggle("is-scene", SCENE_KINDS.has(kind));
  $("d-glyph").style.display = url ? "none" : "";
  if (url) { img.src = url; img.alt = `Picture of ${name}`; }
  $("d-glyph").querySelector("use").setAttribute("href", `#i-${kind}`);
  $("pic-remove").hidden = !url;
  $("pic-replace").textContent = url ? "Replace picture" : "Add a picture";
}

// Your own picture, or none: for when the vision model picked the wrong person.
async function changePicture(method, file) {
  const id = state.selected;
  const node = state.data.nodes.find((n) => n.id === id);
  $("pic-status").textContent = method === "PUT" ? "Cutting it out..." : "Removing...";
  try {
    const answer = await getJSON(`/api/portrait/${id}`, { method, body: file });
    // A new address each time, so the browser doesn't show the old picture from its cache.
    const url = method === "PUT" ? `${answer.portrait}?v=${Date.now()}` : null;
    if (node) {
      node.portrait = url;
      const marker = markers.find((m) => m.node.id === id);
      if (marker) paintDot(marker.wrap.querySelector(".marker-dot"), node);
    }
    if (state.selected === id) showPicture(url, node?.name || "", node?.kind || "character");
    $("pic-status").textContent = method === "PUT" ? "Saved." : "Removed.";
  } catch (error) {
    $("pic-status").textContent = error.message;
  }
}

function fillPlot(appearances) {
  const list = $("pane-plot");
  if (!appearances.length) {
    list.replaceChildren(el("li", "pane-empty", "No plot points up to this reading point."));
    return;
  }
  list.replaceChildren(...appearances.map((a, i) => {
    const li = el("li");
    li.style.setProperty("--i", String(Math.min(i, 12)));
    li.append(el("span", "where", `${a.work}, page ${a.page}`), a.summary);
    if (a.quote) li.append(el("span", "quote", `“${a.quote}”`));
    return li;
  }));
}

function fillLinks(relations) {
  const pane = $("pane-links");
  if (!relations.length) { pane.replaceChildren(el("p", "pane-empty", "No connections up to this reading point.")); return; }
  const groups = {};
  for (const r of relations) (groups[r.kind] ||= []).push(r);
  const parts = [];
  for (const [kind, list] of Object.entries(groups)) {
    const heading = kind.replace("_", " ");
    parts.push(el("h3", null, heading[0].toUpperCase() + heading.slice(1)));
    for (const r of list) {
      const row = el("button", "link-row");
      const meter = el("div", "sentiment");
      const fill = el("span");
      const v = Math.max(-1, Math.min(1, r.sentiment));
      fill.style.left = v < 0 ? `${50 + v * 50}%` : "50%";
      fill.style.width = `${Math.abs(v) * 50}%`;
      fill.style.background = v < 0 ? "var(--alarm)" : "var(--cobalt)";
      meter.append(fill);
      meter.title = `Sentiment ${v.toFixed(1)}`;
      row.append(el("strong", null, r.other), meter, el("small", null, `${r.label}. ${r.work}, page ${r.page}`));
      row.onclick = () => select(r.other_id);
      parts.push(row);
    }
  }
  pane.replaceChildren(...parts);
}

function fillCallbacks(callbacks) {
  const pane = $("pane-callbacks");
  if (!callbacks.length) { pane.replaceChildren(el("li", "pane-empty", "No callbacks found yet. Run litmapr link after adding more issues.")); return; }
  pane.replaceChildren(...callbacks.map((c) => {
    const li = el("li");
    const hop = el("div", "hop");
    hop.append(el("span", null, c.later), el("i"), el("span", null, `${c.earlier} (${c.earlier_entity})`));
    li.append(hop, c.reason);
    return li;
  }));
}

// ---------------------------------------------------------------- tabs
const TABS = ["plot", "links", "callbacks", "narrator"];
let currentTab = "plot";
function showTab(name) {
  currentTab = name;
  for (const tab of TABS) {
    const button = $(`tab-${tab}`);
    const on = tab === name;
    button.setAttribute("aria-selected", String(on));
    button.tabIndex = on ? 0 : -1;
    $(`pane-${tab}`).hidden = !on;
    if (on) {
      const bar = $("tab-bar");
      bar.style.width = `${button.offsetWidth}px`;
      bar.style.transform = `translateX(${button.offsetLeft}px)`;
    }
  }
}
for (const tab of TABS) $(`tab-${tab}`).onclick = () => showTab(tab);
document.querySelector(".tabs").addEventListener("keydown", (event) => {
  if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
  const visible = TABS.filter((t) => !$(`tab-${t}`).hidden);
  const step = event.key === "ArrowRight" ? 1 : -1;
  const next = visible[(visible.indexOf(currentTab) + step + visible.length) % visible.length];
  showTab(next);
  $(`tab-${next}`).focus();
});

// ---------------------------------------------------------------- narrator
// The words stream in from the server as the model writes them. Each finished
// sentence is handed to the browser's own speech engine (no extra software),
// so the narrator starts talking long before it has finished writing.
const speech = "speechSynthesis" in window ? window.speechSynthesis : null;
const prefs = {
  get(key, fallback) { try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch { /* private window: fine */ } },
};
let narration = null; // the narration being written: { controller }
let voices = [];

const voiceOn = () => Boolean(speech) && $("n-voice-on").getAttribute("aria-pressed") === "true";

function voiceScore(v) {
  let score = 0;
  if (/natural|neural|online|premium|enhanced/i.test(v.name)) score += 4; // the good-sounding ones
  if (/^en-(GB|US)/i.test(v.lang)) score += 1;
  return score;
}
function loadVoices() {
  if (!speech) return;
  const all = speech.getVoices();
  voices = all.filter((v) => /^en/i.test(v.lang));
  if (!voices.length) voices = all;
  voices.sort((a, b) => voiceScore(b) - voiceScore(a) || a.name.localeCompare(b.name));
  const select = $("n-voice");
  const wanted = prefs.get("litmapr.voice", "");
  const options = voices.map((v) => new Option(`${v.name} (${v.lang})`, v.name));
  select.replaceChildren(...(options.length ? options : [new Option("System default", "")]));
  if (voices.some((v) => v.name === wanted)) select.value = wanted;
}

function speak(text) {
  if (!voiceOn() || !text.trim()) return;
  const line = new SpeechSynthesisUtterance(text.trim());
  const voice = voices.find((v) => v.name === $("n-voice").value);
  if (voice) { line.voice = voice; line.lang = voice.lang; }
  const briefing = $("n-style").value === "briefing";
  line.rate = briefing ? 1.06 : 0.94; // the storyteller takes its time
  line.pitch = briefing ? 1 : 0.88;
  line.onstart = () => $("dossier").classList.add("is-speaking");
  line.onend = line.onerror = () => setTimeout(refreshNarrateButton, 50);
  speech.speak(line);
}

// Speak every sentence that is complete, starting at `from`. Returns where it stopped.
function speakFinished(text, from) {
  const ends = [...text.slice(from).matchAll(/[.!?…]["”’')]*\s+/g)];
  if (!ends.length) return from;
  const last = ends[ends.length - 1];
  const upTo = from + last.index + last[0].length;
  speak(text.slice(from, upTo));
  return upTo;
}

function isLive() {
  return Boolean(narration) || Boolean(speech && (speech.speaking || speech.pending));
}
function refreshNarrateButton() {
  const live = isLive();
  const button = $("narrate");
  button.textContent = live ? "Stop" : "Narrate";
  button.classList.toggle("is-live", live);
  if (!(speech && (speech.speaking || speech.pending))) $("dossier").classList.remove("is-speaking");
}
function stopNarration() {
  narration?.controller.abort();
  narration = null;
  speech?.cancel();
  refreshNarrateButton();
}

async function narrate() {
  if (isLive()) { stopNarration(); return; }
  const box = $("narration");
  const controller = new AbortController();
  narration = { controller };
  refreshNarrateButton();
  box.classList.remove("is-error");
  const words = document.createTextNode("");
  const caret = el("span", "caret");
  box.replaceChildren(el("span", "n-wait", "Opening the file "), caret);

  const query = new URLSearchParams({ style: $("n-style").value });
  if (state.upto !== null) query.set("upto", state.upto);
  let text = "";
  let spokenTo = 0;
  let shown = 0;
  let streaming = true;
  // Reveal the text a few letters per frame, so even a saved narration types itself out.
  const reveal = () => {
    if (controller.signal.aborted) return; // stopped, or another entry opened
    shown = reducedMotion ? text.length : Math.min(text.length, shown + 3);
    words.textContent = text.slice(0, shown);
    if (shown < text.length || streaming) requestAnimationFrame(reveal);
    else caret.remove();
  };
  try {
    const response = await fetch(`/api/narrate/${state.selected}/stream?${query}`, { signal: controller.signal });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || `The server answered ${response.status}.`);
    }
    box.replaceChildren(words, caret);
    requestAnimationFrame(reveal);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      text += decoder.decode(value, { stream: true });
      const end = text.lastIndexOf("</think>");
      if (end !== -1) {
        // A thinking model wrote out its reasoning first: throw that away and start over.
        text = text.slice(end + "</think>".length).trimStart();
        shown = 0;
        spokenTo = 0;
        speech?.cancel();
      }
      spokenTo = speakFinished(text, spokenTo);
    }
    text = (text + decoder.decode()).trimEnd();
    speak(text.slice(spokenTo));
  } catch (error) {
    if (error.name !== "AbortError") {
      box.classList.add("is-error");
      box.textContent = error.message;
    }
  } finally {
    streaming = false;
    if (narration?.controller === controller) narration = null;
    refreshNarrateButton();
  }
}

function wireNarrator() {
  $("narrate").onclick = narrate;
  $("n-style").value = prefs.get("litmapr.style", "storyteller");
  $("n-style").onchange = () => prefs.set("litmapr.style", $("n-style").value);
  if (!speech) {
    $("n-voice-on").hidden = true;
    $("voice-row").hidden = true;
    document.querySelector(".voice-meter").hidden = true;
    return;
  }
  const on = prefs.get("litmapr.voiceOn", "true") === "true";
  $("n-voice-on").setAttribute("aria-pressed", String(on));
  $("n-voice-on").textContent = on ? "Voice on" : "Voice off";
  $("voice-row").hidden = !on;
  $("n-voice-on").onclick = () => {
    const now = !voiceOn();
    $("n-voice-on").setAttribute("aria-pressed", String(now));
    $("n-voice-on").textContent = now ? "Voice on" : "Voice off";
    $("voice-row").hidden = !now;
    prefs.set("litmapr.voiceOn", String(now));
    if (!now) { speech.cancel(); refreshNarrateButton(); }
  };
  $("n-voice").onchange = () => prefs.set("litmapr.voice", $("n-voice").value);
  loadVoices();
  speech.onvoiceschanged = loadVoices; // voices arrive late in Chrome and Edge
}

// ---------------------------------------------------------------- wiring
function wire() {
  $("close").onclick = closeDossier;
  wireNarrator();
  $("pic-replace").onclick = () => $("pic-file").click();
  $("pic-file").onchange = () => {
    const file = $("pic-file").files[0];
    $("pic-file").value = "";
    if (file) changePicture("PUT", file);
  };
  $("pic-remove").onclick = () => changePicture("DELETE");
  $("timeline").onchange = () => { draw(); graph.d3ReheatSimulation(); };
  const find = () => {
    const wanted = $("search").value.trim().toLowerCase();
    if (!wanted) return;
    const node = state.data.nodes.find(
      (n) => n.name.toLowerCase() === wanted || n.aliases.some((a) => a.toLowerCase() === wanted)
    ) || state.data.nodes.find((n) => n.name.toLowerCase().includes(wanted));
    if (node) select(node.id);
  };
  $("find").onsubmit = (event) => { event.preventDefault(); find(); };
  $("search").onchange = find;
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !$("dossier").hidden) closeDossier();
  });
  window.addEventListener("resize", () => {
    graph.width(innerWidth).height(innerHeight);
    showTab(currentTab);
  });
}

async function start() {
  wire();
  state.config = await getJSON("/api/config");
  await loadWorks();
  await loadGraph();
}

start().catch((error) => {
  $("empty").hidden = false;
  $("empty-title").textContent = "The map couldn't load";
  $("empty-text").textContent = `${error.message} Check that litmapr serve is still running, then reload.`;
});
