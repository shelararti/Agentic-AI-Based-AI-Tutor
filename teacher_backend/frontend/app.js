const state = { subjectId: null, subjectTitle: "", topicId: null, topicTitle: "", module: "teach", mode: "chat" };
const el = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const MODULES = ["teach", "quiz", "code"];
const MODES = ["chat", "lesson", "notes"];

function setStatus(t) { el("status-text").textContent = t; }

async function api(path, options) {
  const res = await fetch(path, options);
  let body = null;
  try { body = await res.json(); } catch (e) { /* non-JSON error body */ }
  if (!res.ok) throw new Error((body && body.detail) || `Request failed (${res.status})`);
  return body;
}
const postJSON = (path, data) => api(path, {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });

// ---- navigation: three modules, plus Teach modes ------------------------
function render() {
  const ready = !!state.topicId;
  el("no-topic").hidden = ready;
  MODULES.forEach((m) => { el(`mod-${m}`).hidden = !ready || state.module !== m; });
  document.querySelectorAll(".module-tab").forEach((b) =>
    b.setAttribute("aria-selected", String(b.dataset.module === state.module)));
  document.querySelectorAll(".seg-btn").forEach((b) =>
    b.setAttribute("aria-selected", String(b.dataset.mode === state.mode)));
  MODES.forEach((m) => { el(`teach-${m}`).hidden = state.mode !== m; });
}

function goModule(m) {
  state.module = MODULES.includes(m) ? m : "teach";
  try { history.replaceState(null, "", `#${state.module}`); } catch (e) { /* ignore */ }
  render();
}
document.querySelectorAll(".module-tab").forEach((b) => b.addEventListener("click", () => goModule(b.dataset.module)));
document.querySelectorAll(".seg-btn").forEach((b) => b.addEventListener("click", () => { state.mode = b.dataset.mode; render(); }));

// ---- sidebar: notes and topic ---------------------------------------------
function pickButton(id, title, meta, active, attr) {
  return `<button class="pick${active ? " is-active" : ""}" type="button" data-${attr}="${esc(id)}" data-title="${esc(title)}">
    ${esc(title)}${meta ? `<small>${esc(meta)}</small>` : ""}</button>`;
}

async function loadSubjects() {
  const subjects = await api("/api/subjects");
  el("subject-list").innerHTML = subjects.map((s) =>
    pickButton(s.id, s.title, `${s.source === "uploaded" ? "your notes" : "ready-made"} · ${s.topic_count} topics`,
      s.id === state.subjectId, "subject")).join("");
  el("subject-list").querySelectorAll(".pick").forEach((c) =>
    c.addEventListener("click", () => selectSubject(c.dataset.subject, c.dataset.title)));
}

async function selectSubject(id, title) {
  state.subjectId = id; state.subjectTitle = title; state.topicId = null;
  const topics = await api(`/api/topics?subject_id=${encodeURIComponent(id)}`);
  document.querySelectorAll("#subject-list .pick").forEach((b) => b.classList.toggle("is-active", b.dataset.subject === id));
  el("topic-empty").hidden = true;
  el("topic-list").innerHTML = topics.map((t) => pickButton(t.id, t.title, "", false, "topic")).join("");
  el("topic-list").querySelectorAll(".pick").forEach((c) =>
    c.addEventListener("click", () => selectTopic(c.dataset.topic, c.dataset.title)));
  el("ask-subject").textContent = title;
  resetAsk();
  render();
}

function selectTopic(id, title) {
  state.topicId = id; state.topicTitle = title;
  document.querySelectorAll("#topic-list .pick").forEach((b) => b.classList.toggle("is-active", b.dataset.topic === id));
  document.querySelectorAll(".topic-name").forEach((n) => (n.textContent = title));
  ["teach", "quiz", "code"].forEach((k) => (el(`out-${k}`).innerHTML = ""));
  el("quiz-score").hidden = true;
  resetTutor();
  render();
}

async function afterSourceCreated(data) {
  el("source-status").textContent = `Created "${data.subject.title}" with ${data.topics.length} topics.`;
  await selectSubject(data.subject.id, data.subject.title);
  await loadSubjects();
}

el("pdf-submit").addEventListener("click", async () => {
  const file = el("pdf-input").files[0];
  if (!file) { el("source-status").textContent = "Choose a PDF first."; return; }
  const form = new FormData();
  form.append("file", file);
  if (el("pdf-title").value.trim()) form.append("title", el("pdf-title").value.trim());
  setStatus("reading PDF…");
  try { await afterSourceCreated(await api("/api/subjects/upload", { method: "POST", body: form })); }
  catch (e) { el("source-status").textContent = e.message; }
  setStatus("ready");
});

el("notes-submit").addEventListener("click", async () => {
  const text = el("notes-text").value;
  if (!text.trim()) { el("source-status").textContent = "Paste some notes first."; return; }
  setStatus("building topics…");
  try { await afterSourceCreated(await postJSON("/api/subjects/from-notes", { title: el("notes-title").value, text })); }
  catch (e) { el("source-status").textContent = e.message; }
  setStatus("ready");
});

// ---- shared chat helpers ------------------------------------------------------
function bubble(logId, cls, html) {
  const log = el(logId);
  const d = document.createElement("div");
  d.className = `msg ${cls}`;
  d.innerHTML = html;
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
  return d;
}

// ---- Teach > Class chat ---------------------------------------------------------
const tutorHistory = [];
const TUTOR_EMPTY = el("tutor-empty").outerHTML;

function resetTutor() {
  tutorHistory.length = 0;
  el("tutor-log").innerHTML = TUTOR_EMPTY;
  el("tutor-log").querySelectorAll(".topic-name").forEach((n) => (n.textContent = state.topicTitle));
  el("tutor-start").addEventListener("click", () => tutorTurn(""));
  el("tutor-chips").hidden = true;
  el("tutor-input").disabled = true; el("tutor-send").disabled = true;
}

async function tutorTurn(message) {
  if (!state.topicId) return;
  const empty = el("tutor-empty");
  if (empty) empty.remove();
  el("tutor-input").disabled = true; el("tutor-send").disabled = true;
  if (message) bubble("tutor-log", "msg-user", esc(message));
  const pending = bubble("tutor-log", "msg-bot msg-pending", "Your teacher is thinking…");
  setStatus("teacher is answering…");
  try {
    const d = await postJSON("/api/teach-chat", {
      topic_id: state.topicId, message, history: tutorHistory,
      level: el("level-select").value, language: el("language-select").value,
    });
    pending.className = "msg msg-bot";
    pending.innerHTML = `${esc(d.reply)}<small>from: ${d.sources.map(esc).join(", ")}</small>`;
    if (message) tutorHistory.push({ role: "user", text: message });
    tutorHistory.push({ role: "assistant", text: d.reply.slice(0, 2000) });
  } catch (e) {
    pending.className = "msg msg-bot msg-error";
    pending.textContent = e.message;
  }
  el("tutor-log").scrollTop = el("tutor-log").scrollHeight;
  el("tutor-chips").hidden = false;
  el("tutor-input").disabled = false; el("tutor-send").disabled = false;
  el("tutor-input").focus();
  setStatus("ready");
}

function tutorSend() {
  const m = el("tutor-input").value.trim();
  if (!m) return;
  el("tutor-input").value = "";
  tutorTurn(m);
}
el("tutor-send").addEventListener("click", tutorSend);
el("tutor-input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.isComposing) tutorSend(); });
document.querySelectorAll("#tutor-chips .chip").forEach((c) =>
  c.addEventListener("click", () => { if (!el("tutor-send").disabled) tutorTurn(c.textContent); }));

// ---- Teach > Ask my notes (ReAct agent) + memory -------------------------------
const askHistory = [];
const ASK_EMPTY = el("ask-log").innerHTML;

function resetAsk() { askHistory.length = 0; el("ask-log").innerHTML = ASK_EMPTY; el("ask-subject").textContent = state.subjectTitle; }

function renderSteps(steps) {
  if (!steps.length) return "";
  const rows = steps.map((s) => `<li>${s.thought ? `<i>${esc(s.thought)}</i><br>` : ""}
    <b>${esc(s.action || "?")}</b>(${esc(s.input)}) → ${esc(s.observation)}</li>`).join("");
  return `<details><summary>How I found this (${steps.length} tool ${steps.length === 1 ? "call" : "calls"})</summary><ol>${rows}</ol></details>`;
}

async function loadMemory() {
  try {
    const { facts } = await api("/api/memory");
    el("mem-count").textContent = facts.length;
    el("mem-list").innerHTML = facts.map((f) => `<li>${esc(f)}</li>`).join("");
  } catch (e) { /* memory is optional */ }
}

async function ask() {
  const q = el("ask-input").value.trim();
  if (!q || !state.subjectId) return;
  el("ask-input").value = "";
  el("ask-btn").disabled = true;
  document.querySelector("#ask-log .chat-empty")?.remove();
  bubble("ask-log", "msg-user", esc(q));
  const pending = bubble("ask-log", "msg-bot msg-pending", "Searching your notes…");
  setStatus("asking…");
  try {
    const d = await postJSON("/api/ask", { subject_id: state.subjectId, question: q,
      language: el("language-select").value, history: askHistory });
    pending.className = "msg msg-bot";
    pending.innerHTML = `${esc(d.answer)}${renderSteps(d.steps)}${d.sources.length ? `<small>from: ${d.sources.map(esc).join(", ")}</small>` : ""}`;
    askHistory.push({ role: "user", text: q }, { role: "assistant", text: d.answer.slice(0, 2000) });
    loadMemory();
  } catch (e) {
    pending.className = "msg msg-bot msg-error";
    pending.textContent = e.message;
  }
  el("ask-log").scrollTop = el("ask-log").scrollHeight;
  el("ask-btn").disabled = false;
  setStatus("ready");
}
el("ask-btn").addEventListener("click", ask);
el("ask-input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.isComposing) ask(); });
el("mem-clear").addEventListener("click", async () => { await api("/api/memory", { method: "DELETE" }); loadMemory(); });

// ---- generated outputs: lesson, quiz, code --------------------------------------
function renderMeta(d) {
  const g = d.grounding;
  const pct = g && g.score != null ? `${Math.round(g.score * 100)}% · ` : "";
  const sources = (d.sources || []).map((s) =>
    `<li><b>${esc(s.topic)}</b>${s.score != null ? ` (${s.score})` : ""}<br>${esc(s.snippet)}…</li>`).join("");
  return `<div class="meta">
    ${g ? `<div><b>Grounding check:</b> ${pct}${esc(g.label)}</div>` : ""}
    <details><summary>Sources used (${(d.sources || []).length}) · retrieval: ${esc(d.retrieval || "n/a")}</summary><ul>${sources}</ul></details>
  </div>`;
}

function renderLesson(d) {
  return `<div class="card"><div class="lesson-text">${esc(d.lesson)}</div></div><div class="card">${renderMeta(d)}</div>`;
}

let quizState = { total: 0, answered: 0, correct: 0 };
function renderQuiz(d) {
  quizState = { total: d.questions.length, answered: 0, correct: 0 };
  const sc = el("quiz-score");
  sc.hidden = false;
  sc.textContent = `Score: 0 / ${quizState.total} answered`;
  const qs = d.questions.map((q, i) => `
    <div class="card quiz-q" data-correct="${q.correct_index}">
      ${q.grounded === false ? `<div class="q-flag">Not clearly supported by your notes. Verify before using.</div>` : ""}
      <div class="q-text">${i + 1}. ${esc(q.question)}</div>
      <div class="opts">${q.options.map((o, j) =>
        `<button class="opt" type="button" data-i="${j}"><span class="letter">${"ABCDEFGH"[j] || j + 1}</span><span>${esc(o)}</span></button>`).join("")}</div>
      <div class="explain" hidden>${q.explanation ? esc(q.explanation) : ""}</div>
    </div>`).join("");
  return qs + `<div class="card">${renderMeta(d)}</div>`;
}

el("out-quiz").addEventListener("click", (e) => {
  const opt = e.target.closest(".opt");
  if (!opt || opt.disabled) return;
  const q = opt.closest(".quiz-q");
  const correct = Number(q.dataset.correct), picked = Number(opt.dataset.i);
  q.querySelectorAll(".opt").forEach((b) => {
    b.disabled = true;
    if (Number(b.dataset.i) === correct) b.classList.add("is-correct");
  });
  if (picked !== correct) opt.classList.add("is-wrong");
  q.querySelector(".explain").hidden = false;
  quizState.answered += 1;
  if (picked === correct) quizState.correct += 1;
  el("quiz-score").textContent = quizState.answered === quizState.total
    ? `Final score: ${quizState.correct} / ${quizState.total}`
    : `Score: ${quizState.correct} correct · ${quizState.answered} of ${quizState.total} answered`;
});

function renderCode(d) {
  return `<div class="card"><div class="q-text">Task</div><p>${esc(d.task)}</p></div>
    ${d.rubric.length ? `<div class="card"><div class="q-text">Rubric</div><ul class="rubric">${d.rubric.map((r) => `<li>${esc(r)}</li>`).join("")}</ul></div>` : ""}
    ${d.solution ? `<div class="card"><button class="btn" id="reveal-solution" type="button">Show sample solution</button>
      <pre class="code" id="solution-block" hidden>${esc(d.solution)}</pre></div>` : ""}
    <div class="card">${renderMeta(d)}</div>`;
}
el("out-code").addEventListener("click", (e) => {
  if (e.target.id !== "reveal-solution") return;
  const block = el("solution-block");
  block.hidden = !block.hidden;
  e.target.textContent = block.hidden ? "Show sample solution" : "Hide sample solution";
});

const RENDERERS = { teach: renderLesson, quiz: renderQuiz, code: renderCode };

document.querySelectorAll(".gen-btn").forEach((btn) =>
  btn.addEventListener("click", async () => {
    const kind = btn.dataset.kind;
    const out = el(`out-${kind}`);
    btn.disabled = true;
    if (kind === "quiz") el("quiz-score").hidden = true;
    out.innerHTML = `<p class="loading">Generating… a small local model can take up to a minute.</p>`;
    setStatus(`generating ${kind}…`);
    try {
      const data = await postJSON("/api/generate", {
        kind, topic_id: state.topicId,
        level: el("level-select").value,
        language: el("language-select").value,
        num_questions: Number(el("num-questions").value),
      });
      out.innerHTML = RENDERERS[kind](data);
    } catch (e) {
      out.innerHTML = `<p class="hint-error">${esc(e.message)}</p>`;
    }
    btn.disabled = false;
    setStatus("ready");
  }));

// ---- init ------------------------------------------------------------------------
(async () => {
  const fromHash = location.hash.replace("#", "");
  if (MODULES.includes(fromHash)) state.module = fromHash;
  render();
  const langs = await api("/api/languages");
  el("language-select").innerHTML = langs.map((l) => `<option>${esc(l)}</option>`).join("");
  await loadSubjects();
  loadMemory();
  try {
    const s = await api("/api/system");
    el("sys-text").textContent =
      `${s.llm.backend} · ${s.llm.model} · embeddings: ${s.embeddings.model} (${s.embeddings.faiss ? "FAISS" : "NumPy"}) · ` +
      `Langfuse: ${s.observability.langfuse_enabled ? "on" : s.observability.langfuse_configured ? "configured" : "off"}`;
  } catch (e) { /* footer is informational */ }
})();
