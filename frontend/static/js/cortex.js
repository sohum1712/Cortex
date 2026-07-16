/**
 * cortex.js — Client-side logic for the Cortex RAG UI.
 * Layout: floating glassmorphic navbar + right-side drawer.
 */
"use strict";

const cortex = (() => {
  const $ = (id) => document.getElementById(id);

  // ── DOM refs ──────────────────────────────────────────────────────────────
  const views = {
    empty:    $("view-empty"),
    ready:    $("view-ready"),
    thinking: $("view-thinking"),
    noInfo:   $("view-no-info"),
  };

  const searchInput    = $("search-input");
  const btnSend        = $("btn-send");
  const btnClearInput  = $("btn-clear-input");
  const charCounter    = $("char-counter");
  const charHint       = $("char-hint");
  const chatHistory    = $("chat-history");
  const thinkingQuery  = $("thinking-query");
  const fileList       = $("file-list");
  const uploadProgress = $("upload-progress");
  const uploadBar      = $("upload-bar");
  const uploadStatus   = $("upload-status-text");
  const urlInput       = $("url-input");
  const toast          = $("toast");
  const toastMsg       = $("toast-msg");
  const toastIcon      = $("toast-icon");
  const drawer         = $("drawer");
  const drawerBackdrop = $("drawer-backdrop");
  const btnScrollTop   = $("btn-scroll-top");
  const mainEl         = $("main");
  const fileCountBadge  = $("file-count-badge");
  const drawerFileCount = $("drawer-file-count");

  // Citations panel refs
  const citationsPanel  = $("citations-panel");
  const citationsList   = $("citations-list");
  const citationsEmpty  = $("citations-empty");
  const citationsCount  = $("citations-count");
  const citationsNote   = $("citations-note");

  // ── State ─────────────────────────────────────────────────────────────────
  let hasFiles    = document.querySelectorAll(".file-item").length > 0;
  let isStreaming = false;
  let toastTimer  = null;
  let drawerOpen  = false;

  const CHAR_LIMIT = 2000;
  const CHAR_WARN  = 1800;

  const NO_INFO_PHRASES = [
    "i could not find this in the uploaded documents",
    "could not find", "no relevant information",
    "not found in", "not available in",
    "don't have information", "do not have information", "unable to find",
  ];

  // ── Markdown ──────────────────────────────────────────────────────────────
  let md = null;
  if (typeof marked !== "undefined") {
    marked.setOptions({ breaks: true, gfm: true });
    md = marked;
  }

  function renderMarkdown(text) {
    if (!md) return `<p style="white-space:pre-wrap">${escapeHtml(text)}</p>`;
    try { return md.parse(text); }
    catch { return `<p style="white-space:pre-wrap">${escapeHtml(text)}</p>`; }
  }

  function enhanceCodeBlocks(container) {
    container.querySelectorAll("pre").forEach((pre) => {
      if (pre.parentElement.classList.contains("code-block-wrap")) return;
      const wrap = document.createElement("div");
      wrap.className = "code-block-wrap";
      pre.parentNode.insertBefore(wrap, pre);
      wrap.appendChild(pre);
      const btn = document.createElement("button");
      btn.className = "btn-copy-code";
      btn.innerHTML = `<span class="material-symbols-outlined">content_copy</span>COPY`;
      btn.addEventListener("click", () => {
        const code = pre.querySelector("code")?.textContent ?? pre.textContent;
        navigator.clipboard.writeText(code).then(() => {
          btn.innerHTML = `<span class="material-symbols-outlined">check</span>COPIED`;
          setTimeout(() => { btn.innerHTML = `<span class="material-symbols-outlined">content_copy</span>COPY`; }, 1800);
        });
      });
      wrap.appendChild(btn);
    });
  }

  // ── Views ─────────────────────────────────────────────────────────────────
  function showView(name) {
    Object.entries(views).forEach(([key, el]) => {
      if (!el) return;
      el.classList.toggle("hidden", key !== name);
    });
  }

  function clearNamedViews() {
    Object.values(views).forEach((el) => el && el.classList.add("hidden"));
  }

  function resolveInitialView() {
    const hasMsgs = chatHistory.children.length > 0;
    if (hasMsgs) { chatHistory.classList.remove("hidden"); clearNamedViews(); return; }
    if (!hasFiles) {
      showView("empty");
      searchInput.disabled = true;
      btnSend.disabled = true;
      btnSend.classList.add("opacity-40", "cursor-not-allowed");
    } else {
      showView("ready");
      searchInput.disabled = false;
      btnSend.disabled = false;
      btnSend.classList.remove("opacity-40", "cursor-not-allowed");
    }
  }

  // ── Toast ─────────────────────────────────────────────────────────────────
  function showToast(msg, type = "info", duration = 3500) {
    clearTimeout(toastTimer);
    const icons = { info:"info", success:"check_circle", error:"error", warning:"warning" };
    toastIcon.textContent = icons[type] ?? "info";
    toastIcon.className = [
      "material-symbols-outlined flex-shrink-0",
      type==="error" ? "text-red-500" : type==="success" ? "text-green-600" :
      type==="warning" ? "text-amber-500" : "text-primary",
    ].join(" ");
    toastMsg.textContent = msg;
    toast.classList.remove("hidden","hide");
    toast.classList.add("show");
    if (duration > 0) toastTimer = setTimeout(dismissToast, duration);
  }

  function dismissToast() {
    clearTimeout(toastTimer);
    toast.classList.add("hide");
    setTimeout(() => toast.classList.add("hidden"), 300);
  }

  // ── File count badges ─────────────────────────────────────────────────────
  function updateBadges(count) {
    [fileCountBadge, drawerFileCount].forEach((el) => {
      if (!el) return;
      if (count > 0) { el.textContent = count; el.classList.remove("hidden"); }
      else           { el.classList.add("hidden"); }
    });
  }

  // ── File list ─────────────────────────────────────────────────────────────
  function renderFileList(files) {
    const noMsg = $("no-files-msg");
    if (noMsg) noMsg.remove();
    fileList.querySelectorAll(".file-item").forEach((el) => el.remove());

    if (!files || files.length === 0) {
      const p = document.createElement("p");
      p.id = "no-files-msg";
      p.className = "font-label-sm text-label-sm text-on-secondary-container opacity-40 text-center py-6";
      p.textContent = "No files uploaded yet.";
      fileList.appendChild(p);
      hasFiles = false;
      updateBadges(0);
    } else {
      files.forEach((name) => {
        const div = document.createElement("div");
        div.className = "file-item flex items-center justify-between px-3 py-2.5 rounded-xl " +
          "bg-white/50 border border-white/60 group hover:bg-white/80 transition-all";
        div.dataset.name = name;
        div.innerHTML = `
          <div class="flex items-center gap-2.5 overflow-hidden min-w-0">
            <span class="material-symbols-outlined text-[15px] flex-shrink-0 text-on-secondary-container"
                  aria-hidden="true">description</span>
            <span class="truncate font-body-md text-body-md text-sm" title="${escapeHtml(name)}">${escapeHtml(name)}</span>
          </div>
          <button class="btn-delete-file flex-shrink-0 ml-2 p-1 rounded-lg opacity-0 group-hover:opacity-100
                         hover:bg-red-50 transition-all text-on-secondary-container hover:text-red-500"
                  aria-label="Remove ${escapeHtml(name)}">
            <span class="material-symbols-outlined text-[15px]">delete</span>
          </button>`;
        fileList.appendChild(div);
      });
      hasFiles = true;
      updateBadges(files.length);
    }
    resolveInitialView();
  }

  function bindDeleteButtons() {
    fileList.addEventListener("click", async (e) => {
      const btn = e.target.closest(".btn-delete-file");
      if (!btn) return;
      const item = btn.closest(".file-item");
      const name = item?.dataset.name;
      if (!name) return;
      if (!confirm(`Remove "${name}" from the knowledge base?`)) return;
      try {
        const res = await fetch("/api/delete", {
          method:"POST", headers:{"Content-Type":"application/json"},
          body: JSON.stringify({ filename: name }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error ?? "Delete failed");
        showToast(`"${name}" removed.`, "success");
        renderFileList(data.files ?? []);
      } catch {
        item.remove();
        const remaining = Array.from(fileList.querySelectorAll(".file-item")).map(el => el.dataset.name);
        renderFileList(remaining);
        showToast(`Removed "${name}".`, "info");
      }
    });
  }

  // ── Utilities ─────────────────────────────────────────────────────────────
  function escapeHtml(str) {
    return String(str)
      .replace(/&/g,"&amp;").replace(/</g,"&lt;")
      .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
  }

  function scrollToBottom() {
    requestAnimationFrame(() => { mainEl.scrollTop = mainEl.scrollHeight; });
  }

  function isNoInfoResponse(text) {
    const lower = text.toLowerCase();
    return NO_INFO_PHRASES.some((p) => lower.includes(p));
  }

  // ── Citations panel ───────────────────────────────────────────────────────

  /**
   * Clears the panel with a quick fade then rebuilds from `sources`.
   * sources: Array of { source: string, content: string, score?: number }
   */
  function updateCitations(sources) {
    if (!citationsList) return;

    // Fade out existing cards
    citationsList.classList.add("fading");

    setTimeout(() => {
      citationsList.innerHTML   = "";
      citationsList.classList.remove("fading");

      const valid = (sources || []).filter((s) => s && s.source);

      if (!valid.length) {
        citationsEmpty.classList.remove("hidden");
        citationsList.classList.add("hidden");
        citationsNote.classList.add("hidden");
        citationsCount.classList.add("hidden");
        return;
      }

      citationsEmpty.classList.add("hidden");
      citationsList.classList.remove("hidden");
      citationsNote.classList.remove("hidden");
      citationsCount.classList.remove("hidden");
      citationsCount.textContent = valid.length;

      valid.forEach((src, idx) => {
        const score   = typeof src.score === "number" ? src.score : 0;
        // Score is typically cosine similarity 0–1; normalise to 0–100 for the bar
        const pct     = Math.round(Math.min(Math.max(score, 0), 1) * 100);
        const isHigh  = pct >= 70;
        const label   = isHigh ? "High" : pct >= 40 ? "Mid" : "Low";
        const name    = src.source.split(/[\\/]/).pop() || src.source;
        const excerpt = (src.content || "").trim().replace(/\s+/g, " ").slice(0, 160);

        const card = document.createElement("div");
        card.className = "source-card";
        card.style.animationDelay = `${idx * 60}ms`;
        card.innerHTML = `
          <div class="source-card-header">
            <span class="material-symbols-outlined source-card-icon">description</span>
            <span class="source-card-name" title="${escapeHtml(src.source)}">${escapeHtml(name)}</span>
          </div>
          ${excerpt ? `<p class="source-card-excerpt">${escapeHtml(excerpt)}…</p>` : ""}
          <div class="source-card-meta">
            <span class="source-badge ${isHigh ? "relevance-high" : ""}">${label}</span>
            ${pct > 0 ? `
            <div class="score-bar-wrap" title="Relevance: ${pct}%">
              <div class="score-bar-fill" style="width:0%" data-target="${pct}"></div>
            </div>` : ""}
          </div>`;
        citationsList.appendChild(card);

        // Animate score bar after paint
        if (pct > 0) {
          requestAnimationFrame(() => {
            setTimeout(() => {
              const fill = card.querySelector(".score-bar-fill");
              if (fill) fill.style.width = pct + "%";
            }, 80 + idx * 60);
          });
        }
      });
    }, 160);
  }

  /** Reset citations to empty state (called on new query start) */
  function clearCitations() {
    if (!citationsList) return;
    citationsList.classList.add("fading");
    setTimeout(() => {
      citationsList.innerHTML = "";
      citationsList.classList.remove("fading", "hidden");
      citationsList.classList.add("hidden");
      citationsEmpty.classList.remove("hidden");
      citationsNote.classList.add("hidden");
      citationsCount.classList.add("hidden");
    }, 160);
  }

  // ── Chat messages ─────────────────────────────────────────────────────────
  function appendUserMessage(text) {
    chatHistory.classList.remove("hidden");
    clearNamedViews();
    const div = document.createElement("div");
    div.className = "msg-user";
    div.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
    chatHistory.appendChild(div);
    scrollToBottom();
  }

  function createAnswerCard(queryText) {
    const wrap = document.createElement("div");
    wrap.className = "msg-assistant";
    wrap.innerHTML = `
      <div class="bubble-wrap">
        <div class="answer-card">
          <div class="answer-header">
            <div class="w-9 h-9 bg-black rounded-xl flex items-center justify-center flex-shrink-0">
              <span class="material-symbols-outlined text-white text-[18px]"
                    style="font-variation-settings:'FILL' 1;" aria-hidden="true">psychology</span>
            </div>
            <div class="min-w-0">
              <p class="font-label-sm text-label-sm uppercase tracking-tighter text-on-secondary-container">
                Cortex Analysis
              </p>
              <p class="font-body-md text-body-md font-bold truncate max-w-[200px] md:max-w-sm"
                 title="${escapeHtml(queryText)}">${escapeHtml(queryText)}</p>
            </div>
          </div>
          <div class="answer-body streaming-cursor"></div>
          <div class="feedback-row hidden">
            <div class="flex items-center gap-4">
              <button class="btn-copy" title="Copy answer">
                <span class="material-symbols-outlined text-base">content_copy</span>Copy
              </button>
              <button class="btn-regen" title="Regenerate">
                <span class="material-symbols-outlined text-base">refresh</span>Regenerate
              </button>
            </div>
            <div class="flex items-center gap-3">
              <button class="btn-thumb-up flex items-center gap-1 hover:text-black transition-colors"
                      aria-label="Mark as helpful">
                <span class="material-symbols-outlined text-base">thumb_up</span>
              </button>
              <button class="btn-thumb-down flex items-center gap-1 hover:text-black transition-colors"
                      aria-label="Mark as not helpful">
                <span class="material-symbols-outlined text-base">thumb_down</span>
              </button>
            </div>
          </div>
        </div>
      </div>`;
    chatHistory.appendChild(wrap);
    scrollToBottom();

    const bodyEl     = wrap.querySelector(".answer-body");
    const feedbackEl = wrap.querySelector(".feedback-row");

    function finalize(fullText) {
      bodyEl.classList.remove("streaming-cursor");
      bodyEl.innerHTML = renderMarkdown(fullText);
      enhanceCodeBlocks(bodyEl);
      feedbackEl.classList.remove("hidden");
      wrap.querySelector(".btn-copy").addEventListener("click", () => {
        navigator.clipboard.writeText(fullText).then(() => showToast("Copied!", "success"));
      });
      wrap.querySelector(".btn-regen").addEventListener("click", () => {
        const q = wrap.querySelector(".answer-card .font-bold")?.getAttribute("title") ||
                  wrap.querySelector(".answer-card .font-bold")?.textContent;
        if (q) sendQuery(q.trim());
      });
      const tu = wrap.querySelector(".btn-thumb-up");
      const td = wrap.querySelector(".btn-thumb-down");
      tu.addEventListener("click", () => {
        tu.querySelector(".material-symbols-outlined").style.fontVariationSettings = "'FILL' 1";
        td.querySelector(".material-symbols-outlined").style.fontVariationSettings = "'FILL' 0";
        showToast("Thanks for the feedback!", "success");
      });
      td.addEventListener("click", () => {
        td.querySelector(".material-symbols-outlined").style.fontVariationSettings = "'FILL' 1";
        tu.querySelector(".material-symbols-outlined").style.fontVariationSettings = "'FILL' 0";
        showToast("Thanks — we'll improve.", "info");
      });
    }

    return { wrap, bodyEl, finalize };
  }

  // ── Send query (SSE) ──────────────────────────────────────────────────────
  async function sendQuery(query) {
    if (isStreaming || !query.trim()) return;
    if (!hasFiles) { showToast("Upload a PDF first.", "warning"); return; }

    isStreaming = true;
    searchInput.value = "";
    autoGrowTextarea(); updateCharCounter();
    searchInput.disabled = true;
    btnSend.disabled = true;
    btnClearInput.classList.add("hidden");

    appendUserMessage(query);
    thinkingQuery.textContent = query.length > 60 ? query.slice(0, 57) + "…" : query;
    showView("thinking");
    scrollToBottom();
    clearCitations(); // reset panel for new query

    const { wrap: cardWrap, bodyEl, finalize } = createAnswerCard(query);
    cardWrap.style.display = "none";
    let fullText  = "";
    let firstToken = true;
    let sources   = []; // collected from SSE

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: query }),
      });
      if (!res.ok) throw new Error(`Server error ${res.status}`);

      const reader  = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer    = "";

      outer: while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          const raw = line.slice(5).trim();
          if (!raw) continue;
          let evt;
          try { evt = JSON.parse(raw); } catch { continue; }
          if (evt.error) throw new Error(evt.error);
          if (evt.token !== undefined) {
            if (firstToken) { clearNamedViews(); cardWrap.style.display = ""; firstToken = false; }
            fullText += evt.token;
            bodyEl.textContent = fullText;
            scrollToBottom();
          }
          if (evt.done) {
            finalize(fullText);
            if (isNoInfoResponse(fullText)) { cardWrap.remove(); showView("noInfo"); }
            // Sources arrive on the done event from the backend
            updateCitations(evt.sources || sources);
            break outer;
          }
          // Also handle sources arriving as a standalone event (future-proofing)
          if (evt.sources) { sources = evt.sources; }
        }
      }
    } catch (err) {
      console.error("Chat error:", err);
      cardWrap.remove();
      showToast(`Error: ${err.message}`, "error", 6000);
      if (chatHistory.children.length === 0 && hasFiles) showView("ready");
      else if (!hasFiles) showView("empty");
    } finally {
      isStreaming = false;
      searchInput.disabled = false;
      btnSend.disabled = false;
      searchInput.focus();
      scrollToBottom();
    }
  }

  // ── Upload files ──────────────────────────────────────────────────────────
  async function uploadFiles(files) {
    if (!files || files.length === 0) return;
    const pdfs = Array.from(files).filter((f) => f.name.toLowerCase().endsWith(".pdf"));
    if (pdfs.length === 0) { showToast("Only PDF files are supported.", "warning"); return; }

    uploadProgress.classList.remove("hidden");
    uploadBar.style.width = "10%";
    uploadStatus.textContent = `Uploading ${pdfs.length} file(s)…`;

    const form = new FormData();
    pdfs.forEach((f) => form.append("files", f));

    try {
      uploadBar.style.width = "40%";
      const res = await fetch("/api/upload", { method:"POST", body:form });
      uploadBar.style.width = "80%";
      if (!res.ok) throw new Error(`Upload failed (${res.status})`);
      const data = await res.json();
      uploadBar.style.width = "100%";

      const indexed = data.results.filter((r) => r.status === "indexed").length;
      const skipped = data.results.filter((r) => r.status.startsWith("skipped")).length;
      const errors  = data.results.filter((r) => r.status.startsWith("error")).length;

      let msg = `Indexed ${indexed} file(s).`;
      if (skipped) msg += ` ${skipped} skipped.`;
      if (errors)  msg += ` ${errors} failed.`;
      showToast(msg, errors ? "warning" : "success");
      renderFileList(data.files ?? []);
      if (!drawerOpen) openDrawer();
    } catch (err) {
      showToast(`Upload error: ${err.message}`, "error");
    } finally {
      setTimeout(() => { uploadProgress.classList.add("hidden"); uploadBar.style.width = "0%"; }, 800);
      // reset all file inputs
      ["file-input","file-input-drawer","file-input-empty"].forEach(id => { const el=$(id); if(el) el.value=""; });
    }
  }

  // ── URL indexing ──────────────────────────────────────────────────────────
  async function indexUrls() {
    const raw = urlInput.value.trim();
    if (!raw) { showToast("Enter at least one URL.", "warning"); return; }
    const urls = raw.split("\n").map((u) => u.trim()).filter(Boolean);
    const btn = $("btn-add-urls");
    btn.disabled = true;
    btn.innerHTML = `<span class="material-symbols-outlined text-base" style="animation:spin 1s linear infinite">refresh</span>Indexing…`;
    try {
      const res  = await fetch("/api/url", { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({urls}) });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? "Unknown error");
      showToast(`Indexed ${data.count} URL(s).`, "success");
      urlInput.value = "";
    } catch (err) {
      showToast(`URL error: ${err.message}`, "error");
    } finally {
      btn.disabled = false;
      btn.innerHTML = `<span class="material-symbols-outlined text-base">language</span>Index URLs`;
    }
  }

  // ── KB actions ────────────────────────────────────────────────────────────
  async function rebuildKB() {
    if (!confirm("Rebuild the knowledge base? This re-indexes all uploaded PDFs.")) return;
    showToast("Rebuilding knowledge base…", "info", 0);
    try {
      const res  = await fetch("/api/rebuild", { method:"POST" });
      const data = await res.json();
      showToast("Knowledge base rebuilt.", "success");
      renderFileList(data.files ?? []);
    } catch (err) { showToast(`Rebuild error: ${err.message}`, "error"); }
  }

  async function clearKB() {
    if (!confirm("Delete ALL uploaded files and the knowledge base?")) return;
    try {
      await fetch("/api/clear", { method:"POST" });
      renderFileList([]);
      chatHistory.innerHTML = "";
      chatHistory.classList.add("hidden");
      showToast("Knowledge base cleared.", "success");
    } catch (err) { showToast(`Clear error: ${err.message}`, "error"); }
  }

  async function resetConversation() {
    try {
      await fetch("/api/reset", { method:"POST" });
      chatHistory.innerHTML = "";
      chatHistory.classList.add("hidden");
      resolveInitialView();
      showToast("Conversation reset.", "success");
    } catch (err) { showToast(`Reset error: ${err.message}`, "error"); }
  }

  // ── Drawer ────────────────────────────────────────────────────────────────
  function openDrawer() {
    drawer.classList.add("open");
    drawerBackdrop.classList.remove("hidden");
    requestAnimationFrame(() => drawerBackdrop.classList.add("visible"));
    drawerOpen = true;
    $("btn-open-drawer").setAttribute("aria-expanded", "true");
    setTimeout(() => $("drawer-close")?.focus(), 320);
  }

  function closeDrawer() {
    drawer.classList.remove("open");
    drawerBackdrop.classList.remove("visible");
    setTimeout(() => drawerBackdrop.classList.add("hidden"), 320);
    drawerOpen = false;
    $("btn-open-drawer").setAttribute("aria-expanded", "false");
    $("btn-open-drawer").focus();
  }

  // ── Search textarea auto-grow + char counter ──────────────────────────────
  function autoGrowTextarea() {
    searchInput.style.height = "auto";
    searchInput.style.height = Math.min(searchInput.scrollHeight, 160) + "px";
  }

  function updateCharCounter() {
    const len = searchInput.value.length;
    charCounter.classList.toggle("hidden", len <= 50);
    charCounter.textContent = `${len}`;
    if (len >= CHAR_WARN) {
      charCounter.classList.add("text-amber-500");
      charHint.textContent = `${CHAR_LIMIT - len} chars remaining`;
      charHint.classList.remove("hidden");
    } else {
      charCounter.classList.remove("text-amber-500");
      charHint.classList.add("hidden");
    }
    btnClearInput.classList.toggle("hidden", len === 0);
  }

  // ── Scroll to top ─────────────────────────────────────────────────────────
  function initScrollToTop() {
    mainEl.addEventListener("scroll", () => {
      btnScrollTop.classList.toggle("visible", mainEl.scrollTop > 300);
    });
    btnScrollTop.addEventListener("click", () => mainEl.scrollTo({ top:0, behavior:"smooth" }));
  }

  // ── Parallax blobs ────────────────────────────────────────────────────────
  function initParallax() {
    const blobs = document.querySelectorAll(".blob");
    if (!blobs.length || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    document.addEventListener("mousemove", (e) => {
      const mx = (e.clientX / window.innerWidth  - 0.5) / 40;
      const my = (e.clientY / window.innerHeight - 0.5) / 40;
      blobs.forEach((b, i) => {
        const f = i === 0 ? 1 : -1.3;
        b.style.transform = `translate(${mx*f*60}px,${my*f*60}px)`;
      });
    });
  }

  // ── Suggestion chips ──────────────────────────────────────────────────────
  function initChips() {
    document.querySelectorAll(".suggestion-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        if (!hasFiles) { showToast("Upload a PDF first.", "warning"); return; }
        searchInput.value = chip.textContent.trim();
        autoGrowTextarea(); updateCharCounter();
        searchInput.focus();
      });
      chip.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); chip.click(); }
      });
    });
  }

  // ── Drag & drop ───────────────────────────────────────────────────────────
  function initDragDrop() {
    const zone = $("drop-zone");
    if (zone) {
      ["dragenter","dragover"].forEach((ev) =>
        zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("border-black/30","bg-black/[0.02]"); })
      );
      ["dragleave","drop"].forEach((ev) =>
        zone.addEventListener(ev, () => zone.classList.remove("border-black/30","bg-black/[0.02]"))
      );
      zone.addEventListener("drop", (e) => { e.preventDefault(); uploadFiles(e.dataTransfer.files); });
    }
    mainEl.addEventListener("dragover", (e) => e.preventDefault());
    mainEl.addEventListener("drop", (e) => { e.preventDefault(); if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files); });
  }

  // ── Keyboard shortcuts ────────────────────────────────────────────────────
  function initKeyboardShortcuts() {
    document.addEventListener("keydown", (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        if (!searchInput.disabled) searchInput.focus();
        else showToast("Upload a PDF first.", "warning");
      }
      if (e.key === "Escape" && drawerOpen) closeDrawer();
    });
  }

  // ── Public ────────────────────────────────────────────────────────────────
  function focusInput() {
    searchInput.focus();
    showView(hasFiles ? "ready" : "empty");
  }

  // ── Init ──────────────────────────────────────────────────────────────────
  function init() {
    resolveInitialView();
    initParallax();
    initChips();
    initDragDrop();
    initScrollToTop();
    initKeyboardShortcuts();
    bindDeleteButtons();
    updateBadges(document.querySelectorAll(".file-item").length);

    // Send
    btnSend.addEventListener("click", () => sendQuery(searchInput.value.trim()));
    searchInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendQuery(searchInput.value.trim()); }
    });
    searchInput.addEventListener("input", () => { autoGrowTextarea(); updateCharCounter(); });
    btnClearInput.addEventListener("click", () => {
      searchInput.value = ""; autoGrowTextarea(); updateCharCounter(); searchInput.focus();
    });

    // All file inputs route to same handler
    ["file-input","file-input-drawer","file-input-empty"].forEach(id => {
      const el = $(id);
      if (el) el.addEventListener("change", () => uploadFiles(el.files));
    });

    // Drawer
    $("btn-open-drawer").addEventListener("click", () => drawerOpen ? closeDrawer() : openDrawer());
    $("drawer-close").addEventListener("click", closeDrawer);
    drawerBackdrop.addEventListener("click", closeDrawer);

    // KB actions
    $("btn-add-urls").addEventListener("click", indexUrls);
    $("btn-rebuild").addEventListener("click", rebuildKB);
    $("btn-clear").addEventListener("click", clearKB);
    $("btn-reset-conv").addEventListener("click", resetConversation);
    $("btn-new-session").addEventListener("click", resetConversation);

    // Toast dismiss
    $("toast-close")?.addEventListener("click", dismissToast);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();

  return { focusInput, sendQuery };
})();
