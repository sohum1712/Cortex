/**
 * cortex.js — Client-side logic for the Cortex RAG UI.
 *
 * UI states
 * ---------
 *   empty    — no files uploaded; search bar disabled
 *   ready    — files present; waiting for first question
 *   thinking — SSE stream in progress (skeleton + shimmer)
 *   answer   — response rendered in chat history
 *   no-info  — model returned a "not found" response
 */

"use strict";

const cortex = (() => {
  // ── DOM refs ─────────────────────────────────────────────────────────────
  const $ = (id) => document.getElementById(id);

  const views = {
    empty:   $("view-empty"),
    ready:   $("view-ready"),
    thinking:$("view-thinking"),
    noInfo:  $("view-no-info"),
  };

  const searchInput    = $("search-input");
  const btnSend        = $("btn-send");
  const chatHistory    = $("chat-history");
  const thinkingQuery  = $("thinking-query");
  const fileInput      = $("file-input");
  const fileList       = $("file-list");
  const uploadProgress = $("upload-progress");
  const uploadBar      = $("upload-bar");
  const uploadStatus   = $("upload-status-text");
  const urlInput       = $("url-input");
  const toast          = $("toast");
  const toastMsg       = $("toast-msg");
  const toastIcon      = $("toast-icon");

  // ── State ─────────────────────────────────────────────────────────────────
  let hasFiles    = document.querySelectorAll(".file-item").length > 0;
  let isStreaming = false;
  let toastTimer  = null;

  // Phrases that indicate the LLM found nothing relevant in the documents
  const NO_INFO_PHRASES = [
    "i could not find this in the uploaded documents",
    "could not find",
    "no relevant information",
    "not found in",
    "not available in",
    "don't have information",
    "do not have information",
    "unable to find",
  ];

  // ── View management ───────────────────────────────────────────────────────

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
    if (hasMsgs) {
      chatHistory.classList.remove("hidden");
      clearNamedViews();
      return;
    }
    if (!hasFiles) {
      showView("empty");
      searchInput.disabled = true;
      btnSend.disabled     = true;
      btnSend.classList.add("opacity-40", "cursor-not-allowed");
    } else {
      showView("ready");
      searchInput.disabled = false;
      btnSend.disabled     = false;
      btnSend.classList.remove("opacity-40", "cursor-not-allowed");
    }
  }

  // ── Toast ──────────────────────────────────────────────────────────────────

  function showToast(msg, type = "info", duration = 3500) {
    clearTimeout(toastTimer);
    const icons = { info: "info", success: "check_circle", error: "error", warning: "warning" };
    toastIcon.textContent = icons[type] ?? "info";
    toastIcon.className = [
      "material-symbols-outlined",
      type === "error"   ? "text-red-500"   :
      type === "success" ? "text-green-600" : "text-primary",
    ].join(" ");
    toastMsg.textContent = msg;
    toast.classList.remove("hidden", "hide");
    toast.classList.add("show");
    toastTimer = setTimeout(() => {
      toast.classList.add("hide");
      setTimeout(() => toast.classList.add("hidden"), 300);
    }, duration);
  }

  // ── File list UI ───────────────────────────────────────────────────────────

  function renderFileList(files) {
    const noMsg = $("no-files-msg");
    if (noMsg) noMsg.remove();
    fileList.querySelectorAll(".file-item").forEach((el) => el.remove());

    if (!files || files.length === 0) {
      const p  = document.createElement("p");
      p.id     = "no-files-msg";
      p.className = "font-label-sm text-label-sm text-on-secondary-container opacity-50 text-center py-4";
      p.textContent = "No files uploaded yet.";
      fileList.appendChild(p);
      hasFiles = false;
    } else {
      files.forEach((name) => {
        const div = document.createElement("div");
        div.className  = "file-item flex items-center p-3 rounded-lg bg-white/10 group";
        div.dataset.name = name;
        div.innerHTML  = `
          <div class="flex items-center gap-3 overflow-hidden">
            <span class="material-symbols-outlined text-sm">description</span>
            <span class="truncate font-body-md text-body-md text-sm">${escapeHtml(name)}</span>
          </div>`;
        fileList.appendChild(div);
      });
      hasFiles = true;
    }
    resolveInitialView();
  }

  // ── Utility ────────────────────────────────────────────────────────────────

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function scrollToBottom() {
    requestAnimationFrame(() => {
      const main = $("main");
      main.scrollTop = main.scrollHeight;
    });
  }

  function isNoInfoResponse(text) {
    const lower = text.toLowerCase();
    return NO_INFO_PHRASES.some((p) => lower.includes(p));
  }

  // ── Chat messages ──────────────────────────────────────────────────────────

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
            <div class="w-10 h-10 bg-black rounded-lg flex items-center justify-center flex-shrink-0">
              <span class="material-symbols-outlined text-white"
                    style="font-variation-settings:'FILL' 1;">psychology</span>
            </div>
            <div>
              <p class="font-label-sm text-label-sm uppercase tracking-tighter text-on-secondary-container">
                Cortex Analysis
              </p>
              <p class="font-body-md text-body-md font-bold truncate max-w-xs">
                ${escapeHtml(queryText)}
              </p>
            </div>
          </div>
          <div class="answer-body streaming-cursor"></div>
          <div class="feedback-row hidden">
            <div class="flex items-center gap-6">
              <button class="btn-copy" title="Copy answer">
                <span class="material-symbols-outlined text-lg">content_copy</span>Copy
              </button>
              <button class="btn-regen" title="Regenerate">
                <span class="material-symbols-outlined text-lg">refresh</span>Regenerate
              </button>
            </div>
            <div class="flex items-center gap-4">
              <span class="material-symbols-outlined text-lg cursor-pointer hover:text-black">thumb_up</span>
              <span class="material-symbols-outlined text-lg cursor-pointer hover:text-black">thumb_down</span>
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
      feedbackEl.classList.remove("hidden");
      wrap.querySelector(".btn-copy").addEventListener("click", () => {
        navigator.clipboard.writeText(fullText)
          .then(() => showToast("Copied to clipboard", "success"));
      });
      wrap.querySelector(".btn-regen").addEventListener("click", () => {
        const q = wrap.querySelector(".answer-card .font-bold")?.textContent;
        if (q) sendQuery(q);
      });
    }

    return { wrap, bodyEl, finalize };
  }

  // ── Core: send query via SSE ───────────────────────────────────────────────

  async function sendQuery(query) {
    if (isStreaming || !query.trim()) return;
    if (!hasFiles) {
      showToast("Upload a PDF first to start querying.", "warning");
      return;
    }

    isStreaming           = true;
    searchInput.value     = "";
    searchInput.disabled  = true;
    btnSend.disabled      = true;

    appendUserMessage(query);

    thinkingQuery.textContent = query.length > 60 ? query.slice(0, 57) + "…" : query;
    showView("thinking");
    scrollToBottom();

    const { wrap: cardWrap, bodyEl, finalize } = createAnswerCard(query);
    cardWrap.style.display = "none";

    try {
      const res = await fetch("/api/chat", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ message: query }),
      });

      if (!res.ok) throw new Error(`Server error ${res.status}`);

      const reader    = res.body.getReader();
      const decoder   = new TextDecoder();
      let buffer      = "";
      let fullText    = "";
      let firstToken  = true;

      outer: while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop();           // hold the incomplete chunk

        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          const raw = line.slice(5).trim();
          if (!raw) continue;

          let evt;
          try { evt = JSON.parse(raw); } catch { continue; }

          if (evt.error)         throw new Error(evt.error);

          if (evt.token !== undefined) {
            if (firstToken) {
              clearNamedViews();
              cardWrap.style.display = "";
              firstToken = false;
            }
            fullText           += evt.token;
            bodyEl.textContent  = fullText;
            scrollToBottom();
          }

          if (evt.done) {
            finalize(fullText);
            if (isNoInfoResponse(fullText)) {
              cardWrap.remove();
              showView("noInfo");
            }
            break outer;
          }
        }
      }
    } catch (err) {
      console.error("Chat error:", err);
      cardWrap.remove();
      showToast(`Error: ${err.message}`, "error", 6000);
      if (chatHistory.children.length === 0 && hasFiles) showView("ready");
      else if (!hasFiles) showView("empty");
    } finally {
      isStreaming           = false;
      searchInput.disabled  = false;
      btnSend.disabled      = false;
      searchInput.focus();
      scrollToBottom();
    }
  }

  // ── File upload ────────────────────────────────────────────────────────────

  async function uploadFiles(files) {
    if (!files || files.length === 0) return;
    const pdfs = Array.from(files).filter((f) => f.name.toLowerCase().endsWith(".pdf"));
    if (pdfs.length === 0) { showToast("Only PDF files are supported.", "warning"); return; }

    uploadProgress.classList.remove("hidden");
    uploadBar.style.width     = "10%";
    uploadStatus.textContent  = `Uploading ${pdfs.length} file(s)…`;

    const form = new FormData();
    pdfs.forEach((f) => form.append("files", f));

    try {
      uploadBar.style.width = "40%";
      const res = await fetch("/api/upload", { method: "POST", body: form });
      uploadBar.style.width = "80%";
      if (!res.ok) throw new Error(`Upload failed (${res.status})`);

      const data    = await res.json();
      uploadBar.style.width = "100%";

      const indexed = data.results.filter((r) => r.status === "indexed").length;
      const skipped = data.results.filter((r) => r.status.startsWith("skipped")).length;
      const errors  = data.results.filter((r) => r.status.startsWith("error")).length;

      let msg = `Indexed ${indexed} file(s).`;
      if (skipped) msg += ` ${skipped} skipped.`;
      if (errors)  msg += ` ${errors} failed.`;

      showToast(msg, errors ? "warning" : "success");
      renderFileList(data.files ?? []);
    } catch (err) {
      showToast(`Upload error: ${err.message}`, "error");
    } finally {
      setTimeout(() => {
        uploadProgress.classList.add("hidden");
        uploadBar.style.width = "0%";
      }, 800);
      fileInput.value = "";
    }
  }

  // ── URL indexing ───────────────────────────────────────────────────────────

  async function indexUrls() {
    const raw  = urlInput.value.trim();
    if (!raw) { showToast("Enter at least one URL.", "warning"); return; }
    const urls = raw.split("\n").map((u) => u.trim()).filter(Boolean);

    const btn     = $("btn-add-urls");
    btn.disabled  = true;
    btn.textContent = "Indexing…";

    try {
      const res  = await fetch("/api/url", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ urls }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? "Unknown error");
      showToast(`Indexed ${data.count} URL(s).`, "success");
      urlInput.value = "";
    } catch (err) {
      showToast(`URL error: ${err.message}`, "error");
    } finally {
      btn.disabled  = false;
      btn.innerHTML = `<span class="material-symbols-outlined text-base">language</span>Index URLs`;
    }
  }

  // ── Knowledge base actions ─────────────────────────────────────────────────

  async function rebuildKB() {
    if (!confirm("Rebuild the knowledge base? This re-indexes all uploaded PDFs.")) return;
    showToast("Rebuilding knowledge base…", "info", 60_000);
    try {
      const res  = await fetch("/api/rebuild", { method: "POST" });
      const data = await res.json();
      showToast("Knowledge base rebuilt.", "success");
      renderFileList(data.files ?? []);
    } catch (err) {
      showToast(`Rebuild error: ${err.message}`, "error");
    }
  }

  async function clearKB() {
    if (!confirm("This will delete ALL uploaded files and the knowledge base. Continue?")) return;
    try {
      await fetch("/api/clear", { method: "POST" });
      renderFileList([]);
      chatHistory.innerHTML = "";
      chatHistory.classList.add("hidden");
      showToast("Knowledge base cleared.", "success");
    } catch (err) {
      showToast(`Clear error: ${err.message}`, "error");
    }
  }

  async function resetConversation() {
    try {
      await fetch("/api/reset", { method: "POST" });
      chatHistory.innerHTML = "";
      chatHistory.classList.add("hidden");
      resolveInitialView();
      showToast("Conversation reset.", "success");
    } catch (err) {
      showToast(`Reset error: ${err.message}`, "error");
    }
  }

  // ── Sidebar toggle ─────────────────────────────────────────────────────────

  function toggleSidebar() {
    $("sidebar").classList.toggle("collapsed");
    $("main").classList.toggle("expanded");
  }

  // ── Mouse parallax on background blobs ────────────────────────────────────

  function initParallax() {
    const blobs = document.querySelectorAll(".blob");
    if (!blobs.length) return;
    document.addEventListener("mousemove", (e) => {
      const mx = (e.clientX / window.innerWidth  - 0.5) / 40;
      const my = (e.clientY / window.innerHeight - 0.5) / 40;
      blobs.forEach((blob, i) => {
        const f = i === 0 ? 1 : -1.3;
        blob.style.transform = `translate(${mx * f * 60}px, ${my * f * 60}px)`;
      });
    });
  }

  // ── Suggestion chips ───────────────────────────────────────────────────────

  function initChips() {
    document.querySelectorAll(".suggestion-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        if (!hasFiles) { showToast("Upload a PDF first.", "warning"); return; }
        searchInput.value = chip.textContent.trim();
        searchInput.focus();
      });
    });
  }

  // ── Drag & drop ────────────────────────────────────────────────────────────

  function initDragDrop() {
    const zone = $("drop-zone");
    if (!zone) return;
    ["dragenter", "dragover"].forEach((evt) =>
      zone.addEventListener(evt, (e) => { e.preventDefault(); zone.classList.add("bg-white/30"); })
    );
    ["dragleave", "drop"].forEach((evt) =>
      zone.addEventListener(evt, () => zone.classList.remove("bg-white/30"))
    );
    zone.addEventListener("drop", (e) => { e.preventDefault(); uploadFiles(e.dataTransfer.files); });
  }

  // ── Public helpers ─────────────────────────────────────────────────────────

  function focusInput() {
    searchInput.focus();
    showView(hasFiles ? "ready" : "empty");
  }

  // ── Initialise ────────────────────────────────────────────────────────────

  function init() {
    resolveInitialView();
    initParallax();
    initChips();
    initDragDrop();

    btnSend.addEventListener("click", () => sendQuery(searchInput.value.trim()));

    searchInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendQuery(searchInput.value.trim());
      }
    });

    fileInput.addEventListener("change", () => uploadFiles(fileInput.files));
    $("btn-add-urls").addEventListener("click",   indexUrls);
    $("btn-rebuild").addEventListener("click",    rebuildKB);
    $("btn-clear").addEventListener("click",      clearKB);
    $("btn-reset-conv").addEventListener("click", resetConversation);
    $("btn-new-session").addEventListener("click",resetConversation);
    $("menu-toggle").addEventListener("click",    toggleSidebar);
    $("nav-files").addEventListener("click", () =>
      fileList.scrollIntoView({ behavior: "smooth" })
    );
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  return { focusInput, sendQuery };
})();
