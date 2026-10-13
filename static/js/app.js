const chatEl = document.getElementById("chat");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("question");
const submitBtn = document.getElementById("submitBtn");
const traceEl = document.getElementById("trace");
const sourceUsedEl = document.getElementById("sourceUsed");
const kbGradeVal = document.getElementById("kbGradeVal");
const webGradeVal = document.getElementById("webGradeVal");
const retryCountVal = document.getElementById("retryCountVal");

const openUploadBtn = document.getElementById("openUpload");
const closeUploadBtn = document.getElementById("closeUpload");
const uploadModal = document.getElementById("uploadModal");
const uploadBtn = document.getElementById("uploadBtn");
const adminKeyInput = document.getElementById("adminKey");
const fileInput = document.getElementById("fileInput");
const fileInfo = document.getElementById("fileInfo");
const uploadStatus = document.getElementById("uploadStatus");

const MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024; // 15MB
const ALLOWED_EXTENSIONS = [".pdf", ".txt", ".md", ".docx"];

// Modal handlers
if (openUploadBtn) {
  openUploadBtn.addEventListener("click", () => {
    uploadModal.classList.remove("hidden");
    if (adminKeyInput) adminKeyInput.focus();
  });
}

if (closeUploadBtn) {
  closeUploadBtn.addEventListener("click", () => {
    uploadModal.classList.add("hidden");
    if (uploadStatus) uploadStatus.innerText = "";
  });
}

// File input selection listener to show file details
if (fileInput) {
  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (file && fileInfo) {
      const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
      fileInfo.classList.remove("hidden");
      fileInfo.innerText = `Selected: ${file.name} (${sizeMb} MB)`;
    } else if (fileInfo) {
      fileInfo.classList.add("hidden");
      fileInfo.innerText = "";
    }
  });
}

// Preset example prompts
document.querySelectorAll(".example").forEach((btn) => {
  btn.addEventListener("click", () => {
    questionInput.value = btn.innerText;
    questionInput.focus();
  });
});

/**
 * Safely escape plain text for HTML insertion
 */
function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/**
 * Validate URL to allow only safe HTTP/HTTPS links
 */
function isSafeUrl(url) {
  if (!url || typeof url !== "string") return false;
  try {
    const parsed = new URL(url);
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch {
    return false;
  }
}

/**
 * Parse inline markdown (bold, italic, code, links) and append safe DOM nodes
 */
function appendInlineMarkdown(parentEl, text) {
  if (!text) return;

  // Unescape escaped markdown formatting delimiters (\**, \*, \___, \__, \_, \`)
  // Leaves math asterisks (e.g. 5 * 10 = 50) intact as they are surrounded by spaces
  const unescapedText = text.replace(/\\(\*{1,3}|_{1,2}|`)/g, "$1");

  // Regex tokenizing in order of precedence:
  // 1. Inline code: `...`
  // 2. Links: [...](...) (handles balanced parens in URL)
  // 3. Bold-Italic: ***...***
  // 4. Bold: **...** or __...__
  // 5. Italic: *...* (not surrounded by whitespace) or _..._ (word-boundary safe)
  const tokenRegex = /(`[^`]+`|\[[^\]]+\]\(((?:[^\s()]|\([^)]*\))+)\)|\*\*\*(?!\s)(.+?)(?<!\s)\*\*\*|\*\*(?!\s)(.+?)(?<!\s)\*\*|__(?!\s)(.+?)(?<!\s)__|(?<!\*)\*(?!\s|\*)(.+?)(?<!\s|\*)\*(?!\*)|(?<!\w)_(?!\s|_)(.+?)(?<!\s|_)_(?!\w))/g;
  let lastIndex = 0;
  let match;

  while ((match = tokenRegex.exec(unescapedText)) !== null) {
    // Append preceding plain text
    if (match.index > lastIndex) {
      parentEl.appendChild(document.createTextNode(unescapedText.substring(lastIndex, match.index)));
    }

    const token = match[0];
    if (token.startsWith("`") && token.endsWith("`")) {
      const code = document.createElement("code");
      code.textContent = token.slice(1, -1);
      parentEl.appendChild(code);
    } else if (token.startsWith("***") && token.endsWith("***") && token.length >= 6) {
      const strong = document.createElement("strong");
      const em = document.createElement("em");
      appendInlineMarkdown(em, token.slice(3, -3));
      strong.appendChild(em);
      parentEl.appendChild(strong);
    } else if (token.startsWith("**") && token.endsWith("**") && token.length >= 4) {
      const strong = document.createElement("strong");
      appendInlineMarkdown(strong, token.slice(2, -2));
      parentEl.appendChild(strong);
    } else if (token.startsWith("__") && token.endsWith("__") && token.length >= 4) {
      const strong = document.createElement("strong");
      appendInlineMarkdown(strong, token.slice(2, -2));
      parentEl.appendChild(strong);
    } else if (token.startsWith("*") && token.endsWith("*") && token.length >= 2) {
      const em = document.createElement("em");
      appendInlineMarkdown(em, token.slice(1, -1));
      parentEl.appendChild(em);
    } else if (token.startsWith("_") && token.endsWith("_") && token.length >= 2) {
      const em = document.createElement("em");
      appendInlineMarkdown(em, token.slice(1, -1));
      parentEl.appendChild(em);
    } else if (token.startsWith("[") && token.includes("](")) {
      const endLabel = token.indexOf("](");
      const label = token.slice(1, endLabel);
      const url = token.slice(endLabel + 2, -1).trim();

      if (isSafeUrl(url)) {
        const a = document.createElement("a");
        a.href = url;
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        a.textContent = label;
        parentEl.appendChild(a);
      } else {
        parentEl.appendChild(document.createTextNode(label));
      }
    }

    lastIndex = match.index + token.length;
  }

  if (lastIndex < unescapedText.length) {
    parentEl.appendChild(document.createTextNode(unescapedText.substring(lastIndex)));
  }
}

/**
 * Render Markdown content (headings, lists, tables, blockquotes, paragraphs) safely into a DOM container
 */
function renderMarkdownToDOM(container, markdownText) {
  if (!markdownText) return;

  // Unescape backslash-escaped markdown formatting characters (e.g. \**, \*, \|, \#, \`, \_)
  // Literal math asterisks surrounded by spaces (5 * 10 = 50) remain intact as literal math
  const cleanMarkdown = markdownText.replace(/\\([*_{}`#|\[\]()])/g, "$1");

  const lines = cleanMarkdown.split("\n");
  let i = 0;

  function isBlockStart(line) {
    if (!line) return false;
    const t = line.trim();
    if (!t) return false;
    if (t.match(/^(#{1,4})\s+/)) return true;
    if (t.startsWith(">")) return true;
    if (t.startsWith("|")) return true;
    if (t.match(/^[-*]\s+/)) return true;
    if (t.match(/^\d+\.\s+/)) return true;
    return false;
  }

  while (i < lines.length) {
    const rawLine = lines[i];
    const trimmed = rawLine.trim();

    if (!trimmed) {
      i++;
      continue;
    }

    // 1. Headings (# h1, ## h2, ### h3, #### h4)
    const headingMatch = trimmed.match(/^(#{1,4})\s+(.*)$/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      const h = document.createElement(`h${level}`);
      appendInlineMarkdown(h, headingMatch[2].trim());
      container.appendChild(h);
      i++;
      continue;
    }

    // 2. Blockquote (> quote)
    if (trimmed.startsWith(">")) {
      const bq = document.createElement("blockquote");
      const quoteText = trimmed.replace(/^>\s*/, "");
      appendInlineMarkdown(bq, quoteText);
      container.appendChild(bq);
      i++;
      continue;
    }

    // 3. Tables (starts with | or contains table separator)
    if (trimmed.startsWith("|")) {
      const tableLines = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        tableLines.push(lines[i].trim());
        i++;
      }

      // Unpack any embedded divider rows or concatenated rows on single lines
      const expandedLines = [];
      tableLines.forEach((tl) => {
        const parts = tl.split(/\\n|<br\s*\/?>/i);
        parts.forEach((p) => {
          const pt = p.trim();
          if (!pt) return;
          const dividerMatch = pt.match(/\|\s*[:]?[-]{2,}[:]?\s*(?:\|\s*[:]?[-]{2,}[:]?\s*)+\|/);
          if (dividerMatch && dividerMatch.index > 0) {
            const before = pt.substring(0, dividerMatch.index).trim();
            const divider = dividerMatch[0].trim();
            const after = pt.substring(dividerMatch.index + dividerMatch[0].length).trim();
            if (before) expandedLines.push(before);
            expandedLines.push(divider);
            if (after) expandedLines.push(after);
          } else {
            expandedLines.push(pt);
          }
        });
      });

      if (expandedLines.length >= 2) {
        const table = document.createElement("table");
        // Row 0 is header
        const headerCols = expandedLines[0]
          .slice(1, expandedLines[0].endsWith("|") ? -1 : undefined)
          .split("|")
          .map((c) => c.trim());
        const numCols = headerCols.length;

        if (numCols > 0) {
          const thead = document.createElement("thead");
          const trHead = document.createElement("tr");
          headerCols.forEach((colText) => {
            const th = document.createElement("th");
            appendInlineMarkdown(th, colText);
            trHead.appendChild(th);
          });
          thead.appendChild(trHead);
          table.appendChild(thead);

          const tbody = document.createElement("tbody");
          const isDivider = (l) => l.replace(/[\s\-|:]/g, "").length === 0;
          const startRow = isDivider(expandedLines[1]) ? 2 : 1;

          const addRowToTbody = (colValues) => {
            const tr = document.createElement("tr");
            colValues.forEach((colText) => {
              const td = document.createElement("td");
              appendInlineMarkdown(td, colText);
              tr.appendChild(td);
            });
            tbody.appendChild(tr);
          };

          for (let r = startRow; r < expandedLines.length; r++) {
            const rowText = expandedLines[r].trim();
            if (!rowText.startsWith("|")) continue;

            const cells = rowText
              .slice(1, rowText.endsWith("|") ? -1 : undefined)
              .split("|")
              .map((c) => c.trim());

            if (cells.length > numCols) {
              // Multiple rows concatenated on a single line
              let cur = [];
              for (let c = 0; c < cells.length; c++) {
                if (cells[c] === "" && (cur.length === 0 || cur.length === numCols)) {
                  if (cur.length === numCols) {
                    addRowToTbody(cur);
                    cur = [];
                  }
                  continue;
                }
                cur.push(cells[c]);
                if (cur.length === numCols) {
                  addRowToTbody(cur);
                  cur = [];
                }
              }
              if (cur.length > 0) {
                while (cur.length < numCols) cur.push("");
                addRowToTbody(cur);
              }
            } else {
              while (cells.length < numCols) cells.push("");
              addRowToTbody(cells);
            }
          }

          table.appendChild(tbody);
          container.appendChild(table);
          continue;
        }
      }
    }

    // 4. Unordered List (- item or * item)
    if (trimmed.match(/^[-*]\s+/)) {
      const ul = document.createElement("ul");
      while (i < lines.length && lines[i].trim().match(/^[-*]\s+/)) {
        const itemText = lines[i].trim().replace(/^[-*]\s+/, "");
        const li = document.createElement("li");
        appendInlineMarkdown(li, itemText);
        ul.appendChild(li);
        i++;
      }
      container.appendChild(ul);
      continue;
    }

    // 5. Ordered List (1. item)
    if (trimmed.match(/^\d+\.\s+/)) {
      const ol = document.createElement("ol");
      while (i < lines.length && lines[i].trim().match(/^\d+\.\s+/)) {
        const itemText = lines[i].trim().replace(/^\d+\.\s+/, "");
        const li = document.createElement("li");
        appendInlineMarkdown(li, itemText);
        ol.appendChild(li);
        i++;
      }
      container.appendChild(ol);
      continue;
    }

    // 6. Regular Paragraph (accumulate consecutive non-empty lines that don't start a new block)
    const pLines = [];
    while (i < lines.length && lines[i].trim() && !isBlockStart(lines[i])) {
      pLines.push(lines[i].trim());
      i++;
    }
    const p = document.createElement("p");
    appendInlineMarkdown(p, pLines.join(" "));
    container.appendChild(p);
  }
}

/**
 * Append chat message bubble to conversation stream.
 */
function appendMessage(sender, text, meta = null) {
  const row = document.createElement("div");
  row.className = `message ${sender}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = sender === "user" ? "You" : "HR";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  if (sender === "user") {
    const p = document.createElement("p");
    p.textContent = text;
    bubble.appendChild(p);
  } else {
    // Render Markdown for assistant responses safely
    renderMarkdownToDOM(bubble, text);
  }

  // Badge for grounded source
  if (meta && meta.source_used) {
    const badge = document.createElement("span");
    badge.className = "answer-source";
    if (meta.source_used === "private_kb") {
      badge.classList.add("source-private");
      badge.textContent = "Verified Private KB";
    } else if (meta.source_used === "web_search") {
      badge.classList.add("source-web");
      badge.textContent = "External Web Search (Requires HR Verification)";
    } else if (meta.source_used === "direct") {
      badge.classList.add("source-direct");
      badge.textContent = "Direct Response";
    } else {
      badge.classList.add("source-insufficient");
      badge.textContent = "Insufficient Evidence";
    }
    bubble.appendChild(badge);
  }

  // Citations list
  if (meta && Array.isArray(meta.citations) && meta.citations.length > 0) {
    const citesContainer = document.createElement("div");
    citesContainer.className = "citations";

    const strong = document.createElement("strong");
    strong.textContent = "Citations:";
    citesContainer.appendChild(strong);
    citesContainer.appendChild(document.createElement("br"));

    meta.citations.forEach((c) => {
      const line = document.createElement("div");
      line.style.marginTop = "4px";

      if (c.type === "web" && isSafeUrl(c.url)) {
        const bullet = document.createTextNode("• ");
        const a = document.createElement("a");
        a.href = c.url;
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        a.textContent = c.title || c.url;
        line.appendChild(bullet);
        line.appendChild(a);
      } else {
        const titleText = c.title || (c.page ? `Internal Document (Page ${c.page})` : "Internal Document");
        line.textContent = `• ${titleText}`;
      }
      citesContainer.appendChild(line);
    });

    bubble.appendChild(citesContainer);
  }

  row.appendChild(avatar);
  row.appendChild(bubble);
  chatEl.appendChild(row);
  chatEl.scrollTop = chatEl.scrollHeight;
  return row;
}

/**
 * Update trace list and real-time execution metrics
 */
function updateTrace(traces, finalSource, metrics = {}) {
  if (traceEl) {
    traceEl.innerHTML = "";
    if (!traces || traces.length === 0) {
      traceEl.innerHTML = '<div class="empty">No trace details available.</div>';
    } else {
      traces.forEach((t) => {
        const item = document.createElement("div");
        item.className = "trace-item";
        item.textContent = t;
        traceEl.appendChild(item);
      });
    }
  }

  if (sourceUsedEl) {
    sourceUsedEl.textContent = finalSource ? finalSource.toUpperCase() : "—";
  }

  if (kbGradeVal) {
    kbGradeVal.textContent = metrics.kb_grade || "—";
  }
  if (webGradeVal) {
    webGradeVal.textContent = metrics.web_grade || "—";
  }
  if (retryCountVal) {
    retryCountVal.textContent = typeof metrics.retry_count === "number" ? metrics.retry_count : "0";
  }
}

// Ingestion API Call
if (uploadBtn) {
  uploadBtn.addEventListener("click", async () => {
    const key = adminKeyInput ? adminKeyInput.value.trim() : "";
    const file = fileInput && fileInput.files ? fileInput.files[0] : null;

    if (!key) {
      uploadStatus.innerText = "Admin authorization key is required.";
      uploadStatus.style.color = "#f87171";
      if (adminKeyInput) adminKeyInput.focus();
      return;
    }

    if (!file) {
      uploadStatus.innerText = "Please select a document to upload.";
      uploadStatus.style.color = "#f87171";
      return;
    }

    // Client-side extension validation
    const lowerName = file.name.toLowerCase();
    const hasValidExt = ALLOWED_EXTENSIONS.some((ext) => lowerName.endsWith(ext));
    if (!hasValidExt) {
      uploadStatus.innerText = `Unsupported format. Allowed: ${ALLOWED_EXTENSIONS.join(", ")}`;
      uploadStatus.style.color = "#f87171";
      return;
    }

    // Client-side size limit validation
    if (file.size > MAX_FILE_SIZE_BYTES) {
      uploadStatus.innerText = `File exceeds max upload limit of 15MB (${(file.size / (1024 * 1024)).toFixed(2)} MB).`;
      uploadStatus.style.color = "#f87171";
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    uploadStatus.innerText = "Parsing, chunking, embedding locally, and upserting into Pinecone...";
    uploadStatus.style.color = "#8bc4ff";
    uploadBtn.disabled = true;

    try {
      const res = await fetch("/api/documents/ingest", {
        method: "POST",
        headers: { "x-admin-key": key },
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) {
        uploadStatus.innerText = data.detail || `Upload failed with status ${res.status}`;
        uploadStatus.style.color = "#f87171";
      } else {
        uploadStatus.innerText = `Success: Indexed ${data.chunks} chunks (${data.file}) into Pinecone (dim=${data.dimension}).`;
        uploadStatus.style.color = "#4ade80";
        if (fileInput) fileInput.value = "";
        if (fileInfo) {
          fileInfo.classList.add("hidden");
          fileInfo.innerText = "";
        }
        // Sensitive admin key is kept in runtime memory only and cleared after operation
        if (adminKeyInput) adminKeyInput.value = "";
      }
    } catch (err) {
      uploadStatus.innerText = `Network or service failure: ${err.message}`;
      uploadStatus.style.color = "#f87171";
    } finally {
      uploadBtn.disabled = false;
    }
  });
}

// Chat API Call
if (chatForm) {
  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = questionInput ? questionInput.value.trim() : "";
    if (!q) return;

    appendMessage("user", q);
    questionInput.value = "";
    submitBtn.disabled = true;

    // Show initial trace and thinking bubble
    const loadingBubble = appendMessage("assistant", "Evaluating company policy knowledge base...");
    updateTrace(["1. Intent Classification: Routing employee query..."], "Routing...");

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });

      const data = await res.json();

      // Remove loading placeholder bubble
      if (loadingBubble && loadingBubble.parentNode) {
        loadingBubble.parentNode.removeChild(loadingBubble);
      }

      if (!res.ok) {
        appendMessage("assistant", `Error: ${data.detail || "Unable to complete request."}`);
        updateTrace([`Workflow halted: ${data.detail || "Server Error"}`], "FAILED");
      } else {
        appendMessage("assistant", data.answer, {
          source_used: data.source_used,
          citations: data.citations,
        });

        updateTrace(data.trace, data.source_used, {
          kb_grade: data.kb_grade,
          web_grade: data.web_grade,
          retry_count: data.retry_count,
        });
      }
    } catch (err) {
      if (loadingBubble && loadingBubble.parentNode) {
        loadingBubble.parentNode.removeChild(loadingBubble);
      }
      appendMessage(
        "assistant",
        "Connection failure: Unable to reach the PeoplePrime HR Copilot service. Please verify your network."
      );
      updateTrace([`Network connection failure: ${err.message}`], "ERROR");
    } finally {
      submitBtn.disabled = false;
      if (questionInput) questionInput.focus();
    }
  });
}
