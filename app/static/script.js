const chat = document.getElementById("chat");
const input = document.getElementById("q");
const sendBtn = document.getElementById("send");

// ── Render a message ────────────────────────────────
function addMsg(role, text, sources) {
  const isUser = role === "You";

  const msg = document.createElement("div");
  msg.className = "msg";

  // Build sources HTML
  let sourcesHtml = "";
  if (sources && sources.length > 0) {
    const clean = (str) => str ? str.replace(/\*\*/g, "").replace(/\s+/g, " ").trim() : "";
    const shorten = (str) => str.length > 40 ? str.substring(0, 37) + "..." : str;
    const chips = sources
      .map((s) => {
        let section = clean(s.section);
        let chapter = clean(s.chapter);
        let label = section ? shorten(section) : chapter ? shorten(chapter) : s.source || "source";
        if (s.page != null) label += `, p.${s.page}`;
        return `<span class="source-chip">${label}</span>`;
      })
      .join("");
    sourcesHtml = `<div class="sources">${chips}</div>`;
  }

  msg.innerHTML = `
    <div class="msg-inner">
      <div class="avatar ${isUser ? "avatar-user" : "avatar-assistant"}">
        ${isUser ? "U" : "AI"}
      </div>
      <div class="msg-content">
        <div class="msg-role">${isUser ? "You" : "Assistant"}</div>
        <div class="msg-text">${text}</div>
        ${sourcesHtml}
      </div>
    </div>`;

  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
  return msg;
}

// ── Typing indicator ────────────────────────────────
function addTyping() {
  const msg = document.createElement("div");
  msg.className = "msg";
  msg.id = "typing";

  msg.innerHTML = `
    <div class="msg-inner">
      <div class="avatar avatar-assistant">AI</div>
      <div class="msg-content">
        <div class="msg-role">Assistant</div>
        <div class="typing-indicator">
          <span></span><span></span><span></span>
        </div>
      </div>
    </div>`;

  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
}

function removeTyping() {
  const el = document.getElementById("typing");
  if (el) el.remove();
}

// ── Send query ──────────────────────────────────────
async function ask() {
  const question = input.value.trim();
  if (!question) return;

  input.value = "";
  addMsg("You", question);

  // Disable input while waiting
  sendBtn.disabled = true;
  input.disabled = true;
  addTyping();

  try {
    const res = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    removeTyping();

    if (!res.ok) {
      const errText = await res.text();
      addMsg("Assistant", "Error: " + errText);
      return;
    }

    const data = await res.json();
    addMsg("Assistant", data.answer, data.sources || []);
  } catch (err) {
    removeTyping();
    addMsg("Assistant", "Network error. Please try again.");
  } finally {
    sendBtn.disabled = false;
    input.disabled = false;
    input.focus();
  }
}

// ── Event listeners ─────────────────────────────────
sendBtn.addEventListener("click", ask);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") ask();
});

// ── Welcome message ─────────────────────────────────
addMsg("Assistant", "Hi! Ask me anything about the thesis.");
