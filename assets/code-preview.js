/* Load the existing research file directly: previews never contain a second,
   potentially stale copy of the model code. All source is inserted as text. */
"use strict";

(async () => {
  const code = document.getElementById("code");
  const error = document.getElementById("source-error");
  const copy = document.getElementById("copy-source");
  if (!code) return;

  let source = "";
  try {
    const response = await fetch(code.dataset.source);
    if (!response.ok) throw new Error("Source unavailable");
    source = await response.text();
    code.textContent = source;
    code.dataset.loaded = "true";
    code.setAttribute("aria-busy", "false");
  } catch {
    code.textContent = "";
    code.setAttribute("aria-busy", "false");
    error.hidden = false;
    return;
  }

  // Highlighting is optional. A blocked or broken highlighter must never
  // turn a successful fetch into a misleading 'file not found' message.
  try {
    if (window.hljs) window.hljs.highlightElement(code);
  } catch {
    code.textContent = source;
  }

  // Clipboard writes require HTTPS (or localhost). Hide this enhancement
  // when unavailable; the source remains selectable and the Raw link works.
  if (copy && window.isSecureContext && navigator.clipboard?.writeText) {
    copy.hidden = false;
    copy.disabled = false;
    copy.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(source);
        copy.textContent = "Copied";
      } catch {
        copy.textContent = "Select text to copy";
      }
      window.setTimeout(() => { copy.textContent = "Copy"; }, 2500);
    });
  }
})();
