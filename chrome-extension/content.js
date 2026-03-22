(() => {
  "use strict";

  const SETTINGS_KEY = "compressorSettings";
  const DEFAULT_BACKEND_BASE = "http://127.0.0.1:8099";
  const FIELD_SELECTOR = "textarea, div[contenteditable='true'], div[contenteditable='plaintext-only']";
  const SCAN_DEBOUNCE_MS = 350;

  const inputControllers = new WeakMap();
  let activePopup = null;

  // Keep button placement synced with host input resize changes.

  const resizeObserver = new ResizeObserver((entries) => {
    for (const entry of entries) {
      const controller = inputControllers.get(entry.target);
      if (controller) {
        positionButton(controller);
        if (activePopup && activePopup.input === controller.input) {
          positionPopup(activePopup.input, activePopup.popup);
        }
      }
    }
  });

  function isEditableField(node) {
    if (!(node instanceof HTMLElement)) {
      return false;
    }
    return node.matches(FIELD_SELECTOR);
  }

  function isVisible(node) {
    if (!(node instanceof HTMLElement)) {
      return false;
    }
    const style = window.getComputedStyle(node);
    if (style.display === "none" || style.visibility === "hidden") {
      return false;
    }
    const rect = node.getBoundingClientRect();
    return rect.width > 20 && rect.height > 20;
  }

  function ensureInjectable(node) {
    if (!isEditableField(node) || !isVisible(node)) {
      return;
    }

    if (node.dataset.optipromptInjected === "1") {
      const existing = inputControllers.get(node);
      if (existing) {
        positionButton(existing);
      }
      return;
    }

    injectOptimizeButton(node);
  }

  // Main scanner that finds dynamic LLM inputs and cleans up removed ones.
  function scanForInputs() {
    const fields = document.querySelectorAll(FIELD_SELECTOR);
    for (const field of fields) {
      ensureInjectable(field);
    }

    for (const [input, controller] of getControllersSnapshot()) {
      if (!document.contains(input)) {
        cleanupController(input, controller);
      }
    }
  }

  function debounce(fn, waitMs) {
    let timer = null;
    return function debounced() {
      if (timer) {
        window.clearTimeout(timer);
      }
      timer = window.setTimeout(() => {
        timer = null;
        fn();
      }, waitMs);
    };
  }

  function cleanBaseUrl(url) {
    return (url || "").trim().replace(/\/+$/, "");
  }

  async function getBackendEndpointCandidates() {
    let configuredBase = DEFAULT_BACKEND_BASE;

    try {
      const stored = await chrome.storage.local.get(SETTINGS_KEY);
      const settings = stored && stored[SETTINGS_KEY] ? stored[SETTINGS_KEY] : {};
      configuredBase = cleanBaseUrl(settings.backendUrl || DEFAULT_BACKEND_BASE) || DEFAULT_BACKEND_BASE;
    } catch (_error) {
      configuredBase = DEFAULT_BACKEND_BASE;
    }

    const candidates = [];

    function pushCandidate(base) {
      const cleaned = cleanBaseUrl(base);
      if (!cleaned) {
        return;
      }
      const endpoint = `${cleaned}/optimize`;
      if (!candidates.includes(endpoint)) {
        candidates.push(endpoint);
      }
    }

    pushCandidate(configuredBase);

    try {
      const parsed = new URL(configuredBase);
      const protocol = parsed.protocol || "http:";
      const port = parsed.port ? `:${parsed.port}` : "";

      if (parsed.hostname === "0.0.0.0") {
        pushCandidate(`${protocol}//127.0.0.1${port}`);
        pushCandidate(`${protocol}//localhost${port}`);
      } else if (parsed.hostname === "localhost") {
        pushCandidate(`${protocol}//127.0.0.1${port}`);
      } else if (parsed.hostname === "127.0.0.1") {
        pushCandidate(`${protocol}//localhost${port}`);
      }
    } catch (_error) {
      // Keep configured endpoint only when URL parsing fails.
    }

    return candidates;
  }

  function getControllersSnapshot() {
    const pairs = [];
    document.querySelectorAll(FIELD_SELECTOR).forEach((input) => {
      const controller = inputControllers.get(input);
      if (controller) {
        pairs.push([input, controller]);
      }
    });
    return pairs;
  }

  function injectOptimizeButton(input) {
    input.dataset.optipromptInjected = "1";

    const button = document.createElement("button");
    button.type = "button";
    button.className = "optiprompt-optimize-btn";
    button.textContent = "⚡ Optimize";
    button.setAttribute("aria-label", "Optimize prompt");

    const controller = {
      input,
      button,
      isLoading: false,
      // Shared position callback used by several DOM events.
      onReposition: () => {
        positionButton(controller);
        if (activePopup && activePopup.input === input) {
          positionPopup(input, activePopup.popup);
        }
      }
    };

    inputControllers.set(input, controller);

    button.addEventListener("click", async (event) => {
      event.preventDefault();
      event.stopPropagation();
      await handleOptimizeClick(controller);
    });

    document.body.appendChild(button);
    resizeObserver.observe(input);

    const events = ["focus", "input", "keyup", "mouseup", "scroll"];
    for (const eventName of events) {
      input.addEventListener(eventName, controller.onReposition, { passive: true });
    }

    window.addEventListener("scroll", controller.onReposition, { passive: true });
    window.addEventListener("resize", controller.onReposition, { passive: true });

    controller.events = events;
    positionButton(controller);
  }

  function cleanupController(input, controller) {
    if (controller.button && controller.button.isConnected) {
      controller.button.remove();
    }

    if (controller.events) {
      for (const eventName of controller.events) {
        input.removeEventListener(eventName, controller.onReposition);
      }
    }

    window.removeEventListener("scroll", controller.onReposition);
    window.removeEventListener("resize", controller.onReposition);

    resizeObserver.unobserve(input);
    inputControllers.delete(input);
    delete input.dataset.optipromptInjected;
  }

  function positionButton(controller) {
    const input = controller.input;
    const button = controller.button;
    if (!document.contains(input) || !document.contains(button) || !isVisible(input)) {
      button.style.display = "none";
      return;
    }

    const rect = input.getBoundingClientRect();
    const spacing = 10;

    button.style.display = "inline-flex";
    button.style.top = `${window.scrollY + rect.bottom - spacing}px`;
    button.style.left = `${window.scrollX + rect.right - spacing}px`;
    button.style.transform = "translate(-100%, -100%)";
  }

  // Uniformly read prompt text from textarea or contenteditable fields.
  function readInputText(input) {
    if (input instanceof HTMLTextAreaElement) {
      return input.value || "";
    }
    return (input.innerText || "").trim();
  }

  // Replace prompt text and keep cursor close to prior relative position.
  function setInputText(input, text) {
    if (input instanceof HTMLTextAreaElement) {
      const oldLength = input.value.length || 1;
      const previousSelection = typeof input.selectionStart === "number"
        ? input.selectionStart
        : input.value.length;
      const ratio = previousSelection / oldLength;
      input.value = text;
      const nextCaret = Math.max(0, Math.min(text.length, Math.round(text.length * ratio)));
      input.focus();
      input.setSelectionRange(nextCaret, nextCaret);
      input.dispatchEvent(new Event("input", { bubbles: true }));
      input.dispatchEvent(new Event("change", { bubbles: true }));
      return;
    }

    const previousOffset = getContentEditableCaretOffset(input);
    const oldLength = (input.innerText || "").length || 1;
    const ratio = previousOffset / oldLength;

    input.innerText = text;
    input.focus();
    const nextOffset = Math.max(0, Math.min(text.length, Math.round(text.length * ratio)));
    setContentEditableCaretOffset(input, nextOffset);
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function getContentEditableCaretOffset(element) {
    const selection = window.getSelection();
    if (!selection || selection.rangeCount === 0) {
      return (element.innerText || "").length;
    }

    const range = selection.getRangeAt(0);
    const preRange = range.cloneRange();
    preRange.selectNodeContents(element);
    preRange.setEnd(range.endContainer, range.endOffset);
    return preRange.toString().length;
  }

  function setContentEditableCaretOffset(element, targetOffset) {
    const selection = window.getSelection();
    if (!selection) {
      return;
    }

    const range = document.createRange();
    const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT, null);

    let accumulated = 0;
    let node = walker.nextNode();
    while (node) {
      const text = node.nodeValue || "";
      const nextAccumulated = accumulated + text.length;
      if (targetOffset <= nextAccumulated) {
        const localOffset = Math.max(0, targetOffset - accumulated);
        range.setStart(node, localOffset);
        range.collapse(true);
        selection.removeAllRanges();
        selection.addRange(range);
        return;
      }
      accumulated = nextAccumulated;
      node = walker.nextNode();
    }

    range.selectNodeContents(element);
    range.collapse(false);
    selection.removeAllRanges();
    selection.addRange(range);
  }

  function closeActivePopup() {
    if (!activePopup) {
      return;
    }
    if (activePopup.popup && activePopup.popup.isConnected) {
      activePopup.popup.remove();
    }
    activePopup = null;
  }

  // Single popup renderer. Only one suggestion/error popup is allowed at once.
  function createPopup(input, options) {
    closeActivePopup();

    const popup = document.createElement("div");
    popup.className = "optiprompt-popup";

    const label = document.createElement("div");
    label.className = "optiprompt-popup-label";
    label.textContent = options.error ? "Optimization Error" : "Optimized";

    const preview = document.createElement("div");
    preview.className = "optiprompt-popup-preview";
    preview.textContent = options.text;

    const actions = document.createElement("div");
    actions.className = "optiprompt-popup-actions";

    const applyBtn = document.createElement("button");
    applyBtn.type = "button";
    applyBtn.className = "optiprompt-popup-btn optiprompt-popup-btn-apply";
    applyBtn.textContent = "Apply";
    applyBtn.disabled = Boolean(options.error);

    const closeBtn = document.createElement("button");
    closeBtn.type = "button";
    closeBtn.className = "optiprompt-popup-btn optiprompt-popup-btn-close";
    closeBtn.textContent = "Close";

    actions.appendChild(applyBtn);
    actions.appendChild(closeBtn);

    popup.appendChild(label);
    popup.appendChild(preview);

    if (options.stats && !options.error) {
      const statsContainer = document.createElement("div");
      statsContainer.className = "optiprompt-popup-stats";

      const metrics = options.stats.metrics || {};
      const tokensSaved = (metrics.original_token_count || 0) - (metrics.optimized_token_count || 0);
      const percent = options.stats.token_reduction_percent || 0;

      let costSavedText = "";
      if (options.stats.estimated_cost_savings) {
        costSavedText = ` • Saved $${Number(options.stats.estimated_cost_savings).toFixed(5)}`;
      }

      statsContainer.innerHTML = `
        <span class="stat-badge">
          <b>${Math.max(0, tokensSaved)}</b> tokens saved
        </span>
        <span class="stat-detail">
          ${percent.toFixed(1)}% compressed${costSavedText}
        </span>
      `;
      popup.appendChild(statsContainer);
    }

    popup.appendChild(actions);

    document.body.appendChild(popup);
    positionPopup(input, popup);

    applyBtn.addEventListener("click", () => {
      setInputText(input, options.text);
      closeActivePopup();
    });

    closeBtn.addEventListener("click", () => {
      closeActivePopup();
    });

    activePopup = { input, popup };
  }

  function positionPopup(input, popup) {
    if (!document.contains(input) || !document.contains(popup) || !isVisible(input)) {
      popup.style.display = "none";
      return;
    }

    const rect = input.getBoundingClientRect();
    const maxWidth = Math.min(440, window.innerWidth - 24);
    const desiredLeft = window.scrollX + rect.left;
    const clampedLeft = Math.max(window.scrollX + 12, Math.min(desiredLeft, window.scrollX + window.innerWidth - maxWidth - 12));

    popup.style.display = "block";
    popup.style.width = `${maxWidth}px`;
    popup.style.left = `${clampedLeft}px`;

    const aboveTop = window.scrollY + rect.top - popup.offsetHeight - 10;
    const belowTop = window.scrollY + rect.bottom + 10;
    const canPlaceAbove = aboveTop >= window.scrollY + 8;
    popup.style.top = `${canPlaceAbove ? aboveTop : belowTop}px`;
  }

  async function handleOptimizeClick(controller) {
    if (controller.isLoading) {
      return;
    }

    const text = readInputText(controller.input).trim();
    if (!text) {
      createPopup(controller.input, {
        text: "Input is empty. Type your prompt and try again.",
        error: true
      });
      return;
    }

    controller.isLoading = true;
    controller.button.classList.add("is-loading");

    try {
      const endpoints = await getBackendEndpointCandidates();
      let finalData = null;
      let lastError = null;

      for (const endpoint of endpoints) {
        try {
          const result = await chrome.runtime.sendMessage({
            type: "OPTIMIZE_PROMPT",
            endpoint: endpoint,
            payload: {
              prompt: text,
              mode: "balanced"
            }
          });

          if (result && result.data) {
            finalData = result.data;
            break;
          }

          if (result && result.error && result.status > 0) {
            // Reachable endpoint with server response: do not keep trying hosts.
            let msg = result.error;
            try {
              const parsed = JSON.parse(msg);
              msg = parsed.detail || msg;
            } catch (e) { }
            throw new Error(msg || `Backend request failed (${result.status})`);
          }

          throw new Error(result ? result.error : "Failed to reach optimization backend.");
        } catch (error) {
          lastError = error;
          continue;
        }
      }

      if (!finalData) {
        throw (lastError || new Error("Failed to reach optimization backend."));
      }

      const optimized = (finalData && finalData.optimized_prompt ? String(finalData.optimized_prompt) : "").trim();

      if (!optimized) {
        throw new Error("Backend returned no optimized_prompt.");
      }

      createPopup(controller.input, {
        text: optimized,
        stats: finalData,
        error: false
      });
    } catch (error) {
      createPopup(controller.input, {
        text: error instanceof Error ? error.message : "Failed to optimize prompt.",
        error: true
      });
    } finally {
      controller.isLoading = false;
      controller.button.classList.remove("is-loading");
      positionButton(controller);
    }
  }

  // safeReadErrorMessage was removed as it is now handled by the background script

  const debouncedScan = debounce(scanForInputs, SCAN_DEBOUNCE_MS);

  // Observe SPA updates where chat inputs appear after initial page load.
  const domObserver = new MutationObserver((mutations) => {
    let shouldScan = false;

    for (const mutation of mutations) {
      if (mutation.type === "childList") {
        for (const node of mutation.addedNodes) {
          if (!(node instanceof HTMLElement)) {
            continue;
          }

          if (isEditableField(node)) {
            ensureInjectable(node);
          }

          if (node.querySelector && node.querySelector(FIELD_SELECTOR)) {
            shouldScan = true;
          }
        }

        for (const node of mutation.removedNodes) {
          if (!(node instanceof HTMLElement)) {
            continue;
          }
          if (isEditableField(node) || (node.querySelector && node.querySelector(FIELD_SELECTOR))) {
            shouldScan = true;
          }
        }
      }

      if (mutation.type === "attributes" && mutation.target instanceof HTMLElement) {
        if (isEditableField(mutation.target)) {
          ensureInjectable(mutation.target);
        }
      }
    }

    if (shouldScan) {
      debouncedScan();
    }
  });

  function init() {
    scanForInputs();
    domObserver.observe(document.documentElement, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["contenteditable", "class", "style"]
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
