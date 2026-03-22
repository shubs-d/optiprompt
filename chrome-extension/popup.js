const STORAGE_KEY = "tokenHistory";
const SETTINGS_KEY = "compressorSettings";
const DEFAULT_BACKEND_URL = "http://0.0.0.0:8099";

const el = {
  backendUrl: document.getElementById("backendUrl"),
  optimizeMode: document.getElementById("optimizeMode"),
  originalText: document.getElementById("originalText"),
  optimizedText: document.getElementById("optimizedText"),
  compressBtn: document.getElementById("compressBtn"),
  analyzeBtn: document.getElementById("analyzeBtn"),
  saveBtn: document.getElementById("saveBtn"),
  originalTokens: document.getElementById("originalTokens"),
  optimizedTokens: document.getElementById("optimizedTokens"),
  savedTokens: document.getElementById("savedTokens"),
  savedPercent: document.getElementById("savedPercent"),
  historyCount: document.getElementById("historyCount"),
  totalSaved: document.getElementById("totalSaved"),
  avgSavings: document.getElementById("avgSavings"),
  bestSaved: document.getElementById("bestSaved"),
  statusText: document.getElementById("statusText"),
  openAnalyticsBtn: document.getElementById("openAnalyticsBtn")
};

let latestAnalysis = null;

function cleanBaseUrl(url) {
  return (url || "").trim().replace(/\/+$/, "");
}

async function getSettings() {
  const stored = await chrome.storage.local.get(SETTINGS_KEY);
  const settings = stored[SETTINGS_KEY] || {};

  return {
    backendUrl: cleanBaseUrl(settings.backendUrl || DEFAULT_BACKEND_URL),
    mode: settings.mode || "balanced"
  };
}

async function saveSettings() {
  const settings = {
    backendUrl: cleanBaseUrl(el.backendUrl.value) || DEFAULT_BACKEND_URL,
    mode: el.optimizeMode.value || "balanced"
  };

  await chrome.storage.local.set({ [SETTINGS_KEY]: settings });
}

function estimateTokens(text) {
  const trimmed = (text || "").trim();
  if (!trimmed) return 0;

  const words = trimmed.match(/\b[\w'-]+\b/g) || [];
  const punctuation = trimmed.match(/[^\s\w]/g) || [];
  const roughByChars = Math.ceil(trimmed.length / 4);
  const roughByLexical = Math.ceil(words.length * 1.25 + punctuation.length * 0.35);

  return Math.max(1, Math.round((roughByChars + roughByLexical) / 2));
}

function buildAnalysis(original, optimized) {
  const originalTokens = estimateTokens(original);
  const optimizedTokens = estimateTokens(optimized);
  const saved = Math.max(0, originalTokens - optimizedTokens);
  const percent = originalTokens === 0 ? 0 : (saved / originalTokens) * 100;

  return {
    id: crypto.randomUUID(),
    timestamp: Date.now(),
    title: original.trim().slice(0, 72) || "Untitled prompt",
    originalTokens,
    optimizedTokens,
    saved,
    percent
  };
}

function renderCurrent(analysis) {
  el.originalTokens.textContent = String(analysis.originalTokens);
  el.optimizedTokens.textContent = String(analysis.optimizedTokens);
  el.savedTokens.textContent = String(analysis.saved);
  el.savedPercent.textContent = `${analysis.percent.toFixed(1)}%`;
}

async function getHistory() {
  const stored = await chrome.storage.local.get(STORAGE_KEY);
  return Array.isArray(stored[STORAGE_KEY]) ? stored[STORAGE_KEY] : [];
}

async function saveHistoryItem(item) {
  const history = await getHistory();
  const next = [item, ...history].slice(0, 200);
  await chrome.storage.local.set({ [STORAGE_KEY]: next });
}

function renderInsights(history) {
  const totalSaved = history.reduce((sum, item) => sum + item.saved, 0);
  const totalPercent = history.reduce((sum, item) => sum + item.percent, 0);
  const bestSaved = history.reduce((max, item) => Math.max(max, item.saved), 0);

  el.historyCount.textContent = String(history.length);
  el.totalSaved.textContent = String(totalSaved);
  el.avgSavings.textContent = history.length ? `${(totalPercent / history.length).toFixed(1)}%` : "0%";
  el.bestSaved.textContent = String(bestSaved);
}

function setStatus(text, isError = false) {
  el.statusText.textContent = text;
  el.statusText.style.color = isError ? "#9f2f1d" : "#315f54";
}

async function sendBackendRequest(type, endpoint, payload) {
  const result = await chrome.runtime.sendMessage({
    type,
    endpoint,
    payload
  });

  if (result && result.data) {
    return result.data;
  }

  const detail = result && result.error ? result.error : "Failed to reach backend.";
  throw new Error(detail);
}

function applyCurrentAnalysis() {
  const original = el.originalText.value;
  const optimized = el.optimizedText.value;

  if (!original.trim() || !optimized.trim()) {
    setStatus("Add both prompt versions before analyzing.", true);
    return;
  }

  latestAnalysis = buildAnalysis(original, optimized);
  renderCurrent(latestAnalysis);
  el.saveBtn.disabled = false;

  setStatus(
    latestAnalysis.saved > 0
      ? `Nice. You saved ${latestAnalysis.saved} estimated tokens.`
      : "No estimated savings yet. Try compressing phrasing further."
  );
}

async function compressPrompt() {
  const prompt = el.originalText.value.trim();
  if (!prompt) {
    setStatus("Add an original prompt first.", true);
    return;
  }

  const backendUrl = cleanBaseUrl(el.backendUrl.value) || DEFAULT_BACKEND_URL;
  const mode = el.optimizeMode.value || "balanced";
  const endpoint = `${backendUrl}/optimize`;

  el.compressBtn.disabled = true;
  setStatus("Compressing with backend...");

  try {
    await saveSettings();

    const data = await sendBackendRequest("OPTIMIZE_PROMPT", endpoint, {
      prompt,
      mode,
      include_candidates: false,
      debug: false,
      seed: 42
    });

    el.optimizedText.value = data.optimized_prompt || "";
    applyCurrentAnalysis();
  } catch (err) {
    setStatus(`Compression failed: ${err.message}`, true);
  } finally {
    el.compressBtn.disabled = false;
  }
}

async function analyzePrompt() {
  const original = el.originalText.value.trim();
  const optimized = el.optimizedText.value.trim();

  if (!original) {
    setStatus("Add an original prompt first.", true);
    return;
  }

  const backendUrl = cleanBaseUrl(el.backendUrl.value) || DEFAULT_BACKEND_URL;
  const endpoint = `${backendUrl}/analyze`;
  const mode = el.optimizeMode.value || "balanced";

  el.analyzeBtn.disabled = true;
  setStatus("Analyzing with backend...");

  try {
    await saveSettings();

    const data = await sendBackendRequest("ANALYZE_PROMPT", endpoint, {
      original_prompt: original,
      optimized_prompt: optimized || null,
      mode,
      seed: 42
    });

    if (data.optimized_prompt && !optimized) {
      el.optimizedText.value = data.optimized_prompt;
    }

    const metrics = data.metrics || {};
    latestAnalysis = {
      id: crypto.randomUUID(),
      timestamp: Date.now(),
      title: original.slice(0, 72) || "Untitled prompt",
      originalTokens: Number(metrics.original_token_count || estimateTokens(original)),
      optimizedTokens: Number(metrics.optimized_token_count || estimateTokens(data.optimized_prompt || optimized)),
      saved: Math.max(0, Number(metrics.original_token_count || 0) - Number(metrics.optimized_token_count || 0)),
      percent: Number(data.token_reduction_percent || 0)
    };

    renderCurrent(latestAnalysis);
    el.saveBtn.disabled = false;
    setStatus(
      latestAnalysis.saved > 0
        ? `Analyzed. You saved ${latestAnalysis.saved} estimated tokens.`
        : "Analysis complete. No token savings detected yet."
    );
  } catch (err) {
    // Fall back to local analysis to preserve existing UX when backend analyze is unavailable.
    applyCurrentAnalysis();
    setStatus(`Backend analyze unavailable, used local estimate. ${err.message}`, true);
  } finally {
    el.analyzeBtn.disabled = false;
  }
}

el.compressBtn.addEventListener("click", compressPrompt);
el.analyzeBtn.addEventListener("click", analyzePrompt);

el.saveBtn.addEventListener("click", async () => {
  if (!latestAnalysis) {
    setStatus("Run analysis before saving.", true);
    return;
  }

  await saveHistoryItem(latestAnalysis);
  const history = await getHistory();
  renderInsights(history);

  el.saveBtn.disabled = true;
  setStatus("Saved to history.");
});

el.openAnalyticsBtn.addEventListener("click", () => {
  chrome.runtime.openOptionsPage();
});

(async function init() {
  const settings = await getSettings();
  el.backendUrl.value = settings.backendUrl;
  el.optimizeMode.value = settings.mode;

  el.backendUrl.addEventListener("change", saveSettings);
  el.optimizeMode.addEventListener("change", saveSettings);

  const history = await getHistory();
  renderInsights(history);
})();
