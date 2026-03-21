const STORAGE_KEY = "tokenHistory";

const els = {
  entriesValue: document.getElementById("entriesValue"),
  totalSavedValue: document.getElementById("totalSavedValue"),
  avgSavingsValue: document.getElementById("avgSavingsValue"),
  bestSaveValue: document.getElementById("bestSaveValue"),
  bars: document.getElementById("bars"),
  historyRows: document.getElementById("historyRows"),
  exportBtn: document.getElementById("exportBtn"),
  clearBtn: document.getElementById("clearBtn"),
  status: document.getElementById("status")
};

async function getHistory() {
  const stored = await chrome.storage.local.get(STORAGE_KEY);
  return Array.isArray(stored[STORAGE_KEY]) ? stored[STORAGE_KEY] : [];
}

function renderStats(history) {
  const totalSaved = history.reduce((sum, item) => sum + item.saved, 0);
  const avgPercent = history.length
    ? history.reduce((sum, item) => sum + item.percent, 0) / history.length
    : 0;
  const bestSaved = history.reduce((max, item) => Math.max(max, item.saved), 0);

  els.entriesValue.textContent = String(history.length);
  els.totalSavedValue.textContent = String(totalSaved);
  els.avgSavingsValue.textContent = `${avgPercent.toFixed(1)}%`;
  els.bestSaveValue.textContent = String(bestSaved);
}

function renderBars(history) {
  els.bars.textContent = "";

  const recent = history.slice(0, 12).reverse();
  const maxSaved = recent.reduce((max, item) => Math.max(max, item.saved), 0) || 1;

  if (!recent.length) {
    const empty = document.createElement("p");
    empty.textContent = "No saved entries yet.";
    empty.style.color = "#3f5d73";
    els.bars.appendChild(empty);
    return;
  }

  for (const item of recent) {
    const bar = document.createElement("div");
    const label = document.createElement("span");

    const height = Math.max(8, Math.round((item.saved / maxSaved) * 150));
    bar.className = "bar";
    bar.style.height = `${height}px`;
    bar.title = `${item.saved} saved (${item.percent.toFixed(1)}%)`;

    const d = new Date(item.timestamp);
    label.textContent = `${d.getMonth() + 1}/${d.getDate()}`;

    bar.appendChild(label);
    els.bars.appendChild(bar);
  }
}

function renderTable(history) {
  els.historyRows.textContent = "";

  if (!history.length) {
    const row = document.createElement("tr");
    row.innerHTML = '<td colspan="5">No history saved yet.</td>';
    els.historyRows.appendChild(row);
    return;
  }

  for (const item of history) {
    const row = document.createElement("tr");
    const date = new Date(item.timestamp).toLocaleString();

    row.innerHTML = `
      <td>${date}</td>
      <td>${item.originalTokens}</td>
      <td>${item.optimizedTokens}</td>
      <td>${item.saved}</td>
      <td>${item.percent.toFixed(1)}%</td>
    `;

    els.historyRows.appendChild(row);
  }
}

function setStatus(text, isError = false) {
  els.status.textContent = text;
  els.status.style.color = isError ? "#9d2f22" : "#3f5d73";
}

async function refresh() {
  const history = await getHistory();
  renderStats(history);
  renderBars(history);
  renderTable(history);
}

els.exportBtn.addEventListener("click", async () => {
  const history = await getHistory();
  if (!history.length) {
    setStatus("No data to export.", true);
    return;
  }

  const blob = new Blob([JSON.stringify(history, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `token-savings-${new Date().toISOString().slice(0, 10)}.json`;
  a.click();
  URL.revokeObjectURL(url);

  setStatus("Exported analytics JSON.");
});

els.clearBtn.addEventListener("click", async () => {
  const confirmDelete = confirm("Clear all saved token history?");
  if (!confirmDelete) return;

  await chrome.storage.local.set({ [STORAGE_KEY]: [] });
  await refresh();
  setStatus("History cleared.");
});

refresh();
