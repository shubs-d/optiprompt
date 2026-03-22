"use strict";

// Lightweight MV3 service worker.
// Reserved for future API proxying and cross-context messaging.
chrome.runtime.onInstalled.addListener(() => {
  // Keeps service worker valid and ready.
});

function toErrorPayload(status, bodyText) {
  try {
    const parsed = JSON.parse(bodyText);
    return { error: parsed.detail || bodyText, status };
  } catch (_err) {
    return { error: bodyText || `HTTP error ${status}`, status };
  }
}

async function proxyBackendRequest(endpoint, payload, method = "POST") {
  const response = await fetch(endpoint, {
    method,
    headers: { "Content-Type": "application/json" },
    body: method.toUpperCase() === "GET" ? undefined : JSON.stringify(payload || {})
  });

  if (!response.ok) {
    const text = await response.text();
    return toErrorPayload(response.status, text);
  }

  const data = await response.json();
  return { data, status: response.status };
}

function buildRequestFromMessage(request) {
  if (request.type === "BACKEND_REQUEST") {
    return {
      endpoint: request.endpoint,
      payload: request.payload || {},
      method: request.method || "POST"
    };
  }

  const endpointByType = {
    OPTIMIZE_PROMPT: "/optimize",
    ANALYZE_PROMPT: "/analyze",
    PREDICT_PROMPT: "/predict",
    OPTIMIZE: "/optimize",
    ANALYZE: "/analyze",
    PREDICT: "/predict"
  };

  const suffix = endpointByType[request.type];
  if (!suffix) {
    return null;
  }

  const endpoint = request.endpoint || `${String(request.baseUrl || "").replace(/\/+$/, "")}${suffix}`;
  return {
    endpoint,
    payload: request.payload || {},
    method: request.method || "POST"
  };
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  const backendRequest = buildRequestFromMessage(request);
  if (!backendRequest) {
    return false;
  }

  proxyBackendRequest(backendRequest.endpoint, backendRequest.payload, backendRequest.method)
    .then((result) => sendResponse(result))
    .catch((error) => {
      sendResponse({ error: error.message || "Failed to fetch from backend", status: 0 });
    });

  // Return true to indicate we will send response asynchronously
  return true;
});
