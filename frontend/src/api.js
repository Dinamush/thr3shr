const API_BASE = "http://127.0.0.1:8000/api";

async function jsonRequest(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `HTTP ${response.status}`);
  }
  return response.json();
}

export const api = {
  getSettings: () => jsonRequest("/settings"),
  saveSettings: (payload) =>
    jsonRequest("/settings", {
      method: "PUT",
      body: JSON.stringify(payload),
    }),
  startRun: (payload) =>
    jsonRequest("/runs/start", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  getRun: (runId) => jsonRequest(`/runs/${runId}`),
  getRunItems: (runId, filters = {}) => {
    const params = new URLSearchParams();
    if (filters.status) params.set("status", filters.status);
    if (filters.needs_review !== undefined)
      params.set("needs_review", String(filters.needs_review));
    const query = params.toString();
    return jsonRequest(`/runs/${runId}/items${query ? `?${query}` : ""}`);
  },
  updateItem: (itemId, payload) =>
    jsonRequest(`/items/${itemId}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
  batchUpdate: (runId, payload) =>
    jsonRequest(`/runs/${runId}/batch`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  migrateRun: (runId, payload) =>
    jsonRequest(`/runs/${runId}/migrate`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
};
