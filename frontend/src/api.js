const API_BASE = "http://127.0.0.1:8000/api";
const STORAGE_KEY = "imageClassifierMockStateV1";

const defaultSettings = {
  root_repo: "",
  categories_root: "",
  confidence_threshold: 0.6,
  default_migrate_mode: "copy",
};
const mockTags = [
  "1girl",
  "solo",
  "blush",
  "black_hair",
  "brown_hair",
  "blue_eyes",
  "short_hair",
  "long_hair",
  "twintails",
  "smile",
  "open_mouth",
  "looking_at_viewer",
];

const mockState = loadMockState();
let backendAvailable = null;

function loadMockState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return { settings: { ...defaultSettings }, nextRunId: 1, runs: {} };
    }
    const parsed = JSON.parse(raw);
    return {
      settings: { ...defaultSettings, ...(parsed.settings || {}) },
      nextRunId: parsed.nextRunId || 1,
      runs: parsed.runs || {},
    };
  } catch (_) {
    return { settings: { ...defaultSettings }, nextRunId: 1, runs: {} };
  }
}

function persistMockState() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(mockState));
}

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

async function request(path, options = {}) {
  if (backendAvailable === false) {
    return mockRequest(path, options);
  }
  try {
    const data = await jsonRequest(path, options);
    backendAvailable = true;
    return data;
  } catch (err) {
    if (backendAvailable === null) {
      backendAvailable = false;
      return mockRequest(path, options);
    }
    throw err;
  }
}

function parseBody(options) {
  if (!options.body) return {};
  return JSON.parse(options.body);
}

function mockCounts(items) {
  const byStatus = {};
  for (const item of items) {
    byStatus[item.status] = (byStatus[item.status] || 0) + 1;
  }
  return Object.entries(byStatus).map(([status, count]) => ({ status, count }));
}

function makeMockItems(runId, selectedFolders, threshold) {
  const pool = selectedFolders.length ? selectedFolders : ["1girl", "solo", "blush"];
  return pool.slice(0, 8).map((tag, idx) => {
    const score = Math.max(0.3, 0.9 - idx * 0.06);
    const needsReview = score < threshold;
    return {
      id: runId * 1000 + idx + 1,
      run_id: runId,
      file_path: `/mock/images/sample_${idx + 1}.jpg`,
      relative_path: `sample_${idx + 1}.jpg`,
      primary_tag: tag,
      primary_score: score,
      secondary_suggestions: pool
        .filter((x) => x !== tag)
        .slice(0, 3)
        .map((x, i) => ({ tag: x, score: Math.max(0.2, score - 0.1 - i * 0.05) })),
      suggested_destination: `${mockState.settings.categories_root || "/mock/categories"}/${tag}`,
      final_tag: tag,
      final_destination: `${mockState.settings.categories_root || "/mock/categories"}/${tag}`,
      status: "proposed",
      needs_review: needsReview,
      review_reason: needsReview ? `Below threshold (${score.toFixed(3)} < ${threshold.toFixed(3)})` : null,
      migrated_to: null,
    };
  });
}

function mockRequest(path, options = {}) {
  const method = (options.method || "GET").toUpperCase();
  const body = parseBody(options);

  if (path === "/settings" && method === "GET") {
    return Promise.resolve(mockState.settings);
  }
  if (path.startsWith("/tags") && method === "GET") {
    const queryString = path.includes("?") ? path.split("?")[1] : "";
    const params = new URLSearchParams(queryString);
    const query = (params.get("query") || "").toLowerCase();
    const limit = Number(params.get("limit") || 50);
    const items = mockTags.filter((t) => t.toLowerCase().includes(query)).slice(0, limit);
    return Promise.resolve({ items, count: items.length });
  }
  if (path === "/settings" && method === "PUT") {
    mockState.settings = { ...defaultSettings, ...body };
    persistMockState();
    return Promise.resolve(mockState.settings);
  }
  if (path === "/runs/start" && method === "POST") {
    mockState.settings = { ...mockState.settings, ...body };
    const runId = mockState.nextRunId++;
    const selectedFolders = body.selected_folders || [];
    const threshold = Number(mockState.settings.confidence_threshold || 0.6);
    const items = makeMockItems(runId, selectedFolders, threshold);
    const mappings = selectedFolders.map((name) => ({
      folder_name: name,
      normalized_name: name.toLowerCase(),
      matched_tag: name.toLowerCase(),
      matched: true,
    }));
    mockState.runs[runId] = {
      run: {
        id: runId,
        root_repo: mockState.settings.root_repo,
        categories_root: mockState.settings.categories_root,
        confidence_threshold: threshold,
      },
      items,
    };
    persistMockState();
    return Promise.resolve({
      run_id: runId,
      stats: {
        total_files: items.length + 2,
        eligible_images: items.length,
        ignored_unsupported: 1,
        ignored_gif: 1,
        failed_to_read: 0,
      },
      mappings,
      unmatched_folders: [],
      created_items: items.length,
    });
  }
  if (path.startsWith("/runs/") && path.endsWith("/items") && method === "GET") {
    const runId = Number(path.split("/")[2]);
    const run = mockState.runs[runId];
    if (!run) return Promise.resolve([]);
    return Promise.resolve(run.items);
  }
  if (path.startsWith("/runs/") && !path.includes("/items") && method === "GET") {
    const runId = Number(path.split("/")[2]);
    const run = mockState.runs[runId];
    if (!run) return Promise.resolve({ run: null, counts: [] });
    return Promise.resolve({ run: run.run, counts: mockCounts(run.items) });
  }
  if (path.startsWith("/items/") && method === "PATCH") {
    const itemId = Number(path.split("/")[2]);
    const run = Object.values(mockState.runs).find((r) => r.items.some((i) => i.id === itemId));
    if (!run) return Promise.resolve(null);
    const item = run.items.find((i) => i.id === itemId);
    if (body.status) item.status = body.status;
    if (body.final_tag) {
      item.final_tag = body.final_tag;
      item.final_destination = `${mockState.settings.categories_root || "/mock/categories"}/${body.final_tag}`;
    }
    item.needs_review = false;
    item.review_reason = null;
    persistMockState();
    return Promise.resolve(item);
  }
  if (path.match(/^\/runs\/\d+\/batch$/) && method === "POST") {
    const runId = Number(path.split("/")[2]);
    const run = mockState.runs[runId];
    if (!run) return Promise.resolve({ updated: 0 });
    let updated = 0;
    for (const item of run.items) {
      if (!body.item_ids.includes(item.id)) continue;
      if (body.status) item.status = body.status;
      if (body.final_tag) {
        item.final_tag = body.final_tag;
        item.final_destination = `${mockState.settings.categories_root || "/mock/categories"}/${body.final_tag}`;
      }
      item.needs_review = false;
      item.review_reason = null;
      updated += 1;
    }
    persistMockState();
    return Promise.resolve({ updated });
  }
  if (path.match(/^\/runs\/\d+\/migrate$/) && method === "POST") {
    const runId = Number(path.split("/")[2]);
    const run = mockState.runs[runId];
    if (!run) return Promise.resolve({ mode: body.mode, total_candidates: 0, migrated_count: 0, failed_count: 0, results: [] });
    const approved = run.items.filter((x) => x.status === "approved");
    const results = approved.map((item) => {
      item.status = "migrated";
      item.migrated_to = `${item.final_destination}/${item.relative_path}`;
      return {
        item_id: item.id,
        source: item.file_path,
        destination: item.migrated_to,
        success: true,
        error: null,
      };
    });
    persistMockState();
    return Promise.resolve({
      mode: body.mode,
      total_candidates: approved.length,
      migrated_count: approved.length,
      failed_count: 0,
      results,
    });
  }

  return Promise.reject(new Error(`Unsupported mock route: ${method} ${path}`));
}

export const api = {
  isOfflineMode: () => backendAvailable === false,
  getTags: (query = "", limit = 50) => {
    const params = new URLSearchParams({ query, limit: String(limit) });
    return request(`/tags?${params.toString()}`);
  },
  getSettings: () => request("/settings"),
  saveSettings: (payload) =>
    request("/settings", {
      method: "PUT",
      body: JSON.stringify(payload),
    }),
  startRun: (payload) =>
    request("/runs/start", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  getRun: (runId) => request(`/runs/${runId}`),
  getRunItems: (runId, filters = {}) => {
    const params = new URLSearchParams();
    if (filters.status) params.set("status", filters.status);
    if (filters.needs_review !== undefined) {
      params.set("needs_review", String(filters.needs_review));
    }
    const query = params.toString();
    return request(`/runs/${runId}/items${query ? `?${query}` : ""}`);
  },
  updateItem: (itemId, payload) =>
    request(`/items/${itemId}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
  batchUpdate: (runId, payload) =>
    request(`/runs/${runId}/batch`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  migrateRun: (runId, payload) =>
    request(`/runs/${runId}/migrate`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
};
