const API_BASE = "http://127.0.0.1:8000/api";
const STORAGE_KEY = "imageClassifierMockStateV1";

const defaultSettings = {
  root_repo: "",
  categories_root: "",
  confidence_threshold: 0.6,
  default_migrate_mode: "copy",
  scan_recursive: true,
  experimental_media_enabled: false,
  experimental_style_detector_enabled: false,
  hybrid_ml_on_review: true,
  tagging_domain: "drawn",
  sfw_classify_mode: false,
  selected_tags: [],
  selected_tags_nsfw: [],
  max_inference_workers: 2,
  inference_batch_size: 8,
  force_cpu_inference: false,
  tagger_model: "wd_swinv2_v3",
  wd_general_threshold: 0.35,
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
const REQUEST_TIMEOUT_MS = 30000;

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
  const controller = new AbortController();
  const timeoutMs = Number(options.timeoutMs) > 0 ? Number(options.timeoutMs) : REQUEST_TIMEOUT_MS;
  const { timeoutMs: _ignored, ...fetchOptions } = options;
  const timeoutId = setTimeout(() => controller.abort("timeout"), timeoutMs);
  try {
    const response = await fetch(`${API_BASE}${path}`, {
      headers: { "Content-Type": "application/json", ...(fetchOptions.headers || {}) },
      ...fetchOptions,
      signal: fetchOptions.signal || controller.signal,
    });
    const text = await response.text();
    const maybeJson = text ? (() => {
      try {
        return JSON.parse(text);
      } catch (_) {
        return null;
      }
    })() : null;
    if (!response.ok) {
      const detail =
        (maybeJson && (maybeJson.detail || maybeJson.message)) || text || "Request failed";
      throw new Error(`HTTP ${response.status} ${response.statusText}: ${detail}`);
    }
    return maybeJson ?? {};
  } catch (err) {
    if (err?.name === "AbortError") {
      throw new Error(`Request timeout after ${timeoutMs / 1000}s`);
    }
    throw err;
  } finally {
    clearTimeout(timeoutId);
  }
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
    const message = String(err?.message || "");
    const networkError =
      err?.name === "TypeError" ||
      message.includes("Failed to fetch") ||
      message.includes("NetworkError") ||
      message.includes("Load failed") ||
      message.includes("CORS request did not succeed") ||
      message.includes("timeout");
    if (backendAvailable === null && networkError) {
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
      global_top_tags: [tag, ...pool.filter((x) => x !== tag)]
        .slice(0, 5)
        .map((x, i) => ({ tag: x, score: Math.max(0.15, score - i * 0.07) })),
      suggested_destination: `${mockState.settings.categories_root || "/mock/categories"}/${tag}`,
      final_tag: tag,
      final_destination: `${mockState.settings.categories_root || "/mock/categories"}/${tag}`,
      status: needsReview ? "proposed" : "approved",
      needs_review: needsReview,
      review_reason: needsReview ? `Below threshold (${score.toFixed(3)} < ${threshold.toFixed(3)})` : null,
      migrated_to: null,
      full_scores: Object.fromEntries(
        [tag, ...pool.filter((x) => x !== tag)].map((name, i) => [
          name,
          Math.max(0.1, score - i * 0.08),
        ])
      ),
    };
  });
}

function computeMockStatus(run) {
  const total = run.total_images || 0;
  const processed = run.processed_images || 0;
  const pct = total > 0 ? Math.min(100, (processed / total) * 100) : 0;
  return {
    run_id: run.id,
    status: run.status,
    total_images: total,
    processed_images: processed,
    failed_images: run.failed_images || 0,
    progress_pct: pct,
    started_at: run.started_at || null,
    finished_at: run.finished_at || null,
    last_error: run.last_error || null,
    cancel_requested: Boolean(run.cancel_requested),
    has_items: (run.items || []).length > 0,
    inference_mode: "single",
    batch_size: 1,
    avg_infer_ms_per_image: run.status === "running" ? 120 : null,
    eta_seconds_remaining:
      run.status === "running" && total > processed
        ? Math.max(1, (total - processed) * 0.12)
        : null,
    eta_finish_at:
      run.status === "running" && total > processed
        ? new Date(Date.now() + Math.max(1, (total - processed) * 120)).toISOString()
        : null,
    queue_seed: null,
    tagger_model: run.tagger_model || mockState.settings.tagger_model || "wd_swinv2_v3",
  };
}

function mockRequest(path, options = {}) {
  const method = (options.method || "GET").toUpperCase();
  const body = parseBody(options);

  if (path === "/settings" && method === "GET") {
    return Promise.resolve(mockState.settings);
  }
  if (path === "/providers" && method === "GET") {
    return Promise.resolve({
      available_providers: ["CPUExecutionProvider"],
      cuda_available: false,
      cpu_available: true,
      forced_cpu: false,
      likely_device: "cpu",
      cuda_usable: false,
      tagger_model: mockState.settings.tagger_model || "wd_swinv2_v3",
      loaded_models: [],
      mock_mode: true,
    });
  }
  if (path === "/models/unload" && method === "POST") {
    return Promise.resolve({
      unloaded: [],
      message: "No tagger sessions were resident in GPU memory.",
    });
  }
  if (path === "/debug/sfw-sources" && method === "GET") {
    return Promise.resolve({
      sources: [
        {
          id: "safebooru",
          label: "Safebooru",
          sfw_policy: "rating:safe",
          max_content_tags: null,
        },
        {
          id: "danbooru",
          label: "Danbooru",
          sfw_policy: "rating:g",
          max_content_tags: 2,
        },
      ],
    });
  }
  if (path === "/debug/sfw-eval" && method === "POST") {
    const tags = Array.isArray(body.tags) ? body.tags.filter(Boolean) : [];
    if (!tags.length) {
      return Promise.reject(new Error("At least one tag is required"));
    }
    const source = body.source || "safebooru";
    if (source === "danbooru" && tags.length > 2) {
      return Promise.reject(new Error("Danbooru allows at most 2 content tag(s)"));
    }
    const count = Math.max(5, Math.min(30, Number(body.count) || 10));
    const threshold = Number(mockState.settings.confidence_threshold || 0.6);
    const items = Array.from({ length: Math.min(count, 6) }, (_, idx) => {
      const pullScores = Object.fromEntries(tags.map((t) => [t, Math.max(0.2, 0.95 - idx * 0.08)]));
      const primary = tags[0] || "1girl";
      const score = pullScores[primary];
      const needsReview = score < threshold;
      return {
        source,
        post_id: String(9000 + idx),
        rating: source === "danbooru" ? "g" : "safe",
        file_name: `${9000 + idx}.jpg`,
        known_tags: tags,
        pull_tag_scores: pullScores,
        global_top_tags: tags.map((t) => ({ tag: t, score: pullScores[t] })),
        primary_tag: needsReview ? null : primary,
        primary_score: needsReview ? null : score,
        needs_review: needsReview,
        review_reason: needsReview ? `Below threshold (${score.toFixed(3)} < ${threshold}).` : null,
        suggested_folder: needsReview ? null : primary,
        secondary_suggestions: needsReview ? [{ tag: primary, score }] : [],
      };
    });
    return Promise.resolve({
      source,
      source_label: source === "danbooru" ? "Danbooru" : "Safebooru",
      sfw_policy: source === "danbooru" ? "rating:g" : "rating:safe",
      query: `${tags.join(" ")} ${source === "danbooru" ? "rating:g" : "rating:safe"}`,
      tags,
      count_requested: count,
      count_evaluated: items.length,
      tagger_model: mockState.settings.tagger_model || "wd_swinv2_v3",
      confidence_threshold: threshold,
      destination_tags: mockState.settings.selected_tags || [],
      recall: tags.map((tag) => ({
        tag,
        present_in_posts: items.length,
        hits_at_threshold: items.filter((it) => Number(it.pull_tag_scores[tag] || 0) >= threshold)
          .length,
        hit_rate: items.length
          ? items.filter((it) => Number(it.pull_tag_scores[tag] || 0) >= threshold).length /
            items.length
          : null,
      })),
      items,
      errors: [],
    });
  }
  if (path === "/scan/preview" && method === "GET") {
    return Promise.resolve({
      root_repo: mockState.settings.root_repo || "/mock/images",
      recursive: Boolean(mockState.settings.scan_recursive),
      experimental_media_enabled: Boolean(mockState.settings.experimental_media_enabled),
      excluded_dirs: mockState.settings.categories_root
        ? [mockState.settings.categories_root]
        : [],
      stats: {
        total_files: 12,
        eligible_images: 8,
        ignored_unsupported: 3,
        ignored_gif: 1,
        failed_to_read: 0,
      },
      sample_paths: ["/mock/images/sample_1.jpg"],
    });
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
    const next = { ...defaultSettings, ...mockState.settings, ...body };
    if (next.tagging_domain === "real_life") {
      next.sfw_classify_mode = false;
    }
    if (next.sfw_classify_mode) {
      next.selected_tags = ["SFW", "scenery"];
    }
    if (!Array.isArray(next.selected_tags_nsfw)) {
      next.selected_tags_nsfw = [];
    }
    mockState.settings = next;
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
      id: runId,
      root_repo: mockState.settings.root_repo,
      categories_root: mockState.settings.categories_root,
      confidence_threshold: threshold,
      tagger_model: mockState.settings.tagger_model || "wd_swinv2_v3",
      status: "pending",
      total_images: items.length,
      processed_images: 0,
      failed_images: 0,
      cancel_requested: 0,
      started_at: new Date().toISOString(),
      finished_at: null,
      last_error: null,
      items,
    };
    persistMockState();
    return Promise.resolve({
      run_id: runId,
      status: "pending",
      stats: null,
      mappings,
      unmatched_folders: [],
      created_items: 0,
      message: "Run queued; poll /api/runs/{run_id}/status for progress.",
    });
  }
  if (path.match(/^\/runs\/\d+\/status$/) && method === "GET") {
    const runId = Number(path.split("/")[2]);
    const run = mockState.runs[runId];
    if (!run) {
      return Promise.reject(new Error("Run not found"));
    }
    if (run.status === "pending") {
      run.status = "running";
    }
    if (run.status === "running" && !run.cancel_requested) {
      run.processed_images = Math.min(run.total_images, run.processed_images + 5);
      if (run.processed_images >= run.total_images) {
        run.status = "completed";
        run.finished_at = new Date().toISOString();
      }
      persistMockState();
    }
    if (run.cancel_requested && run.status === "running") {
      run.status = "cancelled";
      run.finished_at = new Date().toISOString();
      persistMockState();
    }
    return Promise.resolve(computeMockStatus(run));
  }
  if (path.match(/^\/runs\/\d+\/cancel$/) && method === "POST") {
    const runId = Number(path.split("/")[2]);
    const run = mockState.runs[runId];
    if (!run) {
      return Promise.reject(new Error("Run not found"));
    }
    run.cancel_requested = 1;
    if (run.status === "pending") {
      run.status = "cancelled";
      run.finished_at = new Date().toISOString();
    }
    persistMockState();
    return Promise.resolve(computeMockStatus(run));
  }
  if (path.match(/^\/runs\/\d+\/reclassify$/) && method === "POST") {
    const runId = Number(path.split("/")[2]);
    const run = mockState.runs[runId];
    if (!run) {
      return Promise.reject(new Error("Run not found"));
    }
    if (["pending", "running"].includes(run.status)) {
      return Promise.reject(new Error("Cannot reclassify while a run is still pending or running."));
    }
    const threshold = Number(run.confidence_threshold || mockState.settings.confidence_threshold || 0.6);
    const model = body.tagger_model || "wd_eva02_large";
    const idFilter = Array.isArray(body.item_ids) ? new Set(body.item_ids) : null;
    let eligibleCount = 0;
    run.items = run.items.map((item) => {
      if (item.status !== "proposed" || !item.needs_review) return item;
      if (idFilter && !idFilter.has(item.id)) return item;
      eligibleCount += 1;
      const boosted = Math.min(0.99, Number(item.primary_score || 0.4) + 0.35);
      const tag = item.primary_tag || item.secondary_suggestions?.[0]?.tag || "1girl";
      if (boosted >= threshold) {
        return {
          ...item,
          primary_tag: tag,
          primary_score: boosted,
          needs_review: false,
          review_reason: null,
          status: "approved",
          final_tag: item.final_tag || tag,
          suggested_destination: `${mockState.settings.categories_root || "/mock/categories"}/${tag}`,
        };
      }
      return {
        ...item,
        review_reason: `Reclassified with ${model}. Below threshold (${boosted.toFixed(3)} < ${threshold.toFixed(3)}).`,
      };
    });
    if (eligibleCount === 0) {
      return Promise.reject(new Error("No eligible needs-review items to reclassify"));
    }
    run.status = "running";
    run.cancel_requested = 0;
    run.finished_at = null;
    persistMockState();
    setTimeout(() => {
      if (run.cancel_requested) {
        run.status = "cancelled";
      } else {
        run.status = "completed";
      }
      run.finished_at = new Date().toISOString();
      persistMockState();
    }, 400);
    return Promise.resolve({
      run_id: runId,
      status: "running",
      eligible_count: eligibleCount,
      tagger_model: model,
      message: `Reclassify queued for ${eligibleCount} item(s) with ${model}`,
    });
  }
  if (path.startsWith("/runs/") && path.endsWith("/items") && method === "GET") {
    const runId = Number(path.split("/")[2]);
    const run = mockState.runs[runId];
    if (!run) return Promise.resolve([]);
    const visibleCount = Math.max(0, run.processed_images || 0);
    return Promise.resolve(run.items.slice(0, visibleCount));
  }
  if (path.startsWith("/runs/") && !path.includes("/items") && method === "GET") {
    const runId = Number(path.split("/")[2]);
    const run = mockState.runs[runId];
    if (!run) return Promise.resolve({ run: null, counts: [] });
    const visibleItems = run.items.slice(0, Math.max(0, run.processed_images || 0));
    return Promise.resolve({ run, counts: mockCounts(visibleItems) });
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
  if (path.match(/^\/items\/\d+\/scores$/) && method === "GET") {
    const itemId = Number(path.split("/")[2]);
    const run = Object.values(mockState.runs).find((r) => r.items.some((i) => i.id === itemId));
    if (!run) return Promise.reject(new Error("Item not found"));
    const item = run.items.find((i) => i.id === itemId);
    return Promise.resolve({ item_id: item.id, full_scores: item.full_scores || {} });
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
  getTags: (query = "", limit = 50, domain = null) => {
    const params = new URLSearchParams({ query, limit: String(limit) });
    if (domain) params.set("domain", domain);
    return request(`/tags?${params.toString()}`);
  },
  getSettings: () => request("/settings"),
  getProviders: () => request("/providers"),
  unloadModels: () =>
    request("/models/unload", {
      method: "POST",
      body: JSON.stringify({}),
    }),
  getRealLifeStatus: () => request("/real-life/status"),
  previewScan: () => request("/scan/preview"),
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
  getRunStatus: (runId) => request(`/runs/${runId}/status`),
  cancelRun: (runId) =>
    request(`/runs/${runId}/cancel`, {
      method: "POST",
      body: JSON.stringify({}),
    }),
  reclassifyRun: (runId, payload) =>
    request(`/runs/${runId}/reclassify`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  getSfwDebugSources: () => request("/debug/sfw-sources"),
  runSfwDebugEval: (payload) =>
    request("/debug/sfw-eval", {
      method: "POST",
      body: JSON.stringify(payload),
      timeoutMs: 180000,
    }),
  getSfwDebugPreviewUrl: (sourceId, fileName) => {
    if (backendAvailable === false) return null;
    return `${API_BASE}/debug/sfw-eval/preview/${encodeURIComponent(sourceId)}/${encodeURIComponent(fileName)}`;
  },
  runRealismDebugEval: (payload) =>
    request("/debug/realism-eval", {
      method: "POST",
      body: JSON.stringify(payload),
      // Remote download + multi-model GPU scoring can take a long time.
      timeoutMs: 60 * 60 * 1000,
    }),
  getRealismDebugPreviewUrl: (sourceId, fileName) => {
    if (backendAvailable === false) return null;
    return `${API_BASE}/debug/realism-eval/preview/${encodeURIComponent(sourceId)}/${encodeURIComponent(fileName)}`;
  },
  getStyleDetectors: () => request("/debug/style-detectors"),
  runStyleDebugEval: (payload) =>
    request("/debug/style-eval", {
      method: "POST",
      body: JSON.stringify(payload),
      timeoutMs: 60 * 60 * 1000,
    }),
  runTagRecallEval: (payload) =>
    request("/debug/tag-recall-eval", {
      method: "POST",
      body: JSON.stringify(payload),
      // Suite download + dual-model dense scoring can take a while.
      timeoutMs: 60 * 60 * 1000,
    }),
  getTagRecallPreviewUrl: (sourceId, fileName) => {
    if (backendAvailable === false) return null;
    return `${API_BASE}/debug/tag-recall-eval/preview/${encodeURIComponent(sourceId)}/${encodeURIComponent(fileName)}`;
  },
  runTagFpEval: (payload) =>
    request("/debug/tag-fp-eval", {
      method: "POST",
      body: JSON.stringify(payload),
      timeoutMs: 60 * 60 * 1000,
    }),
  getRunItems: (runId, filters = {}) => {
    const params = new URLSearchParams();
    if (filters.status) params.set("status", filters.status);
    if (filters.needs_review !== undefined) {
      params.set("needs_review", String(filters.needs_review));
    }
    if (filters.include_scores) {
      params.set("include_scores", "true");
    }
    const query = params.toString();
    return request(`/runs/${runId}/items${query ? `?${query}` : ""}`, {
      timeoutMs: Number(filters.timeoutMs) > 0 ? Number(filters.timeoutMs) : 120000,
    });
  },
  getItemScores: (itemId) => request(`/items/${itemId}/scores`),
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
      // Large approved sets can take many minutes (copy/move + DB updates).
      timeoutMs: 60 * 60 * 1000,
    }),
  getItemPreviewUrl: (itemId) => {
    if (backendAvailable === false) return null;
    return `${API_BASE}/items/${itemId}/preview`;
  },
};
