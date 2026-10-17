// Shared constants and formatting helpers for the pages.

export const REAL_LIFE_SENSITIVE_TAGS = new Set(["BBC", "Ebony", "Asian"]);

export const TAGGER_MODELS = [
  {
    id: "ml_danbooru",
    label: "ML-Danbooru",
    help: "Loli-only classifier — strong recall on that folder (not used for full multi-folder runs).",
  },
  {
    id: "wd_swinv2_v3",
    label: "WD SwinV2 v3",
    help: "Recommended default — strong general-tag accuracy on anime art.",
  },
  {
    id: "wd_eva02_large",
    label: "WD EVA02 Large",
    help: "Largest WD tagger; slower, often slightly more accurate.",
  },
];

export function taggerLabel(id) {
  return TAGGER_MODELS.find((model) => model.id === id)?.label || id || "";
}

export const ACTIVE_RUN_STATUSES = ["pending", "running"];
export const TERMINAL_RUN_STATUSES = ["completed", "cancelled", "failed"];

const VIDEO_PREVIEW_EXTS = new Set([
  ".mp4",
  ".m4v",
  ".webm",
  ".mov",
  ".mkv",
  ".avi",
  ".flv",
  ".wmv",
]);

function mediaExtFromPath(path) {
  if (!path || typeof path !== "string") return "";
  const clean = path.split(/[?#]/)[0];
  const idx = clean.lastIndexOf(".");
  return idx >= 0 ? clean.slice(idx).toLowerCase() : "";
}

export function isVideoPreviewPath(path) {
  return VIDEO_PREVIEW_EXTS.has(mediaExtFromPath(path));
}

export function formatTagScore(entry) {
  if (!entry) return "";
  return `${entry.tag} (${Number(entry.score).toFixed(3)})`;
}

export function formatDurationSeconds(totalSeconds) {
  const secs = Math.max(0, Math.round(Number(totalSeconds) || 0));
  const hours = Math.floor(secs / 3600);
  const minutes = Math.floor((secs % 3600) / 60);
  const seconds = secs % 60;
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m ${seconds}s`;
  return `${seconds}s`;
}

export function formatEtaFinishAt(iso) {
  if (!iso) return null;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return null;
  return date.toLocaleString(undefined, {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function pathTail(path) {
  const parts = String(path || "").split(/[\\/]+/).filter(Boolean);
  return parts.slice(-2).join("/") || String(path || "");
}

export function formatRunDate(iso) {
  if (!iso) return "";
  // SQLite CURRENT_TIMESTAMP is UTC without a zone marker.
  const normalized = /[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : `${iso.replace(" ", "T")}Z`;
  const date = new Date(normalized);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

export function runProgressText(run) {
  return run.total_images
    ? `${run.processed_images}/${run.total_images}`
    : `${run.processed_images}`;
}

export function runOptionLabel(run) {
  const when = formatRunDate(run.created_at);
  return [
    `Run #${run.run_id}`,
    run.status,
    `${runProgressText(run)} images`,
    pathTail(run.root_repo),
    when,
  ]
    .filter(Boolean)
    .join(" · ");
}

export function runPct(run) {
  if (run.progress_pct != null) return Number(run.progress_pct) || 0;
  const total = Number(run.total_images) || 0;
  return total ? ((Number(run.processed_images) || 0) / total) * 100 : 0;
}
