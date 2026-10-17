// Pages, settings sections and individual settings: what the sidebar, the
// phone tab bar, the settings index and the Ctrl+K jump bar list.

export const PAGES = [
  { id: "run", path: "/", label: "Tag & run", group: "Workflow", icon: "run" },
  { id: "review", path: "/review", label: "Review", group: "Workflow", icon: "review" },
  { id: "runs", path: "/runs", label: "Runs", group: "Workflow", icon: "runs" },
  { id: "settings", path: "/settings", label: "Settings", group: "Setup", icon: "settings" },
  { id: "debug", path: "/debug", label: "Classifier debug", group: "Tools", icon: "debug" },
];

export const PAGE_GROUPS = ["Workflow", "Setup", "Tools"];

export function pageById(id) {
  return PAGES.find((page) => page.id === id) || PAGES[0];
}

export function pageHref(id) {
  return `#${pageById(id).path}`;
}

// "#/review" -> "review".  Unknown paths land on the first page; anything
// under "#/debug" stays the debug page, as before.
export function routeFromHash(hash) {
  const path = String(hash || "").replace(/^#/, "") || "/";
  if (path.startsWith("/debug")) return "debug";
  const match = PAGES.find((page) => page.path !== "/" && path.startsWith(page.path));
  return match ? match.id : "run";
}

export const SETTINGS_SECTIONS = [
  { id: "settings-paths", label: "Paths & scan" },
  { id: "settings-model", label: "Tagger model" },
  { id: "settings-thresholds", label: "Thresholds" },
  { id: "settings-performance", label: "Performance" },
  { id: "settings-classification", label: "Classification" },
  { id: "settings-domain", label: "Tagging domain" },
  { id: "settings-appearance", label: "Appearance" },
];

// One entry per searchable settings unit.  `terms` is the text a search
// matches besides the label (help text, option names).
export const SETTINGS_INDEX = [
  {
    id: "set-root_repo",
    section: "settings-paths",
    label: "Root repository",
    terms: "unsorted images folder source path",
  },
  {
    id: "set-categories_root",
    section: "settings-paths",
    label: "Categories root",
    terms: "destination folders root path",
  },
  {
    id: "set-default_migrate_mode",
    section: "settings-paths",
    label: "Default migrate mode",
    terms: "copy move",
  },
  {
    id: "set-scan_recursive",
    section: "settings-paths",
    label: "Scan subfolders recursively",
    terms: "recursive subfolders",
  },
  {
    id: "set-preview_scan",
    section: "settings-paths",
    label: "Preview scan",
    terms: "eligible images count gif unsupported unreadable",
  },
  {
    id: "set-tagger_model",
    section: "settings-model",
    label: "Tagger model",
    terms: "ML-Danbooru WD SwinV2 v3 EVA02 Large",
  },
  {
    id: "set-confidence_threshold",
    section: "settings-thresholds",
    label: "Assignment confidence",
    terms: "confidence threshold primary suggestion",
  },
  {
    id: "set-wd_general_threshold",
    section: "settings-thresholds",
    label: "WD general threshold",
    terms: "WD14 inference cutoff general tags",
    when: "wd",
  },
  {
    id: "set-max_inference_workers",
    section: "settings-performance",
    label: "Max inference workers",
    terms: "GPU ORT sessions lock workers",
  },
  {
    id: "set-inference_batch_size",
    section: "settings-performance",
    label: "Inference batch size",
    terms: "batch GPU single-image",
  },
  {
    id: "set-force_cpu_inference",
    section: "settings-performance",
    label: "Force CPU inference",
    terms: "cpu gpu cuda",
  },
  {
    id: "set-experimental_media_enabled",
    section: "settings-classification",
    label: "Classify GIF and videos",
    terms: "experimental gif video multi-frame sampling",
  },
  {
    id: "set-experimental_style_detector_enabled",
    section: "settings-classification",
    label: "Real vs anime style gate",
    terms: "experimental style detector real_life onnx needs review",
  },
  {
    id: "set-hybrid_ml_on_review",
    section: "settings-classification",
    label: "Hybrid ML-Danbooru on needs-review",
    terms: "hybrid ml-danbooru merge allowlisted voyeur",
  },
  {
    id: "set-tagging_domain",
    section: "settings-domain",
    label: "Tagging domain",
    terms: "animated drawn real-life photos videos adult tagger vlm sensitive",
  },
  {
    id: "set-theme",
    section: "settings-appearance",
    label: "Theme",
    terms: "appearance palette colour color dark light auto graphite daylight picker",
  },
  {
    id: "set-skin",
    section: "settings-appearance",
    label: "YoRHa skin",
    terms: "nier automata grain scanlines boot intro",
  },
  {
    id: "set-reduce_motion",
    section: "settings-appearance",
    label: "Reduce motion",
    terms: "animation motion transitions instant accessibility",
  },
];

export const DEBUG_SECTIONS = [
  { id: "debug-tag-recall", label: "Tag recall benchmark" },
  { id: "debug-tag-fp", label: "Tag false-positive benchmark" },
  { id: "debug-style", label: "Style detectors" },
  { id: "debug-realism", label: "Real-life vs anime" },
  { id: "debug-sfw", label: "SFW tag recall" },
];

// Settings units that match every search term, keyed by section.  A section
// whose title matches shows all of its units.
export function searchSettings(query, isVisible = () => true) {
  const terms = String(query || "")
    .trim()
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean);
  const units = SETTINGS_INDEX.filter(isVisible);
  const hits = new Set();
  const counts = {};
  for (const section of SETTINGS_SECTIONS) {
    const titleHit =
      terms.length > 0 && terms.every((t) => section.label.toLowerCase().includes(t));
    counts[section.id] = 0;
    for (const unit of units.filter((u) => u.section === section.id)) {
      const hay = `${unit.label} ${unit.terms}`.toLowerCase();
      if (!terms.length || titleHit || terms.every((t) => hay.includes(t))) {
        hits.add(unit.id);
        counts[section.id] += 1;
      }
    }
  }
  return { terms, hits, counts, total: hits.size };
}

// Jump bar ranking: every term must appear in the label, group or extra
// search terms; word starts and label prefixes rank higher, earlier entries
// break ties.
export function scoreJumpEntry(entry, terms) {
  const label = entry.label.toLowerCase();
  const hay = `${label} ${entry.group.toLowerCase()} ${(entry.terms || "").toLowerCase()}`;
  let total = 0;
  for (const term of terms) {
    const at = hay.indexOf(term);
    if (at === -1) return -1;
    total += at === 0 || hay.charAt(at - 1) === " " ? 3 : 1;
    if (label.startsWith(term)) total += 2;
  }
  return total;
}

export function rankJumpEntries(entries, query, limit = 40) {
  const terms = String(query || "")
    .trim()
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean);
  if (!terms.length) {
    return entries.filter((entry) => !entry.quiet).slice(0, limit);
  }
  return entries
    .map((entry, index) => ({ entry, index, score: scoreJumpEntry(entry, terms) }))
    .filter((row) => row.score >= 0)
    .sort((a, b) => b.score - a.score || a.index - b.index)
    .slice(0, limit)
    .map((row) => row.entry);
}

// Static part of the jump index; App.jsx appends one entry per run.
export function staticJumpEntries() {
  return [
    ...PAGES.map((page) => ({
      key: `page-${page.id}`,
      label: page.label,
      group: "Pages",
      route: page.id,
    })),
    // Opens the theme picker instead of going to a page.
    {
      key: "appearance",
      label: "Appearance",
      group: "Change theme",
      terms: "theme palette colour color dark light skin yorha",
      appearance: true,
    },
    ...SETTINGS_SECTIONS.map((section) => ({
      key: `section-${section.id}`,
      label: section.label,
      group: "Settings section",
      route: "settings",
      target: section.id,
    })),
    // The Theme setting is a button that opens the picker, so its entry
    // opens the picker too.
    ...SETTINGS_INDEX.map((unit) => ({
      key: `setting-${unit.id}`,
      label: unit.label,
      group: "Setting",
      route: "settings",
      target: unit.id,
      quiet: true,
      ...(unit.id === "set-theme" ? { appearance: true } : {}),
    })),
    ...DEBUG_SECTIONS.map((section) => ({
      key: `debug-${section.id}`,
      label: section.label,
      group: "Classifier debug",
      route: "debug",
      target: section.id,
      quiet: true,
    })),
  ];
}
