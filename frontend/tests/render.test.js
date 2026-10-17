// Server-renders the app shell and every page in its main states, so a page
// that throws while rendering (a missing field, a bad prop) fails here rather
// than as a blank screen.  Effects do not run under renderToString; this
// checks markup and wiring, not behaviour in a browser.

import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import React from "react";
import { renderToString } from "react-dom/server";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
let server;
let load;

const noop = () => {};
const asyncNoop = async () => {};

function makeApp(overrides = {}) {
  return {
    settings: {
      root_repo: "/dump/inbox",
      categories_root: "/dump/bins",
      confidence_threshold: 0.6,
      default_migrate_mode: "copy",
      scan_recursive: true,
      experimental_media_enabled: false,
      experimental_style_detector_enabled: false,
      hybrid_ml_on_review: true,
      tagging_domain: "drawn",
      sfw_classify_mode: false,
      selected_tags: ["Voyeur"],
      selected_tags_nsfw: [],
      max_inference_workers: 2,
      inference_batch_size: 8,
      force_cpu_inference: false,
      tagger_model: "wd_swinv2_v3",
      wd_general_threshold: 0.35,
    },
    setSettings: noop,
    isDirty: false,
    dirtyCount: 0,
    isWdModel: true,
    loading: false,
    opsLoading: {
      saving: false,
      startingRun: false,
      applyingBatch: false,
      migrating: false,
      reclassifying: false,
      unloadingModels: false,
    },
    offlineMode: false,
    providerInfo: {
      likely_device: "cuda",
      cuda_usable: true,
      active_providers: ["CUDAExecutionProvider"],
      tagger_model: "wd_swinv2_v3",
      forced_cpu: false,
      loaded_models: [],
    },
    realLifeStatus: null,
    realLifeStatusLoading: false,
    runs: [],
    runId: null,
    runStatus: null,
    runActive: false,
    otherActiveRun: undefined,
    anyRunActive: false,
    items: [],
    stats: { total: 0, reviewNeeded: 0, approved: 0, rejected: 0, migrated: 0 },
    selectedIds: [],
    setSelectedIds: noop,
    selectedIdSet: new Set(),
    previewErrors: {},
    setPreviewErrors: noop,
    pendingItemIds: {},
    finalTagDrafts: {},
    expandedScoreRows: {},
    scoreDebugByItem: {},
    scoreLoadingByItem: {},
    migrateMode: "copy",
    setMigrateMode: noop,
    reclassifyModel: "wd_swinv2_v3",
    setReclassifyModel: noop,
    tagQuery: "",
    setTagQuery: noop,
    tagOptions: [],
    selectedTags: ["Voyeur", "Pokemon"],
    sfwClassifyMode: false,
    parkedNsfwTags: [],
    scanPreview: null,
    scanPreviewLoading: false,
    landing: null,
    setLanding: noop,
    setError: noop,
    openRun: noop,
    openRunForReview: noop,
    handleSaveSettings: asyncNoop,
    revertSettings: noop,
    handlePreviewScan: asyncNoop,
    handleStartRun: asyncNoop,
    updateStatus: asyncNoop,
    setFinalTagDraft: noop,
    dropFinalTagDraft: noop,
    commitFinalTag: asyncNoop,
    applyBatch: asyncNoop,
    migrateApproved: asyncNoop,
    cancelActiveRun: asyncNoop,
    handleUnloadModels: asyncNoop,
    handleReclassifyNeedsReview: asyncNoop,
    toggleScoreDebug: asyncNoop,
    setSfwClassifyMode: asyncNoop,
    addSelectedTag: noop,
    addTagsFromInput: asyncNoop,
    removeSelectedTag: noop,
    ...overrides,
  };
}

const RUN = {
  run_id: 12,
  status: "running",
  created_at: "2026-10-16 12:00:00",
  root_repo: "/dump/inbox",
  categories_root: "/dump/bins",
  tagger_model: "wd_swinv2_v3",
  tagging_domain: "drawn",
  total_images: 40,
  processed_images: 10,
  failed_images: 1,
  item_counts: { proposed: 6, approved: 3, migrated: 1 },
};

const ITEMS = [
  {
    id: 1,
    run_id: 12,
    file_path: "/dump/inbox/a/one.png",
    relative_path: "a/one.png",
    primary_tag: "Voyeur",
    primary_score: 0.91,
    secondary_suggestions: [{ tag: "BBC", score: 0.4 }],
    global_top_tags: [{ tag: "1girl", score: 0.99 }],
    final_tag: null,
    status: "proposed",
    needs_review: false,
    review_reason: null,
    migrated_to: null,
  },
  {
    id: 2,
    run_id: 12,
    file_path: "/dump/inbox/b/clip.mp4",
    relative_path: "b/clip.mp4",
    primary_tag: null,
    primary_score: null,
    secondary_suggestions: [],
    global_top_tags: [],
    final_tag: "Pokemon",
    status: "approved",
    needs_review: true,
    review_reason: "below threshold",
    migrated_to: null,
  },
  {
    id: 3,
    run_id: 12,
    file_path: "/dump/inbox/c/done.jpg",
    relative_path: "c/done.jpg",
    primary_tag: "Pokemon",
    primary_score: 0.8,
    secondary_suggestions: [],
    global_top_tags: [],
    final_tag: "Pokemon",
    status: "migrated",
    needs_review: false,
    review_reason: null,
    migrated_to: "/dump/bins/Pokemon/done.jpg",
  },
];

const ACTIVE = {
  runs: [RUN, { ...RUN, run_id: 11, status: "completed", processed_images: 40 }],
  runId: 12,
  runStatus: {
    run_id: 12,
    status: "running",
    total_images: 40,
    processed_images: 10,
    failed_images: 1,
    progress_pct: 25,
    eta_seconds_remaining: 90,
    eta_finish_at: "2026-10-16T12:05:00Z",
    avg_infer_ms_per_image: 120,
    inference_mode: "batched",
    cancel_requested: false,
    tagger_model: "wd_swinv2_v3",
  },
  runActive: true,
  anyRunActive: true,
  items: ITEMS,
  stats: { total: 3, reviewNeeded: 1, approved: 1, rejected: 0, migrated: 1 },
  selectedIds: [1],
  selectedIdSet: new Set([1]),
  scanPreview: {
    stats: {
      eligible_images: 40,
      total_files: 44,
      ignored_gif: 2,
      ignored_unsupported: 1,
      failed_to_read: 1,
    },
    recursive: true,
  },
};

before(async () => {
  // Just enough of a browser for module-level and initial-state reads.
  const store = new Map();
  globalThis.localStorage = {
    getItem: (k) => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => store.set(k, String(v)),
    removeItem: (k) => store.delete(k),
  };
  globalThis.window = { location: { hash: "#/" }, matchMedia: () => ({ matches: false }) };
  globalThis.document = {
    documentElement: { dataset: {}, classList: { contains: () => false } },
  };
  server = await createServer({
    root,
    configFile: `${root}vite.config.js`,
    logLevel: "silent",
    server: { middlewareMode: true, hmr: false },
    appType: "custom",
  });
  load = (path) => server.ssrLoadModule(path);
});

after(async () => {
  await server?.close();
});

// React marks text-node boundaries with <!-- -->; drop them so assertions
// can match the text a user sees.
function render(element) {
  return renderToString(element).replace(/<!-- -->/g, "");
}

async function renderPage(path, app) {
  const { AppContext } = await load("/src/ui.jsx");
  const { AppearanceProvider } = await load("/src/theme/Appearance.jsx");
  const Page = (await load(path)).default;
  return render(
    React.createElement(
      AppContext.Provider,
      { value: app },
      React.createElement(AppearanceProvider, null, React.createElement(Page))
    )
  );
}

test("the whole app renders its shell on the first page", async () => {
  const App = (await load("/src/App.jsx")).default;
  const html = render(React.createElement(App));
  for (const text of ["THR3SHR", "Tag &amp; run", "Jump to a page, setting or run", "Run activity", "Settings"]) {
    assert.ok(html.includes(text), `missing ${text}`);
  }
  assert.ok(html.includes('aria-current="page"'), "current page is marked");
});

test("Tag & run renders idle and with a live run", async () => {
  const idle = await renderPage("/src/pages/RunPage.jsx", makeApp());
  assert.ok(idle.includes("Start run"));
  assert.ok(idle.includes("Preview scan"));
  assert.ok(!idle.includes("Cancel run"));
  // Two columns: the engine lives on the sidebar chip, recent runs on Runs.
  assert.ok(!idle.includes("Unload GPU models"));
  assert.ok(!idle.includes("Recent runs"));

  const live = await renderPage("/src/pages/RunPage.jsx", makeApp(ACTIVE));
  assert.ok(live.includes("Run #12"));
  assert.ok(live.includes("Cancel run"));
  assert.ok(live.includes("Eligible images"));
  assert.ok(
    live.indexOf('id="run-dashboard-title"') < live.indexOf('id="new-run-title"'),
    "a live run's dashboard sits above the New run form"
  );

  const done = await renderPage(
    "/src/pages/RunPage.jsx",
    makeApp({
      ...ACTIVE,
      runStatus: { ...ACTIVE.runStatus, status: "completed" },
      runActive: false,
      anyRunActive: false,
    })
  );
  assert.ok(
    done.indexOf('id="new-run-title"') < done.indexOf('id="run-dashboard-title"'),
    "a finished run's dashboard follows the New run form"
  );

  const realLife = await renderPage(
    "/src/pages/RunPage.jsx",
    makeApp({ settings: { ...makeApp().settings, tagging_domain: "real_life" } })
  );
  assert.ok(realLife.includes("Start real-life tagging"));
  assert.ok(realLife.includes("Animated / drawn domain only"));
});

test("Review renders the empty state and a split list with detail", async () => {
  const empty = await renderPage("/src/pages/ReviewPage.jsx", makeApp());
  assert.ok(empty.includes("No run open"));
  assert.ok(!empty.includes("Open live run"));

  const liveElsewhere = await renderPage(
    "/src/pages/ReviewPage.jsx",
    makeApp({ otherActiveRun: RUN, anyRunActive: true, runs: [RUN] })
  );
  assert.ok(liveElsewhere.includes("Open live run #12"));

  const html = await renderPage("/src/pages/ReviewPage.jsx", makeApp(ACTIVE));
  for (const text of [
    "Review run #12",
    "a/one.png",
    "Approve selected (1)",
    "Reject selected (1)",
    "Migrate approved (1)",
    "Final tag for a/one.png",
    "Select all shown items that are not migrated yet",
    "Global top",
    "sensitive",
  ]) {
    assert.ok(html.includes(text), `missing ${text}`);
  }
  assert.ok(html.includes('class="review-list"'), "list is the default layout");
  assert.ok(!html.includes('class="review-grid"'));
  // Migrate has its own outlined style, distinct from Approve selected.
  assert.match(html, /class="btn btn-migrate"[^>]*>.*Migrate approved \(1\)/);
  // A fresh item's "proposed" chip is neutral, not a warning.
  assert.ok(html.includes("status-chip is-neutral status-proposed"));
  assert.ok(!html.includes("is-warn status-proposed"));
  // Status reads without colour: every chip leads with its glyph.
  assert.match(html, /class="status-chip is-neutral status-proposed"><svg[^>]*class="status-glyph"/);
  assert.match(html, /class="status-chip is-done status-migrated"><svg[^>]*class="status-glyph"/);
});

test("Review remembers the Grid layout and renders tiles", async () => {
  localStorage.setItem("thr3shr-review-layout", "grid");
  try {
    const html = await renderPage("/src/pages/ReviewPage.jsx", makeApp(ACTIVE));
    assert.ok(html.includes('class="review-grid"'));
    assert.ok(!html.includes('class="review-list"'));
    assert.ok(html.includes('data-layout="grid"'));
    assert.match(html, /aria-pressed="true" aria-label="Grid"/);
    // The detail pane's back button returns to the grid.
    assert.match(html, /review-back"[^>]*><svg[^>]*>.*?<\/svg>Grid</);
    for (const text of ["one.png", "clip.mp4", "Select a/one.png", "Final tag for a/one.png"]) {
      assert.ok(html.includes(text), `missing ${text}`);
    }
    // Migrated items cannot be selected in the grid either.
    assert.match(html, /<input type="checkbox" class="grid-check" aria-label="Select c\/done.jpg" disabled=""/);
  } finally {
    localStorage.removeItem("thr3shr-review-layout");
  }
});

test("the sidebar engine chip carries the engine facts and the unload guard", async () => {
  const { AppContext } = await load("/src/ui.jsx");
  const EngineChip = (await load("/src/shell/EngineChip.jsx")).default;
  const engine = { tone: "ok", label: "cuda · wd_swinv2_v3", title: "Active: CUDAExecutionProvider" };
  const chip = (app) =>
    render(
      React.createElement(
        AppContext.Provider,
        { value: app },
        React.createElement(EngineChip, { engine })
      )
    );
  const idle = chip(makeApp());
  assert.ok(idle.includes("cuda · wd_swinv2_v3"));
  assert.ok(idle.includes('aria-expanded="false"'));
  assert.match(idle, /<button type="button" class="btn btn-sm btn-block" title="Drop cached/);

  const busy = chip(makeApp(ACTIVE));
  assert.match(busy, /disabled="" title="Cancel the active run before unloading GPU models"/);

  assert.equal(
    render(
      React.createElement(
        AppContext.Provider,
        { value: makeApp() },
        React.createElement(EngineChip, { engine: null })
      )
    ),
    ""
  );
});

test("every icon the app names exists in the set", async () => {
  const { ICON_NAMES, STATUS_GLYPH_NAMES } = await load("/src/shell/icons.jsx");
  assert.equal(ICON_NAMES.length, 29);
  const { PAGES } = await load("/src/shell/nav.js");
  for (const page of PAGES) assert.ok(ICON_NAMES.includes(page.icon), `no icon ${page.icon}`);
  const used = new Set();
  for (const file of [
    "App.jsx",
    "ClassifierDebugPage.jsx",
    "pages/ReviewPage.jsx",
    "pages/RunPage.jsx",
    "pages/RunsPage.jsx",
    "pages/SettingsPage.jsx",
    "shell/AppShell.jsx",
    "shell/EngineChip.jsx",
    "shell/JumpBar.jsx",
    "shell/RunDock.jsx",
    "theme/Appearance.jsx",
    "theme/ThemePicker.jsx",
  ]) {
    const src = readFileSync(`${root}src/${file}`, "utf8");
    for (const [, name] of src.matchAll(/<Icon name="([a-z]+)"/g)) used.add(name);
    for (const [, name] of src.matchAll(/<StatusGlyph name="([a-z]+)"/g)) {
      assert.ok(STATUS_GLYPH_NAMES.includes(name), `no status glyph ${name}`);
    }
  }
  for (const name of used) assert.ok(ICON_NAMES.includes(name), `no icon ${name}`);
  // Every item, run and flag status has its own glyph.
  for (const status of ["proposed", "reviewed", "approved", "rejected", "migrated", "running", "completed", "cancelled", "failed", "sensitive", "fallback", "queued", "needs"]) {
    assert.ok(STATUS_GLYPH_NAMES.includes(status), `no status glyph ${status}`);
  }
});

test("the app icon files match their source", async () => {
  const { faviconSvg } = await load("/src/shell/appIcon.js");
  assert.equal(readFileSync(`${root}public/favicon.svg`, "utf8"), faviconSvg());
  const html = readFileSync(`${root}index.html`, "utf8");
  for (const href of ["/favicon.ico", "/favicon.svg", "/apple-touch-icon.png", "/manifest.webmanifest"]) {
    assert.ok(html.includes(`href="${href}"`), `index.html links ${href}`);
  }
  const manifest = JSON.parse(readFileSync(`${root}public/manifest.webmanifest`, "utf8"));
  for (const icon of manifest.icons) {
    const png = readFileSync(`${root}public${icon.src}`);
    const [w, h] = icon.sizes.split("x").map(Number);
    assert.deepEqual([png.readUInt32BE(16), png.readUInt32BE(20)], [w, h], icon.src);
  }
  // The desktop icon carries a 256px image (stored as 0 in the ICO header).
  const ico = readFileSync(`${root}public/app.ico`);
  const widths = Array.from({ length: ico.readUInt16LE(4) }, (_, i) => ico.readUInt8(6 + 16 * i) || 256);
  assert.deepEqual(widths, [16, 24, 32, 48, 64, 128, 256]);
});

test("fonts ship with the app instead of loading from Google", () => {
  const html = readFileSync(`${root}index.html`, "utf8");
  assert.ok(!/fonts\.(googleapis|gstatic)\.com/.test(html));
  const main = readFileSync(`${root}src/main.jsx`, "utf8");
  assert.ok(main.includes("@fontsource/figtree"));
  assert.ok(main.includes("@fontsource/fragment-mono"));
});

test("Runs renders an empty state and the run table", async () => {
  assert.ok((await renderPage("/src/pages/RunsPage.jsx", makeApp())).includes("No runs yet"));
  const html = await renderPage("/src/pages/RunsPage.jsx", makeApp(ACTIVE));
  assert.ok(html.includes("Open run #11"));
  assert.ok(html.includes("Viewing"));
});

test("Settings renders every section with the save bar", async () => {
  const html = await renderPage("/src/pages/SettingsPage.jsx", makeApp());
  for (const text of ["Paths &amp; scan", "Tagger model", "Thresholds", "Performance", "Classification", "Tagging domain", "Appearance", "Change theme…", "YoRHa skin", "Reduce motion", "Save settings", "Settings saved"]) {
    assert.ok(html.includes(text), `missing ${text}`);
  }
  const dirty = await renderPage("/src/pages/SettingsPage.jsx", makeApp({ isDirty: true, dirtyCount: 2 }));
  assert.ok(dirty.includes("2 unsaved changes"));
  assert.ok(dirty.includes("Revert"));
});

test("the classifier debug page renders inside the shell layout", async () => {
  const Page = (await load("/src/ClassifierDebugPage.jsx")).default;
  const html = render(React.createElement(Page, { landing: null, onLanded: noop }));
  assert.ok(html.includes("Classifier debug"));
  assert.ok(html.includes('id="debug-tag-recall"'));
  assert.ok(html.includes("Run tag recall benchmark"));
});
