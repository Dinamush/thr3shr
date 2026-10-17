import React, { useEffect, useMemo, useRef, useState } from "react";
import NumberField from "../NumberField";
import Icon from "../shell/icons";
import { TAGGER_MODELS } from "../lib";
import { SETTINGS_SECTIONS, searchSettings } from "../shell/nav";
import { useAppearance } from "../theme/Appearance";
import { motionReduced } from "../theme/appearance";
import { ThemeMini } from "../theme/ThemePicker";
import { PageHead, SwitchRow, useApp } from "../ui";
import { PreviewScanButton, ScanPreview } from "./RunPage";

function Unit({ id, search, className = "", children }) {
  const hit = search.hits.has(id);
  return (
    <div
      id={id}
      className={[
        "form-group",
        "settings-unit",
        className,
        search.terms.length && hit ? "is-search-hit" : "",
      ]
        .filter(Boolean)
        .join(" ")}
      hidden={!hit}
    >
      {children}
    </div>
  );
}

function Section({ id, search, children }) {
  const section = SETTINGS_SECTIONS.find((s) => s.id === id);
  return (
    <section
      id={id}
      className="settings-section"
      aria-labelledby={`${id}-title`}
      hidden={search.terms.length > 0 && !search.counts[id]}
    >
      <h2 className="settings-section-title" id={`${id}-title`}>
        {section.label}
      </h2>
      {children}
    </section>
  );
}

function RealLifeStatus() {
  const { realLifeStatus, realLifeStatusLoading } = useApp();
  return (
    <div className="callout real-life-status">
      <div className="callout-body">
        {realLifeStatusLoading ? (
          <span className="muted">Checking local adult tagger…</span>
        ) : realLifeStatus?.engine ? (
          <>
            <div>
              VLM:{" "}
              <strong>{realLifeStatus.engine.vlm?.available ? "ready" : "unavailable"}</strong>
              {realLifeStatus.engine.vlm?.backend ? ` (${realLifeStatus.engine.vlm.backend})` : ""}
            </div>
            <div>
              Position classifier:{" "}
              <strong>
                {realLifeStatus.engine.position?.available ? "ready" : "unavailable"}
              </strong>
            </div>
            <p className="help-text">{realLifeStatus.engine.message}</p>
            {realLifeStatus.engine.fallback ? (
              <p className="help-text">
                Fallback active: style gate + manual review only until{" "}
                <code>llama-cpp-python</code> and the Qwen2.5-VL GGUF + mmproj are installed.
              </p>
            ) : null}
          </>
        ) : (
          <span className="muted">Real-life tagger status unavailable.</span>
        )}
      </div>
    </div>
  );
}

// Appearance lives in this browser, not on the server: it saves as it changes
// and is not part of the save bar.
function AppearanceSection({ search }) {
  const appearance = useAppearance();
  const { current } = appearance;
  return (
    <Section id="settings-appearance" search={search}>
      <Unit id="set-theme" search={search}>
        <div className="theme-setting">
          <span className="theme-setting-mini">
            <ThemeMini row={current} />
          </span>
          <span className="theme-setting-text">
            <span className="theme-setting-label">Theme · {current.name}</span>
            <span className="help-text">{current.sub}. Saved in this browser.</span>
          </span>
          <button
            type="button"
            className="btn btn-sm"
            aria-haspopup="dialog"
            onClick={(ev) => appearance.openPicker(ev.currentTarget)}
          >
            Change theme…
          </button>
        </div>
      </Unit>
      <Unit id="set-skin" search={search}>
        <SwitchRow
          id="settingsSkin"
          label="YoRHa skin"
          help="Boot intro, film grain and scanlines over any palette."
          checked={appearance.skin}
          onChange={appearance.setSkin}
        />
      </Unit>
      <Unit id="set-reduce_motion" search={search}>
        <SwitchRow
          id="settingsReduceMotion"
          label="Reduce motion"
          help="Makes every animation and transition instant. Your system's reduced-motion setting does the same."
          checked={appearance.reduceMotion}
          onChange={appearance.setReduceMotion}
        />
      </Unit>
    </Section>
  );
}

export default function SettingsPage() {
  const app = useApp();
  const { settings, setSettings, isWdModel, isDirty, dirtyCount, loading, opsLoading, offlineMode } =
    app;
  const [query, setQuery] = useState("");
  const [currentSection, setCurrentSection] = useState(SETTINGS_SECTIONS[0].id);
  const searchRef = useRef(null);
  const search = useMemo(
    () => searchSettings(query, (unit) => unit.when !== "wd" || isWdModel),
    [query, isWdModel]
  );
  const set = (field) => (value) => setSettings((prev) => ({ ...prev, [field]: value }));

  // Scroll-spy: the first visible section is the current one in the index.
  useEffect(() => {
    if (!("IntersectionObserver" in window)) return undefined;
    const visible = new Map();
    const spy = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => visible.set(e.target.id, e.isIntersecting ? e.intersectionRatio : 0));
        const first = SETTINGS_SECTIONS.find((s) => (visible.get(s.id) || 0) > 0);
        if (first) setCurrentSection(first.id);
      },
      { rootMargin: "-64px 0px -55% 0px", threshold: [0, 0.01] }
    );
    SETTINGS_SECTIONS.forEach((s) => {
      const el = document.getElementById(s.id);
      if (el) spy.observe(el);
    });
    return () => spy.disconnect();
  }, []);

  // A jump-bar landing clears any filter that would hide its target, scrolls
  // it into view, flashes it and focuses its control.
  useEffect(() => {
    const target = app.landing?.target;
    if (!target) return undefined;
    setQuery("");
    const frame = requestAnimationFrame(() => {
      const el = document.getElementById(target);
      if (!el) return;
      el.scrollIntoView({ block: "center" });
      el.classList.remove("is-landed");
      void el.offsetWidth;
      el.classList.add("is-landed");
      el.querySelector("input, select, textarea")?.focus({ preventScroll: true });
      app.setLanding(null);
    });
    return () => cancelAnimationFrame(frame);
  }, [app.landing]);

  return (
    <>
      <PageHead
        title="Settings"
        lede="Paths, model, thresholds, performance and appearance. Runs and preview scans save pending changes first."
      />
      <div className="settings-layout">
        <nav className="settings-nav" aria-label="Settings sections">
          <div className="settings-search">
            <label className="sr-only" htmlFor="settingsSearch">
              Search settings
            </label>
            <input
              ref={searchRef}
              id="settingsSearch"
              type="search"
              placeholder="Search settings"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Escape" && query) {
                  e.preventDefault();
                  setQuery("");
                }
              }}
            />
          </div>
          <ol className="settings-nav-list">
            {SETTINGS_SECTIONS.map((section) => {
              const hidden = search.terms.length > 0 && !search.counts[section.id];
              return (
                <li key={section.id}>
                  <a
                    href={`#/settings`}
                    className={`settings-nav-link${hidden ? " is-filtered-out" : ""}`}
                    aria-current={currentSection === section.id ? "true" : undefined}
                    onClick={(e) => {
                      e.preventDefault();
                      document
                        .getElementById(section.id)
                        ?.scrollIntoView({ block: "start", behavior: motionReduced() ? "auto" : "smooth" });
                    }}
                  >
                    {section.label}
                    <span className="settings-nav-count" aria-hidden="true">
                      {search.terms.length && !hidden ? search.counts[section.id] : ""}
                    </span>
                  </a>
                </li>
              );
            })}
          </ol>
          <p className="settings-nav-note">
            Destination tags are picked on Tag &amp; run and save on their own.
          </p>
        </nav>

        <form className="settings-form" onSubmit={app.handleSaveSettings}>
          {search.terms.length > 0 && search.total === 0 ? (
            <p className="settings-no-match">
              No setting matches “{query.trim()}”.
            </p>
          ) : null}

          <Section id="settings-paths" search={search}>
            <Unit id="set-root_repo" search={search}>
              <label htmlFor="root_repo">Root repository</label>
              <input
                id="root_repo"
                className="mono-input"
                value={settings.root_repo}
                onChange={(e) => set("root_repo")(e.target.value)}
                placeholder="Folder of unsorted images"
              />
            </Unit>
            <Unit id="set-categories_root" search={search}>
              <label htmlFor="categories_root">Categories root</label>
              <input
                id="categories_root"
                className="mono-input"
                value={settings.categories_root}
                onChange={(e) => set("categories_root")(e.target.value)}
                placeholder="Destination folders root (not tags.csv)"
              />
            </Unit>
            <Unit id="set-default_migrate_mode" search={search}>
              <label htmlFor="default_migrate_mode">Default migrate mode</label>
              <select
                id="default_migrate_mode"
                value={settings.default_migrate_mode}
                onChange={(e) => set("default_migrate_mode")(e.target.value)}
              >
                <option value="copy">copy</option>
                <option value="move">move</option>
              </select>
            </Unit>
            <Unit id="set-scan_recursive" search={search}>
              <SwitchRow
                id="scan_recursive"
                label="Scan subfolders recursively"
                checked={Boolean(settings.scan_recursive)}
                onChange={set("scan_recursive")}
              />
            </Unit>
            <Unit id="set-preview_scan" search={search}>
              <div className="settings-inline">
                <span className="field-label">Preview scan</span>
                <PreviewScanButton />
              </div>
              <p className="help-text">Counts eligible images under the root before a run.</p>
              <ScanPreview />
            </Unit>
          </Section>

          <Section id="settings-model" search={search}>
            <Unit id="set-tagger_model" search={search}>
              <fieldset className="radio-group">
                <legend className="sr-only">Tagger model</legend>
                {TAGGER_MODELS.map((model) => (
                  <label key={model.id} className="radio-option">
                    <input
                      type="radio"
                      name="tagger_model"
                      value={model.id}
                      checked={settings.tagger_model === model.id}
                      onChange={() => set("tagger_model")(model.id)}
                    />
                    <span className="radio-option-text">
                      <span className="radio-option-label">{model.label}</span>
                      <span className="radio-option-help">{model.help}</span>
                    </span>
                  </label>
                ))}
              </fieldset>
            </Unit>
          </Section>

          <Section id="settings-thresholds" search={search}>
            <Unit id="set-confidence_threshold" search={search}>
              <label htmlFor="confidence_threshold">Assignment confidence</label>
              <NumberField
                id="confidence_threshold"
                className="mono-input field-narrow"
                step={0.01}
                min={0}
                max={1}
                value={settings.confidence_threshold}
                onCommit={set("confidence_threshold")}
              />
              <p className="help-text">
                Selected-tag scores below this are suggestions only (no primary assignment).
              </p>
            </Unit>
            {isWdModel ? (
              <Unit id="set-wd_general_threshold" search={search}>
                <label htmlFor="wd_general_threshold">WD general threshold</label>
                <NumberField
                  id="wd_general_threshold"
                  className="mono-input field-narrow"
                  step={0.01}
                  min={0}
                  max={1}
                  value={settings.wd_general_threshold}
                  onCommit={set("wd_general_threshold")}
                />
                <p className="help-text">WD14 inference cutoff for general tags (default 0.35).</p>
              </Unit>
            ) : null}
          </Section>

          <Section id="settings-performance" search={search}>
            <Unit id="set-max_inference_workers" search={search}>
              <label htmlFor="max_inference_workers">Max inference workers</label>
              <NumberField
                id="max_inference_workers"
                className="mono-input field-narrow"
                min={1}
                max={16}
                integer
                value={settings.max_inference_workers ?? 2}
                onCommit={set("max_inference_workers")}
              />
              <p className="help-text">Keep at 2 for GPU ORT sessions (serialized lock).</p>
            </Unit>
            <Unit id="set-inference_batch_size" search={search}>
              <label htmlFor="inference_batch_size">Inference batch size</label>
              <NumberField
                id="inference_batch_size"
                className="mono-input field-narrow"
                min={1}
                max={64}
                integer
                value={settings.inference_batch_size ?? 8}
                onCommit={set("inference_batch_size")}
              />
              <p className="help-text">
                WD models: 4–8 is a good GPU default. 1 forces single-image mode.
              </p>
            </Unit>
            <Unit id="set-force_cpu_inference" search={search}>
              <SwitchRow
                id="force_cpu_inference"
                label="Force CPU inference"
                checked={Boolean(settings.force_cpu_inference)}
                onChange={set("force_cpu_inference")}
              />
            </Unit>
          </Section>

          <Section id="settings-classification" search={search}>
            <Unit id="set-experimental_media_enabled" search={search}>
              <SwitchRow
                id="experimental_media_enabled"
                label="Classify GIF and videos"
                help="Experimental: length-scaled multi-frame sampling."
                checked={Boolean(settings.experimental_media_enabled)}
                onChange={set("experimental_media_enabled")}
              />
            </Unit>
            <Unit id="set-experimental_style_detector_enabled" search={search}>
              <SwitchRow
                id="experimental_style_detector_enabled"
                label="Real vs anime style gate"
                help={
                  <>
                    Experimental: dedicated ONNX → <code>real_life</code>; uncertain → needs
                    review.
                  </>
                }
                checked={Boolean(settings.experimental_style_detector_enabled)}
                onChange={set("experimental_style_detector_enabled")}
              />
            </Unit>
            <Unit id="set-hybrid_ml_on_review" search={search}>
              <SwitchRow
                id="hybrid_ml_on_review"
                label="Hybrid ML-Danbooru on needs-review"
                help="On needs-review (WD primary), re-run ML-Danbooru and merge allowlisted high-recall tags only (skips Voyeur soft cues). Drawn domain only."
                checked={Boolean(settings.hybrid_ml_on_review)}
                disabled={settings.tagging_domain === "real_life"}
                onChange={set("hybrid_ml_on_review")}
              />
            </Unit>
          </Section>

          <Section id="settings-domain" search={search}>
            <Unit id="set-tagging_domain" search={search}>
              <fieldset className="radio-group">
                <legend className="sr-only">Tagging domain</legend>
                <label className="radio-option">
                  <input
                    type="radio"
                    name="tagging_domain"
                    checked={settings.tagging_domain !== "real_life"}
                    onChange={() => setSettings((prev) => ({ ...prev, tagging_domain: "drawn" }))}
                  />
                  <span className="radio-option-text">
                    <span className="radio-option-label">Animated / drawn</span>
                    <span className="radio-option-help">WD + ML-Danbooru taxonomy.</span>
                  </span>
                </label>
                <label className="radio-option">
                  <input
                    type="radio"
                    name="tagging_domain"
                    checked={settings.tagging_domain === "real_life"}
                    onChange={() =>
                      setSettings((prev) => ({
                        ...prev,
                        tagging_domain: "real_life",
                        experimental_media_enabled: true,
                        sfw_classify_mode: false,
                      }))
                    }
                  />
                  <span className="radio-option-text">
                    <span className="radio-option-label">Real-life photos / videos</span>
                    <span className="radio-option-help">Local adult tagger.</span>
                  </span>
                </label>
              </fieldset>
              <p className="help-text">
                Real-life mode uses an isolated taxonomy under <code>Real Life/&lt;primary&gt;/</code>.
                Sensitive tags (BBC / Ebony / Asian) are suggestions only and need explicit
                Final-tag confirmation. Save settings after switching domains.
              </p>
              {settings.tagging_domain === "real_life" && !offlineMode ? <RealLifeStatus /> : null}
            </Unit>
          </Section>

          <AppearanceSection search={search} />

          <div className={`sticky-save-bar${isDirty ? " is-dirty" : ""}`}>
            <span className="sticky-save-status" role="status">
              {isDirty ? <Icon name="edit" size={14} /> : null}
              {isDirty
                ? `${dirtyCount} unsaved change${dirtyCount === 1 ? "" : "s"}`
                : "Settings saved"}
            </span>
            {isDirty ? (
              <button
                type="button"
                className="btn btn-ghost"
                disabled={opsLoading.saving}
                onClick={app.revertSettings}
              >
                Revert
              </button>
            ) : null}
            <button
              type="submit"
              className="btn btn-primary"
              disabled={loading || opsLoading.saving || !isDirty}
            >
              {opsLoading.saving ? "Saving…" : "Save settings"}
            </button>
          </div>
        </form>
      </div>
    </>
  );
}
