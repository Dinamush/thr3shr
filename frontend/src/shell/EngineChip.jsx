import React, { useEffect, useId, useRef, useState } from "react";
import { useApp } from "../ui";

// Sidebar engine chip.  Clicking it opens the engine facts and the
// "Unload GPU models" action; in the phone "More" sheet the same panel
// opens inline under the chip.
export default function EngineChip({ engine }) {
  const { providerInfo, offlineMode, settings, anyRunActive, opsLoading, handleUnloadModels } =
    useApp();
  const [open, setOpen] = useState(false);
  const rootRef = useRef(null);
  const toggleRef = useRef(null);
  const panelId = useId();

  useEffect(() => {
    if (!open) return undefined;
    const onClick = (ev) => {
      if (rootRef.current && !rootRef.current.contains(ev.target)) setOpen(false);
    };
    const onKey = (ev) => {
      if (ev.key === "Escape") {
        setOpen(false);
        toggleRef.current?.focus();
      }
    };
    document.addEventListener("click", onClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("click", onClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  if (!engine) return null;

  const loaded = providerInfo?.loaded_models || [];
  const rows = providerInfo
    ? [
        ["Device", providerInfo.likely_device || "unknown"],
        ["CUDA", providerInfo.cuda_usable ? "usable" : "not usable", providerInfo.cuda_usable ? "ok" : null],
        ["Active", (providerInfo.active_providers || []).join(", ") || "n/a"],
        ["Tagger", providerInfo.tagger_model || settings.tagger_model],
        ["Forced CPU", providerInfo.forced_cpu ? "yes" : "no"],
        ["Loaded", loaded.length ? loaded.join(", ") : "none"],
      ]
    : [];

  return (
    <div className="engine" data-open={open ? "true" : "false"} ref={rootRef}>
      <button
        type="button"
        className={`engine-chip is-${engine.tone}`}
        ref={toggleRef}
        aria-expanded={open}
        aria-controls={panelId}
        title={`${engine.title} · engine details`}
        onClick={() => setOpen((prev) => !prev)}
      >
        <span className="engine-chip-dot" aria-hidden="true" />
        <span className="engine-chip-label">{engine.label}</span>
      </button>
      <div className="engine-pop" id={panelId} role="region" aria-label="Engine" hidden={!open}>
        <p className="engine-pop-title">Engine</p>
        {offlineMode ? (
          <p className="pane-empty">Demo mode: no inference engine attached.</p>
        ) : !providerInfo ? (
          <p className="pane-empty">Engine status unavailable.</p>
        ) : (
          <>
            <div className="stat-rows">
              {rows.map(([label, value, tone]) => (
                <div key={label} className={`stat-row${tone ? ` is-${tone}` : ""}`}>
                  <span>{label}</span>
                  <span className="stat-row-value">{value}</span>
                </div>
              ))}
            </div>
            <div className="pane-foot">
              <button
                type="button"
                className="btn btn-sm btn-block"
                disabled={anyRunActive || opsLoading.unloadingModels}
                onClick={handleUnloadModels}
                title={
                  anyRunActive
                    ? "Cancel the active run before unloading GPU models"
                    : "Drop cached tagger sessions to free GPU memory"
                }
              >
                {opsLoading.unloadingModels ? "Unloading…" : "Unload GPU models"}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
