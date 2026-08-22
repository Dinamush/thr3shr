import React, { useEffect, useMemo, useState } from "react"
import { api } from "./api"

const DEFAULT_SOURCES = [
  { id: "safebooru", label: "Safebooru", sfw_policy: "rating:safe", max_content_tags: null },
  { id: "danbooru", label: "Danbooru", sfw_policy: "rating:g", max_content_tags: 2 },
]

function pct(value) {
  if (value == null || Number.isNaN(Number(value))) return "n/a"
  return `${(Number(value) * 100).toFixed(1)}%`
}

export default function ClassifierDebugPage() {
  const [sources, setSources] = useState(DEFAULT_SOURCES)
  const [source, setSource] = useState("safebooru")
  const [pullTags, setPullTags] = useState(["1girl", "solo"])
  const [count, setCount] = useState(10)
  const [tagQuery, setTagQuery] = useState("")
  const [tagOptions, setTagOptions] = useState([])
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [settingsSummary, setSettingsSummary] = useState({
    tagger_model: "wd_swinv2_v3",
    confidence_threshold: 0.6,
    selected_tags: [],
    experimental_style_detector_enabled: false,
  })

  const [realismCount, setRealismCount] = useState(12)
  const [realismCompare, setRealismCompare] = useState(true)
  const [realismLoading, setRealismLoading] = useState(false)
  const [realismError, setRealismError] = useState("")
  const [realismResult, setRealismResult] = useState(null)

  const [styleCount, setStyleCount] = useState(12)
  const [styleLoading, setStyleLoading] = useState(false)
  const [styleError, setStyleError] = useState("")
  const [styleResult, setStyleResult] = useState(null)

  const [tagRecallThreshold, setTagRecallThreshold] = useState(0.35)
  const [tagRecallTopK, setTagRecallTopK] = useState(20)
  const [tagRecallLoading, setTagRecallLoading] = useState(false)
  const [tagRecallError, setTagRecallError] = useState("")
  const [tagRecallResult, setTagRecallResult] = useState(null)

  const [tagFpThreshold, setTagFpThreshold] = useState(0.6)
  const [tagFpMinWeight, setTagFpMinWeight] = useState(0.85)
  const [tagFpAlsoWd, setTagFpAlsoWd] = useState(true)
  const [tagFpLoading, setTagFpLoading] = useState(false)
  const [tagFpError, setTagFpError] = useState("")
  const [tagFpResult, setTagFpResult] = useState(null)

  const sourceMeta = useMemo(
    () => sources.find((s) => s.id === source) || sources[0],
    [sources, source]
  )

  useEffect(() => {
    api
      .getSfwDebugSources()
      .then((data) => {
        if (Array.isArray(data?.sources) && data.sources.length) {
          setSources(data.sources)
        }
      })
      .catch(() => {})
    api
      .getSettings()
      .then((data) => {
        const thr = Number(data.confidence_threshold ?? 0.6)
        setSettingsSummary({
          tagger_model: data.tagger_model || "wd_swinv2_v3",
          confidence_threshold: thr,
          selected_tags: Array.isArray(data.selected_tags) ? data.selected_tags : [],
          experimental_style_detector_enabled: Boolean(
            data.experimental_style_detector_enabled
          ),
        })
        setTagFpThreshold(thr)
      })
      .catch((err) => setError(err.message))
  }, [])

  useEffect(() => {
    let cancelled = false
    const timer = setTimeout(() => {
      api
        .getTags(tagQuery, 20)
        .then((data) => {
          if (!cancelled) setTagOptions(data.items || [])
        })
        .catch(() => {
          if (!cancelled) setTagOptions([])
        })
    }, 150)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [tagQuery])

  const handleAddPullTag = (value) => {
    if (!value) return
    const maxTags = sourceMeta?.max_content_tags
    setPullTags((prev) => {
      if (prev.includes(value)) return prev
      if (maxTags != null && prev.length >= maxTags) {
        setError(`${sourceMeta.label} allows at most ${maxTags} content tag(s).`)
        return prev
      }
      setError("")
      return [...prev, value]
    })
  }

  const handleFetchEvaluate = async () => {
    if (pullTags.length === 0) {
      setError("Add at least one tag to pull for the SFW debug eval.")
      return
    }
    setLoading(true)
    setError("")
    try {
      const next = await api.runSfwDebugEval({
        source,
        tags: pullTags,
        count,
      })
      setResult(next)
      const refreshed = await api.getSettings()
      setSettingsSummary({
        tagger_model: refreshed.tagger_model || next.tagger_model,
        confidence_threshold: Number(
          refreshed.confidence_threshold ?? next.confidence_threshold ?? 0.6
        ),
        selected_tags: Array.isArray(refreshed.selected_tags)
          ? refreshed.selected_tags
          : next.destination_tags || [],
        experimental_style_detector_enabled: Boolean(
          refreshed.experimental_style_detector_enabled
        ),
      })
    } catch (err) {
      setError(`SFW debug eval failed: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  const handleRealismEval = async () => {
    setRealismLoading(true)
    setRealismError("")
    try {
      const next = await api.runRealismDebugEval({
        count_per_class: realismCount,
        compare_models: realismCompare,
        tagger_model: realismCompare ? null : settingsSummary.tagger_model,
      })
      setRealismResult(next)
    } catch (err) {
      setRealismError(`Realism eval failed: ${err.message}`)
    } finally {
      setRealismLoading(false)
    }
  }

  const handleStyleEval = async () => {
    setStyleLoading(true)
    setStyleError("")
    try {
      const next = await api.runStyleDebugEval({
        count_per_class: styleCount,
        tagger_model: settingsSummary.tagger_model,
      })
      setStyleResult(next)
    } catch (err) {
      setStyleError(`Style detector eval failed: ${err.message}`)
    } finally {
      setStyleLoading(false)
    }
  }

  const handleTagRecallEval = async () => {
    setTagRecallLoading(true)
    setTagRecallError("")
    try {
      const next = await api.runTagRecallEval({
        models: ["ml_danbooru", "wd_swinv2_v3"],
        threshold: tagRecallThreshold,
        top_k: tagRecallTopK,
        refresh_cache: false,
        include_items: true,
      })
      setTagRecallResult(next)
    } catch (err) {
      setTagRecallError(`Tag recall eval failed: ${err.message}`)
    } finally {
      setTagRecallLoading(false)
    }
  }

  const handleTagFpEval = async () => {
    setTagFpLoading(true)
    setTagFpError("")
    try {
      const next = await api.runTagFpEval({
        models: ["ml_danbooru", "wd_swinv2_v3"],
        tag_threshold: tagFpThreshold,
        route_threshold: tagFpThreshold,
        min_weight: tagFpMinWeight,
        also_wd_threshold: tagFpAlsoWd,
        refresh_cache: false,
      })
      setTagFpResult(next)
    } catch (err) {
      setTagFpError(`Tag FP eval failed: ${err.message}`)
    } finally {
      setTagFpLoading(false)
    }
  }

  const multi = realismResult?.multi_model
  const overall = multi?.overall_conclusion
  const metrics = realismResult?.metrics || {}
  const conclusion = realismResult?.conclusion || {}
  const styleOverall = styleResult?.overall_conclusion || {}
  const styleReports = styleResult?.reports || []
  const tagRecallReports = tagRecallResult?.reports || []
  const tagRecallComparison = tagRecallResult?.comparison || {}
  const tagFpReports = tagFpResult?.reports || []
  const tagFpAlt = tagFpResult?.at_wd_general_threshold

  return (
    <div className="container">
      <header className="app-header">
        <div className="app-nav">
          <a href="#/" className="nav-link">
            THR3SHR
          </a>
          <a href="#/debug" className="nav-link active" aria-current="page">
            Debug
          </a>
        </div>
        <h1>THR3SHR debug</h1>
        <p className="lede">
          Curated tag recall / false-positive benchmarks, SFW pull-tag recall, and real-life vs
          anime separation. Uses saved settings from the main page — no migrate.
        </p>
      </header>

      {(error || realismError || styleError || tagRecallError || tagFpError) && (
        <div className="error">
          {error || realismError || styleError || tagRecallError || tagFpError}
        </div>
      )}

      <section className="panel" aria-label="Tag recall benchmark">
        <h2>Tag recall benchmark</h2>
        <span className="kicker">
          Fixed curated suite: <code>ml_danbooru</code> vs <code>wd_swinv2_v3</code>. Metrics are
          recall@threshold and recall@top‑K on each sample’s desired tags (dense WD scores).
          Pin/download images once via{" "}
          <code>scripts/fetch_tag_recall_suite.py --rebuild</code>.
        </span>
        <div className="grid">
          <label>
            Threshold
            <input
              type="number"
              min="0.05"
              max="1"
              step="0.05"
              value={tagRecallThreshold}
              disabled={tagRecallLoading}
              onChange={(e) =>
                setTagRecallThreshold(
                  Math.max(0.05, Math.min(1, Number(e.target.value) || 0.35))
                )
              }
              aria-label="Tag recall threshold"
            />
          </label>
          <label>
            Top‑K
            <input
              type="number"
              min="1"
              max="200"
              value={tagRecallTopK}
              disabled={tagRecallLoading}
              onChange={(e) =>
                setTagRecallTopK(Math.max(1, Math.min(200, Number(e.target.value) || 20)))
              }
              aria-label="Tag recall top K"
            />
          </label>
        </div>
        <div className="actions">
          <button
            type="button"
            disabled={tagRecallLoading}
            onClick={handleTagRecallEval}
            aria-label="Run tag recall benchmark"
          >
            {tagRecallLoading
              ? "Scoring curated suite…"
              : "Run tag recall benchmark"}
          </button>
        </div>
      </section>

      {tagRecallResult && (
        <section className="panel debug-eval-results" aria-label="Tag recall results">
          <h2>Tag recall results</h2>
          <div className="stats">
            <span>
              Best: <strong>{tagRecallComparison.best_model || "—"}</strong>
            </span>
            <span>Samples: {tagRecallResult.count_samples_cached ?? 0}</span>
            <span>
              thr={tagRecallResult.threshold} · topK={tagRecallResult.top_k}
            </span>
          </div>
          {(tagRecallComparison.pairwise || []).map((pair) => (
            <p className="muted" key={`${pair.model_a}-${pair.model_b}`}>
              Pairwise @threshold: {pair.model_a} wins {pair.a_wins}, {pair.model_b} wins{" "}
              {pair.b_wins}, ties {pair.ties} (n={pair.compared})
            </p>
          ))}
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th scope="col">Model</th>
                  <th scope="col">Micro @thr</th>
                  <th scope="col">Micro @topK</th>
                  <th scope="col">Macro @thr</th>
                  <th scope="col">Macro @topK</th>
                  <th scope="col">Opp</th>
                  <th scope="col">N</th>
                </tr>
              </thead>
              <tbody>
                {tagRecallReports.map((row) => {
                  const s = row.summary || {}
                  return (
                    <tr key={row.tagger_model}>
                      <td>{row.tagger_model}</td>
                      <td>{pct(s.micro_recall_at_threshold)}</td>
                      <td>{pct(s.micro_recall_at_top_k)}</td>
                      <td>{pct(s.macro_recall_at_threshold)}</td>
                      <td>{pct(s.macro_recall_at_top_k)}</td>
                      <td>{s.opportunities ?? 0}</td>
                      <td>{row.count_evaluated}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          {tagRecallReports[0] && (
            <>
              <h3>Per-tag (side by side)</h3>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th scope="col">Tag</th>
                      {tagRecallReports.map((r) => (
                        <th key={`${r.tagger_model}-thr`} scope="col">
                          {r.tagger_model} @thr
                        </th>
                      ))}
                      {tagRecallReports.map((r) => (
                        <th key={`${r.tagger_model}-top`} scope="col">
                          {r.tagger_model} @topK
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {Array.from(
                      new Set(
                        tagRecallReports.flatMap((r) =>
                          Object.keys(r.summary?.per_tag || {})
                        )
                      )
                    )
                      .sort()
                      .map((tag) => (
                        <tr key={tag}>
                          <td>{tag}</td>
                          {tagRecallReports.map((r) => (
                            <td key={`${r.tagger_model}-thr-${tag}`}>
                              {pct(r.summary?.per_tag?.[tag]?.recall_at_threshold)}
                            </td>
                          ))}
                          {tagRecallReports.map((r) => (
                            <td key={`${r.tagger_model}-top-${tag}`}>
                              {pct(r.summary?.per_tag?.[tag]?.recall_at_top_k)}
                            </td>
                          ))}
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
          {tagRecallReports[0]?.items?.length > 0 && (
            <>
              <h3>Samples</h3>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th scope="col">Preview</th>
                      <th scope="col">Bucket</th>
                      <th scope="col">Post</th>
                      {tagRecallReports.map((r) => (
                        <th key={r.tagger_model} scope="col">
                          {r.tagger_model}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(tagRecallReports[0].items || []).map((item) => {
                      const preview = api.getTagRecallPreviewUrl(
                        item.source,
                        item.file_name
                      )
                      return (
                        <tr key={`${item.source}-${item.post_id}`}>
                          <td>
                            {preview ? (
                              <img
                                src={preview}
                                alt=""
                                className="debug-thumb"
                                loading="lazy"
                              />
                            ) : (
                              "—"
                            )}
                          </td>
                          <td>{item.bucket_id}</td>
                          <td>
                            {item.source}/{item.post_id}
                          </td>
                          {tagRecallReports.map((r) => {
                            const match = (r.items || []).find(
                              (x) =>
                                x.source === item.source && x.post_id === item.post_id
                            )
                            const parts = (match?.tag_results || []).map((tr) => {
                              const marks = [
                                tr.hit_at_threshold ? "T" : "·",
                                tr.hit_at_top_k ? "K" : "·",
                              ].join("")
                              const score =
                                tr.score == null ? "—" : Number(tr.score).toFixed(2)
                              return `${tr.tag}:${score}[${marks}]`
                            })
                            return (
                              <td key={`${r.tagger_model}-${item.post_id}`}>
                                <code className="muted">{parts.join(" · ") || "—"}</code>
                              </td>
                            )
                          })}
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </section>
      )}

      <section className="panel" aria-label="Tag false-positive benchmark">
        <h2>Tag false-positive benchmark</h2>
        <span className="kicker">
          Same curated suite. Probes taxonomy evidence tags (weight ≥ min) for your{" "}
          <strong>selected folders</strong> from Settings. FP = score ≥ threshold when the
          Danbooru post does not list that tag. Also reports folder-routing FPs.
        </span>
        <p className="muted">
          Selected:{" "}
          {(settingsSummary.selected_tags || []).length
            ? settingsSummary.selected_tags.join(", ")
            : "(none — pick folders on THR3SHR settings)"}
        </p>
        <div className="grid">
          <label>
            Tag / route threshold
            <input
              type="number"
              min="0.05"
              max="1"
              step="0.05"
              value={tagFpThreshold}
              disabled={tagFpLoading}
              onChange={(e) =>
                setTagFpThreshold(
                  Math.max(0.05, Math.min(1, Number(e.target.value) || 0.6))
                )
              }
              aria-label="Tag false positive threshold"
            />
          </label>
          <label>
            Min evidence weight
            <input
              type="number"
              min="0"
              max="1"
              step="0.05"
              value={tagFpMinWeight}
              disabled={tagFpLoading}
              onChange={(e) =>
                setTagFpMinWeight(
                  Math.max(0, Math.min(1, Number(e.target.value) || 0.85))
                )
              }
              aria-label="Minimum taxonomy evidence weight"
            />
          </label>
        </div>
        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={tagFpAlsoWd}
            disabled={tagFpLoading}
            onChange={(e) => setTagFpAlsoWd(e.target.checked)}
          />
          Also report at WD general threshold (0.35)
        </label>
        <div className="actions">
          <button
            type="button"
            disabled={tagFpLoading || !(settingsSummary.selected_tags || []).length}
            onClick={handleTagFpEval}
            aria-label="Run tag false positive benchmark"
          >
            {tagFpLoading
              ? "Scoring curated suite for FPs…"
              : "Run false-positive benchmark"}
          </button>
        </div>
      </section>

      {tagFpResult && (
        <section className="panel debug-eval-results" aria-label="Tag false positive results">
          <h2>False-positive results</h2>
          <div className="stats">
            <span>Samples: {tagFpResult.count_samples ?? 0}</span>
            <span>Probe tags: {tagFpResult.probe_tag_count ?? 0}</span>
            <span>
              thr={tagFpResult.tag_threshold} · minW={tagFpResult.min_weight}
            </span>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th scope="col">Model</th>
                  <th scope="col">Micro FPR</th>
                  <th scope="col">Macro FPR</th>
                  <th scope="col">FP / neg</th>
                  <th scope="col">Route FPR</th>
                  <th scope="col">Route FP</th>
                  <th scope="col">N</th>
                </tr>
              </thead>
              <tbody>
                {tagFpReports.map((row) => {
                  const tf = row.tag_fp || {}
                  const rf = row.folder_route_fp || {}
                  return (
                    <tr key={row.tagger_model}>
                      <td>{row.tagger_model}</td>
                      <td>{pct(tf.micro_fpr)}</td>
                      <td>{pct(tf.macro_fpr)}</td>
                      <td>
                        {tf.false_positives ?? 0}/{tf.negatives ?? 0}
                      </td>
                      <td>{pct(rf.fpr)}</td>
                      <td>
                        {rf.false_positives ?? 0}/{rf.evaluated ?? 0}
                      </td>
                      <td>{row.count_evaluated}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          {tagFpAlt?.reports?.length > 0 && (
            <>
              <h3>Also at WD general thr={tagFpAlt.tag_threshold}</h3>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th scope="col">Model</th>
                      <th scope="col">Micro FPR</th>
                      <th scope="col">Route FPR</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tagFpAlt.reports.map((row) => (
                      <tr key={`alt-${row.tagger_model}`}>
                        <td>{row.tagger_model}</td>
                        <td>{pct(row.tag_fp?.micro_fpr)}</td>
                        <td>{pct(row.folder_route_fp?.fpr)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}

          <h3>Top false-positive tags (side by side)</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th scope="col">Tag</th>
                  {tagFpReports.map((r) => (
                    <th key={`${r.tagger_model}-fpr`} scope="col">
                      {r.tagger_model} FPR
                    </th>
                  ))}
                  {tagFpReports.map((r) => (
                    <th key={`${r.tagger_model}-cnt`} scope="col">
                      {r.tagger_model} fp/neg
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Array.from(
                  new Set(
                    tagFpReports.flatMap((r) =>
                      (r.top_false_positive_tags || []).map((t) => t.tag)
                    )
                  )
                )
                  .slice(0, 25)
                  .map((tag) => {
                    const rowsByModel = Object.fromEntries(
                      tagFpReports.map((r) => [
                        r.tagger_model,
                        (r.top_false_positive_tags || []).find((t) => t.tag === tag),
                      ])
                    )
                    return (
                      <tr key={tag}>
                        <td>{tag}</td>
                        {tagFpReports.map((r) => (
                          <td key={`${r.tagger_model}-fpr-${tag}`}>
                            {pct(rowsByModel[r.tagger_model]?.fpr)}
                          </td>
                        ))}
                        {tagFpReports.map((r) => {
                          const row = rowsByModel[r.tagger_model]
                          return (
                            <td key={`${r.tagger_model}-cnt-${tag}`}>
                              {row
                                ? `${row.false_positives}/${row.negatives}`
                                : "0"}
                            </td>
                          )
                        })}
                      </tr>
                    )
                  })}
              </tbody>
            </table>
          </div>

          {tagFpReports.some((r) => (r.sample_route_fps || []).length > 0) && (
            <>
              <h3>Folder route false positives</h3>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th scope="col">Preview</th>
                      <th scope="col">Bucket</th>
                      <th scope="col">Truth roots</th>
                      {tagFpReports.map((r) => (
                        <th key={r.tagger_model} scope="col">
                          {r.tagger_model}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {Array.from(
                      new Map(
                        tagFpReports
                          .flatMap((r) => r.sample_route_fps || [])
                          .map((i) => [`${i.source}:${i.post_id}`, i])
                      ).values()
                    ).map((item) => {
                      const preview = api.getTagRecallPreviewUrl(
                        item.source,
                        item.file_name
                      )
                      return (
                        <tr key={`${item.source}-${item.post_id}`}>
                          <td>
                            {preview ? (
                              <img
                                src={preview}
                                alt=""
                                className="debug-thumb"
                                loading="lazy"
                              />
                            ) : (
                              "—"
                            )}
                          </td>
                          <td>{item.bucket_id}</td>
                          <td>{(item.truth || []).join(", ") || "—"}</td>
                          {tagFpReports.map((r) => {
                            const match = (r.sample_route_fps || []).find(
                              (x) =>
                                x.source === item.source && x.post_id === item.post_id
                            )
                            if (!match) {
                              return (
                                <td key={`${r.tagger_model}-${item.post_id}`}>—</td>
                              )
                            }
                            const score =
                              match.score == null
                                ? ""
                                : ` (${Number(match.score).toFixed(2)})`
                            return (
                              <td key={`${r.tagger_model}-${item.post_id}`}>
                                {match.predicted || "—"}
                                {score}
                              </td>
                            )
                          })}
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </section>
      )}

      <section className="panel" aria-label="Style detector compare">
        <h2>Style detectors (real vs anime)</h2>
        <span className="kicker">
          Debug-only compare: dedicated ONNX classifiers (
          <code>deepghs/anime_real_cls</code> via imgutils) vs WD{" "}
          <code>real_life</code> taxonomy. Same photo/anime corpus as below. Enable the
          experimental style gate on the main THR3SHR settings (next to GIF/video) to use
          CAFormer in production runs.
        </span>
        <div className="grid">
          <label>
            Samples per class
            <input
              type="number"
              min="5"
              max="40"
              value={styleCount}
              disabled={styleLoading}
              onChange={(e) =>
                setStyleCount(Math.max(5, Math.min(40, Number(e.target.value) || 12)))
              }
              aria-label="Style detector samples per class"
            />
          </label>
        </div>
        <p className="muted">
          WD baseline uses settings model: <strong>{settingsSummary.tagger_model}</strong>
          {" · "}
          Experimental style gate:{" "}
          <strong>
            {settingsSummary.experimental_style_detector_enabled ? "ON" : "OFF"}
          </strong>{" "}
          (toggle on THR3SHR settings)
        </p>
        <div className="actions">
          <button
            type="button"
            disabled={styleLoading}
            onClick={handleStyleEval}
            aria-label="Run style detector comparison"
          >
            {styleLoading
              ? "Downloading samples & scoring detectors…"
              : "Compare style detectors"}
          </button>
        </div>
      </section>

      {styleResult && (
        <section className="panel debug-eval-results" aria-label="Style detector results">
          <h2>Style detector results</h2>
          <div className="stats">
            <span>
              Decision: <strong>{styleOverall.decision || "—"}</strong>
            </span>
            <span>Best: {styleOverall.best_detector || "—"}</span>
            <span>
              Paths: {styleResult.count_paths} (photos {styleResult.count_photos_fetched},
              anime {styleResult.count_anime_fetched})
            </span>
          </div>
          <p className="muted">{styleOverall.summary}</p>
          <p className="muted">{styleOverall.note}</p>
          <h3>Per detector</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th scope="col">Detector</th>
                  <th scope="col">Decision</th>
                  <th scope="col">Typical P</th>
                  <th scope="col">Typical R</th>
                  <th scope="col">Typical F1</th>
                  <th scope="col">Anime FP</th>
                  <th scope="col">Uncertain</th>
                  <th scope="col">N</th>
                </tr>
              </thead>
              <tbody>
                {styleReports.map((row) => {
                  const t = row.conclusion?.typical_metrics || {}
                  return (
                    <tr key={row.detector_id}>
                      <td>{row.detector_id}</td>
                      <td>{row.conclusion?.decision}</td>
                      <td>{pct(t.precision)}</td>
                      <td>{pct(t.recall)}</td>
                      <td>{pct(t.f1)}</td>
                      <td>{pct(t.anime_false_positive_rate)}</td>
                      <td>{row.conclusion?.uncertain_count ?? 0}</td>
                      <td>{row.count_evaluated}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          {styleReports[0]?.items?.length > 0 && (
            <>
              <h3>Samples ({styleResult.best_detector || styleReports[0].detector_id})</h3>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th scope="col">Preview</th>
                      <th scope="col">Truth</th>
                      {styleReports.map((r) => (
                        <th key={r.detector_id} scope="col">
                          {r.detector_id}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(styleReports[0].items || []).map((item, idx) => {
                      const preview = api.getRealismDebugPreviewUrl(
                        item.source,
                        item.file_name
                      )
                      return (
                        <tr key={item.sample_id}>
                          <td>
                            {preview ? (
                              <img
                                src={preview}
                                alt={item.title || item.sample_id}
                                className="debug-thumb"
                              />
                            ) : (
                              "—"
                            )}
                          </td>
                          <td>
                            {item.label}
                            <div className="muted">{item.bucket}</div>
                          </td>
                          {styleReports.map((r) => {
                            const cell = (r.items || [])[idx]
                            if (!cell) return <td key={r.detector_id}>—</td>
                            return (
                              <td key={r.detector_id}>
                                {cell.predicted_bucket}
                                <div className="muted">
                                  {pct(cell.confidence)} {cell.correct ? "ok" : "miss"}
                                </div>
                              </td>
                            )
                          })}
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </section>
      )}

      <section className="panel" aria-label="Real-life vs anime eval">
        <h2>Real-life vs anime</h2>
        <span className="kicker">
          Remote people portraits (RandomUser CDN) vs Safebooru anime, including realistic /
          photorealistic / 3d edge cases. Scores through production <code>real_life</code> taxonomy
          routing.
        </span>
        <div className="grid">
          <label>
            Samples per class
            <input
              type="number"
              min="5"
              max="40"
              value={realismCount}
              disabled={realismLoading}
              onChange={(e) =>
                setRealismCount(Math.max(5, Math.min(40, Number(e.target.value) || 12)))
              }
              aria-label="Realism samples per class"
            />
          </label>
          <label className="inline-check">
            <input
              type="checkbox"
              checked={realismCompare}
              disabled={realismLoading}
              onChange={(e) => setRealismCompare(e.target.checked)}
            />
            Compare all models (EVA02 / SwinV2 / ML-Danbooru)
          </label>
        </div>
        <p className="muted">
          Current settings model: <strong>{settingsSummary.tagger_model}</strong>
          {!realismCompare ? " (used when compare is off)" : " (ignored while comparing)"}
        </p>
        <div className="actions">
          <button
            type="button"
            disabled={realismLoading}
            onClick={handleRealismEval}
            aria-label="Run real-life versus anime evaluation"
          >
            {realismLoading
              ? "Fetching photos/anime & scoring… (can take a while)"
              : "Run real-life vs anime eval"}
          </button>
        </div>
      </section>

      {realismResult && (
        <section className="panel debug-eval-results" aria-label="Realism eval results">
          <h2>Realism results</h2>
          <div className="stats">
            <span>
              Decision:{" "}
              <strong>
                {(overall && overall.decision) || conclusion.decision || "—"}
              </strong>
            </span>
            <span>
              Evaluated: {realismResult.count_evaluated} (photos fetched{" "}
              {realismResult.count_photos_fetched}, anime {realismResult.count_anime_fetched})
            </span>
            <span>Model shown: {realismResult.tagger_model}</span>
            {overall?.best_model ? <span>Best model: {overall.best_model}</span> : null}
          </div>
          <p className="muted">
            {(overall && overall.summary) || conclusion.summary}
          </p>
          <p className="muted">
            {(overall && overall.video_and_gif) || conclusion.video_note}
          </p>

          <h3>Typical photo vs anime (GO gate)</h3>
          <div className="stats">
            {[
              ["Precision", (conclusion.typical_metrics || metrics).precision],
              ["Recall", (conclusion.typical_metrics || metrics).recall],
              ["F1", (conclusion.typical_metrics || metrics).f1],
              [
                "Anime FP",
                (conclusion.typical_metrics || metrics).anime_false_positive_rate,
              ],
            ].map(([label, value]) => (
              <div key={label} className="metric">
                <span className="label">{label}</span>
                <span className="value">{pct(value)}</span>
              </div>
            ))}
            <div className="metric">
              <span className="label">Edge quarantine</span>
              <span className="value">
                {conclusion.edge_quarantine_count ?? "—"}/{conclusion.edge_sample_count ?? "—"}
              </span>
            </div>
          </div>
          <h3>Including photoreal/3d edges</h3>
          <div className="stats">
            <div className="metric">
              <span className="label">Precision</span>
              <span className="value">{pct(metrics.precision)}</span>
            </div>
            <div className="metric">
              <span className="label">Recall</span>
              <span className="value">{pct(metrics.recall)}</span>
            </div>
            <div className="metric">
              <span className="label">Anime FP rate</span>
              <span className="value">{pct(metrics.anime_false_positive_rate)}</span>
            </div>
          </div>

          {multi?.reports?.length ? (
            <>
              <h3>Per-model</h3>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>Decision</th>
                      <th>P</th>
                      <th>R</th>
                      <th>F1</th>
                      <th>Anime FP</th>
                      <th>N</th>
                    </tr>
                  </thead>
                  <tbody>
                    {multi.reports.map((row) => (
                      <tr key={row.tagger_model}>
                        <td>{row.tagger_model}</td>
                        <td>{row.conclusion?.decision}</td>
                        <td>{pct(row.metrics?.precision)}</td>
                        <td>{pct(row.metrics?.recall)}</td>
                        <td>{pct(row.metrics?.f1)}</td>
                        <td>{pct(row.metrics?.anime_false_positive_rate)}</td>
                        <td>{row.count_evaluated}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : null}

          {realismResult.by_label && Object.keys(realismResult.by_label).length > 0 ? (
            <>
              <h3>Edge / label breakdown</h3>
              <div className="stats">
                {Object.entries(realismResult.by_label).map(([label, row]) => (
                  <div key={label} className="metric">
                    <span className="label">{label}</span>
                    <span className="value">
                      acc {pct(row.accuracy)} (n={row.count})
                    </span>
                  </div>
                ))}
              </div>
            </>
          ) : null}

          {realismResult.errors?.length > 0 && (
            <div className="error">{realismResult.errors.join(" · ")}</div>
          )}

          <h3>Samples</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Preview</th>
                  <th>Truth</th>
                  <th>Predicted</th>
                  <th>Evidence</th>
                  <th>Folder</th>
                </tr>
              </thead>
              <tbody>
                {(realismResult.items || []).map((item) => {
                  const previewUrl = api.getRealismDebugPreviewUrl(item.source, item.file_name)
                  const ev = item.evidence_scores || {}
                  const evText = ["realistic", "photorealistic", "photo_(medium)", "3d"]
                    .map((tag) => `${tag}:${Number(ev[tag] || 0).toFixed(2)}`)
                    .join(" ")
                  return (
                    <tr
                      key={item.sample_id}
                      className={item.correct ? "" : "needs-review"}
                    >
                      <td>
                        {previewUrl ? (
                          <img
                            src={previewUrl}
                            alt={item.title || item.sample_id}
                            className="debug-thumb"
                          />
                        ) : (
                          <span className="muted">n/a</span>
                        )}
                      </td>
                      <td>
                        {item.bucket}
                        <div className="muted">
                          {item.label} · {item.query}
                        </div>
                      </td>
                      <td>
                        {item.predicted_bucket}{" "}
                        {item.correct ? (
                          <span className="muted">ok</span>
                        ) : (
                          <strong>miss</strong>
                        )}
                      </td>
                      <td>{evText}</td>
                      <td>
                        {item.primary_folder
                          ? `${item.primary_folder} (${Number(item.primary_score || 0).toFixed(3)})`
                          : "—"}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <section className="panel" aria-label="THR3SHR debug controls">
        <h2>SFW tag recall</h2>
        <span className="kicker">
          Safebooru / Danbooru only. Change tagger and destination tags on the main THR3SHR page,
          then return here.
        </span>
        <div className="grid">
          <label>
            Source
            <select
              value={source}
              onChange={(e) => {
                const next = e.target.value
                setSource(next)
                const meta = sources.find((s) => s.id === next)
                if (meta?.max_content_tags != null) {
                  setPullTags((prev) => prev.slice(0, meta.max_content_tags))
                }
              }}
              disabled={loading}
              aria-label="SFW image source"
            >
              {sources.map((src) => (
                <option key={src.id} value={src.id}>
                  {src.label} ({src.sfw_policy})
                </option>
              ))}
            </select>
          </label>
          <label>
            Sample count
            <input
              type="number"
              min="5"
              max="30"
              value={count}
              disabled={loading}
              onChange={(e) =>
                setCount(Math.max(5, Math.min(30, Number(e.target.value) || 10)))
              }
              aria-label="SFW sample count"
            />
          </label>
        </div>
        <p className="muted">
          Model: <strong>{settingsSummary.tagger_model}</strong> · Threshold:{" "}
          <strong>{Number(settingsSummary.confidence_threshold).toFixed(2)}</strong> · Destination
          tags:{" "}
          {settingsSummary.selected_tags.length
            ? settingsSummary.selected_tags.join(", ")
            : "(none — classify preview will need review)"}
          {sourceMeta?.max_content_tags != null
            ? ` · ${sourceMeta.label} max content tags: ${sourceMeta.max_content_tags}`
            : ""}
        </p>
        <label>
          Tags to pull
          <input
            value={tagQuery}
            onChange={(e) => setTagQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault()
                const value = tagQuery.trim()
                if (value) {
                  handleAddPullTag(value)
                  setTagQuery("")
                }
              }
            }}
            placeholder="Search or type a tag, Enter to add"
            disabled={loading}
            aria-label="Search tags to pull"
          />
        </label>
        <div className="tag-suggestions" role="listbox" aria-label="Tag suggestions for debug pull">
          {tagOptions.map((tag) => (
            <button
              key={tag}
              type="button"
              className="chip"
              disabled={loading || pullTags.includes(tag)}
              onClick={() => handleAddPullTag(tag)}
            >
              {tag}
            </button>
          ))}
        </div>
        <div className="stats">
          {pullTags.length === 0 ? (
            <span className="muted">No pull tags yet</span>
          ) : (
            pullTags.map((tag) => (
              <span key={tag} className="chip">
                {tag}
                <button
                  type="button"
                  onClick={() => setPullTags((prev) => prev.filter((t) => t !== tag))}
                  aria-label={`Remove pull tag ${tag}`}
                  disabled={loading}
                >
                  ×
                </button>
              </span>
            ))
          )}
        </div>
        <div className="actions">
          <button
            type="button"
            disabled={loading || pullTags.length === 0}
            onClick={handleFetchEvaluate}
            aria-label="Fetch and evaluate SFW samples"
          >
            {loading ? "Fetching & evaluating…" : "Fetch & evaluate"}
          </button>
        </div>
      </section>

      {result && (
        <section className="panel debug-eval-results" aria-label="THR3SHR debug results">
          <h2>SFW results</h2>
          <div className="stats">
            <span>
              Query: <code>{result.query}</code>
            </span>
            <span>
              Evaluated: {result.count_evaluated}/{result.count_requested}
            </span>
          </div>
          {result.errors?.length > 0 && (
            <div className="error">{result.errors.join(" · ")}</div>
          )}
          <h3>Recall @ {Number(result.confidence_threshold).toFixed(2)}</h3>
          <div className="stats">
            {(result.recall || []).map((row) => (
              <div key={row.tag} className="metric">
                <span className="label">{row.tag}</span>
                <span className="value">
                  {row.hit_rate == null
                    ? "n/a"
                    : `${(row.hit_rate * 100).toFixed(0)}% (${row.hits_at_threshold}/${row.present_in_posts})`}
                </span>
              </div>
            ))}
          </div>
          <h3>Classify preview</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Preview</th>
                  <th>Post</th>
                  <th>Pull scores</th>
                  <th>Primary</th>
                  <th>Review</th>
                </tr>
              </thead>
              <tbody>
                {(result.items || []).map((item) => {
                  const previewUrl = api.getSfwDebugPreviewUrl(item.source, item.file_name)
                  const pullText = Object.entries(item.pull_tag_scores || {})
                    .map(([tag, score]) =>
                      `${tag}:${score == null ? "—" : Number(score).toFixed(2)}`
                    )
                    .join(" ")
                  return (
                    <tr
                      key={`${item.source}-${item.post_id}`}
                      className={item.needs_review ? "needs-review" : ""}
                    >
                      <td>
                        {previewUrl ? (
                          <img
                            src={previewUrl}
                            alt={`Post ${item.post_id}`}
                            className="debug-thumb"
                          />
                        ) : (
                          <span className="muted">n/a</span>
                        )}
                      </td>
                      <td>
                        {item.post_id}
                        <div className="muted">{(item.known_tags || []).join(", ")}</div>
                      </td>
                      <td>{pullText}</td>
                      <td>
                        {item.primary_tag
                          ? `${item.primary_tag} (${Number(item.primary_score).toFixed(3)})`
                          : "—"}
                      </td>
                      <td>{item.needs_review ? item.review_reason || "needs review" : "ok"}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  )
}
