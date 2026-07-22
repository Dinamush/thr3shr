import React, { useEffect, useMemo, useState } from "react"
import { api } from "./api"

const DEFAULT_SOURCES = [
  { id: "safebooru", label: "Safebooru", sfw_policy: "rating:safe", max_content_tags: null },
  { id: "danbooru", label: "Danbooru", sfw_policy: "rating:g", max_content_tags: 2 },
]

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
  })

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
        setSettingsSummary({
          tagger_model: data.tagger_model || "wd_swinv2_v3",
          confidence_threshold: Number(data.confidence_threshold ?? 0.6),
          selected_tags: Array.isArray(data.selected_tags) ? data.selected_tags : [],
        })
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
      })
    } catch (err) {
      setError(`SFW debug eval failed: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <header className="app-header">
        <div className="app-nav">
          <a href="#/" className="nav-link">
            Classifier
          </a>
          <a href="#/debug" className="nav-link active" aria-current="page">
            Classifier debug
          </a>
        </div>
        <h1>Classifier debug</h1>
        <p className="lede">
          Download SFW samples for chosen tags, measure recall @ threshold, and preview destination
          assignment. Uses saved settings from the main page — no migrate.
        </p>
      </header>

      {error && <div className="error">{error}</div>}

      <section className="panel" aria-label="Classifier debug controls">
        <h2>Fetch samples</h2>
        <span className="kicker">
          Safebooru / Danbooru only. Change tagger and destination tags on the main Classifier page,
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
        <section className="panel debug-eval-results" aria-label="Classifier debug results">
          <h2>Results</h2>
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
