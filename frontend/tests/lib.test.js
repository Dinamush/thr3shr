import assert from "node:assert/strict";
import { test } from "node:test";
import {
  formatDurationSeconds,
  formatTagScore,
  isVideoPreviewPath,
  pathTail,
  runOptionLabel,
  runPct,
  taggerLabel,
} from "../src/lib.js";

test("durations read as h/m/s", () => {
  assert.equal(formatDurationSeconds(5), "5s");
  assert.equal(formatDurationSeconds(125), "2m 5s");
  assert.equal(formatDurationSeconds(3720), "1h 2m");
  assert.equal(formatDurationSeconds(-4), "0s");
});

test("paths keep their last two segments on either separator", () => {
  assert.equal(pathTail("/srv/dump/a/b.png"), "a/b.png");
  assert.equal(pathTail("C:\\dump\\x\\y.jpg"), "x/y.jpg");
  assert.equal(pathTail(""), "");
});

test("video previews are detected by extension, ignoring query strings", () => {
  assert.ok(isVideoPreviewPath("a/b.MP4"));
  assert.ok(isVideoPreviewPath("clip.webm?x=1"));
  assert.ok(!isVideoPreviewPath("a/b.png"));
  assert.ok(!isVideoPreviewPath(null));
});

test("run labels and progress", () => {
  const run = {
    run_id: 7,
    status: "completed",
    processed_images: 3,
    total_images: 4,
    root_repo: "/dump/inbox",
    created_at: "",
  };
  assert.equal(runOptionLabel(run), "Run #7 · completed · 3/4 images · dump/inbox");
  assert.equal(runPct(run), 75);
  assert.equal(runPct({ ...run, progress_pct: 12.5 }), 12.5);
  assert.equal(runPct({ ...run, total_images: 0 }), 0);
});

test("tag scores and tagger names", () => {
  assert.equal(formatTagScore({ tag: "solo", score: 0.91234 }), "solo (0.912)");
  assert.equal(taggerLabel("wd_swinv2_v3"), "WD SwinV2 v3");
  assert.equal(taggerLabel("custom"), "custom");
});
