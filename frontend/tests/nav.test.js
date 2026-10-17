import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import {
  DEBUG_SECTIONS,
  PAGES,
  SETTINGS_INDEX,
  SETTINGS_SECTIONS,
  rankJumpEntries,
  routeFromHash,
  searchSettings,
  staticJumpEntries,
} from "../src/shell/nav.js";

const source = (path) => readFileSync(new URL(`../src/${path}`, import.meta.url), "utf8");

test("hash routes map to pages, as before for home and debug", () => {
  assert.equal(routeFromHash(""), "run");
  assert.equal(routeFromHash("#/"), "run");
  assert.equal(routeFromHash("#/nowhere"), "run");
  assert.equal(routeFromHash("#/debug"), "debug");
  assert.equal(routeFromHash("#/debug/anything"), "debug");
  assert.equal(routeFromHash("#/review"), "review");
  assert.equal(routeFromHash("#/runs"), "runs");
  assert.equal(routeFromHash("#/settings"), "settings");
});

test("every page has a unique id and path", () => {
  assert.equal(new Set(PAGES.map((p) => p.id)).size, PAGES.length);
  assert.equal(new Set(PAGES.map((p) => p.path)).size, PAGES.length);
});

test("every indexed setting and section exists on the Settings page", () => {
  const page = source("pages/SettingsPage.jsx");
  for (const unit of SETTINGS_INDEX) {
    assert.ok(page.includes(`<Unit id="${unit.id}"`), `Settings page renders ${unit.id}`);
    assert.ok(SETTINGS_SECTIONS.some((s) => s.id === unit.section), `${unit.id} section`);
  }
  for (const section of SETTINGS_SECTIONS) {
    assert.ok(page.includes(`<Section id="${section.id}"`), `Settings page renders ${section.id}`);
  }
  // A unit missing from the index would never match a search, so it would hide.
  const rendered = [...page.matchAll(/<Unit id="([^"]+)"/g)].map((m) => m[1]);
  assert.deepEqual(rendered.sort(), SETTINGS_INDEX.map((u) => u.id).sort());
});

test("every debug benchmark anchor exists on the debug page", () => {
  const page = source("ClassifierDebugPage.jsx");
  for (const section of DEBUG_SECTIONS) {
    assert.ok(page.includes(`id="${section.id}"`), section.id);
  }
});

test("settings search matches labels and help terms, title hits show the section", () => {
  const all = searchSettings("");
  assert.equal(all.total, SETTINGS_INDEX.length);

  const cpu = searchSettings("cpu");
  assert.ok(cpu.hits.has("set-force_cpu_inference"));
  assert.equal(cpu.counts["settings-paths"], 0);

  const section = searchSettings("thresholds");
  assert.ok(section.hits.has("set-confidence_threshold"));
  assert.ok(section.hits.has("set-wd_general_threshold"));

  const none = searchSettings("zzzz-no-match");
  assert.equal(none.total, 0);

  const notWd = searchSettings("WD14", (unit) => unit.when !== "wd");
  assert.ok(!notWd.hits.has("set-wd_general_threshold"), "hidden units are not counted");
});

test("jump bar ranks label starts first and hides quiet entries until searched", () => {
  const entries = staticJumpEntries();
  const keys = entries.map((e) => e.key);
  assert.equal(new Set(keys).size, keys.length, "unique keys");

  const idle = rankJumpEntries(entries, "");
  assert.ok(idle.every((e) => !e.quiet));
  assert.ok(idle.some((e) => e.route === "review"));

  const batch = rankJumpEntries(entries, "batch");
  assert.equal(batch[0].target, "set-inference_batch_size");

  const multi = rankJumpEntries(entries, "wd thr");
  assert.equal(multi[0].target, "set-wd_general_threshold");

  assert.deepEqual(rankJumpEntries(entries, "zzzz-no-match"), []);

  // Theme searches open the picker rather than a page.
  assert.ok(rankJumpEntries(entries, "theme").slice(0, 2).every((e) => e.appearance));
  assert.ok(rankJumpEntries(entries, "palette")[0].appearance);
});
