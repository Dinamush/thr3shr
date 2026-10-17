import assert from "node:assert/strict";
import { test } from "node:test";
import { paletteCss } from "../src/theme/paletteCss.js";
import {
  AUTO_DARK,
  AUTO_LIGHT,
  AUTO_THEME,
  DEFAULT_PALETTE,
  THEMES,
  THEME_GROUPS,
  TOKEN_NAMES,
  isKnownTheme,
  resolveTheme,
  themeMenu,
} from "../src/theme/themes.js";

function luminance(hex) {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
  const lin = (c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
}

function contrast(a, b) {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

test("every palette sets every token as a hex colour", () => {
  for (const theme of THEMES) {
    for (const name of TOKEN_NAMES) {
      assert.match(theme.tokens[name] || "", /^#[0-9a-f]{6}$/i, `${theme.slug} --${name}`);
    }
    assert.deepEqual(Object.keys(theme.tokens).sort(), [...TOKEN_NAMES].sort(), theme.slug);
  }
});

test("palette slugs are unique, CSS-safe and grouped", () => {
  const slugs = THEMES.map((theme) => theme.slug);
  assert.equal(new Set(slugs).size, slugs.length);
  assert.ok(!slugs.includes(AUTO_THEME));
  const groups = new Set(THEME_GROUPS.map((group) => group.id));
  for (const theme of THEMES) {
    assert.match(theme.slug, /^[a-z-]+$/, "index.html only accepts [a-z-] names");
    assert.ok(groups.has(theme.group), `${theme.slug} group`);
    assert.ok(["light", "dark"].includes(theme.scheme), `${theme.slug} scheme`);
  }
});

test("the SHUCK3R palette set is all here", () => {
  const expected = [
    "graphite", "daylight", "nier", "blush", "petal", "mucosa", "coral", "plum",
    "rosewood", "terracotta", "vermilion", "mahogany", "ebony", "bordeaux", "onyx",
  ];
  assert.deepEqual(THEMES.map((theme) => theme.slug), expected);
});

test("text and state colours clear 4.5:1 on their surfaces", () => {
  for (const { slug, tokens: t } of THEMES) {
    for (const fg of ["tx", "tx2", "tx3", "ac-tx", "ok", "warn", "bad"]) {
      for (const bg of ["sf", "bg"]) {
        const ratio = contrast(t[fg], t[bg]);
        assert.ok(ratio >= 4.5, `${slug}: --${fg} on --${bg} is ${ratio.toFixed(2)}:1`);
      }
    }
    assert.ok(contrast(t["on-ac"], t.ac) >= 4.5, `${slug}: --on-ac on --ac`);
    assert.ok(contrast(t["on-bad"], t.bad) >= 4.5, `${slug}: --on-bad on --bad`);
  }
});

test("Auto resolves by the OS scheme and unknown names fall back to Auto", () => {
  assert.equal(resolveTheme(AUTO_THEME, true), AUTO_DARK);
  assert.equal(resolveTheme(AUTO_THEME, false), AUTO_LIGHT);
  assert.equal(resolveTheme("mucosa", true), "mucosa");
  assert.equal(resolveTheme("no-such-theme", false), AUTO_LIGHT);
  assert.ok(isKnownTheme("auto"));
  assert.ok(isKnownTheme("bordeaux"));
  assert.ok(!isKnownTheme("no-such-theme"));
  assert.ok(!isKnownTheme(null));
});

test("the Appearance menu lists Auto first and every palette once", () => {
  const rows = themeMenu().flatMap((group) => group.themes);
  assert.equal(rows[0].slug, AUTO_THEME);
  assert.deepEqual(
    rows.slice(1).map((row) => row.slug).sort(),
    THEMES.map((theme) => theme.slug).sort()
  );
  for (const row of rows) assert.equal(row.chips.length, 4, row.slug);
});

test("generated CSS has one block per palette, the default on :root", () => {
  const css = paletteCss();
  for (const theme of THEMES) {
    assert.ok(css.includes(`[data-theme="${theme.slug}"]`), theme.slug);
    assert.ok(css.includes(`--ac-tx: ${theme.tokens["ac-tx"]}`), `${theme.slug} accent text`);
  }
  assert.ok(css.startsWith(":root,") || css.includes(`:root,\n[data-theme="${DEFAULT_PALETTE}"]`));
  assert.ok(css.includes('[data-theme="nier"] {\n  --r-sm: 0'), "YoRHa stays square");
});

test("a new palette entry needs nothing but its data", () => {
  const extra = {
    ...THEMES[0],
    slug: "test-palette",
    group: "dark",
    tokens: { ...THEMES[0].tokens },
  };
  const css = paletteCss([...THEMES, extra]);
  assert.ok(css.includes('[data-theme="test-palette"] {'));
  assert.ok(css.includes('[data-theme="test-palette"]'), "dark shadow block");
});
