// Writes the app icon files in public/ from src/shell/appIcon.js:
//   favicon.svg             16px tab icon, light and dark aware
//   favicon.ico             16, 32, 48
//   app.ico                 desktop shortcut icon, 16 to 256
//   apple-touch-icon.png    180, full bleed (iOS cuts its own corners)
//   icon-192.png, icon-512.png   web app manifest icons
// Each size uses the detail level drawn for it.  Re-run after changing the
// artwork:  npm run icons

import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { Resvg } from "@resvg/resvg-js";
import { appIconSvg, faviconSvg, levelFor } from "../src/shell/appIcon.js";

const publicDir = fileURLToPath(new URL("../public/", import.meta.url));

function png(px, options = {}) {
  const svg = appIconSvg(levelFor(px), options);
  return new Resvg(svg, { fitTo: { mode: "width", value: px } }).render().asPng();
}

// ICO with PNG-compressed entries (read by Windows Vista and later and every
// current browser).
function ico(sizes) {
  const images = sizes.map((px) => ({ px, data: png(px) }));
  const header = Buffer.alloc(6 + 16 * images.length);
  header.writeUInt16LE(0, 0);
  header.writeUInt16LE(1, 2);
  header.writeUInt16LE(images.length, 4);
  let offset = header.length;
  images.forEach(({ px, data }, i) => {
    const entry = 6 + 16 * i;
    header.writeUInt8(px >= 256 ? 0 : px, entry);
    header.writeUInt8(px >= 256 ? 0 : px, entry + 1);
    header.writeUInt8(0, entry + 2);
    header.writeUInt8(0, entry + 3);
    header.writeUInt16LE(1, entry + 4);
    header.writeUInt16LE(32, entry + 6);
    header.writeUInt32LE(data.length, entry + 8);
    header.writeUInt32LE(offset, entry + 12);
    offset += data.length;
  });
  return Buffer.concat([header, ...images.map((image) => image.data)]);
}

const outputs = {
  "favicon.svg": faviconSvg(),
  "favicon.ico": ico([16, 32, 48]),
  "app.ico": ico([16, 24, 32, 48, 64, 128, 256]),
  "apple-touch-icon.png": png(180, { bleed: true }),
  "icon-192.png": png(192),
  "icon-512.png": png(512),
};

for (const [name, data] of Object.entries(outputs)) {
  writeFileSync(publicDir + name, data);
  console.log(`public/${name}`);
}
