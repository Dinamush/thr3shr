import React, { useEffect, useRef, useState } from "react";
import { BOOT_SEEN_STORAGE_KEY } from "./themes";
import { motionReduced } from "./appearance";

const LINES = [
  ["Commencing System Check"],
  ["Memory Unit: Green", "green"],
  ["Initializing Tagger Pods"],
  ["Loading tags.csv Vocabulary"],
  ["Vitals: Green", "green"],
  ["Calibrating Confidence Thresholds"],
  ["Black Box Temperature: Normal"],
  ["Mounting Destination Bins"],
  ["Initializing Review Queue"],
  ["Equipment Authentication: Complete"],
  ["Equipment Status: Green", "green"],
  ["All Systems Green", "green"],
  ["Threshing Preparations Complete"],
];

// YoRHa skin boot intro, once per browser profile.  index.html adds the
// .nier-booting class before first paint only when the skin is on and the
// intro has not been seen; without that class this renders nothing.
export default function YorhaBoot() {
  const [active, setActive] = useState(() =>
    document.documentElement.classList.contains("nier-booting")
  );
  const [shown, setShown] = useState(0);
  const [glitch, setGlitch] = useState(false);
  const overlayRef = useRef(null);
  const dismissRef = useRef(null);

  useEffect(() => {
    if (!active) return undefined;
    const timers = [];
    const dismiss = () => {
      timers.forEach(clearTimeout);
      document.documentElement.classList.remove("nier-booting");
      document.body.classList.remove("nier-boot-active");
      try {
        localStorage.setItem(BOOT_SEEN_STORAGE_KEY, "1");
      } catch {
        /* ignore */
      }
      setActive(false);
    };
    dismissRef.current = dismiss;
    const onKey = (ev) => {
      if (ev.key === "Escape" || ev.key === "Enter" || ev.key === " ") {
        ev.preventDefault();
        dismiss();
      }
    };
    document.body.classList.add("nier-boot-active");
    document.addEventListener("keydown", onKey);
    overlayRef.current?.focus();

    if (motionReduced()) {
      setShown(LINES.length);
      timers.push(setTimeout(dismiss, 400));
    } else {
      let idx = 0;
      let delay = 280;
      const revealNext = () => {
        if (idx >= LINES.length) {
          if (Math.random() < 0.35) setGlitch(true);
          timers.push(setTimeout(dismiss, 1400));
          return;
        }
        idx += 1;
        setShown(idx);
        if (idx > 4) delay = Math.max(90, delay - 35);
        timers.push(setTimeout(revealNext, delay + Math.random() * 120));
      };
      timers.push(setTimeout(revealNext, 500));
    }
    return () => {
      timers.forEach(clearTimeout);
      document.removeEventListener("keydown", onKey);
    };
  }, [active]);

  if (!active) return null;
  return (
    <div
      ref={overlayRef}
      className={`nier-boot${glitch ? " nier-boot--glitch" : ""}`}
      role="dialog"
      aria-label="System boot"
      aria-modal="true"
      tabIndex={-1}
      onClick={() => dismissRef.current?.()}
    >
      <p className="nier-boot__title">
        L O A D I N G &nbsp; - &nbsp; B O O T I N G &nbsp; T H R 3 S H R &nbsp; S Y S T E M . . .
      </p>
      <ul className="nier-boot__log" aria-live="polite">
        {LINES.map(([text, tone], idx) => (
          <li
            key={text}
            className={[
              "nier-boot__line",
              tone === "green" ? "nier-boot__line--green" : "",
              idx < shown ? "is-visible" : "",
            ]
              .filter(Boolean)
              .join(" ")}
          >
            {text}
            {idx === LINES.length - 1 ? (
              <span className="nier-boot__cursor" aria-hidden="true">
                _
              </span>
            ) : null}
          </li>
        ))}
      </ul>
      <p className="nier-boot__motto">YoRHa &mdash; For the Glory of Mankind</p>
    </div>
  );
}
