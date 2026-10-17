import React, { useState } from "react";

// Number input that can be cleared and retyped. The text the user types is
// kept as a draft; valid in-range values commit as they are typed (so the
// unsaved-changes bar reacts), and on blur an empty or out-of-range draft is
// clamped or reverted instead of being coerced to 0 mid-edit.
export default function NumberField({
  value,
  onCommit,
  min,
  max,
  step = 1,
  integer = false,
  ...inputProps
}) {
  const [draft, setDraft] = useState(null);
  const normalize = (raw) => {
    const parsed = Number(raw);
    if (String(raw).trim() === "" || !Number.isFinite(parsed)) return null;
    return integer ? Math.round(parsed) : parsed;
  };
  return (
    <input
      {...inputProps}
      type="number"
      min={min}
      max={max}
      step={step}
      value={draft ?? String(value ?? "")}
      onChange={(e) => {
        const text = e.target.value;
        setDraft(text);
        const next = normalize(text);
        if (next !== null && next >= min && next <= max && next !== value) onCommit(next);
      }}
      onBlur={() => {
        if (draft === null) return;
        const next = normalize(draft);
        if (next !== null) {
          const clamped = Math.min(max, Math.max(min, next));
          if (clamped !== value) onCommit(clamped);
        }
        setDraft(null);
      }}
    />
  );
}
