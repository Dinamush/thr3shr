import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
// Order matters: palette overrides must follow the token defaults.
import "./theme/tokens.css";
import "virtual:thr3shr-palettes.css";
import "./styles.css";
import "./theme/skin-yorha.css";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
