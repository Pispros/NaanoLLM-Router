/** @type {import('tailwindcss').Config} */
// Build with:
//   npx tailwindcss@3 -i ./tailwind.input.css -o ./web/tailwind.css --minify
// (the forms + container-queries plugins below mirror the old
//  cdn.tailwindcss.com?plugins=forms,container-queries URL)
module.exports = {
  content: ["./web/index.html"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        "background": "#081425",
        "on-background": "#d8e3fb",
        "surface": "#081425",
        "surface-dim": "#081425",
        "surface-bright": "#2f3a4c",
        "surface-container-lowest": "#040e1f",
        "surface-container-low": "#111c2d",
        "surface-container": "#152031",
        "surface-container-high": "#1f2a3c",
        "surface-container-highest": "#2a3548",
        "surface-variant": "#2a3548",
        "on-surface": "#d8e3fb",
        "on-surface-variant": "#c6c6cd",
        "primary": "#bec6e0",
        "on-primary": "#283044",
        "primary-fixed": "#dae2fd",
        "primary-fixed-dim": "#bec6e0",
        "primary-container": "#0f172a",
        "on-primary-container": "#798098",
        "secondary": "#c1c7cf",
        "secondary-container": "#41474e",
        "on-secondary-container": "#afb6bd",
        "tertiary": "#b9c8de",
        "outline": "#909097",
        "outline-variant": "#45464d",
        "error": "#ffb4ab",
        "error-container": "#93000a",
        "on-error": "#690005",
        "on-error-container": "#ffdad6",
        "inverse-surface": "#d8e3fb",
        "inverse-on-surface": "#263143",
        "planner": "#8b8cff",
        "coder": "#2dd4bf",
        "router": "#d946ef",
        "ok": "#3fb950",
        "warn": "#eab308"
      },
      borderRadius: { DEFAULT: "0.125rem", lg: "0.25rem", xl: "0.5rem", full: "0.75rem" },
      spacing: { "margin-page": "48px", gutter: "24px", "container-max": "1280px", "section-gap": "64px" },
      fontFamily: {
        "headline-xl": ["Raleway"], "headline-lg": ["Raleway"], "headline-md": ["Raleway"],
        "label-caps": ["Montserrat"], "body-lg": ["Montserrat"], "body-md": ["Montserrat"], "body-sm": ["Montserrat"]
      },
      fontSize: {
        "headline-xl": ["40px", { lineHeight: "1.2", letterSpacing: "-0.02em", fontWeight: "700" }],
        "headline-lg": ["32px", { lineHeight: "1.3", fontWeight: "600" }],
        "headline-md": ["24px", { lineHeight: "1.4", fontWeight: "600" }],
        "label-caps": ["12px", { lineHeight: "1", letterSpacing: "0.1em", fontWeight: "600" }],
        "body-lg": ["18px", { lineHeight: "1.6", fontWeight: "400" }],
        "body-md": ["16px", { lineHeight: "1.5", fontWeight: "400" }],
        "body-sm": ["14px", { lineHeight: "1.5", fontWeight: "400" }]
      }
    }
  },
  plugins: [
    require("@tailwindcss/forms"),
    require("@tailwindcss/container-queries")
  ]
};
