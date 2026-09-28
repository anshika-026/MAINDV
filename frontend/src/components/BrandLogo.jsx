import { useState } from "react";

// The Dewin logo, served from frontend/public/dewin-logo.png. Falls back to
// the plain "D" mark if that file is missing, so the header never shows a
// broken image.
export default function BrandLogo({ size = 28 }) {
  const [missing, setMissing] = useState(false);
  if (missing) {
    return (
      <div
        className="rounded-md bg-brand-500 flex items-center justify-center text-white font-bold text-sm shrink-0"
        style={{ width: size, height: size }}
      >
        D
      </div>
    );
  }
  return (
    <img
      src="/dewin-logo.png"
      alt="Dewin"
      className="object-contain shrink-0 rounded-md"
      style={{ height: size, width: "auto" }}
      onError={() => setMissing(true)}
    />
  );
}
