"use client";

interface ModeToggleProps {
  mode: "emotional" | "rational";
  onModeChange: (mode: "emotional" | "rational") => void;
}

export default function ModeToggle({ mode, onModeChange }: ModeToggleProps) {
  return (
    <div className="flex items-center gap-4 text-[11px] tracking-wide uppercase">
      <button
        onClick={() => onModeChange("emotional")}
        className={`transition-colors px-1 ${
          mode === "emotional"
            ? "text-text underline underline-offset-4 decoration-accent"
            : "text-muted hover:text-text"
        }`}
      >
        [S] 感性
      </button>
      <button
        onClick={() => onModeChange("rational")}
        className={`transition-colors px-1 ${
          mode === "rational"
            ? "text-text underline underline-offset-4 decoration-accent"
            : "text-muted hover:text-text"
        }`}
      >
        [R] 理性
      </button>
    </div>
  );
}
