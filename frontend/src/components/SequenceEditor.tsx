import type { ReactElement } from "react";

const MAX_CONTINUATIONS = 5;

export function SequenceEditor({
  segments,
  setSegments,
  durationSec,
  busy,
}: {
  segments: string[];
  setSegments: (next: string[]) => void;
  durationSec: number;
  busy: boolean;
}): ReactElement {
  const update = (index: number, value: string) =>
    setSegments(segments.map((text, i) => (i === index ? value : text)));
  const total = durationSec + segments.length * Math.max(0, durationSec - 5 / 24);

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 space-y-3">
      <div className="flex items-center justify-between gap-2">
        <div>
          <h2 className="text-lg font-semibold tracking-tight">Sequence</h2>
          <p className="text-[11px] text-zinc-500 mt-1">
            Segment 1 is the prompt above. Each continuation starts from the previous segment&apos;s last frames and
            sound; all segments are merged into one video (~{total.toFixed(1)} s).
          </p>
        </div>
        <button
          type="button"
          disabled={busy || segments.length >= MAX_CONTINUATIONS}
          onClick={() => setSegments([...segments, ""])}
          className="shrink-0 px-3 py-1.5 rounded-lg border border-zinc-700 text-xs text-zinc-200 hover:bg-zinc-800 disabled:text-zinc-600"
        >
          + Add continuation
        </button>
      </div>
      {segments.map((text, index) => (
        <div key={index} className="space-y-1">
          <div className="flex items-center justify-between text-[11px] text-zinc-400">
            <span>Segment {index + 2}</span>
            <button
              type="button"
              disabled={busy}
              onClick={() => setSegments(segments.filter((_, i) => i !== index))}
              className="text-zinc-500 hover:text-zinc-200"
            >
              Remove
            </button>
          </div>
          <textarea
            value={text}
            onChange={(event) => update(index, event.target.value)}
            rows={3}
            placeholder="What happens next…"
            className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-sm text-zinc-100 resize-y"
          />
        </div>
      ))}
    </div>
  );
}
