import { useEffect, useRef, useState, type ReactElement } from "react";

export type SceneItem = { key: string; clipId: string; start: number; end: number; duration: number };
export type SceneClip = { id: string; url: string; prompt: string };

const fmt = (seconds: number) => `${seconds.toFixed(1)}s`;
const length = (item: SceneItem) => Math.max(0, (item.end || item.duration) - item.start);

export function SceneBuilder({
  items,
  setItems,
  clips,
  exporting,
  onExport,
  onExtend,
}: {
  items: SceneItem[];
  setItems: (update: (prev: SceneItem[]) => SceneItem[]) => void;
  clips: SceneClip[];
  exporting: boolean;
  onExport: () => void;
  onExtend: (clipId: string) => void;
}): ReactElement {
  const player = useRef<HTMLVideoElement>(null);
  const [selected, setSelected] = useState(0);
  const [playingAll, setPlayingAll] = useState(false);
  const url = (clipId: string) => clips.find((clip) => clip.id === clipId)?.url ?? "";
  const current = items[Math.min(selected, items.length - 1)];
  const total = items.reduce((sum, item) => sum + length(item), 0);
  const ready = items.length > 0 && items.every((item) => item.duration > 0);

  const patch = (key: string, change: Partial<SceneItem>) =>
    setItems((prev) => prev.map((item) => (item.key === key ? { ...item, ...change } : item)));
  const move = (index: number, by: number) =>
    setItems((prev) => {
      const next = [...prev];
      const target = index + by;
      if (target < 0 || target >= next.length) return prev;
      [next[index], next[target]] = [next[target], next[index]];
      return next;
    });

  // Show the selected shot from its trim-in point; while "Play all" runs, hop to the next shot at each trim-out.
  useEffect(() => {
    const video = player.current;
    if (!video || !current) return;
    video.currentTime = current.start;
    if (playingAll) void video.play();
  }, [current?.key, current?.start, playingAll]); // eslint-disable-line react-hooks/exhaustive-deps

  if (items.length === 0) {
    return (
      <div className="glass flex min-h-[40vh] flex-col items-center justify-center rounded-3xl p-8 text-center">
        <p className="text-xl font-semibold tracking-tight text-zinc-100">Your scene is empty</p>
        <p className="mt-2 max-w-md text-sm text-zinc-400">
          Add clips with <span className="text-zinc-200">＋ Scene</span>, or <span className="text-zinc-200">Extend</span> one to
          continue it from its last frames. Trim and reorder shots here, then export a single video.
        </p>
      </div>
    );
  }

  return (
    <div className="glass space-y-4 rounded-3xl p-4">
      <div className="overflow-hidden rounded-2xl bg-black">
        <video
          ref={player}
          src={current ? url(current.clipId) : undefined}
          controls
          className="mx-auto max-h-[32vh] w-full"
          onTimeUpdate={(event) => {
            const video = event.currentTarget;
            if (!current || video.currentTime < (current.end || current.duration)) return;
            if (playingAll && selected < items.length - 1) setSelected(selected + 1);
            else {
              video.pause();
              setPlayingAll(false);
            }
          }}
        />
      </div>

      <div className="flex flex-wrap items-center gap-2 text-xs text-zinc-400">
        <button
          type="button"
          onClick={() => {
            setSelected(0);
            setPlayingAll(true);
          }}
          className="chip"
        >
          ▶ Play scene
        </button>
        <span>
          {items.length} shot{items.length === 1 ? "" : "s"} · {fmt(total)}
        </span>
        <button
          type="button"
          disabled={!ready || exporting}
          onClick={onExport}
          className="ml-auto rounded-full bg-gradient-to-br from-white to-zinc-300 px-4 py-1.5 text-xs font-semibold text-zinc-900 disabled:from-zinc-700 disabled:to-zinc-700 disabled:text-zinc-400"
        >
          {exporting ? "Exporting…" : "Export video"}
        </button>
      </div>

      <div className="flex gap-2 overflow-x-auto pb-1" role="list" aria-label="Timeline">
        {items.map((item, index) => {
          const active = index === selected;
          return (
            <div
              key={item.key}
              role="listitem"
              className={`shrink-0 rounded-2xl border p-1.5 transition ${
                active ? "border-white/40 bg-white/10" : "border-white/10 bg-white/[0.03] hover:bg-white/[0.06]"
              }`}
              style={{ width: `${Math.max(150, Math.min(320, length(item) * 26))}px` }}
            >
              <button type="button" onClick={() => setSelected(index)} className="block w-full" aria-label={`Shot ${index + 1}`}>
                <video
                  src={`${url(item.clipId)}#t=${item.start}`}
                  muted
                  preload="metadata"
                  onLoadedMetadata={(event) => {
                    const duration = event.currentTarget.duration;
                    if (item.duration !== duration) patch(item.key, { duration, end: item.end || duration });
                  }}
                  className="h-20 w-full rounded-xl bg-black object-cover"
                />
              </button>
              <div className="mt-1 flex items-center justify-between px-0.5 text-[11px] text-zinc-400">
                <span>
                  {index + 1} · {fmt(length(item))}
                </span>
                <span className="flex gap-1">
                  <button type="button" onClick={() => move(index, -1)} disabled={index === 0} aria-label="Move left" className="hover:text-white disabled:opacity-30">
                    ←
                  </button>
                  <button type="button" onClick={() => move(index, 1)} disabled={index === items.length - 1} aria-label="Move right" className="hover:text-white disabled:opacity-30">
                    →
                  </button>
                  <button type="button" onClick={() => onExtend(item.clipId)} title="Continue this shot from its last frames" className="hover:text-white">
                    ↪
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setItems((prev) => prev.filter((x) => x.key !== item.key));
                      setSelected((s) => Math.max(0, Math.min(s, items.length - 2)));
                    }}
                    aria-label="Remove shot"
                    className="hover:text-red-300"
                  >
                    ✕
                  </button>
                </span>
              </div>
              {active && item.duration > 0 && (
                <div className="mt-1 space-y-1 px-0.5 text-[10px] text-zinc-500">
                  <label className="flex items-center gap-2">
                    In
                    <input
                      type="range"
                      min={0}
                      max={item.duration}
                      step={1 / 24}
                      value={item.start}
                      onChange={(event) => patch(item.key, { start: Math.min(Number(event.target.value), (item.end || item.duration) - 0.1) })}
                      className="w-full accent-white"
                    />
                    <span className="w-9 text-right font-mono">{fmt(item.start)}</span>
                  </label>
                  <label className="flex items-center gap-2">
                    Out
                    <input
                      type="range"
                      min={0}
                      max={item.duration}
                      step={1 / 24}
                      value={item.end || item.duration}
                      onChange={(event) => patch(item.key, { end: Math.max(Number(event.target.value), item.start + 0.1) })}
                      className="w-full accent-white"
                    />
                    <span className="w-9 text-right font-mono">{fmt(item.end || item.duration)}</span>
                  </label>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
