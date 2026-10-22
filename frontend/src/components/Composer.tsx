import { useRef, type Dispatch, type ReactElement, type ReactNode, type SetStateAction } from "react";
import { computeAlignedFrames, computeSize } from "../math";
import { ASPECTS, type GenerationState } from "./GenerationControls";

export type Task = "ref2v" | "i2v";

export type SlotState = {
  mediaId: string | null;
  previewUrl: string | null;
  name: string;
};

export function emptySlots(count: number): SlotState[] {
  return Array.from({ length: count }, () => ({ mediaId: null, previewUrl: null, name: "" }));
}
export type SlotKind = "image" | "video" | "audio" | "video-audio" | "keyframe";

const MEGAPIXELS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0];
const DURATIONS = Array.from({ length: 15 }, (_, i) => i + 1);
const MAX_SCENES = 6;
const ACCEPT: Record<SlotKind, string> = {
  image: "image/*",
  keyframe: "image/*",
  video: "video/*",
  audio: "audio/*",
  "video-audio": "audio/*",
};
const TAG: Record<"image" | "video" | "audio", string> = { image: "Picture", video: "Video", audio: "Audio" };

export function Composer({
  task,
  onTask,
  prompt,
  setPrompt,
  segments,
  setSegments,
  images,
  videos,
  videoAudios,
  audios,
  keyframes,
  settings,
  setSettings,
  seedText,
  setSeedText,
  ready,
  busy,
  onPick,
  onClear,
  onGenerate,
}: {
  task: Task;
  onTask: (task: Task) => void;
  prompt: string;
  setPrompt: (value: string) => void;
  segments: string[];
  setSegments: (next: string[]) => void;
  images: SlotState[];
  videos: SlotState[];
  videoAudios: SlotState[];
  audios: SlotState[];
  keyframes: SlotState[];
  settings: GenerationState;
  setSettings: Dispatch<SetStateAction<GenerationState>>;
  seedText: string;
  setSeedText: (value: string) => void;
  ready: boolean;
  busy: boolean;
  onPick: (kind: SlotKind, index: number, file: File) => void;
  onClear: (kind: SlotKind, index: number) => void;
  onGenerate: () => void;
}): ReactElement {
  const area = useRef<HTMLTextAreaElement>(null);
  const picker = useRef<HTMLInputElement>(null);
  const pending = useRef<{ kind: SlotKind; index: number } | null>(null);

  const pick = (kind: SlotKind, index: number) => {
    if (index < 0 || !picker.current) return;
    pending.current = { kind, index };
    picker.current.accept = ACCEPT[kind];
    picker.current.click();
  };

  const insertTag = (tag: string) => {
    const node = area.current;
    const start = node?.selectionStart ?? prompt.length;
    const end = node?.selectionEnd ?? prompt.length;
    const spacer = start > 0 && !/\s/.test(prompt[start - 1]) ? " " : "";
    setPrompt(`${prompt.slice(0, start)}${spacer}${tag}${prompt.slice(end)}`);
    const caret = start + spacer.length + tag.length;
    requestAnimationFrame(() => {
      node?.focus();
      node?.setSelectionRange(caret, caret);
    });
  };

  // Tags number the attached items of each kind in order, matching how references are sent.
  const filled = (slots: SlotState[]) => slots.flatMap((slot, index) => (slot.mediaId ? [index] : []));
  const firstEmpty = (slots: SlotState[]) => slots.findIndex((slot) => !slot.mediaId);
  const [width, height] = computeSize(settings.aspectRatio, settings.megapixels);
  const canGenerate = ready && !busy && prompt.trim().length > 0;

  const chips = (kind: "image" | "video" | "audio", slots: SlotState[]) =>
    filled(slots).map((index, order) => {
      const slot = slots[index];
      const tag = `<${TAG[kind]} ${order + 1}>`;
      const paired = kind === "video" ? videoAudios[index] : undefined;
      return (
        <div key={`${kind}-${index}`} className="group relative w-24 shrink-0 rounded-xl border border-white/10 bg-zinc-900">
          <Preview kind={kind} url={slot.previewUrl} name={slot.name} />
          <button
            type="button"
            onClick={() => insertTag(tag)}
            className="absolute left-1 top-1 rounded-md bg-black/70 px-1.5 py-0.5 text-[10px] font-medium text-sky-300 hover:text-sky-100"
            title="Insert this tag into the prompt"
          >
            {tag}
          </button>
          <RemoveButton disabled={busy} label={`Remove ${tag}`} onClick={() => onClear(kind, index)} />
          {kind === "video" && (
            <button
              type="button"
              disabled={busy}
              onClick={() => (paired?.mediaId ? onClear("video-audio", index) : pick("video-audio", index))}
              className="block w-full truncate px-1.5 py-1 text-left text-[10px] text-amber-300/90 hover:text-amber-200"
              title={paired?.mediaId ? "Remove the paired audio" : "Pair a different soundtrack with this video"}
            >
              {paired?.mediaId ? `♪ ${paired.name} ✕` : "♪ own soundtrack"}
            </button>
          )}
        </div>
      );
    });

  return (
    <div className="rounded-3xl border border-white/10 bg-zinc-900/90 shadow-2xl shadow-black/50 backdrop-blur">
      <input
        ref={picker}
        type="file"
        hidden
        onChange={(event) => {
          const file = event.target.files?.[0];
          event.target.value = "";
          if (file && pending.current) onPick(pending.current.kind, pending.current.index, file);
        }}
      />

      <div className="flex flex-wrap items-center gap-2 px-4 pt-3">
        <div className="inline-flex rounded-full bg-zinc-950 p-1" role="tablist" aria-label="Task">
          {(
            [
              ["i2v", "I2V", "Frames to video"],
              ["ref2v", "Ref2V", "Ingredients to video"],
            ] as const
          ).map(([id, label, hint]) => (
            <button
              key={id}
              type="button"
              role="tab"
              aria-selected={task === id}
              disabled={busy}
              onClick={() => onTask(id)}
              className={`rounded-full px-3 py-1 text-xs font-semibold transition ${
                task === id ? "bg-white text-zinc-900" : "text-zinc-400 hover:text-white"
              }`}
            >
              {label}{" "}
              <span className={`hidden font-normal sm:inline ${task === id ? "text-zinc-600" : "text-zinc-500"}`}>· {hint}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="flex gap-2 overflow-x-auto px-4 pt-3">
        {task === "i2v" ? (
          (["First frame", "Last frame"] as const).map((label, index) => {
            const slot = keyframes[index];
            return slot.mediaId ? (
              <div key={label} className="relative w-28 shrink-0 rounded-xl border border-white/10 bg-zinc-900">
                <Preview kind="image" url={slot.previewUrl} name={slot.name} />
                <span className="absolute left-1 top-1 rounded-md bg-black/70 px-1.5 py-0.5 text-[10px] text-zinc-200">{label}</span>
                <RemoveButton disabled={busy} label={`Remove ${label}`} onClick={() => onClear("keyframe", index)} />
              </div>
            ) : (
              <AddTile key={label} label={label} hint="optional" disabled={busy || !ready} onClick={() => pick("keyframe", index)} />
            );
          })
        ) : (
          <>
            {chips("image", images)}
            {chips("video", videos)}
            {chips("audio", audios)}
            <AddTile label="Image" hint={`${filled(images).length}/9`} disabled={busy || !ready || firstEmpty(images) < 0} onClick={() => pick("image", firstEmpty(images))} />
            <AddTile label="Video" hint={`${filled(videos).length}/3`} disabled={busy || !ready || firstEmpty(videos) < 0} onClick={() => pick("video", firstEmpty(videos))} />
            <AddTile label="Audio" hint={`${filled(audios).length}/3`} disabled={busy || !ready || firstEmpty(audios) < 0} onClick={() => pick("audio", firstEmpty(audios))} />
          </>
        )}
      </div>

      <div className="px-4 pt-3">
        <textarea
          ref={area}
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && (event.metaKey || event.ctrlKey) && canGenerate) onGenerate();
          }}
          rows={3}
          spellCheck={false}
          aria-label="Prompt"
          placeholder={
            task === "i2v"
              ? "Describe the shot, the motion and the sound…"
              : "Describe the shot; address ingredients as <Picture 1>, <Video 1>, <Audio 1>…"
          }
          className="max-h-60 w-full resize-y bg-transparent text-[15px] leading-6 text-zinc-100 placeholder:text-zinc-500 focus:outline-none"
        />
        {segments.map((text, index) => (
          <div key={index} className="mt-2 border-t border-white/5 pt-2">
            <div className="flex items-center justify-between text-[11px] text-zinc-500">
              <span>Scene {index + 2} · continues from the last frames of scene {index + 1}</span>
              <button type="button" disabled={busy} onClick={() => setSegments(segments.filter((_, i) => i !== index))} className="hover:text-zinc-200">
                Remove
              </button>
            </div>
            <textarea
              value={text}
              onChange={(event) => setSegments(segments.map((t, i) => (i === index ? event.target.value : t)))}
              rows={2}
              aria-label={`Scene ${index + 2} prompt`}
              placeholder="What happens next…"
              className="w-full resize-y bg-transparent text-sm leading-6 text-zinc-100 placeholder:text-zinc-600 focus:outline-none"
            />
          </div>
        ))}
      </div>

      <div className="flex flex-wrap items-center gap-2 px-3 pb-3 pt-1">
        <Pill label="Aspect ratio">
          <select
            value={settings.aspectRatio}
            onChange={(event) => setSettings((prev) => ({ ...prev, aspectRatio: event.target.value }))}
            className="pill-select"
          >
            {ASPECTS.map((aspect) => (
              <option key={aspect} value={aspect}>
                {aspect.split(" ")[0]}
              </option>
            ))}
          </select>
        </Pill>
        <Pill label="Resolution">
          <select
            value={settings.megapixels}
            onChange={(event) => setSettings((prev) => ({ ...prev, megapixels: Number(event.target.value) }))}
            className="pill-select"
          >
            {MEGAPIXELS.map((mp) => {
              const [w, h] = computeSize(settings.aspectRatio, mp);
              return (
                <option key={mp} value={mp}>
                  {w}×{h}
                </option>
              );
            })}
          </select>
        </Pill>
        <Pill label="Duration">
          <select
            value={settings.durationSec}
            onChange={(event) => setSettings((prev) => ({ ...prev, durationSec: Number(event.target.value) }))}
            className="pill-select"
          >
            {(DURATIONS.includes(settings.durationSec) ? DURATIONS : [...DURATIONS, settings.durationSec].sort((a, b) => a - b)).map((s) => (
              <option key={s} value={s}>
                {s}s
              </option>
            ))}
          </select>
        </Pill>
        <Pill label="Seed">
          <input
            value={seedText}
            onChange={(event) => setSeedText(event.target.value.replace(/[^\d]/g, ""))}
            placeholder="random"
            inputMode="numeric"
            aria-label="Seed"
            className="w-24 bg-transparent text-xs text-zinc-100 placeholder:text-zinc-500 focus:outline-none"
          />
          <button
            type="button"
            onClick={() => setSeedText(String(Math.floor(Math.random() * 2 ** 32)))}
            className="text-zinc-400 hover:text-white"
            title="Pick a random seed you can reuse"
            aria-label="Random seed"
          >
            ⚄
          </button>
        </Pill>
        <button
          type="button"
          disabled={busy || segments.length >= MAX_SCENES - 1}
          onClick={() => setSegments([...segments, ""])}
          className="rounded-full border border-white/10 px-3 py-1.5 text-xs text-zinc-300 hover:bg-white/5 disabled:text-zinc-600"
          title="Add a scene that continues from this one's last frames; all scenes are merged into one video"
        >
          + Extend
        </button>
        <span className="ml-auto hidden text-[11px] text-zinc-500 sm:inline">
          {width}×{height} · {computeAlignedFrames(settings.durationSec)} frames
          {segments.length > 0 ? ` · ${segments.length + 1} scenes` : ""}
        </span>
        <button
          type="button"
          disabled={!canGenerate}
          onClick={onGenerate}
          aria-label={busy ? "Generating" : "Generate"}
          title="Generate (Ctrl+Enter)"
          className="ml-auto flex h-10 w-10 items-center justify-center rounded-full bg-white text-lg font-bold text-zinc-900 transition hover:bg-zinc-200 disabled:bg-zinc-700 disabled:text-zinc-500 sm:ml-0"
        >
          {busy ? <span className="h-4 w-4 animate-spin rounded-full border-2 border-zinc-500 border-t-zinc-900" /> : "→"}
        </button>
      </div>
    </div>
  );
}

function Preview({ kind, url, name }: { kind: "image" | "video" | "audio"; url: string | null; name: string }): ReactElement {
  if (url && kind === "image") return <img src={url} alt={name} className="h-20 w-full rounded-t-xl object-cover" />;
  if (url && kind === "video") return <video src={url} muted className="h-20 w-full rounded-t-xl bg-black object-cover" />;
  return (
    <div className="flex h-20 w-full items-end rounded-t-xl bg-gradient-to-br from-zinc-800 to-zinc-900 p-1.5">
      <span className="truncate text-[10px] text-zinc-400">♪ {name}</span>
    </div>
  );
}

function RemoveButton({ label, disabled, onClick }: { label: string; disabled: boolean; onClick: () => void }): ReactElement {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      aria-label={label}
      className="absolute right-1 top-1 h-5 w-5 rounded-full bg-black/70 text-[11px] leading-5 text-zinc-300 hover:text-white"
    >
      ✕
    </button>
  );
}

function AddTile({ label, hint, disabled, onClick }: { label: string; hint: string; disabled: boolean; onClick: () => void }): ReactElement {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className="flex h-20 w-24 shrink-0 flex-col items-center justify-center rounded-xl border border-dashed border-white/15 text-xs text-zinc-400 transition hover:border-white/40 hover:text-white disabled:opacity-40"
    >
      <span className="text-lg leading-none">+</span>
      <span className="mt-1">{label}</span>
      <span className="text-[10px] text-zinc-600">{hint}</span>
    </button>
  );
}

function Pill({ label, children }: { label: string; children: ReactNode }): ReactElement {
  return (
    <label className="flex items-center gap-1.5 rounded-full border border-white/10 bg-zinc-950/60 px-3 py-1.5 text-xs text-zinc-400" title={label}>
      <span className="sr-only">{label}</span>
      {children}
    </label>
  );
}
