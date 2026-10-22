import { useCallback, useEffect, useMemo, useRef, useState, type Dispatch, type SetStateAction } from "react";
import {
  SecureClient,
  type EnvelopeResult,
  type GpuInfo,
  type ProgressEvent,
} from "./cryptoClient";
import { Composer, emptySlots, type SlotKind, type SlotState, type Task } from "./components/Composer";
import {
  EMPTY_MODELS,
  FACTORY_DEFAULTS,
  GenerationControls,
  stepsFor,
  type GenerationState,
  type ModelLists,
} from "./components/GenerationControls";
import { PanicFlushButton } from "./components/PanicFlushButton";
import { SceneBuilder, type SceneItem } from "./components/SceneBuilder";
import { SystemMonitor } from "./components/SystemMonitor";
import { computeAlignedFrames, computeSize } from "./math";

// Each task has its own FastH3 checkpoint; switching tabs picks it when it is installed.
const TASK_MODEL: Record<Task, RegExp> = { ref2v: /omniref/i, i2v: /fasth3_8step_v2/i };
const TASK_LABEL: Record<Task, string> = { ref2v: "Ref2V", i2v: "I2V" };

const ROLE_CLASS: Record<string, string> = {
  active_full: "text-emerald-300",
  active_tight_partial: "text-amber-300",
  bypassed_busy: "text-zinc-500",
};

const ROLE_LABEL: Record<string, string> = {
  active_full: "free",
  active_tight_partial: "tight",
  bypassed_busy: "busy",
};

type Clip = { id: string; url: string; prompt: string; meta: string; took?: number };

function fmtDuration(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
  return `${Math.floor(s / 3600)}h ${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}m`;
}

/** Remaining ms from the progress rate since the latest sample at least 20 s old (model loading makes the
 * whole-job average misleading early on); null until there is a rate to go on. */
function estimateEta(history: [number, number][], now: number): number | null {
  const last = history[history.length - 1];
  if (!last || last[1] < 0.05 || last[1] >= 1) return null;
  let past = history[0];
  for (const sample of history) if (last[0] - sample[0] >= 20_000) past = sample;
  const rate = (last[1] - past[1]) / (last[0] - past[0]);
  return rate > 0 ? Math.max(0, (1 - last[1]) / rate - (now - last[0])) : null;
}
type JobInfo = Omit<Clip, "id" | "url"> & { after?: string };

export function App() {
  const client = useMemo(() => new SecureClient(), []);
  const urls = useRef(new Set<string>());
  const settingsDialog = useRef<HTMLDialogElement>(null);
  const systemDialog = useRef<HTMLDialogElement>(null);
  const [systemOpen, setSystemOpen] = useState(false);
  const [queue, setQueue] = useState<{ position: number; ahead: number } | null>(null);
  const [stopping, setStopping] = useState(false);
  const [ready, setReady] = useState(false);
  const [status, setStatus] = useState("Securing session");
  const [error, setError] = useState("");
  const [prompt, setPrompt] = useState("");
  const [settings, setSettings] = useState<GenerationState>(FACTORY_DEFAULTS);
  const [models, setModels] = useState<ModelLists>(EMPTY_MODELS);
  const [gpus, setGpus] = useState<GpuInfo[]>([]);
  const [images, setImages] = useState<SlotState[]>(() => emptySlots(9));
  const [videos, setVideos] = useState<SlotState[]>(() => emptySlots(3));
  const [videoAudios, setVideoAudios] = useState<SlotState[]>(() => emptySlots(3));
  const [audios, setAudios] = useState<SlotState[]>(() => emptySlots(3));
  const [keyframes, setKeyframes] = useState<SlotState[]>(() => emptySlots(2));
  const [task, setTask] = useState<Task>("ref2v");
  const [segments, setSegments] = useState<string[]>([]);
  const slotSetters: Record<SlotKind, Dispatch<SetStateAction<SlotState[]>>> = {
    image: setImages,
    video: setVideos,
    "video-audio": setVideoAudios,
    audio: setAudios,
    keyframe: setKeyframes,
  };
  const [controlId, setControlId] = useState<string | null>(null);
  const [controlName, setControlName] = useState("");
  const [seedText, setSeedText] = useState("");
  const [busy, setBusy] = useState(false);
  const [flushing, setFlushing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState("");
  const [detail, setDetail] = useState("");
  const [activityLog, setActivityLog] = useState<string[]>([]);
  const [jobId, setJobId] = useState<string | null>(null);
  const [clips, setClips] = useState<Clip[]>([]);
  const activeJob = useRef<string | null>(null);
  const jobStart = useRef(0);
  const progressHistory = useRef<[number, number][]>([]);
  const [now, setNow] = useState(() => Date.now());
  const jobInfo = useRef<Record<string, JobInfo>>({});
  const [scene, setScene] = useState<SceneItem[]>([]);
  const [view, setView] = useState<"clips" | "scene">("clips");
  const [continueFrom, setContinueFrom] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);
  const pulled = useRef(new Set<string>());

  const pushLog = useCallback((entry: string) => {
    setActivityLog((prev) => {
      if (prev[prev.length - 1] === entry) return prev;
      return [...prev.slice(-14), entry];
    });
  }, []);

  const track = useCallback((url: string) => {
    urls.current.add(url);
    return url;
  }, []);

  const revokeAll = useCallback(() => {
    for (const url of urls.current) {
      URL.revokeObjectURL(url);
    }
    urls.current.clear();
    pulled.current.clear();
    setClips([]);
    setScene([]);
    setContinueFrom(null);
  }, []);

  const refreshModels = useCallback(async () => {
    const result = await client.call("list_models", {});
    if (!result.ok) {
      throw new Error(result.error || "list_models_failed");
    }
    const available = result.metadata.available as ModelLists | undefined;
    const nextModels: ModelLists = {
      diffussion: available?.diffussion ?? [],
      vae: available?.vae ?? [],
      clip: available?.clip ?? [],
      lora: available?.lora ?? [],
      controlnet: available?.controlnet ?? [],
    };
    setModels(nextModels);
    setSettings((prev) => {
      const pick = (list: string[], current: string, hint?: string): string => {
        if (list.length === 0) return current;
        if (list.includes(current)) return current;
        if (hint) {
          const matched = list.find((item) => item.toLowerCase().includes(hint.toLowerCase()));
          if (matched) return matched;
        }
        return list[0];
      };
      return {
        ...prev,
        diffusionModel: pick(nextModels.diffussion, prev.diffusionModel),
        clipModel: pick(nextModels.clip, prev.clipModel),
        videoVaeModel: pick(nextModels.vae, prev.videoVaeModel, "video"),
        audioVaeModel: pick(nextModels.vae, prev.audioVaeModel, "audio"),
        loraModel: pick(nextModels.lora, prev.loraModel),
        controlnetModel:
          prev.controlnetModel && nextModels.controlnet.includes(prev.controlnetModel)
            ? prev.controlnetModel
            : (nextModels.controlnet[0] ?? ""),
      };
    });
    setGpus((result.metadata.gpu_topology as GpuInfo[] | undefined) ?? []);
  }, [client]);

  useEffect(() => {
    let stop = false;
    void (async () => {
      try {
        await client.handshake();
        client.connect((event) => {
          if (stop) return;
          handleProgress(event);
        });
        await refreshModels();
        if (!stop) {
          setReady(true);
          setStatus("Session ready");
        }
      } catch {
        if (!stop) setError("Secure session failed");
      }
    })();
    return () => {
      stop = true;
      client.reset();
    };
    // The client instance is stable. Progress handling closes over latest job via ref.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [client]);

  function handleProgress(event: ProgressEvent) {
    if (activeJob.current && event.job_id !== activeJob.current) return;
    if (event.gpus && event.gpus.length > 0) {
      setGpus(event.gpus);
    }
    setStage(event.stage);
    setProgress(event.progress);
    const history = progressHistory.current;
    if (!history.length || event.progress > history[history.length - 1][1]) history.push([Date.now(), event.progress]);
    if (event.detail) {
      setDetail(event.detail);
      setStatus(`${event.stage}: ${event.detail}`);
      pushLog(`[${Math.round(event.progress * 100)}%] ${event.stage} — ${event.detail}`);
    } else {
      setStatus(event.stage);
    }
    if (event.done) {
      setBusy(false);
      setStopping(false);
      setQueue(null);
      const info = jobInfo.current[event.job_id];
      if (info && info.took === undefined && jobStart.current) info.took = Date.now() - jobStart.current;
      if (!event.ok && event.detail === "aborted") {
        setError("");
        setStatus("Stopped");
        pushLog("[■] Stopped — GPU memory released");
      } else if (event.ok) {
        setStatus("Ready");
        pushLog("[100%] Complete — fetching encrypted MP4");
        void pullOutput(event.job_id);
      } else {
        setError(event.detail || "inference_failed");
        setStatus(`Failed: ${event.detail || "inference_failed"}`);
        pushLog(`[ERR] ${event.detail || "inference_failed"}`);
      }
    }
  }

  useEffect(() => {
    if (!busy || !jobId) return;
    const timer = window.setInterval(() => {
      void (async () => {
        try {
          const res = await client.call("job_status", { job_id: jobId });
          if (!res.ok) return;
          const m = res.metadata;
          const ev: ProgressEvent = {
            job_id: String(m.job_id ?? jobId),
            stage: String(m.stage ?? "running"),
            progress: typeof m.progress === "number" ? m.progress : 0,
            detail: String(m.detail ?? ""),
            done: Boolean(m.done),
            ok: Boolean(m.ok),
            gpus: Array.isArray(m.gpus) ? (m.gpus as GpuInfo[]) : undefined,
          };
          setQueue(
            typeof m.queue_position === "number" ? { position: m.queue_position, ahead: Number(m.jobs_ahead ?? 0) } : null,
          );
          handleProgress(ev);
        } catch {
          // Ignore transient polling errors while WebSocket is active
        }
      })();
    }, 1000);
    return () => window.clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [busy, jobId, client]);

  async function pullOutput(id: string) {
    // Both the WebSocket and the status poll report completion; fetch each output once.
    if (pulled.current.has(id)) return;
    pulled.current.add(id);
    try {
      const result = await client.call("download_output", { job_id: id });
      if (!result.ok) {
        pulled.current.delete(id);
        if (result.error === "not_ready") return;
        setError(result.error || "download_failed");
        return;
      }
      const file = result.files[0];
      if (!file) return;
      const copy = new Uint8Array(new ArrayBuffer(file.data.byteLength));
      copy.set(file.data);
      const url = track(URL.createObjectURL(new Blob([copy], { type: "video/mp4" })));
      const { after, ...info } = jobInfo.current[id] ?? { prompt: "", meta: "" };
      setClips((prev) => [{ id, url, ...info }, ...prev]);
      if (after) {
        // An extension lands right after the shot it continues; the source joins the scene if it was not in it.
        setScene((prev) => {
          const shot = newShot(id);
          const at = prev.map((item) => item.clipId).lastIndexOf(after);
          if (at < 0) return [...prev, newShot(after), shot];
          return [...prev.slice(0, at + 1), shot, ...prev.slice(at + 1)];
        });
      }
      setStatus("Playback ready");
    } catch (err) {
      pulled.current.delete(id);
      setError(err instanceof Error ? err.message : "download_failed");
    }
  }

  async function uploadSlot(kind: "image" | "video" | "audio" | "control", file: File): Promise<string> {
    const data = new Uint8Array(await file.arrayBuffer());
    const result: EnvelopeResult = await client.call(
      "upload_media",
      { kind, filename: file.name },
      [{ name: file.name, mime: file.type || "application/octet-stream", kind, data }],
    );
    if (!result.ok || typeof result.metadata.media_id !== "string") {
      throw new Error(result.error || "upload_failed");
    }
    return result.metadata.media_id;
  }

  function replaceSlot(setter: Dispatch<SetStateAction<SlotState[]>>, index: number, next: SlotState) {
    setter((prev) => prev.map((slot, slotIndex) => (slotIndex === index ? next : slot)));
  }

  async function onPick(kind: SlotKind, index: number, file: File) {
    setError("");
    try {
      const uploadKind = kind === "video-audio" ? "audio" : kind === "keyframe" ? "image" : kind;
      const mediaId = await uploadSlot(uploadKind, file);
      const previewUrl = track(URL.createObjectURL(file));
      replaceSlot(slotSetters[kind], index, { mediaId, previewUrl, name: file.name });
    } catch (err) {
      setError(err instanceof Error ? err.message : "upload_failed");
    }
  }

  function clearPreview(slot: SlotState) {
    if (slot.previewUrl) {
      URL.revokeObjectURL(slot.previewUrl);
      urls.current.delete(slot.previewUrl);
    }
  }

  function onClear(kind: SlotKind, index: number) {
    const blank: SlotState = { mediaId: null, previewUrl: null, name: "" };
    slotSetters[kind]((prev) => {
      clearPreview(prev[index]);
      return prev.map((slot, slotIndex) => (slotIndex === index ? blank : slot));
    });
    if (kind === "video") onClear("video-audio", index);
  }

  function newShot(clipId: string): SceneItem {
    return { key: `${clipId}-${Math.random().toString(36).slice(2, 8)}`, clipId, start: 0, end: 0, duration: 0 };
  }

  function extendClip(clipId: string) {
    setContinueFrom(clipId);
    setPrompt("");
  }

  async function exportScene() {
    setExporting(true);
    setError("");
    try {
      const result = await client.call("merge_clips", {
        clips: scene.map((item) => ({ job_id: item.clipId, start: item.start, end: item.end || item.duration })),
      });
      if (!result.ok || typeof result.metadata.job_id !== "string") {
        setError(result.error || "merge_failed");
        return;
      }
      const seconds = Number(result.metadata.seconds ?? 0);
      jobInfo.current[result.metadata.job_id] = {
        prompt: `Scene export · ${scene.length} shots`,
        meta: `Edited · ${result.metadata.width}×${result.metadata.height} · ${seconds.toFixed(1)}s`,
      };
      await pullOutput(result.metadata.job_id);
      setView("clips");
    } catch (err) {
      setError(err instanceof Error ? err.message : "merge_failed");
    } finally {
      setExporting(false);
    }
  }

  function selectTask(next: Task) {
    setTask(next);
    const model = models.diffussion.find((name) => TASK_MODEL[next].test(name));
    if (model) setSettings((prev) => ({ ...prev, diffusionModel: model, steps: stepsFor(model, prev.mode) }));
  }

  async function onControlFile(file: File | null) {
    if (!file) {
      setControlId(null);
      setControlName("");
      return;
    }
    try {
      const mediaId = await uploadSlot("control", file);
      setControlId(mediaId);
      setControlName(file.name);
    } catch (err) {
      setError(err instanceof Error ? err.message : "upload_failed");
    }
  }

  function handleResetToDefaults() {
    setSettings({ ...FACTORY_DEFAULTS });
    setPrompt("");
    setSegments([]);
    setSeedText("");
  }

  useEffect(() => {
    if (!busy) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [busy]);

  async function stopJob() {
    if (!jobId) return;
    setStopping(true);
    setDetail("Stopping — the worker halts at its next checkpoint and frees the GPUs");
    pushLog("[■] Stop requested");
    try {
      const result = await client.call("cancel_job", { job_id: jobId });
      if (!result.ok) {
        setStopping(false);
        setError(result.error || "cancel_failed");
      }
    } catch (err) {
      setStopping(false);
      setError(err instanceof Error ? err.message : "cancel_failed");
    }
  }

  async function generate() {
    setError("");
    setBusy(true);
    jobStart.current = Date.now();
    progressHistory.current = [];
    setQueue(null);
    setNow(Date.now());
    setProgress(0.01);
    setStage("queued");
    setDetail("Validating request & checking system VRAM/RAM gates...");
    setActivityLog(["[1%] queued — Validating request & checking system VRAM/RAM gates..."]);
    setStatus("Submitting");
    const [width, height] = computeSize(settings.aspectRatio, settings.megapixels);
    const frames = computeAlignedFrames(settings.durationSec);
    const seed = seedText.trim() === "" ? null : Number(seedText);
    const scenes = segments.map((text) => text.trim()).filter(Boolean);
    try {
      const result = await client.call("generate_video", {
        models: {
          diffusion: settings.diffusionModel,
          video_vae: settings.videoVaeModel,
          audio_vae: settings.audioVaeModel,
          clip: settings.clipModel,
          lora: settings.loraModel,
          controlnet: settings.controlnetEnabled ? settings.controlnetModel || null : null,
        },
        mode: settings.mode,
        steps: settings.steps,
        lora_strength: settings.loraStrength,
        prompt,
        duration_sec: settings.durationSec,
        aspect_ratio: settings.aspectRatio,
        megapixels: settings.megapixels,
        fps: settings.fps,
        ref_image_size: settings.refImageSize,
        sampler_name: settings.samplerName,
        scheduler: settings.scheduler,
        denoise: settings.denoise,
        seed: seed !== null && Number.isFinite(seed) ? Math.floor(seed) : null,
        controlnet_config: {
          enabled: settings.controlnetEnabled,
          control_media_id: controlId,
          strength: settings.controlnetStrength,
          start_percent: settings.controlStart,
          end_percent: settings.controlEnd,
        },
        task,
        segments: scenes,
        continue_from: continueFrom,
        keyframes: task === "i2v" ? { first: keyframes[0].mediaId, last: keyframes[1].mediaId } : {},
        references:
          task === "ref2v"
            ? {
                images: images.map((slot) => slot.mediaId).filter((id): id is string => Boolean(id)),
                videos: videos.flatMap((slot) => (slot.mediaId ? [slot.mediaId] : [])),
                video_audios: videos.flatMap((slot, index) =>
                  slot.mediaId ? [videoAudios[index]?.mediaId ?? null] : [],
                ),
                audios: audios.map((slot) => slot.mediaId).filter((id): id is string => Boolean(id)),
              }
            : {},
      });
      if (!result.ok || typeof result.metadata.job_id !== "string") {
        setBusy(false);
        setError(result.error || "generate_failed");
        pushLog(`[ERR] ${result.error || "generate_failed"}`);
        return;
      }
      const id = result.metadata.job_id;
      const size = task === "i2v" && keyframes[0].mediaId ? "first-frame aspect" : `${width}×${height}`;
      jobInfo.current[id] = {
        prompt: [prompt, ...scenes].join("  →  "),
        meta: `${TASK_LABEL[task]}${continueFrom ? " · extension" : ""} · ${continueFrom ? "matches source" : size} · ${settings.durationSec}s${scenes.length ? ` × ${scenes.length + 1} scenes` : ""}`,
        after: continueFrom ?? undefined,
      };
      setContinueFrom(null);
      activeJob.current = id;
      setJobId(id);
      setDetail(`Job accepted (${width}×${height} · ${frames} frames)`);
      pushLog(`[2%] accepted — Job ${id.slice(0, 8)} (${width}×${height}, ${frames}f)`);
      setStatus(`${width}×${height} · ${frames} frames`);
    } catch (err) {
      setBusy(false);
      setError(err instanceof Error ? err.message : "generate_failed");
    }
  }

  async function flushAll() {
    setFlushing(true);
    setError("");
    let flushError = "";
    try {
      await client.call("flush_all", {});
    } catch (err) {
      flushError = err instanceof Error ? err.message : "flush_failed";
    }
    revokeAll();
    setImages(emptySlots(9));
    setVideos(emptySlots(3));
    setVideoAudios(emptySlots(3));
    setAudios(emptySlots(3));
    setKeyframes(emptySlots(2));
    setControlId(null);
    setControlName("");
    setJobId(null);
    activeJob.current = null;
    setBusy(false);
    setProgress(0);
    setStage("");
    setDetail("");
    setActivityLog([]);
    client.reset();
    try {
      await client.handshake();
      client.connect((event) => handleProgress(event));
      await refreshModels();
      setReady(true);
      if (flushError) {
        setError(`Server flush failed (${flushError}); local media cleared and session reset`);
        setStatus("Flush incomplete");
      } else {
        setStatus("Media & VRAM flushed");
      }
    } catch {
      setReady(false);
      setError("Secure session failed");
    } finally {
      setFlushing(false);
    }
  }

  const pct = Math.round(progress * 100);
  const eta = estimateEta(progressHistory.current, now);
  const extending = clips.find((clip) => clip.id === continueFrom);
  const tab = (id: "clips" | "scene", label: string) => (
    <button
      type="button"
      role="tab"
      aria-selected={view === id}
      onClick={() => setView(id)}
      className={`rounded-full px-4 py-1.5 text-xs font-semibold transition ${
        view === id ? "bg-white text-zinc-900" : "text-zinc-300 hover:bg-white/10"
      }`}
    >
      {label}
    </button>
  );

  return (
    <div className="app-bg flex h-screen flex-col overflow-x-hidden text-zinc-100">
      <header className="glass-bar flex items-center justify-between gap-3 px-5 py-3">
        <div className="flex min-w-0 items-baseline gap-3">
          <h1 className="shrink-0 text-base font-semibold tracking-tight">H3 Studio</h1>
          <p className="hidden truncate text-xs text-zinc-400 sm:block">
            {status}
            {jobId ? ` · ${jobId.slice(0, 8)}` : ""}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2 sm:gap-3">
          {ready && gpus.length > 0 && (
            <div className="hidden items-center gap-2 font-mono text-[10px] md:flex">
              {gpus.map((gpu) => {
                const role = gpu.role || "bypassed_busy";
                return (
                  <span key={gpu.index} className={ROLE_CLASS[role] ?? ROLE_CLASS.bypassed_busy}>
                    GPU{gpu.index} {ROLE_LABEL[role] ?? role}
                  </span>
                );
              })}
            </div>
          )}
          <button
            type="button"
            onClick={() => {
              systemDialog.current?.showModal();
              setSystemOpen(true);
            }}
            className="chip whitespace-nowrap"
          >
            📊 System
          </button>
          <button type="button" onClick={() => settingsDialog.current?.showModal()} className="chip whitespace-nowrap">
            ⚙ Settings
          </button>
          <PanicFlushButton busy={flushing || !ready} onFlush={() => void flushAll()} />
        </div>
      </header>

      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-6xl space-y-5 px-4 py-6">
          <div className="glass inline-flex rounded-full p-1" role="tablist" aria-label="View">
            {tab("clips", `Clips${clips.length ? ` · ${clips.length}` : ""}`)}
            {tab("scene", `Scene builder${scene.length ? ` · ${scene.length}` : ""}`)}
          </div>
          {(busy || (stage && stage !== "done" && !clips.length)) && (
            <section className="glass rounded-3xl p-4" aria-live="polite">
              <div className="flex items-center justify-between text-sm">
                <span className="flex items-center gap-3 font-medium">
                  {busy ? (stopping ? "Stopping…" : queue ? "In queue" : "Generating") : stage}
                  {busy && (
                    <button
                      type="button"
                      disabled={stopping}
                      onClick={() => void stopJob()}
                      className="rounded-full border border-red-400/40 px-2.5 py-0.5 text-[11px] font-medium text-red-200 hover:bg-red-500/15 disabled:opacity-50"
                      title="Stop this generation and free the GPUs"
                    >
                      ■ Stop
                    </button>
                  )}
                </span>
                <span className="font-mono text-xs text-zinc-300">
                  {pct}%
                  {busy && jobStart.current > 0 && (
                    <>
                      <span className="text-zinc-500"> · </span>⏱ {fmtDuration(now - jobStart.current)}
                      <span className="text-zinc-500"> · </span>
                      {queue
                        ? `#${queue.position} in line · ${queue.ahead} job${queue.ahead === 1 ? "" : "s"} ahead`
                        : `ETA ${eta === null ? "estimating…" : `~${fmtDuration(eta)}`}`}
                    </>
                  )}
                </span>
              </div>
              <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-white/10">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-indigo-300 via-white to-teal-200 transition-all duration-300"
                  style={{ width: `${pct}%` }}
                />
              </div>
              <p className="mt-2 truncate text-xs text-zinc-400">{detail || stage}</p>
              {activityLog.length > 0 && (
                <details className="mt-2">
                  <summary className="cursor-pointer text-[11px] text-zinc-500">Activity</summary>
                  <div className="mt-1 max-h-40 space-y-1 overflow-y-auto font-mono text-[11px] text-zinc-400">
                    {activityLog.map((line, idx) => (
                      <div key={`${idx}-${line}`} className="truncate">
                        {line}
                      </div>
                    ))}
                  </div>
                </details>
              )}
            </section>
          )}
          {error && (
            <p role="alert" className="rounded-2xl border border-red-400/30 bg-red-500/10 px-4 py-2 text-sm text-red-200 backdrop-blur">
              {error}
            </p>
          )}
          {view === "scene" ? (
            <SceneBuilder
              items={scene}
              setItems={setScene}
              clips={clips}
              exporting={exporting}
              onExport={() => void exportScene()}
              onExtend={extendClip}
            />
          ) : clips.length === 0 && !busy ? (
            <div className="flex min-h-[40vh] flex-col items-center justify-center text-center">
              <p className="bg-gradient-to-br from-white to-zinc-400 bg-clip-text text-3xl font-semibold tracking-tight text-transparent">
                What do you want to make?
              </p>
              <p className="mt-3 max-w-md text-sm text-zinc-400">
                {task === "i2v"
                  ? "Add an optional first and last frame, describe the shot and its sound, then press →."
                  : "Add ingredients (pictures, videos, audio), reference them in the prompt by tag, then press →."}{" "}
                Use + Extend for more scenes, or build a timeline in Scene builder.
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              {clips.map((clip) => (
                <figure key={clip.id} className="glass group overflow-hidden rounded-3xl">
                  <video src={clip.url} controls className="max-h-[420px] w-full bg-black" />
                  <figcaption className="space-y-2 p-3">
                    <p className="line-clamp-2 text-sm text-zinc-200">{clip.prompt || "Untitled"}</p>
                    <div className="flex flex-wrap items-center gap-2 text-[11px] text-zinc-400">
                      <span className="mr-auto">
                        {clip.meta}
                        {clip.took ? ` · ⏱ ${fmtDuration(clip.took)}` : ""}
                      </span>
                      <button
                        type="button"
                        onClick={() => {
                          setScene((prev) => [...prev, newShot(clip.id)]);
                          setView("scene");
                        }}
                        className="chip"
                      >
                        ＋ Scene
                      </button>
                      <button type="button" disabled={busy} onClick={() => extendClip(clip.id)} className="chip" title="Generate what happens next, starting from this clip's last frames">
                        ↪ Extend
                      </button>
                      <a href={clip.url} download={`h3_${clip.id.slice(0, 8)}.mp4`} className="chip">
                        ↓
                      </a>
                    </div>
                  </figcaption>
                </figure>
              ))}
            </div>
          )}
        </div>
      </main>

      <div className="px-4 pb-4">
        <div className="mx-auto max-w-4xl">
          <Composer
            task={task}
            onTask={selectTask}
            prompt={prompt}
            setPrompt={setPrompt}
            segments={segments}
            setSegments={setSegments}
            images={images}
            videos={videos}
            videoAudios={videoAudios}
            audios={audios}
            keyframes={keyframes}
            settings={settings}
            setSettings={setSettings}
            seedText={seedText}
            setSeedText={setSeedText}
            ready={ready && !flushing}
            busy={busy}
            extending={continueFrom ? extending?.prompt || "a clip" : null}
            onCancelExtend={() => setContinueFrom(null)}
            onPick={(kind, index, file) => void onPick(kind, index, file)}
            onClear={onClear}
            onGenerate={() => void generate()}
            onStop={() => void stopJob()}
            stopping={stopping}
          />
        </div>
      </div>

      <dialog
        ref={systemDialog}
        onClose={() => setSystemOpen(false)}
        onClick={(event) => {
          if (event.target === event.currentTarget) systemDialog.current?.close();
        }}
        className="glass-strong w-[min(56rem,calc(100vw-2rem))] rounded-3xl p-0 text-zinc-100 backdrop:bg-black/60 backdrop:backdrop-blur-sm"
      >
        <div className="flex items-center justify-between border-b border-white/10 px-6 py-4">
          <h2 className="text-base font-semibold">System</h2>
          <button
            type="button"
            onClick={() => systemDialog.current?.close()}
            aria-label="Close system panel"
            className="rounded-full px-2 text-zinc-400 hover:text-white"
          >
            ✕
          </button>
        </div>
        <div className="max-h-[75vh] overflow-y-auto p-6">
          <SystemMonitor client={client} open={systemOpen} />
        </div>
      </dialog>

      <dialog
        ref={settingsDialog}
        onClick={(event) => {
          if (event.target === event.currentTarget) settingsDialog.current?.close();
        }}
        className="glass-strong w-[min(56rem,calc(100vw-2rem))] rounded-3xl p-0 text-zinc-100 backdrop:bg-black/60 backdrop:backdrop-blur-sm"
      >
        <div className="flex items-center justify-between border-b border-white/10 px-6 py-4">
          <h2 className="text-base font-semibold">Settings</h2>
          <button
            type="button"
            onClick={() => settingsDialog.current?.close()}
            aria-label="Close settings"
            className="rounded-full px-2 text-zinc-400 hover:text-white"
          >
            ✕
          </button>
        </div>
        <div className="max-h-[75vh] overflow-y-auto p-6">
          <GenerationControls
            state={settings}
            setState={setSettings}
            availableModels={models}
            onResetDefaults={handleResetToDefaults}
            onRefreshModels={() => void refreshModels()}
            controlName={controlName}
            onControlFile={(file) => void onControlFile(file)}
          />
        </div>
      </dialog>
    </div>
  );
}
