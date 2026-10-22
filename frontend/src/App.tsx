import { useCallback, useEffect, useMemo, useRef, useState, type Dispatch, type SetStateAction } from "react";
import {
  SecureClient,
  type EnvelopeResult,
  type GpuInfo,
  type ProgressEvent,
} from "./cryptoClient";
import {
  EMPTY_MODELS,
  FACTORY_DEFAULTS,
  GenerationControls,
  stepsFor,
  type GenerationState,
  type ModelLists,
} from "./components/GenerationControls";
import { PanicFlushButton } from "./components/PanicFlushButton";
import { PromptTagEditor } from "./components/PromptTagEditor";
import { KeyframeRack, ReferenceRack, emptySlots, type SlotState } from "./components/ReferenceRack";
import { computeAlignedFrames, computeSize } from "./math";

const DEFAULT_PROMPT =
  "Bold comic-book ink style, crisp outlines, dramatic rim light. Use <Picture 1> and <Picture 2> as identity references.";

type Task = "ref2v" | "i2v";
type SlotKind = "image" | "video" | "audio" | "video-audio" | "keyframe";

const TASKS: { id: Task; label: string; model: RegExp }[] = [
  { id: "ref2v", label: "Ref2V", model: /omniref/i },
  { id: "i2v", label: "I2V", model: /fasth3_8step_v2/i },
];

const ROLE_CLASS: Record<string, string> = {
  active_full: "border-emerald-700 bg-emerald-500/15 text-emerald-300",
  active_tight_partial: "border-amber-700 bg-amber-500/15 text-amber-300",
  bypassed_busy: "border-zinc-700 bg-zinc-800 text-zinc-400",
};

const ROLE_LABEL: Record<string, string> = {
  active_full: "full",
  active_tight_partial: "tight",
  bypassed_busy: "busy",
};

export function App() {
  const client = useMemo(() => new SecureClient(), []);
  const urls = useRef(new Set<string>());
  const insertRef = useRef<(tag: string) => void>(() => undefined);
  const [ready, setReady] = useState(false);
  const [status, setStatus] = useState("Securing session");
  const [error, setError] = useState("");
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [settings, setSettings] = useState<GenerationState>(FACTORY_DEFAULTS);
  const [models, setModels] = useState<ModelLists>(EMPTY_MODELS);
  const [gpus, setGpus] = useState<GpuInfo[]>([]);
  const [images, setImages] = useState<SlotState[]>(() => emptySlots(9));
  const [videos, setVideos] = useState<SlotState[]>(() => emptySlots(3));
  const [videoAudios, setVideoAudios] = useState<SlotState[]>(() => emptySlots(3));
  const [audios, setAudios] = useState<SlotState[]>(() => emptySlots(3));
  const [keyframes, setKeyframes] = useState<SlotState[]>(() => emptySlots(2));
  const [task, setTask] = useState<Task>("ref2v");
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
  const [outputUrl, setOutputUrl] = useState<string | null>(null);
  const activeJob = useRef<string | null>(null);

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
    setOutputUrl(null);
  }, []);

  const bindInsert = useCallback((insert: (tag: string) => void) => {
    insertRef.current = insert;
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
    if (event.detail) {
      setDetail(event.detail);
      setStatus(`${event.stage}: ${event.detail}`);
      pushLog(`[${Math.round(event.progress * 100)}%] ${event.stage} — ${event.detail}`);
    } else {
      setStatus(event.stage);
    }
    if (event.done) {
      setBusy(false);
      if (event.ok) {
        setStatus("Ready to download");
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
    try {
      const result = await client.call("download_output", { job_id: id });
      if (!result.ok) {
        if (result.error === "not_ready") return;
        setError(result.error || "download_failed");
        return;
      }
      const file = result.files[0];
      if (!file) return;
      const copy = new Uint8Array(new ArrayBuffer(file.data.byteLength));
      copy.set(file.data);
      const blob = new Blob([copy], { type: "video/mp4" });
      const url = track(URL.createObjectURL(blob));
      setOutputUrl((previous) => {
        if (previous) URL.revokeObjectURL(previous);
        return url;
      });
      setStatus("Playback ready");
    } catch (err) {
      setError(err instanceof Error ? err.message : "download_failed");
    }
  }

  async function uploadSlot(
    kind: "image" | "video" | "audio" | "control",
    file: File,
  ): Promise<string> {
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

  function replaceSlot(
    setter: Dispatch<SetStateAction<SlotState[]>>,
    index: number,
    next: SlotState,
  ) {
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
  }

  function selectTask(next: Task) {
    setTask(next);
    // Each task has its own FastH3 checkpoint; switch to it when it is installed.
    const model = models.diffussion.find((name) => TASKS.find((t) => t.id === next)?.model.test(name));
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
    setPrompt(DEFAULT_PROMPT);
    setSeedText("");
  }

  async function generate() {
    setError("");
    setBusy(true);
    setProgress(0.01);
    setStage("queued");
    setDetail("Validating request & checking system VRAM/RAM gates...");
    setActivityLog(["[1%] queued — Validating request & checking system VRAM/RAM gates..."]);
    setStatus("Submitting");
    const [width, height] = computeSize(settings.aspectRatio, settings.megapixels);
    const frames = computeAlignedFrames(settings.durationSec);
    const seed = seedText.trim() === "" ? null : Number(seedText);
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
      activeJob.current = result.metadata.job_id;
      setJobId(result.metadata.job_id);
      setDetail(`Job accepted (${width}×${height} · ${frames} frames)`);
      pushLog(`[2%] accepted — Job ${result.metadata.job_id.slice(0, 8)} (${width}×${height}, ${frames}f)`);
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

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      <header className="border-b border-zinc-800 px-6 py-4 flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-0">
          <h1 className="text-xl font-semibold tracking-tight">MiniMax-H3 {task === "ref2v" ? "Ref2VA" : "I2VA"}</h1>
          <p className="text-xs text-zinc-500 mt-1">
            {status}
            {jobId ? ` · ${jobId.slice(0, 8)}` : ""}
          </p>
          {ready && (
            <div className="mt-2 flex flex-wrap gap-1.5">
              {gpus.length === 0 ? (
                <span className="text-[11px] text-zinc-500">No GPU reported</span>
              ) : (
                gpus.map((gpu) => {
                  const role = gpu.role || "bypassed_busy";
                  const pct = Math.round(gpu.allocated_weight_pct ?? 0);
                  return (
                    <span
                      key={gpu.index}
                      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 font-mono text-[10px] ${ROLE_CLASS[role] ?? ROLE_CLASS.bypassed_busy}`}
                    >
                      GPU{gpu.index} {pct}% {ROLE_LABEL[role] ?? role}
                    </span>
                  );
                })
              )}
            </div>
          )}
        </div>
        <PanicFlushButton busy={flushing || !ready} onFlush={() => void flushAll()} />
      </header>
      <nav className="px-4 pt-4 flex gap-2" role="tablist" aria-label="Task">
        {TASKS.map((t) => (
          <button
            key={t.id}
            type="button"
            role="tab"
            aria-selected={task === t.id}
            disabled={busy}
            onClick={() => selectTask(t.id)}
            className={`px-4 py-2 rounded-lg text-sm font-semibold border ${
              task === t.id
                ? "bg-blue-600 border-blue-500 text-white"
                : "bg-zinc-900 border-zinc-800 text-zinc-300 hover:text-white disabled:text-zinc-600"
            }`}
          >
            {t.label}
          </button>
        ))}
      </nav>
      <main className="grid grid-cols-1 xl:grid-cols-[320px_minmax(0,1fr)_420px] gap-4 p-4">
        {task === "ref2v" ? (
          <ReferenceRack
            images={images}
            videos={videos}
            videoAudios={videoAudios}
            audios={audios}
            busy={!ready || flushing}
            onPick={(kind, index, file) => void onPick(kind, index, file)}
            onClear={onClear}
            onInsert={(tag) => insertRef.current(tag)}
          />
        ) : (
          <KeyframeRack
            keyframes={keyframes}
            busy={!ready || flushing}
            onPick={(index, file) => void onPick("keyframe", index, file)}
            onClear={(index) => onClear("keyframe", index)}
          />
        )}
        <section className="space-y-4">
          <PromptTagEditor prompt={prompt} setPrompt={setPrompt} bindInsert={bindInsert} showTags={task === "ref2v"} />
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 space-y-4">
            <div className="flex flex-wrap items-end justify-between gap-3">
              <label className="text-xs text-zinc-400">
                Seed
                <input
                  value={seedText}
                  onChange={(event) => setSeedText(event.target.value.replace(/[^\d]/g, ""))}
                  placeholder="random"
                  className="mt-1 block w-40 bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-sm text-zinc-100"
                />
              </label>
              <button
                type="button"
                disabled={!ready || busy || flushing}
                onClick={() => void generate()}
                className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:bg-zinc-800 disabled:text-zinc-500 text-sm font-semibold"
              >
                {busy ? "Generating" : "Generate"}
              </button>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between items-center text-xs text-zinc-400">
                <span className="font-medium text-zinc-300">
                  {stage ? `${stage}${detail ? ` — ${detail}` : ""}` : "idle"}
                </span>
                <span className="font-mono text-emerald-400">{Math.round(progress * 100)}%</span>
              </div>
              <div className="h-2 rounded-full bg-zinc-950 overflow-hidden">
                <div
                  className="h-full bg-emerald-500 transition-all duration-300"
                  style={{ width: `${Math.round(progress * 100)}%` }}
                />
              </div>
              {activityLog.length > 0 && (
                <div className="bg-zinc-950/90 border border-zinc-800/80 rounded-lg p-2.5 max-h-32 overflow-y-auto font-mono text-[11px] text-zinc-400 space-y-1">
                  {activityLog.map((line, idx) => (
                    <div key={`${idx}-${line}`} className="truncate">
                      {line}
                    </div>
                  ))}
                </div>
              )}
            </div>
            {error && <p className="text-sm text-red-400">{error}</p>}
            {outputUrl && (
              <video src={outputUrl} controls className="w-full rounded-lg bg-black max-h-[480px]" />
            )}
          </div>
        </section>
        <GenerationControls
          state={settings}
          setState={setSettings}
          availableModels={models}
          onResetDefaults={handleResetToDefaults}
          onRefreshModels={() => void refreshModels()}
          controlName={controlName}
          onControlFile={(file) => void onControlFile(file)}
        />
      </main>
    </div>
  );
}
