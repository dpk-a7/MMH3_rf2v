import type { Dispatch, ReactElement, ReactNode, SetStateAction } from "react";
import { computeAlignedFrames, computeDimensions } from "../math";

export interface GenerationState {
  diffusionModel: string;
  videoVaeModel: string;
  audioVaeModel: string;
  clipModel: string;
  loraModel: string;
  controlnetModel: string;
  mode: "normal" | "turbo";
  steps: number;
  loraStrength: number;
  aspectRatio: string;
  megapixels: number;
  durationSec: number;
  fps: number;
  refImageSize: "match" | "max";
  scheduler: "simple" | "beta" | "normal";
  samplerName: string;
  denoise: number;
  controlnetEnabled: boolean;
  controlnetStrength: number;
  controlStart: number;
  controlEnd: number;
}

export const FACTORY_DEFAULTS: GenerationState = {
  diffusionModel: "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
  videoVaeModel: "minimax_h3_video_vae_fp16.safetensors",
  audioVaeModel: "minimax_h3_audio_vae_fp32.safetensors",
  clipModel: "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
  loraModel: "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
  controlnetModel: "",
  mode: "normal",
  steps: 20,
  loraStrength: 1.0,
  aspectRatio: "16:9 (Widescreen)",
  megapixels: 0.4,
  durationSec: 5.0,
  fps: 24,
  refImageSize: "match",
  scheduler: "simple",
  samplerName: "res_multistep",
  denoise: 1.0,
  controlnetEnabled: false,
  controlnetStrength: 0.8,
  controlStart: 0,
  controlEnd: 0.85,
};

export const ASPECTS = [
  "16:9 (Widescreen)",
  "9:16 (Portrait)",
  "1:1 (Square)",
  "4:3 (Standard)",
  "3:4 (Portrait Standard)",
  "21:9 (Ultra-wide)",
];

export type ModelLists = {
  diffussion: string[];
  vae: string[];
  clip: string[];
  lora: string[];
  controlnet: string[];
};

export const EMPTY_MODELS: ModelLists = {
  diffussion: [],
  vae: [],
  clip: [],
  lora: [],
  controlnet: [],
};

export function GenerationControls({
  state,
  setState,
  availableModels,
  onResetDefaults,
  onRefreshModels,
  controlName,
  onControlFile,
}: {
  state: GenerationState;
  setState: Dispatch<SetStateAction<GenerationState>>;
  availableModels: ModelLists;
  onResetDefaults: () => void;
  onRefreshModels: () => void;
  controlName: string;
  onControlFile: (file: File | null) => void;
}): ReactElement {
  const handleModeChange = (newMode: "normal" | "turbo") => {
    setState((prev) => ({
      ...prev,
      mode: newMode,
      steps: newMode === "turbo" ? 4 : 20,
    }));
  };

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 space-y-6 text-zinc-100">
      <div className="flex flex-wrap items-center justify-between border-b border-zinc-800 pb-3 gap-2">
        <h2 className="text-lg font-semibold tracking-tight">Model & Sampling Controls</h2>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onRefreshModels}
            className="px-2.5 py-1.5 text-xs font-medium rounded-lg bg-zinc-950 hover:bg-zinc-800 text-emerald-400 border border-zinc-800 transition"
            title="Rescan model/diffussion, model/clip, model/vae, model/lora, and model/controlnet subfolders"
          >
            ↻ Rescan Subfolders
          </button>
          <button
            type="button"
            onClick={onResetDefaults}
            className="px-3 py-1.5 text-xs font-medium rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-200 border border-zinc-700 transition"
            title="Reset all models, mode, steps, resolution, and duration to factory defaults"
          >
            ↺ Reset to Defaults
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Field label={`Diffusion Model (model/diffussion/ · ${availableModels.diffussion.length} found)`}>
          <select
            value={state.diffusionModel}
            onChange={(event) => {
              const name = event.target.value;
              // FastH3 checkpoints are distilled to 8 steps.
              setState((prev) => ({
                ...prev,
                diffusionModel: name,
                steps: /fasth3/i.test(name) ? 8 : prev.mode === "turbo" ? 4 : 20,
              }));
            }}
            className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-sm"
          >
            {renderModelOptions(availableModels.diffussion, state.diffusionModel, "diff")}
          </select>
        </Field>
        <Field label={`CLIP / Text Encoder (model/clip/ · ${availableModels.clip.length} found)`}>
          <select
            value={state.clipModel}
            onChange={(event) => setState((prev) => ({ ...prev, clipModel: event.target.value }))}
            className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-sm"
          >
            {renderModelOptions(availableModels.clip, state.clipModel, "clip")}
          </select>
        </Field>
        <Field label={`Video VAE [FP16] (model/vae/ · ${availableModels.vae.length} found)`}>
          <select
            value={state.videoVaeModel}
            onChange={(event) => setState((prev) => ({ ...prev, videoVaeModel: event.target.value }))}
            className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-sm"
          >
            {renderModelOptions(availableModels.vae, state.videoVaeModel, "vvae")}
          </select>
        </Field>
        <Field label={`Audio VAE [FP32] (model/vae/ · ${availableModels.vae.length} found)`}>
          <select
            value={state.audioVaeModel}
            onChange={(event) => setState((prev) => ({ ...prev, audioVaeModel: event.target.value }))}
            className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-sm"
          >
            {renderModelOptions(availableModels.vae, state.audioVaeModel, "avae")}
          </select>
        </Field>
        <Field label={`LoRA Model (model/lora/ · ${availableModels.lora.length} found)`}>
          <select
            value={state.loraModel}
            onChange={(event) => setState((prev) => ({ ...prev, loraModel: event.target.value }))}
            className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-sm"
          >
            {renderModelOptions(availableModels.lora, state.loraModel, "lora")}
          </select>
        </Field>
        <Field label={`ControlNet Model (model/controlnet/ · ${availableModels.controlnet.length} found)`}>
          <select
            value={state.controlnetModel}
            onChange={(event) => {
              const nextModel = event.target.value;
              setState((prev) => ({
                ...prev,
                controlnetModel: nextModel,
                controlnetEnabled: nextModel !== "" ? prev.controlnetEnabled : false,
              }));
            }}
            className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-sm"
          >
            <option value="">None (Disabled)</option>
            {availableModels.controlnet.map((model) => (
              <option key={`cn-top-${model}`} value={model}>
                {model}
              </option>
            ))}
          </select>
        </Field>
      </div>

      <div className="bg-zinc-950/70 border border-zinc-800 rounded-lg p-4 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <span className="text-sm font-medium">Generation Mode</span>
          <div className="inline-flex rounded-lg bg-zinc-900 p-1 border border-zinc-800">
            <button
              type="button"
              onClick={() => handleModeChange("normal")}
              className={`px-3 py-1 text-xs font-medium rounded-md transition ${
                state.mode === "normal" ? "bg-blue-600 text-white" : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              Normal Mode (Default 20 Steps)
            </button>
            <button
              type="button"
              onClick={() => handleModeChange("turbo")}
              className={`px-3 py-1 text-xs font-medium rounded-md transition ${
                state.mode === "turbo"
                  ? "bg-amber-500 text-zinc-950 font-semibold"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              ⚡ Turbo LoRA Mode (Default 4 Steps)
            </button>
          </div>
        </div>

        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-zinc-400">
              Sampling Steps ({state.mode === "turbo" ? "Turbo LoRA Active" : "Full Quality"})
            </span>
            <span className="font-mono font-semibold text-zinc-100">{state.steps} steps</span>
          </div>
          <input
            type="range"
            min={1}
            max={state.mode === "turbo" ? 20 : 50}
            step={1}
            value={state.steps}
            onChange={(event) => setState((prev) => ({ ...prev, steps: Number(event.target.value) }))}
            className="w-full accent-blue-500 cursor-pointer"
          />
          <div className="flex justify-between text-[10px] text-zinc-500 mt-0.5">
            <span>1</span>
            <span>{state.mode === "turbo" ? "Recommended: 4" : "Recommended: 20"}</span>
            <span>{state.mode === "turbo" ? "20" : "50"}</span>
          </div>
        </div>

        {state.mode === "turbo" && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2 border-t border-zinc-800/80">
            <Field label={`Turbo LoRA File (model/lora/ · ${availableModels.lora.length} found)`} tone="amber">
              <select
                value={state.loraModel}
                onChange={(event) => setState((prev) => ({ ...prev, loraModel: event.target.value }))}
                className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-1.5 text-xs"
              >
                {renderModelOptions(availableModels.lora, state.loraModel, "turbo-lora")}
              </select>
            </Field>
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-amber-400">LoRA Strength</span>
                <span className="font-mono">{state.loraStrength.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min={0.1}
                max={1.5}
                step={0.05}
                value={state.loraStrength}
                onChange={(event) =>
                  setState((prev) => ({ ...prev, loraStrength: Number(event.target.value) }))
                }
                className="w-full accent-amber-500"
              />
            </div>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Field label="Aspect Ratio">
          <select
            value={state.aspectRatio}
            onChange={(event) => setState((prev) => ({ ...prev, aspectRatio: event.target.value }))}
            className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-sm"
          >
            {ASPECTS.map((aspect) => (
              <option key={aspect} value={aspect}>
                {aspect}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Output FPS">
          <select
            value={state.fps}
            onChange={(event) => setState((prev) => ({ ...prev, fps: Number(event.target.value) }))}
            className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-sm"
          >
            {[24, 30, 60].map((fps) => (
              <option key={fps} value={fps}>
                {fps} fps
              </option>
            ))}
          </select>
        </Field>
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-zinc-400">Megapixels & Output Size</span>
            <span className="font-mono text-emerald-400">
              {state.megapixels.toFixed(1)} MP → {computeDimensions(state.aspectRatio, state.megapixels)}
            </span>
          </div>
          <input
            type="range"
            min={0.2}
            max={2}
            step={0.1}
            value={state.megapixels}
            onChange={(event) => setState((prev) => ({ ...prev, megapixels: Number(event.target.value) }))}
            className="w-full accent-emerald-500"
          />
        </div>
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-zinc-400">Duration (24fps Causal Alignment)</span>
            <span className="font-mono text-emerald-400">
              {state.durationSec.toFixed(1)}s → {computeAlignedFrames(state.durationSec)} frames
            </span>
          </div>
          <input
            type="range"
            min={1}
            max={15}
            step={0.5}
            value={state.durationSec}
            onChange={(event) => setState((prev) => ({ ...prev, durationSec: Number(event.target.value) }))}
            className="w-full accent-emerald-500"
          />
        </div>
        <Field label="Reference Image Size">
          <select
            value={state.refImageSize}
            onChange={(event) =>
              setState((prev) => ({ ...prev, refImageSize: event.target.value as "match" | "max" }))
            }
            className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-sm"
          >
            <option value="match">match</option>
            <option value="max">max (2048 short edge)</option>
          </select>
        </Field>
        <Field label="Scheduler">
          <select
            value={state.scheduler}
            onChange={(event) =>
              setState((prev) => ({
                ...prev,
                scheduler: event.target.value as GenerationState["scheduler"],
              }))
            }
            className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-sm"
          >
            <option value="simple">simple</option>
            <option value="beta">beta</option>
            <option value="normal">normal</option>
          </select>
        </Field>
        <Field label="Sampler">
          <input
            readOnly
            value={state.samplerName}
            className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-sm text-zinc-400"
          />
        </Field>
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-zinc-400">Denoise</span>
            <span className="font-mono">{state.denoise.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={state.denoise}
            onChange={(event) => setState((prev) => ({ ...prev, denoise: Number(event.target.value) }))}
            className="w-full accent-zinc-200"
          />
        </div>
      </div>

      <div className="border border-zinc-800 rounded-lg p-4 space-y-3">
        <label className="flex items-center justify-between text-sm">
          <span>ControlNet</span>
          <input
            type="checkbox"
            checked={state.controlnetEnabled}
            onChange={(event) => {
              const checked = event.target.checked;
              setState((prev) => ({
                ...prev,
                controlnetEnabled: checked,
                controlnetModel:
                  checked && !prev.controlnetModel && availableModels.controlnet.length > 0
                    ? availableModels.controlnet[0]
                    : prev.controlnetModel,
              }));
            }}
            className="accent-emerald-500"
          />
        </label>
        {state.controlnetEnabled && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Field label={`ControlNet File (model/controlnet/ · ${availableModels.controlnet.length} found)`}>
              <select
                value={state.controlnetModel}
                onChange={(event) => setState((prev) => ({ ...prev, controlnetModel: event.target.value }))}
                className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-sm"
              >
                <option value="">Select a ControlNet</option>
                {availableModels.controlnet.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Control Media">
              <input
                type="file"
                accept="image/*,video/*"
                onChange={(event) => {
                  const file = event.target.files?.[0] ?? null;
                  event.target.value = "";
                  onControlFile(file);
                }}
                className="block w-full text-[11px] text-zinc-400 file:mr-2 file:rounded-md file:border-0 file:bg-zinc-800 file:px-2 file:py-1 file:text-zinc-200"
              />
              {controlName && <p className="text-[10px] text-zinc-500 truncate mt-1">{controlName}</p>}
            </Field>
            <Slider
              label="Strength"
              min={0}
              max={2}
              step={0.05}
              value={state.controlnetStrength}
              onChange={(value) => setState((prev) => ({ ...prev, controlnetStrength: value }))}
            />
            <Slider
              label="Start"
              min={0}
              max={0.95}
              step={0.05}
              value={state.controlStart}
              onChange={(value) => setState((prev) => ({ ...prev, controlStart: value }))}
            />
            <Slider
              label="End"
              min={0.05}
              max={1}
              step={0.05}
              value={state.controlEnd}
              onChange={(value) => setState((prev) => ({ ...prev, controlEnd: value }))}
            />
          </div>
        )}
      </div>
    </div>
  );
}

function renderModelOptions(options: string[], current: string, prefix: string): ReactElement[] {
  const merged = withCurrent(options, current);
  return merged.map((model) => {
    const onDisk = options.includes(model);
    return (
      <option key={`${prefix}-${model}`} value={model}>
        {onDisk ? model : `${model} (not found in folder)`}
      </option>
    );
  });
}

function withCurrent(options: string[], current: string): string[] {
  if (current && !options.includes(current)) {
    return [...options, current];
  }
  return options.length > 0 ? options : current ? [current] : [];
}

function Field({
  label,
  tone = "zinc",
  children,
}: {
  label: string;
  tone?: "zinc" | "amber";
  children: ReactNode;
}): ReactElement {
  return (
    <div>
      <label className={`block text-xs mb-1 ${tone === "amber" ? "text-amber-400" : "text-zinc-400"}`}>
        {label}
      </label>
      {children}
    </div>
  );
}

function Slider({
  label,
  min,
  max,
  step,
  value,
  onChange,
}: {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (value: number) => void;
}): ReactElement {
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-zinc-400">{label}</span>
        <span className="font-mono">{value.toFixed(2)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="w-full accent-emerald-500"
      />
    </div>
  );
}
