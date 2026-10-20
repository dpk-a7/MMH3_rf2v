import type { ReactElement } from "react";

export type SlotState = {
  mediaId: string | null;
  previewUrl: string | null;
  name: string;
};

export function emptySlots(count: number): SlotState[] {
  return Array.from({ length: count }, () => ({
    mediaId: null,
    previewUrl: null,
    name: "",
  }));
}

export function ReferenceRack({
  images,
  videos,
  videoAudios,
  audios,
  busy,
  onPick,
  onClear,
  onInsert,
}: {
  images: SlotState[];
  videos: SlotState[];
  videoAudios: SlotState[];
  audios: SlotState[];
  busy: boolean;
  onPick: (kind: "image" | "video" | "audio" | "video-audio", index: number, file: File) => void;
  onClear: (kind: "image" | "video" | "audio" | "video-audio", index: number) => void;
  onInsert: (tag: string) => void;
}): ReactElement {
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 space-y-5">
      <div>
        <h2 className="text-lg font-semibold tracking-tight">Reference Rack</h2>
        <p className="text-xs text-zinc-500 mt-1">9 images, 3 videos (each uses its own soundtrack unless you pair other audio), 3 standalone audios.</p>
      </div>
      <SlotGrid
        title="Images"
        accept="image/*"
        slots={images}
        tag={(index) => `<Picture ${index + 1}>`}
        busy={busy}
        onPick={(index, file) => onPick("image", index, file)}
        onClear={(index) => onClear("image", index)}
        onInsert={onInsert}
      />
      <SlotGrid
        title="Videos"
        accept="video/*"
        slots={videos}
        tag={(index) => `<Video ${index + 1}>`}
        busy={busy}
        onPick={(index, file) => onPick("video", index, file)}
        onClear={(index) => onClear("video", index)}
        onInsert={onInsert}
        paired={videoAudios}
        onPairedPick={(index, file) => onPick("video-audio", index, file)}
        onPairedClear={(index) => onClear("video-audio", index)}
      />
      <SlotGrid
        title="Audios"
        accept="audio/*"
        slots={audios}
        tag={(index) => `<Audio ${index + 1}>`}
        busy={busy}
        onPick={(index, file) => onPick("audio", index, file)}
        onClear={(index) => onClear("audio", index)}
        onInsert={onInsert}
      />
    </div>
  );
}

function SlotGrid({
  title,
  accept,
  slots,
  tag,
  busy,
  onPick,
  onClear,
  onInsert,
  paired,
  onPairedPick,
  onPairedClear,
}: {
  title: string;
  accept: string;
  slots: SlotState[];
  tag: (index: number) => string;
  busy: boolean;
  onPick: (index: number, file: File) => void;
  onClear: (index: number) => void;
  onInsert: (tag: string) => void;
  paired?: SlotState[];
  onPairedPick?: (index: number, file: File) => void;
  onPairedClear?: (index: number) => void;
}): ReactElement {
  return (
    <section className="space-y-2">
      <h3 className="text-xs uppercase tracking-wide text-zinc-400">{title}</h3>
      <div className="grid grid-cols-1 gap-2">
        {slots.map((slot, index) => (
          <div key={`${title}-${index}`} className="rounded-lg border border-zinc-800 bg-zinc-950 p-2 space-y-2">
            <div className="flex items-center justify-between gap-2">
              <button
                type="button"
                onClick={() => onInsert(tag(index))}
                className="text-[11px] font-medium text-sky-300 hover:text-sky-200"
              >
                {tag(index)}
              </button>
              {slot.mediaId && (
                <button
                  type="button"
                  onClick={() => onClear(index)}
                  className="text-[11px] text-zinc-500 hover:text-zinc-200"
                >
                  Clear
                </button>
              )}
            </div>
            <input
              type="file"
              accept={accept}
              disabled={busy}
              onChange={(event) => {
                const file = event.target.files?.[0];
                event.target.value = "";
                if (file) onPick(index, file);
              }}
              className="block w-full text-[11px] text-zinc-400 file:mr-2 file:rounded-md file:border-0 file:bg-zinc-800 file:px-2 file:py-1 file:text-zinc-200"
            />
            {slot.previewUrl && accept.startsWith("image") && (
              <img src={slot.previewUrl} alt="" className="h-16 w-full object-cover rounded-md" />
            )}
            {slot.previewUrl && accept.startsWith("video") && (
              <video src={slot.previewUrl} className="h-16 w-full rounded-md bg-black" muted />
            )}
            {slot.previewUrl && accept.startsWith("audio") && (
              <audio src={slot.previewUrl} controls className="w-full h-8" />
            )}
            {slot.name && <p className="text-[10px] text-zinc-500 truncate">{slot.name}</p>}
            {paired && onPairedPick && onPairedClear && (
              <div className="border-t border-zinc-800 pt-2 space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-amber-400">
                    {paired[index]?.mediaId ? "Paired audio" : "Audio: video's own soundtrack"}
                  </span>
                  {paired[index]?.mediaId && (
                    <button
                      type="button"
                      onClick={() => onPairedClear(index)}
                      className="text-[10px] text-zinc-500"
                    >
                      Clear
                    </button>
                  )}
                </div>
                <input
                  type="file"
                  accept="audio/*"
                  disabled={busy}
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    event.target.value = "";
                    if (file) onPairedPick(index, file);
                  }}
                  className="block w-full text-[11px] text-zinc-400 file:mr-2 file:rounded-md file:border-0 file:bg-zinc-800 file:px-2 file:py-1 file:text-zinc-200"
                />
                {paired[index]?.name && (
                  <p className="text-[10px] text-zinc-500 truncate">{paired[index]?.name}</p>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}
