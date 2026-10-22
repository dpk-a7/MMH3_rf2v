import type { ReactElement } from "react";

export function PanicFlushButton({
  busy,
  onFlush,
}: {
  busy: boolean;
  onFlush: () => void;
}): ReactElement {
  return (
    <button
      type="button"
      onClick={onFlush}
      disabled={busy}
      className="whitespace-nowrap rounded-full border border-red-500/40 px-3 py-1.5 text-xs font-medium text-red-300 transition hover:bg-red-500/15 disabled:text-red-900"
      title="Abort jobs, zero-wipe uploads and outputs, flush GPU memory, and revoke local previews"
    >
      Flush media
    </button>
  );
}
