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
      className="px-3 py-2 text-sm font-semibold rounded-lg bg-red-600 hover:bg-red-500 disabled:bg-red-900 disabled:text-red-200 text-white border border-red-400 transition"
      title="Abort jobs, zero-wipe uploads and outputs, flush GPU memory, and revoke local previews"
    >
      🔥 Flush All Media Now
    </button>
  );
}
