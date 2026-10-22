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
      title="Abort your jobs and zero-wipe your uploads and generated videos. Other users are not affected."
    >
      Flush my media
    </button>
  );
}
