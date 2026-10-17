import { useEffect, useRef, type ReactElement } from "react";

const PICTURES = [1, 2, 3, 4, 5, 6, 7, 8, 9];
const VIDEOS = [1, 2, 3];
const AUDIOS = [1, 2, 3];

export function PromptTagEditor({
  prompt,
  setPrompt,
  bindInsert,
}: {
  prompt: string;
  setPrompt: (value: string) => void;
  bindInsert: (insert: (tag: string) => void) => void;
}): ReactElement {
  const area = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bindInsert((tag: string) => {
      const node = area.current;
      const value = node?.value ?? prompt;
      const start = node?.selectionStart ?? value.length;
      const end = node?.selectionEnd ?? value.length;
      const spacer = start > 0 && value[start - 1] !== " " && value[start - 1] !== "\n" ? " " : "";
      const next = `${value.slice(0, start)}${spacer}${tag}${value.slice(end)}`;
      setPrompt(next);
      const caret = start + spacer.length + tag.length;
      requestAnimationFrame(() => {
        node?.focus();
        node?.setSelectionRange(caret, caret);
      });
    });
  }, [bindInsert, prompt, setPrompt]);

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold tracking-tight">Prompt</h2>
        <span className="text-[11px] text-zinc-500">Tags address the reference rack in order</span>
      </div>
      <textarea
        ref={area}
        value={prompt}
        onChange={(event) => setPrompt(event.target.value)}
        rows={7}
        className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-sm leading-6 text-zinc-100 resize-y"
        spellCheck={false}
      />
      <div className="space-y-2">
        <TagRow label="Pictures" tags={PICTURES.map((n) => `<Picture ${n}>`)} onInsert={(tag) => insertFromButton(area.current, prompt, setPrompt, tag)} />
        <TagRow label="Videos" tags={VIDEOS.map((n) => `<Video ${n}>`)} onInsert={(tag) => insertFromButton(area.current, prompt, setPrompt, tag)} />
        <TagRow label="Audios" tags={AUDIOS.map((n) => `<Audio ${n}>`)} onInsert={(tag) => insertFromButton(area.current, prompt, setPrompt, tag)} />
      </div>
    </div>
  );
}

function TagRow({
  label,
  tags,
  onInsert,
}: {
  label: string;
  tags: string[];
  onInsert: (tag: string) => void;
}): ReactElement {
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <span className="text-[11px] uppercase tracking-wide text-zinc-500 w-16">{label}</span>
      {tags.map((tag) => (
        <button
          key={tag}
          type="button"
          onClick={() => onInsert(tag)}
          className="px-2 py-1 rounded-md bg-zinc-950 border border-zinc-800 text-[11px] text-zinc-300 hover:border-zinc-600"
        >
          {tag}
        </button>
      ))}
    </div>
  );
}

function insertFromButton(
  node: HTMLTextAreaElement | null,
  prompt: string,
  setPrompt: (value: string) => void,
  tag: string,
): void {
  const value = node?.value ?? prompt;
  const start = node?.selectionStart ?? value.length;
  const end = node?.selectionEnd ?? value.length;
  const spacer = start > 0 && value[start - 1] !== " " && value[start - 1] !== "\n" ? " " : "";
  const next = `${value.slice(0, start)}${spacer}${tag}${value.slice(end)}`;
  setPrompt(next);
  const caret = start + spacer.length + tag.length;
  requestAnimationFrame(() => {
    node?.focus();
    node?.setSelectionRange(caret, caret);
  });
}
