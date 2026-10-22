import { useEffect, useState, type ReactElement } from "react";
import type { SecureClient } from "../cryptoClient";

type GpuStat = {
  index: number;
  name: string;
  util_percent: number | null;
  mem_used_mb: number | null;
  mem_total_mb: number | null;
  temp_c: number | null;
  power_w: number | null;
  power_limit_w: number | null;
  fan_percent: number | null;
};

type Stats = {
  gpus: GpuStat[];
  cpu: { percent: number; cores: number; temp_c: number | null; load1: number | null };
  ram: { container_used: number | null; container_limit: number | null; host_used: number; host_total: number };
  queue: { running: number; queued: number };
};

const GB = 1024 ** 3;

export function SystemMonitor({ client, open }: { client: SecureClient; open: boolean }): ReactElement {
  const [stats, setStats] = useState<Stats | null>(null);
  const [failed, setFailed] = useState(false);

  // Poll only while the panel is open.
  useEffect(() => {
    if (!open) return;
    let stop = false;
    const tick = async () => {
      try {
        const result = await client.call("system_stats", {});
        if (stop) return;
        setFailed(!result.ok);
        if (result.ok) setStats(result.metadata as unknown as Stats);
      } catch {
        if (!stop) setFailed(true);
      }
    };
    void tick();
    const timer = window.setInterval(() => void tick(), 2000);
    return () => {
      stop = true;
      window.clearInterval(timer);
    };
  }, [client, open]);

  if (!stats) {
    return <p className="text-sm text-zinc-400">{failed ? "System stats are unavailable." : "Reading system stats…"}</p>;
  }
  const { cpu, ram, queue } = stats;
  const limit = ram.container_limit && ram.container_limit < ram.host_total * 4 ? ram.container_limit : null;

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Tile label="Generating" value={String(queue.running)} />
        <Tile label="In queue" value={String(queue.queued)} />
        <Tile label="CPU" value={`${cpu.percent.toFixed(0)}%`} hint={`${cpu.cores} cores · load ${cpu.load1?.toFixed(2) ?? "—"}`} />
        <Tile label="CPU temp" value={cpu.temp_c === null ? "—" : `${cpu.temp_c.toFixed(0)}°C`} tone={heat(cpu.temp_c, 75, 90)} />
      </div>

      <section className="space-y-2">
        <h3 className="text-xs uppercase tracking-wide text-zinc-400">Memory</h3>
        {limit && ram.container_used !== null && (
          <Meter label="App (container) RAM" used={ram.container_used / GB} total={limit / GB} unit="GB" />
        )}
        <Meter label="Host RAM" used={ram.host_used / GB} total={ram.host_total / GB} unit="GB" />
        <Meter label="CPU" used={cpu.percent} total={100} unit="%" />
      </section>

      <section className="space-y-3">
        <h3 className="text-xs uppercase tracking-wide text-zinc-400">GPUs</h3>
        {stats.gpus.length === 0 && <p className="text-sm text-zinc-500">No GPU reported.</p>}
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          {stats.gpus.map((gpu) => (
            <div key={gpu.index} className="rounded-2xl border border-white/10 bg-white/[0.03] p-3 space-y-2">
              <div className="flex items-center justify-between gap-2 text-sm">
                <span className="truncate font-medium">
                  GPU{gpu.index} · {gpu.name}
                </span>
                <span className={`shrink-0 font-mono text-xs ${TONE[heat(gpu.temp_c, 80, 88)]}`}>
                  {gpu.temp_c === null ? "—" : `${gpu.temp_c}°C`}
                </span>
              </div>
              <Meter label="Utilisation" used={gpu.util_percent ?? 0} total={100} unit="%" />
              <Meter label="VRAM" used={(gpu.mem_used_mb ?? 0) / 1024} total={(gpu.mem_total_mb ?? 0) / 1024} unit="GB" />
              <p className="text-[11px] text-zinc-500">
                Power {gpu.power_w === null ? "—" : `${gpu.power_w.toFixed(0)} W`}
                {gpu.power_limit_w ? ` / ${gpu.power_limit_w.toFixed(0)} W` : ""}
                {gpu.fan_percent !== null ? ` · fan ${gpu.fan_percent}%` : ""}
              </p>
            </div>
          ))}
        </div>
      </section>
      <p className="text-[11px] text-zinc-500">Refreshes every 2 s while open.</p>
    </div>
  );
}

const TONE = { ok: "text-emerald-300", warm: "text-amber-300", hot: "text-red-300" } as const;

function heat(temp: number | null, warm: number, hot: number): keyof typeof TONE {
  if (temp === null || temp < warm) return "ok";
  return temp < hot ? "warm" : "hot";
}

function Tile({ label, value, hint, tone = "ok" }: { label: string; value: string; hint?: string; tone?: keyof typeof TONE }): ReactElement {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.03] px-3 py-2">
      <p className="text-[11px] text-zinc-500">{label}</p>
      <p className={`text-lg font-semibold ${tone === "ok" ? "text-zinc-100" : TONE[tone]}`}>{value}</p>
      {hint && <p className="text-[10px] text-zinc-500">{hint}</p>}
    </div>
  );
}

function Meter({ label, used, total, unit }: { label: string; used: number; total: number; unit: string }): ReactElement {
  const pct = total > 0 ? Math.min(100, (used / total) * 100) : 0;
  const bar = pct > 90 ? "from-red-400 to-red-300" : pct > 70 ? "from-amber-300 to-amber-200" : "from-indigo-300 to-teal-200";
  return (
    <div>
      <div className="flex justify-between text-[11px] text-zinc-400">
        <span>{label}</span>
        <span className="font-mono">
          {unit === "%" ? `${used.toFixed(0)}%` : `${used.toFixed(1)} / ${total.toFixed(1)} ${unit}`}
        </span>
      </div>
      <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-white/10" role="meter" aria-label={label} aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100}>
        <div className={`h-full rounded-full bg-gradient-to-r ${bar}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
