import logging
import os
import queue
import socket
import sys
import threading

logging.disable(logging.CRITICAL)
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import msgpack
import torch
import torch.distributed as dist

from memory_manager import PinnedModelRegistry
from parallel_shard import CTX
from pipeline_h3 import BadReference, JobAborted, MuxFailed, ResourceLimitExceeded, run_job

WRITE_LOCK = threading.Lock()
JOBS_LOCK = threading.Lock()
JOBS = {}
CMD_Q = queue.Queue()
MAX_FRAME = 64 * 1024 * 1024


def init_parallel():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", str(rank)))
    CTX.rank = rank
    CTX.world = world
    CTX.local_rank = local
    cores = os.cpu_count() or 8
    threads = max(2, min(16, cores // max(1, world * 2)))
    for env_key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(env_key, str(threads))
    try:
        torch.set_num_threads(threads)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        for sdp_fn in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_math_sdp"):
            fn = getattr(torch.backends.cuda, sdp_fn, None)
            if callable(fn):
                fn(True)
        CTX.device = torch.device(f"cuda:{local}")
    else:
        CTX.device = torch.device("cpu")
    if world > 1 and torch.cuda.is_available():
        dist.init_process_group(backend="nccl")
    elif world > 1:
        CTX.world = 1
        CTX.rank = 0
        if rank != 0:
            raise SystemExit(0)


def resolve_model_root():
    env = os.environ.get("MODEL_ROOT")
    if env:
        return env
    if os.path.isdir("/app/model"):
        return "/app/model"
    here = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))
    return here


def select_optimal_gpus():
    """Return (csv_devices, count) skipping GPUs heavily occupied by external workloads."""
    preset = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if preset and preset.lower() not in ("all", "none", "-1"):
        devs = [d.strip() for d in preset.split(",") if d.strip()]
        if devs:
            return ",".join(devs), len(devs)
    import subprocess

    rows = []
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.free", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        for line in proc.stdout.decode("ascii", "ignore").splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                rows.append((parts[0], float(parts[1]), float(parts[2])))
    except Exception:
        rows = []
    if not rows:
        return "0", 1
    healthy = [idx for idx, total, free in rows if free >= max(8192.0, total * 0.40)]
    if healthy:
        return ",".join(healthy), len(healthy)
    best = max(rows, key=lambda item: item[2])[0]
    return str(best), 1


def sock_path():
    return os.environ.get("IPC_SOCK", "/dev/shm/minimax_h3.sock")


def read_exact(conn, size):
    buf = bytearray()
    while len(buf) < size:
        chunk = conn.recv(size - len(buf))
        if not chunk:
            raise ConnectionError("eof")
        buf.extend(chunk)
    return bytes(buf)


def read_msg(conn):
    length = int.from_bytes(read_exact(conn, 4), "little")
    if length <= 0 or length > MAX_FRAME:
        raise ConnectionError("frame")
    payload = read_exact(conn, length)
    return msgpack.unpackb(payload, raw=False)


def write_msg(conn, obj):
    payload = msgpack.packb(obj, use_bin_type=True)
    header = len(payload).to_bytes(4, "little")
    with WRITE_LOCK:
        conn.sendall(header + payload)


def mark_abort(job_id):
    with JOBS_LOCK:
        if job_id:
            event = JOBS.get(job_id)
            if event is not None:
                event.set()
            return
        for event in JOBS.values():
            event.set()


def has_active_jobs():
    with JOBS_LOCK:
        return len(JOBS) > 0


def emit(conn, job_id, stage, frac, detail):
    try:
        write_msg(
            conn,
            {
                "type": "progress",
                "job_id": job_id,
                "payload": {"stage": stage, "progress": frac, "detail": detail},
            },
        )
    except Exception:
        return


def client_thread(conn, registry):
    try:
        message = read_msg(conn)
        kind = message.get("type")
        job_id = message.get("job_id") or ""
        if kind == "abort":
            mark_abort(job_id)
            write_msg(conn, {"type": "result", "job_id": job_id, "payload": {"ok": True}})
            return
        if kind == "flush":
            mark_abort(None)
            if not has_active_jobs() and CMD_Q.empty() and not CTX.active():
                registry.flush_vram()
                write_msg(conn, {"type": "result", "job_id": "", "payload": {"ok": True}})
                return
            box = queue.Queue(1)
            CMD_Q.put({"type": "flush", "box": box})
            try:
                resp = box.get(timeout=8.0)
            except queue.Empty:
                resp = {"type": "result", "job_id": "", "payload": {"ok": True}}
            write_msg(conn, resp)
            return
        if kind != "generate":
            write_msg(conn, {"type": "error", "job_id": job_id, "payload": {"error": "bad_op"}})
            return
        payload = message.get("payload") or {}
        if not job_id:
            job_id = str(payload.get("job_id") or "")
        emit(conn, job_id, "worker_accepted", 0.01, "Connected to inference worker")
        event = threading.Event()
        with JOBS_LOCK:
            JOBS[job_id] = event
        box = queue.Queue(1)
        CMD_Q.put(
            {
                "type": "generate",
                "payload": payload,
                "box": box,
                "conn": conn,
                "event": event,
                "job_id": job_id,
            }
        )
        try:
            write_msg(conn, box.get())
        finally:
            with JOBS_LOCK:
                JOBS.pop(job_id, None)
    except Exception:
        try:
            write_msg(conn, {"type": "error", "job_id": "", "payload": {"error": "inference_failed"}})
        except Exception:
            return
    finally:
        try:
            conn.close()
        except Exception:
            return


def serve(path, registry):
    directory = os.path.dirname(path) or "/dev/shm"
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    os.chmod(path, 0o600)
    server.listen(16)
    while True:
        conn, _addr = server.accept()
        threading.Thread(target=client_thread, args=(conn, registry), daemon=True).start()


def error_frame(job_id, code):
    return {"type": "error", "job_id": job_id, "payload": {"error": code}}


def coordinator(registry):
    while True:
        cmd = CMD_Q.get()
        if CTX.active():
            dist.broadcast_object_list(
                [{"type": cmd["type"], "payload": cmd.get("payload")}],
                src=0,
            )
        box = cmd["box"]
        job_id = cmd.get("job_id") or ""
        try:
            if cmd["type"] == "generate":
                payload = cmd["payload"]
                job_id = str(payload.get("job_id") or job_id)

                def _progress(stage, frac, detail, _conn=cmd.get("conn"), _job=job_id):
                    if _conn is not None:
                        emit(_conn, _job, stage, frac, detail)

                run_job(payload, registry, _progress, cmd.get("event"))
                box.put({"type": "result", "job_id": job_id, "payload": {"ok": True}})
            elif cmd["type"] == "flush":
                registry.flush_vram()
                box.put({"type": "result", "job_id": "", "payload": {"ok": True}})
            else:
                box.put(error_frame(job_id, "bad_op"))
        except JobAborted:
            box.put(error_frame(job_id, "aborted"))
        except FileNotFoundError:
            box.put(error_frame(job_id, "model_missing"))
        except BadReference:
            box.put(error_frame(job_id, "bad_reference"))
        except MuxFailed:
            box.put(error_frame(job_id, "mux_failed"))
        except ResourceLimitExceeded as exc:
            box.put(error_frame(job_id, str(exc) or "insufficient_vram"))
        except Exception:
            box.put(error_frame(job_id, "inference_failed"))


def follower(registry):
    while True:
        holder = [None]
        dist.broadcast_object_list(holder, src=0)
        cmd = holder[0] or {}
        kind = cmd.get("type")
        try:
            if kind == "generate":
                run_job(cmd.get("payload") or {}, registry, None, None)
            elif kind == "flush":
                registry.flush_vram()
        except Exception:
            continue


def main():
    try:
        null_out = open(os.devnull, "w", encoding="utf-8")
        null_err = open(os.devnull, "w", encoding="utf-8")
        sys.stdout = null_out
        sys.stderr = null_err
    except OSError:
        pass
    init_parallel()
    registry = PinnedModelRegistry(resolve_model_root())
    if CTX.rank != 0:
        follower(registry)
        return
    threading.Thread(target=serve, args=(sock_path(), registry), daemon=True).start()
    coordinator(registry)


if __name__ == "__main__":
    main()
