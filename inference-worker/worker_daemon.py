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

from memory_manager import ModelRegistry, ResourceLimitExceeded
from pipeline_h3 import BadReference, JobAborted, MuxFailed, run_job

WRITE_LOCK = threading.Lock()
JOBS_LOCK = threading.Lock()
JOBS = {}
CURRENT_JOB = None
CMD_Q = queue.Queue()
MAX_FRAME = 64 * 1024 * 1024


def init_runtime():
    cores = os.cpu_count() or 8
    threads = max(4, min(16, cores // 2))
    for env_key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(env_key, str(threads))
    try:
        torch.set_num_threads(threads)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")


def resolve_model_root():
    env = os.environ.get("MODEL_ROOT")
    if env:
        return env
    if os.path.isdir("/app/model"):
        return "/app/model"
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))


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


def _interrupt_comfy():
    if "comfy.model_management" not in sys.modules:
        return
    try:
        import comfy.model_management as mm

        mm.interrupt_current_processing(True)
    except Exception:
        return


def mark_abort(job_id):
    with JOBS_LOCK:
        if job_id:
            event = JOBS.get(job_id)
            if event is not None:
                event.set()
            if job_id == CURRENT_JOB:
                _interrupt_comfy()
            return
        for event in JOBS.values():
            event.set()
        _interrupt_comfy()


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


def client_thread(conn):
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
            CMD_Q.put({"type": "flush", "box": None})
            write_msg(conn, {"type": "result", "job_id": "", "payload": {"ok": True}})
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
        CMD_Q.put({"type": "generate", "payload": payload, "box": box, "conn": conn, "event": event, "job_id": job_id})
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


def serve(path):
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
        threading.Thread(target=client_thread, args=(conn,), daemon=True).start()


def error_frame(job_id, code):
    return {"type": "error", "job_id": job_id, "payload": {"error": code}}


def _is_oom(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def coordinator(registry):
    """Runs generate and flush commands strictly one at a time."""
    global CURRENT_JOB
    while True:
        cmd = CMD_Q.get()
        if cmd["type"] == "flush":
            try:
                registry.flush_vram()
            except Exception:
                pass
            continue
        box = cmd["box"]
        job_id = str((cmd.get("payload") or {}).get("job_id") or cmd.get("job_id") or "")
        if cmd["event"].is_set():
            box.put(error_frame(job_id, "aborted"))
            continue

        def _progress(stage, frac, detail, _conn=cmd.get("conn"), _job=job_id):
            emit(_conn, _job, stage, frac, detail)

        with JOBS_LOCK:
            CURRENT_JOB = job_id
        try:
            run_job(cmd["payload"], registry, _progress, cmd["event"])
            box.put({"type": "result", "job_id": job_id, "payload": {"ok": True}})
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
        except Exception as exc:
            if _is_oom(exc):
                registry.trim_all_memory()
                box.put(error_frame(job_id, "insufficient_vram"))
            else:
                box.put(error_frame(job_id, "inference_failed"))
        finally:
            with JOBS_LOCK:
                CURRENT_JOB = None


def main():
    try:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
    except OSError:
        pass
    init_runtime()
    registry = ModelRegistry(resolve_model_root())
    threading.Thread(target=serve, args=(sock_path(),), daemon=True).start()
    coordinator(registry)


if __name__ == "__main__":
    main()
