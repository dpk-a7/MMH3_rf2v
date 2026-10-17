import logging
import os
import queue
import socket
import sys
import threading

logging.disable(logging.CRITICAL)
try:
    _NULL_OUT = open(os.devnull, "w", encoding="utf-8")
    _NULL_ERR = open(os.devnull, "w", encoding="utf-8")
    sys.stdout = _NULL_OUT
    sys.stderr = _NULL_ERR
except OSError:
    _NULL_OUT = None
    _NULL_ERR = None

os.environ.setdefault("PYTHONWARNINGS", "ignore")

import msgpack
import torch
import torch.distributed as dist

from memory_manager import PinnedModelRegistry
from parallel_shard import CTX
from pipeline_h3 import BadReference, JobAborted, MuxFailed, run_job

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
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
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
            box = queue.Queue(1)
            CMD_Q.put({"type": "flush", "box": box})
            write_msg(conn, box.get())
            return
        if kind != "generate":
            write_msg(conn, {"type": "error", "job_id": job_id, "payload": {"error": "bad_op"}})
            return
        payload = message.get("payload") or {}
        if not job_id:
            job_id = str(payload.get("job_id") or "")
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
    init_parallel()
    registry = PinnedModelRegistry(resolve_model_root())
    if CTX.rank != 0:
        follower(registry)
        return
    threading.Thread(target=serve, args=(sock_path(),), daemon=True).start()
    coordinator(registry)


if __name__ == "__main__":
    main()
