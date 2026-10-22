# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp>=2.3,<3", "httpx>=0.27"]
# ///
"""MCP server that lets an agent generate MiniMax-H3 videos through the app's agent API.

Run on the agent's machine (stdio transport):

    MMH3_URL=https://192.168.0.112:8443 MMH3_AGENT_TOKEN=... MMH3_INSECURE_TLS=1 uv run mcp/mmh3_mcp.py

The server generates its TLS certificate itself, so either point MMH3_CA_CERT at a copy of it or set
MMH3_INSECURE_TLS=1. File arguments are paths on the agent's machine; they are uploaded for the job.
"""

import asyncio
import os
from pathlib import Path

import httpx
from mcp.server.mcpserver import Context, MCPServer

BASE = os.environ.get("MMH3_URL", "https://localhost:8443").rstrip("/") + "/api/v1/agent"
TOKEN = os.environ.get("MMH3_AGENT_TOKEN", "")
VERIFY = os.environ.get("MMH3_CA_CERT") or not os.environ.get("MMH3_INSECURE_TLS")

mcp = MCPServer("mmh3-video")


def _client():
    return httpx.AsyncClient(base_url=BASE, headers={"Authorization": f"Bearer {TOKEN}"}, verify=VERIFY,
                             timeout=httpx.Timeout(600.0, connect=15.0))


async def _json(response):
    if response.status_code >= 400:
        raise RuntimeError(f"{response.status_code}: {response.text}")
    return response.json()


async def _upload(client, path, kind):
    file = Path(path).expanduser()
    response = await client.post("/media", params={"kind": kind, "filename": file.name}, content=file.read_bytes())
    return (await _json(response))["media_id"]


@mcp.tool()
async def list_models() -> dict:
    """Model files installed on the server, the defaults, and GPU status."""
    async with _client() as client:
        return await _json(await client.get("/models"))


@mcp.tool()
async def generate_video(
    prompt: str,
    ctx: Context,
    task: str = "ref2v",
    reference_images: list[str] | None = None,
    reference_videos: list[str] | None = None,
    reference_audios: list[str] | None = None,
    first_frame: str | None = None,
    last_frame: str | None = None,
    segments: list[str] | None = None,
    duration_sec: float = 5.0,
    megapixels: float = 0.4,
    aspect_ratio: str = "16:9",
    seed: int | None = None,
    continue_from: str | None = None,
    output_path: str | None = None,
    wait: bool = True,
) -> dict:
    """Generate a video with synchronized audio (24 fps, MP4).

    task: "ref2v" uses reference_images/videos/audios (address them in the prompt as <Picture 1>, <Video 1>,
    <Audio 1> in order; a reference video's own soundtrack is used). "i2v" animates from an optional first_frame
    to an optional last_frame (neither = text-to-video); the output takes the first frame's aspect ratio.
    segments: extra prompts; each continues from the previous segment's last frames and all are merged into
    one video (up to 5). continue_from: job_id of a finished clip to extend; the new clip starts from its last
    frames, matches its size and can be joined to it with merge_clips. duration_sec is per segment (1-15). megapixels 0.2-2.0 sets the size
    (0.4 = 864x480 at 16:9). aspect_ratio: 16:9, 9:16, 1:1, 4:3, 3:4 or 21:9.
    With wait=True the call returns after the MP4 is saved to output_path (default ./mmh3_<job>.mp4);
    a 5 s clip takes about 2-5 minutes. With wait=False it returns the job_id for get_job/download_video.
    """
    async with _client() as client:
        refs = {
            "images": [await _upload(client, p, "image") for p in reference_images or []],
            "videos": [await _upload(client, p, "video") for p in reference_videos or []],
            "audios": [await _upload(client, p, "audio") for p in reference_audios or []],
        }
        keyframes = {
            "first": await _upload(client, first_frame, "image") if first_frame else None,
            "last": await _upload(client, last_frame, "image") if last_frame else None,
        }
        body = {"task": task, "prompt": prompt, "duration_sec": duration_sec, "megapixels": megapixels,
                "aspect_ratio": aspect_ratio, "segments": segments or [], "seed": seed, "continue_from": continue_from,
                "references": refs, "keyframes": keyframes}
        job_id = (await _json(await client.post("/jobs", json=body)))["job_id"]
        if not wait:
            return {"job_id": job_id, "status": "queued"}
        while True:
            job = await _json(await client.get(f"/jobs/{job_id}"))
            await ctx.report_progress(float(job.get("progress") or 0.0), 1.0, job.get("detail") or job.get("stage"))
            if job.get("done"):
                break
            await asyncio.sleep(5)
        if not job.get("ok"):
            return {"job_id": job_id, "status": job.get("status"), "error": job.get("error") or job.get("detail")}
    return await download_video(job_id, output_path)


@mcp.tool()
async def merge_clips(clips: list[dict], output_path: str | None = None) -> dict:
    """Join finished clips into one MP4, in order, each trimmed to [start, end] seconds.

    clips: [{"job_id": "...", "start": 0.0, "end": 5.0}, ...]; sizes are fitted to the first clip.
    """
    async with _client() as client:
        job_id = (await _json(await client.post("/merge", json={"clips": clips})))["job_id"]
    return await download_video(job_id, output_path)


@mcp.tool()
async def cancel_job(job_id: str) -> dict:
    """Stop a queued or running job; a running job halts at its next checkpoint and the GPUs are freed."""
    async with _client() as client:
        return await _json(await client.post(f"/jobs/{job_id}/cancel"))


@mcp.tool()
async def get_job(job_id: str) -> dict:
    """Status of a job: status, stage, progress (0-1), detail, done, ok, error."""
    async with _client() as client:
        return await _json(await client.get(f"/jobs/{job_id}"))


@mcp.tool()
async def download_video(job_id: str, output_path: str | None = None) -> dict:
    """Save a finished job's MP4 to output_path (default ./mmh3_<job>.mp4) and return the path."""
    async with _client() as client:
        response = await client.get(f"/jobs/{job_id}/video")
        if response.status_code >= 400:
            raise RuntimeError(f"{response.status_code}: {response.text}")
    path = Path(output_path or f"mmh3_{job_id[:8]}.mp4").expanduser().resolve()
    path.write_bytes(response.content)
    return {"job_id": job_id, "status": "done", "video_path": str(path), "bytes": len(response.content)}


if __name__ == "__main__":
    mcp.run()
