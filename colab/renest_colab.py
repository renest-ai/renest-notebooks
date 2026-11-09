#!/usr/bin/env python3
"""Renest Colab Runner: Launches ComfyUI and creates a public WebUI tunnel on Google Colab.

Rules & Architecture:
1. Environment must be pre-restored via `renest restore`.
2. Serves as a host supervisor calling `renest start` and running cloudflared tunnel.
3. Does not modify renest CLI or server internals. Pure standard library implementation.
4. User-facing text in English per project constitution.
5. Clean lifecycle: guarantees termination of both child processes on any exit path.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Optional


CLOUDFLARED_URLS = {
    ("Linux", "x86_64"): "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
    ("Linux", "aarch64"): "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64",
    ("Darwin", "x86_64"): "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64",
    ("Darwin", "arm64"): "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-arm64",
}


def ensure_cloudflared(dest_dir: str = "/tmp") -> str:
    """Ensure cloudflared binary is available, downloading standalone binary if missing."""
    system_bin = shutil.which("cloudflared")
    if system_bin:
        return system_bin

    local_bin = os.path.join(dest_dir, "cloudflared")
    if os.path.isfile(local_bin) and os.access(local_bin, os.X_OK):
        return local_bin

    os_type = platform.system()
    machine = platform.machine()
    download_url = CLOUDFLARED_URLS.get((os_type, machine))
    if not download_url:
        download_url = CLOUDFLARED_URLS[("Linux", "x86_64")]

    print(f"[Renest Colab] Preparing Cloudflare tunnel binary ({os_type}/{machine})...", flush=True)
    try:
        urllib.request.urlretrieve(download_url, local_bin)
        os.chmod(local_bin, 0o755)
    except Exception as e:
        raise RuntimeError(f"Failed to download cloudflared: {e}. Check network connection.") from e

    return local_bin


def extract_tunnel_url(log_line: str) -> Optional[str]:
    """Extract trycloudflare.com URL from cloudflared log line."""
    match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", log_line)
    if match:
        return match.group(0)
    return None


def wait_for_http_ready(port: int, timeout: float = 60.0) -> bool:
    """Poll local HTTP port until ComfyUI answers HTTP 200, or timeout expires."""
    deadline = time.monotonic() + timeout
    endpoints = [f"http://127.0.0.1:{port}/system_stats", f"http://127.0.0.1:{port}/"]
    while time.monotonic() < deadline:
        for url in endpoints:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Renest-Colab-HealthCheck/1.0"})
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    if resp.status == 200:
                        return True
            except (urllib.error.URLError, OSError):
                pass
        time.sleep(1.0)
    return False


def _terminate_proc(proc: Optional[subprocess.Popen], timeout: float = 5.0) -> None:
    """Terminate a subprocess and its entire process group, ensuring zero orphans.

    Reaps direct child processes with wait() to avoid zombie/defunct entries.
    Raises RuntimeError if any process in the group remains alive after SIGKILL.
    """
    if proc is None:
        return

    pid = proc.pid
    try:
        pgid = os.getpgid(pid)
    except (AttributeError, OSError, ProcessLookupError):
        pgid = pid

    def _group_alive() -> bool:
        try:
            os.killpg(pgid, 0)
            return True
        except (AttributeError, OSError, ProcessLookupError):
            return proc.poll() is None

    def _reap_proc() -> None:
        try:
            proc.wait(timeout=0.2)
        except (subprocess.TimeoutExpired, OSError, AttributeError):
            pass

    if not _group_alive():
        _reap_proc()
        return

    # 1. Try graceful SIGINT first to allow clean application teardown
    try:
        try:
            os.killpg(pgid, signal.SIGINT)
        except (AttributeError, OSError, ProcessLookupError):
            proc.send_signal(signal.SIGINT)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if not _group_alive():
                _reap_proc()
                return
            time.sleep(0.1)
    except OSError:
        pass

    # 2. Fall back to SIGTERM
    try:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (AttributeError, OSError, ProcessLookupError):
            proc.terminate()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not _group_alive():
                _reap_proc()
                return
            time.sleep(0.1)
    except OSError:
        pass

    # 3. Force SIGKILL on process group
    try:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (AttributeError, OSError, ProcessLookupError):
            proc.kill()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if not _group_alive():
                _reap_proc()
                return
            time.sleep(0.05)
    except OSError:
        pass

    _reap_proc()
    if _group_alive():
        raise RuntimeError(
            f"Process teardown failed: process group {pgid} (PID {pid}) still active after SIGKILL"
        )


def _cleanup_procs(app_proc: Optional[subprocess.Popen], tunnel_proc: Optional[subprocess.Popen]) -> None:
    """Ensure both child processes are terminated and waited for, preventing orphan processes.

    Raises RuntimeError if teardown fails to terminate either process group.
    """
    errors = []
    for name, p, t in [("tunnel", tunnel_proc, 3.0), ("app", app_proc, 5.0)]:
        try:
            _terminate_proc(p, timeout=t)
        except Exception as e:
            errors.append(f"{name}: {e}")
    if errors:
        msg = f"Process teardown failed: {'; '.join(errors)}"
        print(f"[Renest Colab] Error: {msg}", file=sys.stderr, flush=True)
        raise RuntimeError(msg)


def launch_webui(
    workspace: str,
    port: int = 8188,
    listen: str = "0.0.0.0",
    cloudflared_path: Optional[str] = None,
    timeout_tunnel: float = 30.0,
    timeout_http: float = 60.0,
) -> int:
    """Launch `renest start` and cloudflared tunnel, continuously draining pipes and guarding lifecycle."""
    if not os.path.isdir(workspace):
        print(f"Error: Workspace directory does not exist: {workspace}", file=sys.stderr)
        print("Please run `renest restore` first to restore the environment.", file=sys.stderr)
        return 1

    cf_bin = cloudflared_path or ensure_cloudflared()

    app_proc: Optional[subprocess.Popen] = None
    tunnel_proc: Optional[subprocess.Popen] = None

    def _sig_handler(signum, frame):
        raise KeyboardInterrupt()

    prev_int = signal.signal(signal.SIGINT, _sig_handler)
    prev_term = signal.signal(signal.SIGTERM, _sig_handler)

    try:
        print(f"[Renest Colab] Starting ComfyUI via `renest start` (port {port})...", flush=True)
        start_cmd = [
            "renest",
            "start",
            "--dir",
            workspace,
            "--listen",
            listen,
            "--port",
            str(port),
        ]

        app_proc = subprocess.Popen(
            start_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        tunnel_cmd = [
            cf_bin,
            "tunnel",
            "--url",
            f"http://127.0.0.1:{port}",
        ]

        print("[Renest Colab] Establishing Cloudflare tunnel...", flush=True)
        tunnel_proc = subprocess.Popen(
            tunnel_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        tunnel_url_holder: list[str] = []
        tunnel_found_event = threading.Event()

        # Continuous pipe draining threads to avoid OS 64KB pipe buffer deadlock
        def drain_tunnel():
            if not tunnel_proc or not tunnel_proc.stdout:
                return
            for line in tunnel_proc.stdout:
                if not tunnel_found_event.is_set():
                    url = extract_tunnel_url(line)
                    if url:
                        tunnel_url_holder.append(url)
                        tunnel_found_event.set()

        def drain_app():
            if not app_proc or not app_proc.stdout:
                return
            for line in app_proc.stdout:
                if any(k in line for k in ("Starting server", "Import times", "To see the GUI", "Traceback", "Error")):
                    print(f"[ComfyUI] {line.strip()}", flush=True)

        t_tunnel = threading.Thread(target=drain_tunnel, daemon=True)
        t_app = threading.Thread(target=drain_app, daemon=True)
        t_tunnel.start()
        t_app.start()

        # 1. Wait for tunnel URL
        if not tunnel_found_event.wait(timeout=timeout_tunnel):
            if tunnel_proc.poll() is not None:
                print(f"Error: cloudflared exited prematurely (code {tunnel_proc.returncode}).", file=sys.stderr)
                return tunnel_proc.returncode or 1
            if app_proc.poll() is not None:
                print(f"Error: renest start exited prematurely (code {app_proc.returncode}).", file=sys.stderr)
                return app_proc.returncode or 1
            print(f"Error: Timed out waiting for Cloudflare tunnel URL ({timeout_tunnel}s).", file=sys.stderr)
            return 1

        tunnel_url = tunnel_url_holder[0]

        # 2. Health check ComfyUI HTTP endpoint: MUST succeed before announcing ready
        print(f"[Renest Colab] Waiting for ComfyUI to answer on port {port}...", flush=True)
        http_ready = wait_for_http_ready(port=port, timeout=timeout_http)
        if not http_ready:
            print(f"Error: ComfyUI HTTP health check timed out after {timeout_http}s.", file=sys.stderr)
            if app_proc.poll() is not None:
                print(f"Error: ComfyUI process terminated with exit code {app_proc.returncode}", file=sys.stderr)
                return app_proc.returncode or 1
            return 1

        # 3. Both tunnel URL and HTTP health check passed: declare ready
        banner = [
            "",
            "═" * 70,
            "  🎉 ComfyUI WebUI is ready!",
            f"  👉 Public WebUI URL: {tunnel_url}",
            "     (Unauthenticated public tunnel via Cloudflare. All models loaded)",
            "═" * 70,
            "",
        ]
        print("\n".join(banner), flush=True)

        # 4. Supervise until interrupt or child process termination
        while True:
            if app_proc.poll() is not None:
                print(f"\n[Renest Colab] ComfyUI process exited (code {app_proc.returncode}).", file=sys.stderr)
                return app_proc.returncode
            if tunnel_proc.poll() is not None:
                print(f"\n[Renest Colab] cloudflared process exited (code {tunnel_proc.returncode}).", file=sys.stderr)
                return tunnel_proc.returncode
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[Renest Colab] Shutting down services...", flush=True)
        return 0
    finally:
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)
        # Guarantee both processes are cleaned up on any exit path
        _cleanup_procs(app_proc, tunnel_proc)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Renest Colab Helper: Launches ComfyUI and establishes public tunnel."
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    launch_p = subparsers.add_parser("launch", help="Launch ComfyUI and tunnel")
    launch_p.add_argument(
        "--dir",
        default="/content/ComfyUI",
        help="Workspace directory (default: /content/ComfyUI)",
    )
    launch_p.add_argument(
        "--port",
        type=int,
        default=8188,
        help="ComfyUI listen port (default: 8188)",
    )
    launch_p.add_argument(
        "--listen",
        default="0.0.0.0",
        help="ComfyUI listen host (default: 0.0.0.0)",
    )
    launch_p.add_argument(
        "--timeout-tunnel",
        type=float,
        default=30.0,
        help="Max seconds to wait for tunnel URL (default: 30.0)",
    )
    launch_p.add_argument(
        "--timeout-http",
        type=float,
        default=60.0,
        help="Max seconds to wait for ComfyUI HTTP readiness (default: 60.0)",
    )

    args = parser.parse_args(argv)

    try:
        if args.subcommand == "launch":
            return launch_webui(
                workspace=args.dir,
                port=args.port,
                listen=args.listen,
                timeout_tunnel=args.timeout_tunnel,
                timeout_http=args.timeout_http,
            )
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
