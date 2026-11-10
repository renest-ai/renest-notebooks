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
from collections import deque
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
    if os.path.isfile(local_bin) and os.access(local_bin, os.X_OK) and os.path.getsize(local_bin) > 1_000_000:
        return local_bin

    os_type = platform.system()
    machine = platform.machine()
    download_url = CLOUDFLARED_URLS.get((os_type, machine))
    if not download_url:
        download_url = CLOUDFLARED_URLS[("Linux", "x86_64")]

    print(f"[Renest Colab] Downloading Cloudflare tunnel binary ({os_type}/{machine})...", flush=True)
    tmp_bin = local_bin + ".tmp"
    try:
        urllib.request.urlretrieve(download_url, tmp_bin)
        os.chmod(tmp_bin, 0o755)
        os.replace(tmp_bin, local_bin)
    except Exception as e:
        raise RuntimeError(f"Failed to download cloudflared: {e}. Check network connection.") from e

    return local_bin


def extract_tunnel_url(log_line: str) -> Optional[str]:
    """Extract trycloudflare.com URL from cloudflared log line."""
    match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", log_line)
    if match:
        return match.group(0)
    return None


def wait_for_http_ready(port: int, timeout: float = 120.0) -> bool:
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


def wait_for_public_url(url: str, timeout: float = 20.0) -> tuple[bool, str]:
    """Poll a public tunnel URL until DNS + edge routing return HTTP 200 end-to-end."""
    deadline = time.monotonic() + timeout
    probe_url = f"{url.rstrip('/')}/system_stats"
    last_err = "timed out waiting for HTTP 200"
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(probe_url, headers={"User-Agent": "Renest-Colab-HealthCheck/1.0"})
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                if 200 <= resp.status < 400:
                    return True, ""
                last_err = f"HTTP {resp.status}"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # Non-ComfyUI root/fallback path may still answer 404 on /system_stats while / answers 200
                try:
                    req_root = urllib.request.Request(url.rstrip("/") + "/", headers={"User-Agent": "Renest-Colab-HealthCheck/1.0"})
                    with urllib.request.urlopen(req_root, timeout=4.0) as resp2:
                        if 200 <= resp2.status < 400:
                            return True, ""
                except Exception as e2:
                    last_err = str(e2)
            else:
                last_err = f"HTTP {e.code}"
        except (urllib.error.URLError, OSError) as e:
            last_err = str(e)
        time.sleep(1.0)
    return False, last_err


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


def _dump_tail(label: str, tail: deque[str]) -> None:
    if not tail:
        return
    print(f"--- Last {len(tail)} lines from {label} ---", file=sys.stderr, flush=True)
    for line in tail:
        print(f"  [{label}] {line}", file=sys.stderr, flush=True)


def _stop_tunnel(tunnel_proc: Optional[subprocess.Popen]) -> None:
    """Terminate an optional tunnel process without failing the caller."""
    if tunnel_proc is None:
        return
    try:
        _terminate_proc(tunnel_proc, timeout=2.0)
    except Exception as exc:
        print(f"[Renest Colab] Warning during tunnel cleanup: {exc}", file=sys.stderr, flush=True)


def launch_webui(
    workspace: str,
    port: int = 8188,
    listen: Optional[str] = None,
    cloudflared_path: Optional[str] = None,
    timeout_tunnel: float = 45.0,
    timeout_http: float = 120.0,
    allow_local_fallback: bool = False,
) -> int:
    """Launch `renest start` and optional cloudflared tunnel, draining pipes and guarding lifecycle."""
    if not os.path.isdir(workspace):
        print(f"Error: Workspace directory does not exist: {workspace}", file=sys.stderr, flush=True)
        print("Please run `renest restore` first to restore the environment.", file=sys.stderr, flush=True)
        return 1

    cf_bin: Optional[str] = cloudflared_path
    if cf_bin is None:
        try:
            cf_bin = ensure_cloudflared()
        except RuntimeError as exc:
            if not allow_local_fallback:
                raise
            print(f"[Renest Colab] Warning: Cloudflare binary unavailable ({exc}); continuing with local port {port}.", file=sys.stderr, flush=True)
            cf_bin = None

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
            "--port",
            str(port),
        ]
        if listen:
            start_cmd.extend(["--listen", listen])

        app_proc = subprocess.Popen(
            start_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        app_tail: deque[str] = deque(maxlen=40)
        tunnel_tail: deque[str] = deque(maxlen=30)
        tunnel_url_holder: list[str] = []
        tunnel_found_event = threading.Event()

        def drain_app():
            if not app_proc or not app_proc.stdout:
                return
            for line in app_proc.stdout:
                clean = line.rstrip()
                if clean:
                    app_tail.append(clean)
                if any(
                    k in line
                    for k in (
                        "[start]",
                        "Starting server",
                        "Import times",
                        "To see the GUI",
                        "Traceback",
                        "Error",
                        "FATAL",
                    )
                ):
                    print(f"[ComfyUI] {clean}", flush=True)

        t_app = threading.Thread(target=drain_app, daemon=True)
        t_app.start()

        t_tunnel: Optional[threading.Thread] = None
        if cf_bin is not None:
            tunnel_cmd = [
                cf_bin,
                "tunnel",
                "--protocol",
                "http2",
                "--no-autoupdate",
                "--url",
                f"http://127.0.0.1:{port}",
            ]
            print("[Renest Colab] Establishing Cloudflare tunnel (HTTP/2)...", flush=True)
            tunnel_proc = subprocess.Popen(
                tunnel_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )

            def drain_tunnel():
                if not tunnel_proc or not tunnel_proc.stdout:
                    return
                for line in tunnel_proc.stdout:
                    clean = line.rstrip()
                    if clean:
                        tunnel_tail.append(clean)
                    if not tunnel_found_event.is_set():
                        url = extract_tunnel_url(line)
                        if url:
                            tunnel_url_holder.append(url)
                            tunnel_found_event.set()

            t_tunnel = threading.Thread(target=drain_tunnel, daemon=True)
            t_tunnel.start()

            # 1. Wait for tunnel URL, polling early-exit status of both children
            tunnel_deadline = time.monotonic() + timeout_tunnel
            while not tunnel_found_event.is_set():
                if app_proc.poll() is not None:
                    t_app.join(timeout=1.0)
                    print(
                        f"Error: renest start exited prematurely (code {app_proc.returncode}).",
                        file=sys.stderr,
                        flush=True,
                    )
                    _dump_tail("ComfyUI", app_tail)
                    return app_proc.returncode or 1
                if tunnel_proc.poll() is not None:
                    t_tunnel.join(timeout=1.0)
                    msg = f"cloudflared exited prematurely (code {tunnel_proc.returncode})."
                    if allow_local_fallback:
                        print(f"[Renest Colab] Warning: {msg} Continuing with local WebUI fallback.", file=sys.stderr, flush=True)
                        _dump_tail("cloudflared", tunnel_tail)
                        tunnel_proc = None
                        break
                    print(f"Error: {msg}", file=sys.stderr, flush=True)
                    _dump_tail("cloudflared", tunnel_tail)
                    return tunnel_proc.returncode or 1
                if time.monotonic() >= tunnel_deadline:
                    msg = f"Timed out waiting for Cloudflare tunnel URL ({timeout_tunnel}s)."
                    if allow_local_fallback:
                        print(f"[Renest Colab] Warning: {msg} Continuing with local WebUI fallback.", file=sys.stderr, flush=True)
                        _dump_tail("cloudflared", tunnel_tail)
                        _stop_tunnel(tunnel_proc)
                        tunnel_proc = None
                        break
                    print(f"Error: {msg}", file=sys.stderr, flush=True)
                    _dump_tail("cloudflared", tunnel_tail)
                    _dump_tail("ComfyUI", app_tail)
                    return 1
                tunnel_found_event.wait(timeout=0.25)

        tunnel_url: Optional[str] = tunnel_url_holder[0] if tunnel_url_holder else None
        if tunnel_url:
            print(f"[Renest Colab] Tunnel endpoint ready: {tunnel_url}", flush=True)

        # 2. Health check ComfyUI HTTP endpoint: fail fast if app_proc exits prematurely
        print(f"[Renest Colab] Waiting for ComfyUI to answer on port {port}...", flush=True)
        http_deadline = time.monotonic() + timeout_http
        http_ready = False
        while True:
            if app_proc.poll() is not None:
                t_app.join(timeout=1.0)
                print(
                    f"Error: ComfyUI process terminated with exit code {app_proc.returncode}",
                    file=sys.stderr,
                    flush=True,
                )
                _dump_tail("ComfyUI", app_tail)
                return app_proc.returncode or 1
            remaining = http_deadline - time.monotonic()
            if remaining <= 0:
                break
            slice_timeout = min(1.0, max(0.05, remaining))
            if wait_for_http_ready(port=port, timeout=slice_timeout):
                http_ready = True
                break

        if not http_ready:
            print(f"Error: ComfyUI HTTP health check timed out after {timeout_http}s.", file=sys.stderr, flush=True)
            _dump_tail("ComfyUI", app_tail)
            if app_proc.poll() is not None:
                print(
                    f"Error: ComfyUI process terminated with exit code {app_proc.returncode}",
                    file=sys.stderr,
                    flush=True,
                )
                return app_proc.returncode or 1
            return 1

        # 3. Verify tunnel end-to-end before declaring public URL ready
        if tunnel_url:
            print(f"[Renest Colab] Verifying Cloudflare tunnel reachability ({tunnel_url})...", flush=True)
            tunnel_ok, tunnel_err = wait_for_public_url(tunnel_url, timeout=20.0)
            if not tunnel_ok:
                msg = f"Cloudflare tunnel {tunnel_url} failed end-to-end check ({tunnel_err})."
                _dump_tail("cloudflared", tunnel_tail)
                if allow_local_fallback:
                    print(f"[Renest Colab] Warning: {msg} Continuing with Colab Proxy.", file=sys.stderr, flush=True)
                    _stop_tunnel(tunnel_proc)
                    tunnel_proc = None
                    tunnel_url = None
                else:
                    print(f"Error: {msg}", file=sys.stderr, flush=True)
                    return 1

        # 4. Declare ready (with verified Cloudflare tunnel URL or local Colab fallback marker)
        if tunnel_url:
            banner = [
                "",
                "═" * 70,
                "  🎉 ComfyUI WebUI is ready!",
                f"  👉 Public WebUI URL (verified HTTP 200): {tunnel_url}",
                "     (If *.trycloudflare.com is blocked on your network, use the Colab Proxy link below.)",
                "═" * 70,
                "",
            ]
        else:
            banner = [
                "",
                "═" * 70,
                f"  🎉 ComfyUI WebUI is running locally on port {port}!",
                f"  [Renest Colab] LOCAL_WEBUI_READY port={port}",
                "  👉 Cloudflare tunnel was unavailable; use the Colab Proxy link below.",
                "═" * 70,
                "",
            ]
        print("\n".join(banner), flush=True)

        # 4. Supervise until interrupt or child process termination
        while True:
            if app_proc.poll() is not None:
                t_app.join(timeout=1.0)
                print(f"\n[Renest Colab] ComfyUI process exited (code {app_proc.returncode}).", file=sys.stderr, flush=True)
                _dump_tail("ComfyUI", app_tail)
                return app_proc.returncode
            if tunnel_proc is not None and tunnel_proc.poll() is not None:
                if t_tunnel is not None:
                    t_tunnel.join(timeout=1.0)
                msg = f"cloudflared process exited (code {tunnel_proc.returncode})."
                if allow_local_fallback:
                    print(f"\n[Renest Colab] Warning: {msg} Keeping ComfyUI alive on local port {port}.", file=sys.stderr, flush=True)
                    _dump_tail("cloudflared", tunnel_tail)
                    print(f"[Renest Colab] LOCAL_WEBUI_READY port={port}", flush=True)
                    tunnel_proc = None
                else:
                    print(f"\n[Renest Colab] {msg}", file=sys.stderr, flush=True)
                    _dump_tail("cloudflared", tunnel_tail)
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
        default=None,
        help="Optional ComfyUI listen host override (only valid when nest records a listen flag)",
    )
    launch_p.add_argument(
        "--timeout-tunnel",
        type=float,
        default=45.0,
        help="Max seconds to wait for tunnel URL (default: 45.0)",
    )
    launch_p.add_argument(
        "--timeout-http",
        type=float,
        default=120.0,
        help="Max seconds to wait for ComfyUI HTTP readiness (default: 120.0)",
    )
    launch_p.add_argument(
        "--allow-local-fallback",
        action="store_true",
        help="Keep ComfyUI running locally if cloudflared quick tunnel fails or times out",
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
                allow_local_fallback=args.allow_local_fallback,
            )
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr, flush=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
