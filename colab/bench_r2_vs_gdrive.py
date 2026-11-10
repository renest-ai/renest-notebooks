#!/usr/bin/env python3
"""Controlled Google Colab benchmark: Renest R2 vs Google Drive on identical hardware and files.

Measures end-to-end Renest restore (stages S0-S5) and controlled same-machine,
same-file transfer + SHA-256 verification across:
1. Renest R2 (multi-stream ranged download via `renest restore`)
2. Google Drive via `drive.mount` FUSE copy-to-local + SHA-256
3. Google Drive direct FUSE read + SHA-256
4. Google Drive HTTP shared-file download (`gdown`) + SHA-256

All timings are machine-recorded into JSON and CSV artifacts for blog analysis.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from typing import Any, Optional


DEFAULT_R2_DIR = "/content/bench_r2"
DEFAULT_DRIVE_COPY_DIR = "/content/bench_gdrive_copy"
DEFAULT_GDOWN_DIR = "/content/bench_gdown"
DEFAULT_DRIVE_MOUNT = "/content/drive"

CSV_COLUMNS = [
    "ts_utc",
    "gpu",
    "vram_gib",
    "cpu_count",
    "ram_gib",
    "region_hint",
    "arm",
    "nest_label",
    "files",
    "total_bytes",
    "total_gib",
    "download_or_copy_s",
    "sha256_s",
    "total_s",
    "effective_mbps",
    "sha256_verified",
    "cache_dropped",
    "status",
    "note",
]


@dataclass
class FileSpec:
    rel_path: str
    size_bytes: int
    sha256: str


@dataclass
class ArmResult:
    arm: str
    status: str
    files: int
    total_bytes: int
    download_or_copy_s: float
    sha256_s: float
    total_s: float
    effective_mbps: float
    sha256_verified: bool
    cache_dropped: bool
    note: str = ""
    details: dict[str, Any] | None = None


def sha256_file(path: Path) -> str:
    """Compute SHA-256 using C-level file_digest (Python 3.11+)."""
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def try_drop_page_cache() -> bool:
    """Best-effort Linux page cache drop between controlled read tests."""
    drop_path = Path("/proc/sys/vm/drop_caches")
    if not drop_path.exists():
        return False
    try:
        os.sync()
        drop_path.write_text("3\n", encoding="utf-8")
        return True
    except OSError:
        return False


def collect_machine_facts(workspace_root: str = "/content") -> dict[str, Any]:
    """Capture Colab VM hardware, storage, OS, and optional network region hint."""
    facts: dict[str, Any] = {
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count() or 0,
        "gpu": "unknown",
        "vram_gib": 0.0,
        "ram_gib": 0.0,
        "disk_free_gib": 0.0,
        "region_hint": "unknown",
    }
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        facts["ram_gib"] = round((page_size * phys_pages) / (1024**3), 2)
    except (ValueError, OSError, AttributeError):
        pass

    target_disk = workspace_root if os.path.exists(workspace_root) else "/"
    try:
        facts["disk_free_gib"] = round(shutil.disk_usage(target_disk).free / (1024**3), 2)
    except OSError:
        pass

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5.0,
        ).strip()
        if out:
            first = out.splitlines()[0].split(",")
            facts["gpu"] = first[0].strip()
            if len(first) > 1:
                facts["vram_gib"] = round(float(first[1].strip()) / 1024.0, 2)
    except Exception:
        pass

    try:
        req = urllib.request.Request("https://ipinfo.io/json", headers={"User-Agent": "Renest-Bench/1.0"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            info = json.loads(resp.read().decode("utf-8"))
            parts = [info.get("country", ""), info.get("region", ""), info.get("org", "")]
            facts["region_hint"] = " / ".join(p for p in parts if p) or "unknown"
    except Exception:
        pass

    return facts


def parse_grant_input(raw: str, out_path: Path) -> str:
    """Normalize a bare ULID token, JSON string, CLI snippet, or file path into a grant argument."""
    code = raw.strip()
    if os.path.isfile(code):
        return code
    m_json = re.search(r"(\{[\s\S]*\})", code)
    if m_json:
        try:
            parsed = json.loads(m_json.group(1))
            if "grant_id" in parsed:
                out_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
                return str(out_path)
        except ValueError:
            pass
    if "--grant" in code:
        m = re.search(r"--grant\s+([^\s]+)", code)
        if m and m.group(1).strip("\"'") != "grant.json":
            code = m.group(1).strip("\"'")
    if os.path.isfile(code):
        return code
    m_ulid = re.search(r"[0-9A-HJKMNP-TV-Z]{26}", code, re.IGNORECASE)
    clean_id = m_ulid.group(0) if m_ulid else code
    envelope = {"grant_version": "2", "grant_id": clean_id, "origin": "https://api.renest.ai"}
    out_path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    return str(out_path)


def discover_model_files(workspace: Path, min_bytes: int = 100 * 1024 * 1024, rel_paths: Optional[list[str]] = None) -> list[FileSpec]:
    """Identify target model files in a restored workspace and compute their reference SHA-256."""
    candidates: list[Path] = []
    if rel_paths:
        for rp in rel_paths:
            p = workspace / rp
            if not p.is_file():
                raise FileNotFoundError(f"Expected benchmark file not found: {p}")
            candidates.append(p)
    else:
        for p in sorted(workspace.rglob("*")):
            if not p.is_file() or any(part in (".venv", ".renest", "__pycache__", ".git") for part in p.parts):
                continue
            if p.stat().st_size >= min_bytes:
                candidates.append(p)

    specs: list[FileSpec] = []
    for p in candidates:
        rel = str(p.relative_to(workspace)).replace("\\", "/")
        specs.append(FileSpec(rel_path=rel, size_bytes=p.stat().st_size, sha256=sha256_file(p)))
    return specs


def copy_with_fsync(src: Path, dst: Path, bufsize: int = 8 * 1024 * 1024) -> int:
    """Copy a file to local disk and flush OS buffers so timing includes actual write completion."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with src.open("rb") as rf, dst.open("wb") as wf:
        while chunk := rf.read(bufsize):
            wf.write(chunk)
            total += len(chunk)
        wf.flush()
        os.fsync(wf.fileno())
    return total


def _mbps(byte_count: int, seconds: float) -> float:
    if byte_count <= 0 or seconds <= 0:
        return 0.0
    return round((byte_count * 8.0) / 1e6 / seconds, 2)


def run_r2_restore_arm(grant_raw: str, target_dir: Path, nest_label: str) -> tuple[ArmResult, dict[str, Any]]:
    """Run cold `renest --json restore` from Cloudflare R2 and extract exact S0-S5 metrics."""
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    grant_arg = parse_grant_input(grant_raw, target_dir.parent / "bench_grant.json")
    cmd = ["renest", "--json", "restore", "--grant", grant_arg, "--dir", str(target_dir), "--yes", "--force"]

    t0 = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    result_ev: dict[str, Any] = {}
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") == "stage_start":
            print(f"[R2] -> {ev.get('stage')}: {ev.get('desc', '')}", flush=True)
        elif ev.get("type") == "stage_done":
            print(f"[R2] done {ev.get('stage')} in {ev.get('duration_s')}s ({ev.get('detail', '')})", flush=True)
        elif ev.get("type") == "result":
            result_ev = ev
    rc = proc.wait()
    wall_s = round(time.monotonic() - t0, 3)

    metrics = result_ev.get("metrics") or {}
    stages = result_ev.get("stages") or {}
    xfer_bytes = int(metrics.get("transfer_bytes") or 0)
    xfer_s = float(metrics.get("transfer_seconds") or stages.get("S1") or 0.0)
    verify_s = float(metrics.get("verify_seconds") or stages.get("S2") or 0.0)
    blobs = result_ev.get("blobs") or {}

    arm = ArmResult(
        arm="r2_full_restore",
        status="ok" if rc == 0 else f"failed_exit_{rc}",
        files=int(blobs.get("downloaded") or blobs.get("total") or 0),
        total_bytes=xfer_bytes,
        download_or_copy_s=round(xfer_s, 3),
        sha256_s=round(verify_s, 3),
        total_s=round(float(metrics.get("total_seconds") or wall_s), 3),
        effective_mbps=_mbps(xfer_bytes, xfer_s),
        sha256_verified=(rc == 0),
        cache_dropped=False,
        note=f"nest={nest_label}; S0={stages.get('S0')}s S1={stages.get('S1')}s S2={stages.get('S2')}s S3={stages.get('S3')}s S4={stages.get('S4')}s S5={stages.get('S5')}s",
        details={
            "exit_code": rc,
            "stages": stages,
            "metrics": metrics,
            "precheck_overall": (result_ev.get("precheck") or {}).get("overall"),
            "recipe": result_ev.get("recipe"),
            "evidence_dir": result_ev.get("evidence_dir"),
        },
    )
    return arm, result_ev


def benchmark_r2_local_hash(workspace: Path, specs: list[FileSpec]) -> ArmResult:
    """Measure standalone SHA-256 verification speed on the local R2-restored model files."""
    cache_dropped = try_drop_page_cache()
    t0 = time.monotonic()
    matched = True
    total_bytes = 0
    for sp in specs:
        p = workspace / sp.rel_path
        digest = sha256_file(p)
        total_bytes += p.stat().st_size
        if digest != sp.sha256:
            matched = False
    hash_s = round(time.monotonic() - t0, 3)
    return ArmResult(
        arm="local_ssd_sha256",
        status="ok" if matched else "hash_mismatch",
        files=len(specs),
        total_bytes=total_bytes,
        download_or_copy_s=0.0,
        sha256_s=hash_s,
        total_s=hash_s,
        effective_mbps=_mbps(total_bytes, hash_s),
        sha256_verified=matched,
        cache_dropped=cache_dropped,
        note="Standalone SHA-256 read baseline on local /content disk",
    )


def ensure_drive_mounted(mount_point: Path) -> tuple[bool, float, str]:
    """Mount Google Drive inside Colab if not already mounted."""
    my_drive = mount_point / "MyDrive"
    if my_drive.is_dir():
        return True, 0.0, "already_mounted"
    t0 = time.monotonic()
    try:
        from google.colab import drive  # type: ignore[import-not-found]

        drive.mount(str(mount_point))
        elapsed = round(time.monotonic() - t0, 3)
        return my_drive.is_dir(), elapsed, "mounted_fresh"
    except Exception as exc:
        return False, round(time.monotonic() - t0, 3), f"mount_failed: {exc}"


def _resolve_drive_source(drive_dir: Path, sp: FileSpec) -> Optional[Path]:
    """Locate a matching file in Drive either by full relative path or basename."""
    by_rel = drive_dir / sp.rel_path
    if by_rel.is_file():
        return by_rel
    by_name = drive_dir / Path(sp.rel_path).name
    if by_name.is_file():
        return by_name
    return None


def benchmark_drive_mount_copy(drive_dir: Path, dst_dir: Path, specs: list[FileSpec], mount_s: float = 0.0) -> ArmResult:
    """Copy exact model files from Google Drive FUSE mount to local SSD and verify SHA-256."""
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    sources: list[tuple[FileSpec, Path]] = []
    for sp in specs:
        src = _resolve_drive_source(drive_dir, sp)
        if src is None:
            return ArmResult(
                arm="gdrive_mount_copy",
                status="missing_source_file",
                files=0,
                total_bytes=0,
                download_or_copy_s=0.0,
                sha256_s=0.0,
                total_s=0.0,
                effective_mbps=0.0,
                sha256_verified=False,
                cache_dropped=False,
                note=f"Missing in Drive dir {drive_dir}: {sp.rel_path}",
            )
        sources.append((sp, src))

    t_copy0 = time.monotonic()
    copied_bytes = 0
    for sp, src in sources:
        dst = dst_dir / sp.rel_path
        copied_bytes += copy_with_fsync(src, dst)
    copy_s = round(time.monotonic() - t_copy0, 3)

    cache_dropped = try_drop_page_cache()
    t_hash0 = time.monotonic()
    verified = True
    for sp, _ in sources:
        if sha256_file(dst_dir / sp.rel_path) != sp.sha256:
            verified = False
    sha_s = round(time.monotonic() - t_hash0, 3)
    total_s = round(copy_s + sha_s, 3)

    return ArmResult(
        arm="gdrive_mount_copy",
        status="ok" if verified else "hash_mismatch",
        files=len(sources),
        total_bytes=copied_bytes,
        download_or_copy_s=copy_s,
        sha256_s=sha_s,
        total_s=total_s,
        effective_mbps=_mbps(copied_bytes, copy_s),
        sha256_verified=verified,
        cache_dropped=cache_dropped,
        note=f"drive.mount copy to local SSD + SHA-256 (mount_s={mount_s})",
    )


def benchmark_drive_fuse_direct_hash(drive_dir: Path, specs: list[FileSpec]) -> ArmResult:
    """Read and SHA-256 verify directly from Google Drive FUSE mount (no local copy)."""
    sources: list[tuple[FileSpec, Path]] = []
    for sp in specs:
        src = _resolve_drive_source(drive_dir, sp)
        if src is None:
            return ArmResult(
                arm="gdrive_fuse_direct_read",
                status="missing_source_file",
                files=0,
                total_bytes=0,
                download_or_copy_s=0.0,
                sha256_s=0.0,
                total_s=0.0,
                effective_mbps=0.0,
                sha256_verified=False,
                cache_dropped=False,
                note=f"Missing in Drive dir {drive_dir}: {sp.rel_path}",
            )
        sources.append((sp, src))

    cache_dropped = try_drop_page_cache()
    t0 = time.monotonic()
    verified = True
    total_bytes = 0
    for sp, src in sources:
        total_bytes += src.stat().st_size
        if sha256_file(src) != sp.sha256:
            verified = False
    read_s = round(time.monotonic() - t0, 3)
    return ArmResult(
        arm="gdrive_fuse_direct_read",
        status="ok" if verified else "hash_mismatch",
        files=len(sources),
        total_bytes=total_bytes,
        download_or_copy_s=read_s,
        sha256_s=read_s,
        total_s=read_s,
        effective_mbps=_mbps(total_bytes, read_s),
        sha256_verified=verified,
        cache_dropped=cache_dropped,
        note="Direct FUSE read + SHA-256 from /content/drive without copying",
    )


def benchmark_gdown(gdown_map: dict[str, str], dst_dir: Path, specs: list[FileSpec]) -> ArmResult:
    """Download exact files from Google Drive via gdown HTTP API and verify SHA-256."""
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    spec_by_rel = {sp.rel_path: sp for sp in specs}
    spec_by_name = {Path(sp.rel_path).name: sp for sp in specs}

    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
    except subprocess.CalledProcessError as exc:
        return ArmResult("gdrive_gdown", f"pip_failed_{exc.returncode}", 0, 0, 0.0, 0.0, 0.0, 0.0, False, False, str(exc))

    t_dl0 = time.monotonic()
    downloaded: list[tuple[FileSpec, Path]] = []
    for key, file_id in gdown_map.items():
        sp = spec_by_rel.get(key) or spec_by_name.get(Path(key).name)
        if sp is None:
            return ArmResult("gdrive_gdown", "unknown_spec_file", 0, 0, 0.0, 0.0, 0.0, 0.0, False, False, f"No matching reference file for {key}")
        out_file = dst_dir / sp.rel_path
        out_file.parent.mkdir(parents=True, exist_ok=True)
        res = subprocess.run(
            ["gdown", "--id", file_id, "-O", str(out_file)],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0 or not out_file.is_file():
            err_tail = (res.stderr or res.stdout or "gdown failed").strip().splitlines()[-3:]
            return ArmResult(
                arm="gdrive_gdown",
                status=f"gdown_error_{res.returncode}",
                files=len(downloaded),
                total_bytes=sum(p.stat().st_size for _, p in downloaded),
                download_or_copy_s=round(time.monotonic() - t_dl0, 3),
                sha256_s=0.0,
                total_s=round(time.monotonic() - t_dl0, 3),
                effective_mbps=0.0,
                sha256_verified=False,
                cache_dropped=False,
                note=" | ".join(err_tail)[:300],
            )
        downloaded.append((sp, out_file))
    dl_s = round(time.monotonic() - t_dl0, 3)

    cache_dropped = try_drop_page_cache()
    t_hash0 = time.monotonic()
    verified = True
    total_bytes = 0
    for sp, out_file in downloaded:
        total_bytes += out_file.stat().st_size
        if sha256_file(out_file) != sp.sha256:
            verified = False
    sha_s = round(time.monotonic() - t_hash0, 3)
    total_s = round(dl_s + sha_s, 3)

    return ArmResult(
        arm="gdrive_gdown",
        status="ok" if verified else "hash_mismatch",
        files=len(downloaded),
        total_bytes=total_bytes,
        download_or_copy_s=dl_s,
        sha256_s=sha_s,
        total_s=total_s,
        effective_mbps=_mbps(total_bytes, dl_s),
        sha256_verified=verified,
        cache_dropped=cache_dropped,
        note="gdown HTTP download + SHA-256",
    )


def parse_gdown_arg(raw: str) -> dict[str, str]:
    """Parse 'path1=id1,path2=id2' or path to a JSON map into {file_key: drive_file_id}."""
    if os.path.isfile(raw):
        return json.loads(Path(raw).read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"Invalid --gdown-ids entry '{pair}'; expected 'filename=DRIVE_FILE_ID'")
        k, v = pair.split("=", 1)
        mapping[k.strip()] = v.strip()
    return mapping


def write_outputs(out_prefix: Path, nest_label: str, facts: dict[str, Any], specs: list[FileSpec], arms: list[ArmResult]) -> tuple[Path, Path]:
    """Write structured JSON evidence and CSV rows for blog publication."""
    json_path = Path(f"{out_prefix}.json")
    csv_path = Path(f"{out_prefix}.csv")
    json_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "nest_label": nest_label,
        "machine": facts,
        "reference_files": [asdict(s) for s in specs],
        "arms": [asdict(a) for a in arms],
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for a in arms:
            writer.writerow(
                {
                    "ts_utc": facts.get("ts_utc", ""),
                    "gpu": facts.get("gpu", ""),
                    "vram_gib": facts.get("vram_gib", 0.0),
                    "cpu_count": facts.get("cpu_count", 0),
                    "ram_gib": facts.get("ram_gib", 0.0),
                    "region_hint": facts.get("region_hint", ""),
                    "arm": a.arm,
                    "nest_label": nest_label,
                    "files": a.files,
                    "total_bytes": a.total_bytes,
                    "total_gib": round(a.total_bytes / (1024**3), 3),
                    "download_or_copy_s": a.download_or_copy_s,
                    "sha256_s": a.sha256_s,
                    "total_s": a.total_s,
                    "effective_mbps": a.effective_mbps,
                    "sha256_verified": a.sha256_verified,
                    "cache_dropped": a.cache_dropped,
                    "status": a.status,
                    "note": a.note,
                }
            )
    return json_path, csv_path


def format_markdown_summary(facts: dict[str, Any], arms: list[ArmResult]) -> str:
    """Render a clean Markdown summary table for immediate inspection."""
    lines = [
        f"### Colab Benchmark Summary ({facts.get('gpu')} / {facts.get('ram_gib')} GiB RAM / {facts.get('region_hint')})",
        "",
        "| Arm | Status | Size (GiB) | Transfer/Copy (s) | SHA-256 (s) | Total (s) | Speed (Mbps) | Verified |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for a in arms:
        gib = a.total_bytes / (1024**3)
        lines.append(
            f"| `{a.arm}` | {a.status} | {gib:.2f} | {a.download_or_copy_s:.1f} | {a.sha256_s:.1f} | {a.total_s:.1f} | {a.effective_mbps:.1f} | {a.sha256_verified} |"
        )
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Controlled Colab benchmark: Renest R2 vs Google Drive")
    parser.add_argument("--nest-label", default="z-image-turbo", help="Label for this nest sample (e.g. klein-4b, z-image-turbo)")
    parser.add_argument("--grant", default="", help="Renest grant code or grant.json path for cold R2 restore")
    parser.add_argument("--r2-dir", default=DEFAULT_R2_DIR, help="Target workspace directory for R2 restore")
    parser.add_argument("--skip-r2-restore", action="store_true", help="Reuse existing --r2-dir instead of running cold `renest restore`")
    parser.add_argument("--files", nargs="*", default=None, help="Optional relative file paths inside workspace to benchmark")
    parser.add_argument("--drive-dir", default="", help="Path inside mounted Google Drive containing the same files (e.g. /content/drive/MyDrive/renest-bench)")
    parser.add_argument("--drive-copy-dir", default=DEFAULT_DRIVE_COPY_DIR, help="Local destination directory for Drive FUSE copy test")
    parser.add_argument("--skip-fuse-direct", action="store_true", help="Skip direct FUSE SHA-256 read on mounted Drive")
    parser.add_argument("--gdown-ids", default="", help="Optional 'relpath=DRIVE_ID,...' or JSON file for gdown HTTP comparison")
    parser.add_argument("--gdown-dir", default=DEFAULT_GDOWN_DIR, help="Local destination directory for gdown test")
    parser.add_argument("--out-dir", default="/content", help="Directory for JSON/CSV benchmark reports")
    args = parser.parse_args(argv)

    facts = collect_machine_facts(args.out_dir)
    r2_dir = Path(args.r2_dir)
    arms: list[ArmResult] = []

    if not args.skip_r2_restore:
        if not args.grant:
            parser.error("--grant is required unless --skip-r2-restore is set")
        r2_arm, _ = run_r2_restore_arm(args.grant, r2_dir, args.nest_label)
        arms.append(r2_arm)

    if not r2_dir.is_dir():
        raise RuntimeError(f"Workspace directory not found for reference files: {r2_dir}")

    specs = discover_model_files(r2_dir, rel_paths=args.files)
    if not specs:
        raise RuntimeError(f"No reference model files (>=100 MiB) found in {r2_dir}")

    arms.append(benchmark_r2_local_hash(r2_dir, specs))

    if args.drive_dir:
        drive_path = Path(args.drive_dir)
        mounted, mount_s, mount_note = ensure_drive_mounted(Path(DEFAULT_DRIVE_MOUNT))
        if not mounted:
            arms.append(ArmResult("gdrive_mount_copy", "mount_failed", 0, 0, mount_s, 0.0, mount_s, 0.0, False, False, mount_note))
        else:
            arms.append(benchmark_drive_mount_copy(drive_path, Path(args.drive_copy_dir), specs, mount_s=mount_s))
            if not args.skip_fuse_direct:
                arms.append(benchmark_drive_fuse_direct_hash(drive_path, specs))

    if args.gdown_ids:
        gmap = parse_gdown_arg(args.gdown_ids)
        arms.append(benchmark_gdown(gmap, Path(args.gdown_dir), specs))

    ts_tag = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    out_prefix = Path(args.out_dir) / f"colab_r2_vs_gdrive_{args.nest_label}_{ts_tag}"
    json_path, csv_path = write_outputs(out_prefix, args.nest_label, facts, specs, arms)
    print("\n" + format_markdown_summary(facts, arms))
    print(f"\nSaved JSON report: {json_path}")
    print(f"Saved CSV rows:    {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
