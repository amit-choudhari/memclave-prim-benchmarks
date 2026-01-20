#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import subprocess
import hashlib
import time
import shutil
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


# ---------------------------
# Bench config
# ---------------------------
DEFAULT_BENCH_DIRS = [
    "BFS", "BS", "GEMV", "HST-L", "HST-S", "MLP", "NW", "RED",
    "SCAN-RSS", "SCAN-SSA", "SEL", "SpMV", "TRNS", "TS", "UNI", "VA",
]

EXCLUDE_BIN_NAMES = {
    "dpu_code", "dpu", "dpu_host", "gemv_dpu", "trns_dpu", "bfs_dpu", "nw_dpu"
}


# ---------------------------
# Zenodo dataset config (BFS)
# ---------------------------
# Try multiple URL forms; Zenodo/CDN can be flaky on large files.
ZENODO_BFS_URLS = [
    "https://zenodo.org/records/18307126/files/bfs-data.tar.zst?download=1",
    "https://zenodo.org/records/18307126/files/bfs-data.tar.zst",
]
# From bfs-data.tar.zst.sha256 (your uploaded checksum file)
ZENODO_BFS_SHA256 = "1fe6b7b185509cd567fd530f378aa15c48ff43ad5be8d2c9707e93ff0ada7f3a"

BFS_MARKERS = [
    Path("data") / "LiveJournal1",
    Path("data") / "loc-gowalla",
    Path("data") / "roadNet-PA",
]


# ---------------------------
# Helpers
# ---------------------------
def is_executable(p: Path) -> bool:
    return p.is_file() and os.access(str(p), os.X_OK)


def pick_host_binary(bench_dir: Path) -> Path | None:
    bin_dir = bench_dir / "bin"
    if not bin_dir.is_dir():
        return None

    preferred = [
        bin_dir / "host_code",
        bin_dir / "host",
        bin_dir / f"{bench_dir.name.lower()}_host",
    ]
    for p in preferred:
        if is_executable(p):
            return p

    candidates = []
    for p in sorted(bin_dir.iterdir()):
        if not is_executable(p):
            continue
        name = p.name
        low = name.lower()
        if name in EXCLUDE_BIN_NAMES:
            continue
        if "dpu" in low and "host" not in low:
            continue
        candidates.append(p)

    if not candidates:
        return None

    for p in candidates:
        if "host" in p.name.lower():
            return p

    return candidates[0]


def classify(output: str, rc: int) -> tuple[bool, str]:
    if rc != 0:
        return False, f"rc={rc}"
    if "ERROR" in output:
        return False, "found ERROR"
    if "OK" in output:
        return True, "found OK"
    return False, "no OK/ERROR marker"


# ---------------------------
# Make
# ---------------------------
def run_make(bench_dir: Path, jobs: int | None, target: str | None) -> tuple[int, str]:
    cmd: List[str] = ["make"]
    if jobs and jobs > 0:
        cmd += [f"-j{jobs}"]
    if target:
        cmd += [target]

    proc = subprocess.run(
        cmd,
        cwd=str(bench_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode, (proc.stdout or "")


# ---------------------------
# Download + extract (BFS)
# ---------------------------
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(urls: list[str], dest: Path, *, timeout: int = 60, retries: int = 6) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    transient = {429, 502, 503, 504}
    curl_path = shutil.which("curl")

    def _rm_tmp():
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass

    def try_urllib(url: str) -> None:
        _rm_tmp()
        req = urllib.request.Request(url, headers={"User-Agent": "memclave-prim-benchmarks/1.0"})

        with urllib.request.urlopen(req, timeout=timeout) as r, tmp.open("wb") as f:
            total = r.headers.get("Content-Length")
            total = int(total) if total and total.isdigit() else None
            done = 0

            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = (done / total) * 100
                    print(
                        f"      download: {done/1e6:.1f}/{total/1e6:.1f} MB ({pct:.1f}%)",
                        end="\r",
                        flush=True,
                    )
                else:
                    print(f"      download: {done/1e6:.1f} MB", end="\r", flush=True)

        print()
        tmp.replace(dest)

    def try_curl(url: str) -> None:
        _rm_tmp()
        cmd = [
            "curl", "-L", "--fail",
            "--retry", "8",
            "--retry-delay", "2",
            "--connect-timeout", "15",
            "-o", str(tmp),
            url,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"curl failed (rc={proc.returncode}):\n{proc.stdout}")
        tmp.replace(dest)

    last_err: Exception | None = None

    for url in urls:
        print(f"      trying: {url}")

        # urllib retry loop
        for attempt in range(retries):
            try:
                try_urllib(url)
                return
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code in transient:
                    sleep_s = min(2 ** attempt, 30)
                    print(f"      HTTP {e.code} (transient), retrying in {sleep_s}s...")
                    time.sleep(sleep_s)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError) as e:
                last_err = e
                sleep_s = min(2 ** attempt, 30)
                print(f"      network/timeout error, retrying in {sleep_s}s... ({e})")
                time.sleep(sleep_s)
                continue
            except Exception as e:
                last_err = e
                break  # try next URL

        # curl fallback for this URL
        if curl_path:
            print("      urllib failed repeatedly; trying curl fallback...")
            try:
                try_curl(url)
                return
            except Exception as e:
                last_err = e

    raise RuntimeError(f"download failed for all URLs: {last_err}")


def extract_tar_zst(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try tar --zstd
    try:
        proc = subprocess.run(
            ["tar", "--zstd", "-xf", str(archive), "-C", str(out_dir)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if proc.returncode == 0:
            return
    except FileNotFoundError:
        pass

    # Fallback: zstd -dc | tar -xf -
    zstd = subprocess.Popen(["zstd", "-dc", str(archive)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    tar = subprocess.run(
        ["tar", "-xf", "-", "-C", str(out_dir)],
        stdin=zstd.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    zstd.stdout.close()
    zstd_err = zstd.stderr.read().decode("utf-8", errors="replace")
    zstd.wait()

    if tar.returncode != 0 or zstd.returncode != 0:
        msg = (tar.stdout or "") + ("\n[zstd stderr]\n" + zstd_err if zstd_err else "")
        raise RuntimeError(f"extract failed (tar rc={tar.returncode}, zstd rc={zstd.returncode}):\n{msg}")


def ensure_bfs_data(bench_dir: Path, allow_download: bool = True) -> tuple[bool, str]:
    data_dir = bench_dir / "data"
    markers_abs = [bench_dir / m for m in BFS_MARKERS]

    if all(p.exists() for p in markers_abs):
        return True, "datasets present"

    if not allow_download:
        missing = [str(p) for p in markers_abs if not p.exists()]
        return False, f"datasets missing (offline): {', '.join(missing)}"

    cache_dir = bench_dir / "data" / ".cache"
    archive = cache_dir / "bfs-data.tar.zst"

    print("==> BFS data missing. Downloading from Zenodo...")
    if not archive.exists():
        download_file(ZENODO_BFS_URLS, archive, timeout=60, retries=6)
    else:
        print(f"      using cached archive: {archive}")

    print("      verifying sha256...")
    got = sha256_file(archive)
    if got.lower() != ZENODO_BFS_SHA256.lower():
        try:
            archive.unlink()
        except OSError:
            pass
        return False, f"sha256 mismatch for {archive.name} (got {got}, expected {ZENODO_BFS_SHA256})"

    print(f"      extracting into: {data_dir}")
    extract_tar_zst(archive, data_dir)

    if all(p.exists() for p in markers_abs):
        return True, "downloaded + extracted"
    missing = [str(p) for p in markers_abs if not p.exists()]
    return False, f"extracted but markers still missing: {', '.join(missing)}"


# ---------------------------
# Args
# ---------------------------
def parse_args(argv: List[str]) -> tuple[List[str], bool, int | None, str | None, bool]:
    """
    Returns: (selected_benchmarks, do_make, jobs, make_target, allow_download)
    """
    do_make = True
    allow_download = True
    jobs: int | None = None
    make_target: str | None = None

    selected: List[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--list":
            print("Benchmarks:")
            for b in DEFAULT_BENCH_DIRS:
                print(f"  {b}")
            sys.exit(0)
        elif a == "--no-make":
            do_make = False
        elif a == "--no-download":
            allow_download = False
        elif a in ("-j", "--jobs"):
            if i + 1 >= len(argv):
                raise SystemExit("Missing value for --jobs")
            jobs = int(argv[i + 1])
            i += 1
        elif a == "--make-target":
            if i + 1 >= len(argv):
                raise SystemExit("Missing value for --make-target")
            make_target = argv[i + 1]
            i += 1
        else:
            selected.append(a)
        i += 1

    if not selected:
        selected = DEFAULT_BENCH_DIRS

    return selected, do_make, jobs, make_target, allow_download


# ---------------------------
# Main
# ---------------------------
def main():
    root = Path.cwd()
    selected, do_make, jobs, make_target, allow_download = parse_args(sys.argv[1:])

    logdir = root / "logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir.mkdir(parents=True, exist_ok=True)

    passed: List[str] = []
    failed: List[Tuple[str, str]] = []

    print(f"Root         : {root}")
    print(f"Logs         : {logdir}")
    if do_make:
        make_desc = f"enabled (jobs={jobs if jobs is not None else 'default'}, target={make_target or 'default'})"
    else:
        make_desc = "disabled"
    print(f"Make         : {make_desc}")
    print(f"Auto-download: {'yes' if allow_download else 'no'}")
    print()

    for bench in selected:
        bench_dir = root / bench
        if not bench_dir.is_dir():
            failed.append((bench, "missing dir"))
            print(f"[FAIL] {bench}: missing directory")
            print()
            continue

        # Ensure BFS datasets (before build/run so we don't waste time)
        if bench == "BFS":
            ok, reason = ensure_bfs_data(bench_dir, allow_download=allow_download)
            if not ok:
                failed.append((bench, reason))
                print(f"[FAIL] {bench}: {reason}")
                print()
                continue
            print(f"[OK]   {bench}: {reason}")
            print()

        # 1) Build
        if do_make:
            mf = bench_dir / "Makefile"
            if not mf.exists():
                failed.append((bench, "missing Makefile"))
                print(f"[FAIL] {bench}: Makefile not found")
                print()
                continue

            print(f"==> Building {bench}: make {'-j'+str(jobs) if jobs else ''} {make_target or ''}".strip())
            rc, out = run_make(bench_dir, jobs=jobs, target=make_target)
            make_log = logdir / f"{bench}.make.log"
            make_log.write_text(out, encoding="utf-8", errors="replace")

            if rc != 0:
                failed.append((bench, f"make failed (rc={rc})"))
                print(f"[FAIL] {bench}: make failed (rc={rc})")
                print(f"      make log: {make_log}")
                print()
                continue
            else:
                print(f"[OK]   {bench}: make succeeded")
                print(f"      make log: {make_log}")
                print()

        # 2) Find host binary
        host_bin = pick_host_binary(bench_dir)
        if host_bin is None:
            failed.append((bench, "no host binary found"))
            print(f"[FAIL] {bench}: no runnable host binary found under {bench}/bin/")
            print()
            continue

        # 3) Run host binary
        log_path = logdir / f"{bench}.run.log"
        cmd = [str(host_bin)]
        print(f"==> Running {bench}: (cwd={bench_dir.name}) ./{host_bin.relative_to(bench_dir)}")

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(bench_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            out = proc.stdout or ""
        except Exception as e:
            failed.append((bench, f"exception: {e}"))
            print(f"[FAIL] {bench}: exception: {e}")
            print()
            continue

        log_path.write_text(out, encoding="utf-8", errors="replace")

        ok, reason = classify(out, proc.returncode)
        if ok:
            passed.append(bench)
            print(f"[PASS] {bench}: {reason}")
        else:
            failed.append((bench, reason))
            print(f"[FAIL] {bench}: {reason}")

        print(f"      run log: {log_path}")
        print()

    print("============== Summary ==============")
    print(f"PASSED ({len(passed)}):")
    for b in passed:
        print(f"  - {b}")
    print()
    print(f"FAILED ({len(failed)}):")
    for b, why in failed:
        print(f"  - {b}: {why}")
    print("=====================================")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
