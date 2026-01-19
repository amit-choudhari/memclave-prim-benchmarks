#!/usr/bin/env python3
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

# Directories to consider as benchmarks (you can edit this list)
DEFAULT_BENCH_DIRS = [
    "BFS", "BS", "GEMV", "HST-L", "HST-S", "MLP", "NW", "RED",
    "SCAN-RSS", "SCAN-SSA", "SEL", "SpMV", "TRNS", "TS", "UNI", "VA", "SpMV", "NW",
]

EXCLUDE_BIN_NAMES = {
    "dpu_code", "dpu", "dpu_host", "gemv_dpu", "trns_dpu", "bfs_dpu", "nw_dpu"
}

def is_executable(p: Path) -> bool:
    return p.is_file() and os.access(str(p), os.X_OK)

def pick_host_binary(bench_dir: Path) -> Path | None:
    bin_dir = bench_dir / "bin"
    if not bin_dir.is_dir():
        return None

    # Common names first
    candidates = []
    preferred = [
        bin_dir / "host_code",
        bin_dir / "host",
        bin_dir / f"{bench_dir.name.lower()}_host",
    ]
    for p in preferred:
        if is_executable(p):
            return p

    # Otherwise: choose an executable in bin/ that isn't "dpu_*" / "dpu_code"
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

    # Prefer ones with "host" in the filename
    for p in candidates:
        if "host" in p.name.lower():
            return p

    return candidates[0]

def classify(output: str, rc: int) -> tuple[bool, str]:
    # PASS if contains OK and not ERROR, and rc==0
    # FAIL if contains ERROR or rc!=0 or no OK marker
    if rc != 0:
        return False, f"rc={rc}"
    if "ERROR" in output:
        return False, "found ERROR"
    if "OK" in output:
        return True, "found OK"
    return False, "no OK/ERROR marker"

def main():
    root = Path.cwd()

    # Args: optional benchmark names; --list
    args = sys.argv[1:]
    if args and args[0] == "--list":
        print("Benchmarks:")
        for b in DEFAULT_BENCH_DIRS:
            print(f"  {b}")
        sys.exit(0)

    selected = args if args else DEFAULT_BENCH_DIRS

    logdir = root / "logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir.mkdir(parents=True, exist_ok=True)

    passed, failed = [], []

    print(f"Root   : {root}")
    print(f"Logs   : {logdir}")
    print()

    for bench in selected:
        bench_dir = root / bench
        if not bench_dir.is_dir():
            failed.append((bench, "missing dir"))
            print(f"[FAIL] {bench}: missing directory")
            print()
            continue

        host_bin = pick_host_binary(bench_dir)
        if host_bin is None:
            failed.append((bench, "no host binary found"))
            print(f"[FAIL] {bench}: no runnable host binary found under {bench}/bin/")
            print()
            continue

        log_path = logdir / f"{bench}.log"
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

        print(f"      log: {log_path}")
        print()

    # Summary
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
