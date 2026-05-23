#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import List

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

def http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req) as resp:
        return resp.read()

def find_local_zip_url(html: str, contest: str) -> str:
    """
    Heuristically locate the Local Tools ZIP URL from the task page HTML.
    Priority:
      1) A ZIP link near the words "ローカル版" / "Local" / "local"
      2) Any ZIP under https://img.atcoder.jp/{contest}/, excluding Windows binaries
      3) Fallback to https://img.atcoder.jp/{contest}/tools.zip
    """
    # 1) Search near "Local" wording
    for kw in ["ローカル版", "Local", "local"]:
        for m in re.finditer(kw, html, flags=re.I):
            frag = html[m.start(): m.start() + 800]  # scan nearby region
            z = re.search(
                rf'href="(https://img\.atcoder\.jp/{re.escape(contest)}/[^"]+?\.zip)"',
                frag,
            )
            if z:
                url = z.group(1)
                if not re.search(r'windows|win32|x86_64-pc-windows', url, flags=re.I):
                    return url

    # 2) Scan whole page for ZIPs under img.atcoder.jp/<contest>/
    all_zip = re.findall(
        rf'href="(https://img\.atcoder\.jp/{re.escape(contest)}/[^"]+?\.zip)"', html
    )
    cand = [u for u in all_zip if not re.search(r'windows|win32|x86_64-pc-windows', u, flags=re.I)]
    if cand:
        # Prefer names containing tools/local/tester/vis
        cand.sort(key=lambda u: (0 if re.search(r'tools|local|tester|vis', u, flags=re.I) else 1, len(u)))
        return cand[0]

    # 3) Fallback
    return f"https://img.atcoder.jp/{contest}/tools.zip"

def detect_tools_root(extract_dir: Path) -> Path:
    """
    From extracted contents, find the buildable tools directory that has Cargo.toml.
    Preference:
      - {extract}/tools
      - {extract}/<single top-level dir> (if it contains Cargo.toml or tools/)
      - {extract} if it has Cargo.toml
      - first directory found by recursive search containing Cargo.toml
    """
    t = extract_dir / "tools"
    if t.is_dir():
        return t

    entries = [p for p in extract_dir.iterdir() if not p.name.startswith("__MACOSX")]
    if len(entries) == 1 and entries[0].is_dir():
        one = entries[0]
        if (one / "Cargo.toml").exists():
            return one
        if (one / "tools").is_dir():
            return one / "tools"

    if (extract_dir / "Cargo.toml").exists():
        return extract_dir

    for p in extract_dir.rglob("Cargo.toml"):
        return p.parent

    raise RuntimeError("No Cargo.toml found in the extracted archive.")

def copytree_force(src: Path, dst: Path, force: bool):
    if dst.exists():
        if not force:
            raise RuntimeError(f"{dst} already exists. Use --force to overwrite.")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

def cargo_build_and_collect_bins(tools_dir: Path) -> List[Path]:
    """
    Run cargo build --release --message-format=json and collect built executable paths.
    """
    try:
        proc = subprocess.run(
            ["cargo", "build", "--release", "--message-format=json"],
            cwd=str(tools_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "cargo not found. Please install Rust and ensure cargo is on your PATH."
        )

    bins: List[Path] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("reason") == "compiler-artifact":
            target = obj.get("target", {})
            kinds = target.get("kind", [])
            if "bin" in kinds:
                exe = obj.get("executable")
                if exe:
                    bins.append(Path(exe))

    if proc.returncode != 0:
        raise RuntimeError("cargo build failed.\n\n" + proc.stdout)

    # Fallback: scan target/release for executable-looking files
    if not bins:
        rel = tools_dir / "target" / "release"
        if rel.is_dir():
            for p in rel.iterdir():
                if is_probably_executable(p):
                    bins.append(p)

    if not bins:
        raise RuntimeError("No executables detected. Check the build logs.")

    return bins

def is_probably_executable(p: Path) -> bool:
    if not p.is_file():
        return False
    if os.name == "nt":
        return p.suffix.lower() == ".exe"
    return p.suffix == "" and os.access(p, os.X_OK)

def main():
    ap = argparse.ArgumentParser(description="Set up AtCoder Heuristic Contest local tools.")
    ap.add_argument("contest", help="Contest ID (e.g., ahc051)")
    ap.add_argument("--force", action="store_true", help="Overwrite ./tools if it exists")
    args = ap.parse_args()

    contest = args.contest.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", contest):
        ap.error("Invalid contest ID.")

    task_url = f"https://atcoder.jp/contests/{contest}/tasks/{contest}_a?lang=en"
    print(f"[1/5] Fetching task page: {task_url}")
    try:
        html = http_get(task_url).decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as e:
        sys.exit(f"Failed to fetch task page: {e}")
    except Exception as e:
        sys.exit(f"Error while fetching task page: {e}")

    print("[2/5] Locating Local Tools ZIP URL...")
    zip_url = find_local_zip_url(html, contest)
    print(f"    ZIP URL: {zip_url}")

    print("[3/5] Downloading & extracting...")
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        zpath = tdir / "tools.zip"
        try:
            data = http_get(zip_url)
        except Exception as e:
            sys.exit(f"Failed to download ZIP: {e}\nURL: {zip_url}")

        zpath.write_bytes(data)
        try:
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(tdir)
        except zipfile.BadZipFile:
            sys.exit("Failed to extract ZIP (file may be corrupted).")

        tools_src = detect_tools_root(tdir)
        tools_dst = Path.cwd() / "tools"
        print(f"    Destination directory: {tools_dst}")
        copytree_force(tools_src, tools_dst, args.force)

    print("[4/5] Building (cargo build --release)...")
    bins = cargo_build_and_collect_bins(Path("tools"))
    for b in bins:
        print(f"    Built: {b}")

    print("[5/5] Copying executables to tools/ ...")
    copied = []
    for b in bins:
        dst = Path("tools") / b.name
        shutil.copy2(b, dst)
        copied.append(dst)

    print("\nDone. Executables placed at:")
    for c in copied:
        print(f"  - {c}")
    print("\nCheck README or run with --help as provided by each tool.")

if __name__ == "__main__":
    main()
