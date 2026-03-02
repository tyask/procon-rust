#!/usr/bin/env python3

import argparse
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def extract_contest_id(url: str) -> str:
    m = re.search(r"/contests/([A-Za-z0-9_-]+)", url)
    if not m:
        raise ValueError(f"Cannot detect contest id from URL: {url}")
    return m.group(1)

def parse_target(target: str) -> tuple[str, str]:
    """
    Returns (contest_id, seed_url).
    - URL mode: target includes /contests/{id}
    - ID mode: target itself is treated as contest id (e.g. ahc061)
    """
    target = target.strip()
    if "/contests/" in target:
        return extract_contest_id(target), target

    if not re.fullmatch(r"[A-Za-z0-9_-]+", target):
        raise ValueError(
            f"Invalid contest id: {target}. Use something like 'ahc061' or pass a contest URL."
        )
    contest = target
    return contest, f"https://atcoder.jp/contests/{contest}"


def parse_zip_candidates(html: str, base_url: str, contest: str) -> list[str]:
    urls = []
    for href in re.findall(r'href="([^"]+?\.zip)"', html, flags=re.I):
        u = urllib.parse.urljoin(base_url, href)
        if f"img.atcoder.jp/{contest}/" not in u:
            continue
        if re.search(r"windows|win32|x86_64-pc-windows", u, flags=re.I):
            continue
        urls.append(u)
    return urls


def pick_zip_url(candidates: list[str], contest: str) -> str | None:
    if not candidates:
        return None
    # Prefer likely local tools names.
    candidates = sorted(
        set(candidates),
        key=lambda u: (
            0 if re.search(r"tools|local|tester|vis", u, flags=re.I) else 1,
            len(u),
            u,
        ),
    )
    return candidates[0]


def find_zip_url(start_url: str, contest: str) -> str:
    tried = []
    fallback_pages = [
        start_url,
        f"https://atcoder.jp/contests/{contest}/tasks/{contest}_a?lang=en",
        f"https://atcoder.jp/contests/{contest}/tasks/{contest}_a?lang=ja",
    ]
    for page in fallback_pages:
        if page in tried:
            continue
        tried.append(page)
        try:
            html = http_get(page).decode("utf-8", errors="ignore")
        except Exception:
            continue
        candidates = parse_zip_candidates(html, page, contest)
        zip_url = pick_zip_url(candidates, contest)
        if zip_url:
            return zip_url
    # Common fallback path used in many AHC pages.
    return f"https://img.atcoder.jp/{contest}/tools.zip"


def safe_extract(zf: zipfile.ZipFile, dst: Path) -> None:
    dst = dst.resolve()
    for info in zf.infolist():
        out = (dst / info.filename).resolve()
        if not str(out).startswith(str(dst) + "/") and out != dst:
            raise RuntimeError(f"Unsafe path in zip: {info.filename}")
    zf.extractall(dst)


def detect_tools_root(extract_dir: Path) -> Path:
    d = extract_dir / "tools"
    if d.is_dir():
        return d

    entries = [p for p in extract_dir.iterdir() if p.name != "__MACOSX"]
    if len(entries) == 1 and entries[0].is_dir():
        one = entries[0]
        if (one / "tools").is_dir():
            return one / "tools"
        return one
    return extract_dir


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download and extract AtCoder AHC local tools into ./tools"
    )
    ap.add_argument(
        "url_or_contest",
        help="AHC page URL or contest id (e.g. https://.../contests/ahc061/... or ahc061)",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite ./tools if exists")
    args = ap.parse_args()

    try:
        contest, seed_url = parse_target(args.url_or_contest)
    except ValueError as e:
        sys.exit(str(e))

    print(f"[1/3] Detecting zip URL from: {seed_url}")
    zip_url = find_zip_url(seed_url, contest)
    print(f"      zip: {zip_url}")

    dst = Path.cwd() / "tools"
    if dst.exists():
        if not args.force:
            sys.exit(f"{dst} already exists. Use --force to overwrite.")
        shutil.rmtree(dst)

    print("[2/3] Downloading and extracting...")
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        zpath = tdir / "tools.zip"
        try:
            zpath.write_bytes(http_get(zip_url))
        except urllib.error.HTTPError as e:
            sys.exit(f"Failed to download zip: {e}\nURL: {zip_url}")
        except Exception as e:
            sys.exit(f"Failed to download zip: {e}\nURL: {zip_url}")

        try:
            with zipfile.ZipFile(zpath) as zf:
                safe_extract(zf, tdir)
        except zipfile.BadZipFile:
            sys.exit("Downloaded file is not a valid zip archive.")

        src = detect_tools_root(tdir)
        shutil.copytree(src, dst)

    print(f"[3/3] Done. Extracted to: {dst}")


if __name__ == "__main__":
    main()
