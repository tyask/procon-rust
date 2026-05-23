#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# 対象セクション開始/終了
SEC_START = re.compile(r'^\[package\.metadata\.cargo-compete\.bin\]\s*$')
SEC_END   = re.compile(r'^\s*\[.*\]\s*$')  # 次のテーブル開始で終了

# 1行インライン表にマッチ:
#   <indent><key> = { ... problem = "<URL>" ... }<eol?>
#  グループ:
#   (1)=prefix( problem 値直前まで) / (2)=key / (3)=URL / (4)=suffix(閉じクオート以降) / (5)=EOL
LINE_PAT = re.compile(
    r'^(\s*([A-Za-z0-9_-]+)\s*=\s*{[^}]*?\bproblem\s*=\s*")'  # (1)prefix (2)key
    r'([^"]*)'                                                # (3)url
    r'(".*)'                                                  # (4)suffix
    r'(\r?\n?)$'                                              # (5)eol（改行無でもOK）
)

def base_url_from_args(contest: Optional[str], url: Optional[str]) -> str:
    """-a か -u から基底 URL を決定（末尾スラッシュ除去）。"""
    if contest:
        return f"https://atcoder.jp/contests/{contest}/tasks/{contest}"
    assert url is not None
    return url.rstrip("/")

def rewrite_line(line: str, base_url: str) -> Tuple[bool, str]:
    """
    1 行が「key = { ... problem = "<URL>" ... }」の形なら、
    problem の値を「{base_url}_{key}」へ置換する。改行コードを保持。
    """
    m = LINE_PAT.match(line)
    if not m:
        return False, line
    key = m.group(2)
    new_line = f'{m.group(1)}{base_url}_{key}{m.group(4)}{m.group(5)}'
    return True, new_line

def rewrite_text(text: str, base_url: str) -> Tuple[str, int]:
    """
    ファイル全体テキストに対して、対象セクション内のインライン行のみ problem を置換。
    戻り値: (書き換え後テキスト, 置換件数)
    """
    out_lines = []
    in_sec = False
    replaced = 0

    for line in text.splitlines(keepends=True):
        if SEC_START.match(line):
            in_sec = True
            out_lines.append(line)
            continue
        if in_sec and SEC_END.match(line):
            in_sec = False
            out_lines.append(line)
            continue

        if in_sec:
            changed, new_line = rewrite_line(line, base_url)
            if changed:
                replaced += 1
                out_lines.append(new_line)
            else:
                out_lines.append(line)
        else:
            out_lines.append(line)

    return "".join(out_lines), replaced

def update_cargo_toml(toml_path: Path, base_url: str) -> int:
    """
    toml_path の Cargo.toml を読み、problem の URL を {base_url}_{key} に更新して上書き保存。
    戻り値: 置換行数
    """
    if not toml_path.exists():
        raise FileNotFoundError(f"{toml_path} not found.")

    text = toml_path.read_text(encoding="utf-8")
    new_text, replaced = rewrite_text(text, base_url)
    toml_path.write_text(new_text, encoding="utf-8")
    return replaced

def run_cargo_compete_download() -> None:
    """cargo compete download --overwrite を実行。エラーは例外で上位へ。"""
    print("Running: cargo compete download --overwrite")
    subprocess.run(["cargo", "compete", "download", "--overwrite"], check=True)

def run_compete_refresh(script_dir: Path, rs_files: List[Path]) -> None:
    """
    同ディレクトリの compete_refresh.py を、src/bin/*.rs を引数にして実行する。
    """
    refresh_py = script_dir / "compete_refresh.py"
    if not refresh_py.exists():
        print(f"WARNING: {refresh_py} not found. Skipping compete_refresh.", file=sys.stderr)
        return

    args = [sys.executable, str(refresh_py)] + [str(p) for p in rs_files]
    print("Running:", " ".join(args))
    subprocess.run(args, check=True)

def main() -> None:
    p = argparse.ArgumentParser(
        description="Rewrite Cargo.toml (cargo-compete bin section) problem URLs, download, and refresh."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("-a", "--atcoder", metavar="CONTEST",
                   help="Contest ID (e.g. abc422) -> https://atcoder.jp/contests/{contest}/tasks/{contest}")
    g.add_argument("-u", "--url", metavar="URL",
                   help="Base URL without suffix (e.g. https://atcoder.jp/contests/abc422/tasks/abc422)")
    p.add_argument("--toml", default="Cargo.toml", help="Path to Cargo.toml (default: ./Cargo.toml)")
    p.add_argument("--no-download", action="store_true", help="Skip running `cargo compete download --overwrite`.")
    args = p.parse_args()

    base_url = base_url_from_args(args.atcoder, args.url)
    toml = Path(args.toml)

    # 1) Cargo.toml 更新
    try:
        replaced = update_cargo_toml(toml, base_url)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if replaced == 0:
        print("WARNING: No inline entries with problem= found inside [package.metadata.cargo-compete.bin].",
              file=sys.stderr)
    print(f"Updated {toml} ({replaced} entr{'y' if replaced==1 else 'ies'})")

    # 2) cargo compete download
    if not args.no_download:
        try:
            run_cargo_compete_download()
        except FileNotFoundError:
            print("ERROR: `cargo` not found in PATH.", file=sys.stderr); sys.exit(2)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: cargo compete failed with exit code {e.returncode}.", file=sys.stderr); sys.exit(e.returncode)

    # 3) 最後に compete_refresh.py を src/bin/*.rs を引数にして実行
    script_dir = Path(__file__).resolve().parent
    rs_files = sorted(Path("src/bin").glob("*.rs"))
    if not rs_files:
        print("WARNING: No Rust files matched src/bin/*.rs. Running refresh with no arguments.", file=sys.stderr)

    try:
        run_compete_refresh(script_dir, rs_files)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: compete_refresh.py failed with exit code {e.returncode}.", file=sys.stderr); sys.exit(e.returncode)
    except FileNotFoundError:
        # 既に run_compete_refresh 内で warning を出しているためここでは沈黙でもOK
        pass

if __name__ == "__main__":
    main()
