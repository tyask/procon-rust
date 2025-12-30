#!/bin/env python
import argparse
import os
import re
import shutil
import subprocess
import time
import urllib.request

"""
Rustのソースファイルに以下のような文字列を埋め込むことでその問題のテストケースをダウンロードしテストを実行する.
// CONTEST(abc200-a)

テストケースのダウンロード及び実行はcargo competeを用いる.
既にテストケースがダウンロードされている場合は再ダウンロードはしない.
"""

def lookup_problem(src):
    pattern = re.compile(r'CONTEST\(((?P<CONTEST2>.*):)?(?P<CONTEST>.*)-(?P<PROBLEM>.*?)\)(?P<FORCE> -f)?')
    with open(src, 'r', encoding='utf-8') as f:
        matched = [line for line in f.readlines() if pattern.search(line)]
        if not matched:
            return None
        m = pattern.search(matched[0])
        contest2 = m.group('CONTEST2')
        contest = m.group('CONTEST')
        problem = m.group('PROBLEM')
        force = m.group('FORCE') is not None
        return (contest2, contest, problem.lower(), force)

def lookup_bin(src):
    return os.path.splitext(os.path.basename(src))[0].lower()

def lookup_cargo(src):
    return os.path.normpath(os.path.join(os.path.dirname(src), '..', '..', 'Cargo.toml'))

def generate_problem_url(prob):
    contest2, contest, problem, _ = prob
    contest2 = contest if contest2 is None else contest2
    return 'https://atcoder.jp/contests/{}/tasks/{}_{}'.format(contest2, contest, problem)

def can_open_url(url):
    try:
        with urllib.request.urlopen(url) as res:
            return True
    except urllib.error.URLError as e:
        print("Failed to open url: {} (code={}, reason={})".format(url, e.code, e.reason))
        return False

def write_url_to_cargo(cargo, bin, url):
    # [package.metadata.cargo-compete.bin]
    # a = { problem = "https://atcoder.jp/contests/abc200/tasks/abc200_a" } (*)
    # b = { problem = "https://atcoder.jp/contests/abc200/tasks/abc200_a" } (*)
    # 
    # (*)の行を置換する
    p1 = re.compile(r'{}\s*=\s*{{\s*problem\s*=\s*"(?P<URL>.*?)"\s*}}'.format(bin))
    tmp = 'Cargo_tmp.toml'
    need_download_testcases = False
    with open(tmp, 'w') as w:
        with open(cargo, 'r') as f:
            for line in f.readlines():
                m1 = p1.search(line)
                if m1:
                    prev = m1.group("URL")
                    print("Prev URL: {}".format(prev))
                    w.write('{} = {{ problem = "{}" }}\n'.format(bin, url))
                    need_download_testcases = prev != url
                else:
                    w.write(line)

    shutil.move(tmp, cargo)
    return need_download_testcases

def cargo_compete_downlowad_test(bin):
    return subprocess.run('cargo compete retrieve t --overwrite --bin {}'.format(bin),
                          shell=True,
                          stderr=subprocess.PIPE,
                          encoding='utf-8')

def cargo_compete_test(bin):
    subprocess.run('cargo compete test {}'.format(bin), shell=True)

def download_testcases(bin):
    print('Will download testcases...')
    while True:
        ret = cargo_compete_downlowad_test(bin)
        print(ret.stderr, end='')
        if ret.returncode == 0:
            print('Successfully downloaded testcases!')
            break

        if '429 Too Many Requests' in ret.stderr:
            print('Failed to download testcases. Retrying...')
            time.sleep(1)
        else:
            print('Failed to download testcases')
            return False

    return True

def parse_args():
    parser = argparse.ArgumentParser(description='This script downloads testcases and executes it')
    parser.add_argument('-s', '--src', required=True, help='target source file')
    parser.add_argument('-f', '--force-download', default=False, action="store_true", help='force download testcases if true')
    return parser.parse_args()

def main():
    args = parse_args()
    src = args.src
    prob = lookup_problem(src)
    if not prob:
        print("Missing lines for specifying problem. src={}".format(src))
        return

    cargo = lookup_cargo(src)
    bin = lookup_bin(src)
    url = generate_problem_url(prob)
    force_download = prob[3] or args.force_download

    if not can_open_url(url):
        return

    print('bin={}'.format(bin))
    print('url={}'.format(url))
    print('cargo={}'.format(cargo))
    print('force_download={}'.format(force_download))

    if write_url_to_cargo(cargo, bin, url) or force_download:
        ok = download_testcases(bin)
    else:
        ok = True
        print('Already downloaded testcases')

    if ok:
        print('Executing compete test')
        cargo_compete_test(bin)

if __name__ == '__main__':
    main()
