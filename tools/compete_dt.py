import os
import re
import shutil
import subprocess
import sys

"""
Rustのソースファイルに以下のような文字列を埋め込むことでその問題のテストケースをダウンロードしテストを実行する.
// CONTEST(abc200-a)

テストケースのダウンロード及び実行はcargo competeを用いる.
既にテストケースがダウンロードされている場合は再ダウンロードはしない.
"""

def lookup_problem(src):
    pattern = re.compile('CONTEST\((?P<CONTEST>.*?)-(?P<PROBLEM>.*?)\)')
    with open(src, 'r') as f:
        matched = [line for line in f.readlines() if pattern.search(line)]
        if not matched:
            return None
        m = pattern.search(matched[0])
        contest = m.group('CONTEST')
        problem = m.group('PROBLEM')
        return (contest, problem)

def lookup_bin(src):
    return os.path.splitext(os.path.basename(src))[0]

def lookup_cargo(src):
    return os.path.normpath(os.path.join(os.path.dirname(src), '..', '..', 'Cargo.toml'))

def generate_problem_url(contest, problem):
    return 'https://atcoder.jp/contests/{0}/tasks/{0}_{1}'.format(contest, problem)

def write_url_to_cargo(cargo, bin, url):
    # a = { problem = "https://atcoder.jp/contests/abc200/tasks/abc200_a" }
    # pattern = re.compile('({}\s+=\s+\{\s+problem\s+=\s+")(.*?)("\s+\})'.format(bin))
    pattern = re.compile('({}\s*=\s*{{\s*problem\s*=\s*")(?P<URL>.*?)("\s*}})'.format(bin))
    tmp = 'Cargo_tmp.toml'
    changed = False
    with open(tmp, 'w') as w:
        with open(cargo, 'r') as f:
            for line in f.readlines():
                m = pattern.search(line)
                if m:
                    prev = m.group("URL")
                    print("Prev URL: {}".format(prev))
                    changed = prev != url
                    w.write(pattern.sub('\\1' + url + '\\3', line))
                else:
                    w.write(line)
    shutil.move(tmp, cargo)
    return changed
    
def cargo_compete_downlowad_test():
    subprocess.run('cargo compete retrieve t --overwrite', shell=True)

def cargo_compete_test(bin):
    subprocess.run('cargo compete test {}'.format(bin), shell=True)

def main():
    src = sys.argv[1]
    prob = lookup_problem(src)
    if not prob:
        print("Missing lines for specifying problem. src={}".format(src))
        return

    cargo = lookup_cargo(src)
    bin = lookup_bin(src)
    url = generate_problem_url(prob[0], prob[1])

    print('bin={}'.format(bin))
    print('url={}'.format(url))
    print('cargo={}'.format(cargo))

    if write_url_to_cargo(cargo, bin, url):
        print('will download testcases...')
        cargo_compete_downlowad_test()
    else:
        print('Already Downloaded testcases')

    cargo_compete_test(bin)

if __name__ == '__main__':
    main()