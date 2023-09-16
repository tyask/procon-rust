#!/bin/env python
import argparse
import os
import re
import subprocess as sb
import sys
import time
from concurrent.futures import ProcessPoolExecutor

"""
ヒューリスティックコンテストの問題を実行しスコアを出力する.
"""

class Result:
    def __init__(self, case, inf, outf, visout, stderr, elapsed):
        self.case = case
        self.inf = inf
        self.outf = outf
        self.visout = visout
        self.stderr = stderr
        self.elapsed = elapsed
        self.score = self._score()

    def _score(self):
        # ビジュアライザの出力からスコアを取得する. 問題に応じて変更する必要あり.
        for line in self.visout.split('\n'):
            m = re.search('Score = (\d+)', line)
            if m:
                return int(m[1])
        return 0

    def print(self):
        print(self.stderr, end='')
        print('{:04d} SCORE={:11,d}, ELAPSED={:.2f}s'.format(self.case, self.score, self.elapsed))

    def clip(self):
        sb.run('clip < {}'.format(self.outf), shell=True, check=True, stderr=sb.DEVNULL)

class Context:
    def __init__(self, args):
        self.bin = os.path.basename(os.path.dirname(__file__)) + "-a"
        self.target_dir = 'target'
        self.tools = 'tools'
        self.exe = os.path.abspath(os.path.join(self.target_dir, 'release', self.bin + '.exe'))
        self.vis = os.path.abspath(os.path.join(self.tools, 'vis.exe'))
        cs = self._resolve_cases(args)
        self.cases = self._parse_cases(cs) if len(cs) else range(0, 5)
        self.max_workers = 5

    def _resolve_cases(self, args):
        cases = list(args.cases)
        if args.file:
            with open(args.file) as f:
                for line in f.readlines():
                    if not line.startswith('#'):
                        cases.extend(line.split(' '))
        return cases

    def _parse_cases(self, cases):
        ret = []
        for s in cases:
            ss = s.split('-')
            if len(ss) == 1:
                ret.append(int(ss[0]))
            else:
                ret.extend(range(int(ss[0]), int(ss[1])+1))
        return ret

    def cargo_build(self):
        sb.run('cargo build --release --bin {} --target-dir {} -q'.format(self.bin, self.target_dir),
            shell=True, check=True, stderr=sb.DEVNULL)

    def input_file(self, case):
        return os.path.join(self.tools, 'in', '{:04d}.txt'.format(case))

    def output_file(self, case):
        return os.path.join(self.tools, 'out', '{:04d}.txt'.format(case))

    def exe_test(self, inf, outf):
        return sb.run('{} < {} > {}'.format(self.exe, inf, outf), shell=True, check=True, capture_output=True, text=True).stderr

    def exe_vis(self, inf, outf):
        return sb.run('{} {} {}'.format(self.vis, inf, outf), shell=True, check=True, capture_output=True, text=True).stdout

    def execute(self, case):
        inf  = self.input_file(case)
        outf = self.output_file(case)
        os.makedirs(os.path.dirname(outf), exist_ok=True)

        # execute test
        st = time.time()
        stderr = self.exe_test(inf, outf)
        en = time.time()
        elapsed = en - st

        # evaluate
        visout = self.exe_vis(inf, outf)
        return Result(case, inf, outf, visout, stderr, elapsed)

    def execute_with_multiprocess(self):
        with ProcessPoolExecutor(max_workers=self.max_workers) as e:
            fs = [e.submit(self.execute, cs) for cs in self.cases]
            for f in fs:
                yield f.result()

    def execute_with_singleprocess(self):
        for cs in self.cases:
            yield self.execute(cs)

    def execute_bare(self):
        for cs in self.cases:
            inf  = self.input_file(cs)
            outf = self.output_file(cs)
            sb.run('{} < {}'.format(self.exe, inf), shell=True, check=True, text=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Executing program for huristic contest')
    parser.add_argument('cases', metavar='CASES', nargs='*', help='cases to execute')
    parser.add_argument('-b', '--bare', action='store_true', help='execute bare')
    parser.add_argument('-f', '--file', help='file the the cases are written')
    parser.add_argument('-in', '--indir', help='directory which has input files', default='in')
    parser.add_argument('-t', '--test', action='store_true', help='execute parameter test')
    parser.add_argument('-a', '--a', help='execute file', default='a')
    return parser.parse_args()

def main():
    args = parse_args()
    ctx = Context(args)
    ctx.cargo_build()
    execute(ctx, args)

def execute(ctx, args):
    if args.bare:
        ctx.execute_bare()
        return

    res = ctx.execute_with_multiprocess()
    total = 0
    for r in res:
        r.print()
        r.clip()
        total += r.score
    print('TOTAL={:,}'.format(total))

if __name__ == '__main__':
    main()
