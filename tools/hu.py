#!/bin/env python
import os
import subprocess as sb
import sys
import time
from concurrent.futures import ProcessPoolExecutor

"""
ヒューリスティックコンテストの問題を実行しスコアを出力する.
"""

class Result:
    def __init__(self, case, inf, outf, visout, elapsed):
        self.case = case
        self.inf = inf
        self.outf = outf
        self.visout = visout
        self.elapsed = elapsed
        self.score = self._score()

    def _score(self):
        # ビジュアライザの出力からスコアを取得する. 問題に応じて変更する必要あり.
        return int(self.visout.split(' ')[2])

    def format(self):
        return '{:04d} SCORE={:11,d}, ELAPSED={:.2f}s'.format(self.case, self.score, self.elapsed)

class Context:
    def __init__(self, cases):
        self.bin = os.path.basename(os.path.dirname(__file__)) + "-a"
        self.target_dir = 'target'
        self.tools = 'tools'
        self.exe = os.path.abspath(os.path.join(self.target_dir, 'release', self.bin + '.exe'))
        self.vis = os.path.abspath(os.path.join(self.tools, 'vis.exe'))
        self.cases = self._parse_cases(cases) if len(cases) else [0]
        self.max_workers = 5

    def _parse_cases(self, cases):
        ret = []
        for s in cases:
            ss = s.split('-')
            if len(ss) == 1:
                ret.append(int(ss[0]))
            else:
                ret.extend(range(int(ss[0]), int(ss[1])+1))
        ret = list(set(ret))
        ret.sort()
        return ret

    def cargo_build(self):
        sb.run('cargo build --release --bin {} --target-dir {} -q'.format(self.bin, self.target_dir),
            shell=True, check=True, stderr=sb.DEVNULL)

    def input_file(self, case):
        return '{}/in/{:04d}.txt'.format(self.tools, case)

    def output_file(self, case):
        return '{}/out/{:04d}.txt'.format(self.tools, case)

    def exe_test(self, inf, outf):
        sb.run('{} < {} > {}'.format(self.exe, inf, outf), shell=True, check=True)

    def exe_vis(self, inf, outf):
        return sb.run('{} {} {}'.format(self.vis, inf, outf), shell=True, check=True, capture_output=True, text=True).stdout

    def execute(self, case):
        inf  = self.input_file(case)
        outf = self.output_file(case)
        os.makedirs(os.path.dirname(outf), exist_ok=True)

        # execute test
        st = time.time()
        self.exe_test(inf, outf)
        en = time.time()
        elapsed = en - st

        # evaluate
        visout = self.exe_vis(inf, outf)
        return Result(case, inf, outf, visout, elapsed)

    def execute_with_multiprocess(self):
        with ProcessPoolExecutor(max_workers=self.max_workers) as e:
            fs = [e.submit(self.execute, case) for case in self.cases]
            for f in fs:
                yield f.result()

def main():
    ctx = Context(sys.argv[1:])
    ctx.cargo_build()
    res = ctx.execute_with_multiprocess()
    total = 0
    for r in res:
        print(r.format())
        total += r.score

    print('TOTAL={:,}'.format(total))

if __name__ == '__main__':
    main()
