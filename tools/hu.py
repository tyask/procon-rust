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
        #print(self.stderr, end='')
        cmts = self._lookup_comments()
        print('{:04d} SCORE[{:11,d}] ELAPSED[{:.2f}s] CMTS[{}]'.format(self.case, self.score, self.elapsed, cmts))

    def clip(self):
        sb.run('clip < {}'.format(self.outf), shell=True, check=True, stderr=sb.DEVNULL)

    def _lookup_comments(self):
        cmts = ''
        for line in self.stderr.split('\n'):
            if line.startswith('# '):
                if cmts:
                    cmts += '/'
                cmts += line[2:]
        return cmts

class Context:
    def __init__(self, args):
        self.bin = os.path.basename(os.path.dirname(__file__)) + "-" + args.a
        self.target_dir = 'target'
        self.tools = 'tools'
        self.exe = os.path.abspath(os.path.join(self.target_dir, 'release', self.bin + '.exe'))
        self.vis = os.path.abspath(os.path.join(self.tools, 'vis.exe'))
        self.tester = os.path.abspath(os.path.join(self.tools, 'tester.exe'))
        self.cases = self._parse_cases(args)
        self.args = args
        self.max_workers = 5

    def _parse_cases(self, args):
        if not args.cases:
            return range(0, 5)

        # (1 2 3-5) => (1 2 3 4 5)
        ret = []
        for s in args.cases:
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

    def cmd(self, inf):
        os.environ['INPUT_FILE'] = inf
        return '{} < {}'.format(self.exe, inf)

    def run(self, inf, outf):
        return sb.run('{} > {}'.format(self.cmd(inf), outf), shell=True, check=True, capture_output=True, text=True).stderr

    def exe_vis(self, inf, outf):
        return sb.run('{} {} {}'.format(self.vis, inf, outf), shell=True, check=True, capture_output=True, text=True).stdout

    def execute_case(self, case):
        inf  = self.input_file(case)
        outf = self.output_file(case)
        os.makedirs(os.path.dirname(outf), exist_ok=True)

        # execute test
        st = time.time()
        stderr = self.run(inf, outf)
        en = time.time()
        elapsed = en - st

        # evaluate
        visout = self.exe_vis(inf, outf)
        return Result(case, inf, outf, visout, stderr, elapsed)

    def execute_with_multiprocess(self):
        with ProcessPoolExecutor(max_workers=self.max_workers) as e:
            fs = [e.submit(self.execute_case, cs) for cs in self.cases]
            for f in fs:
                yield f.result()

    def execute_with_singleprocess(self):
        for cs in self.cases:
            yield self.execute_case(cs)

    def run_only(self):
        for cs in self.cases:
            sb.run(self.cmd(self.input_file(cs)), shell=True, check=True, text=True)

    def execute(self):
        if self.args.run:
            self.run_only()
            return;
        elif self.args.single:
            res = self.execute_with_singleprocess()
        elif self.args.multi:
            res = self.execute_with_multiprocess()
        else:
            print('Invalid type: {}'.format(self.type))
            return;

        total = 0
        for r in res:
            r.print()
            total += r.score
        else:
            r.clip()
        print('TOTAL={:,}'.format(total))


def parse_args():
    parser = argparse.ArgumentParser(description='Executing program for huristic contest')
    parser.add_argument('cases', metavar='CASES', nargs='*', help='cases to execute')
    parser.add_argument('-a', '--a', help='execute file', default='a')
    parser.add_argument('-r', '--run', action='store_true', help='run')
    parser.add_argument('-s', '--single', action='store_true', help='execute program on single thread')
    parser.add_argument('-m', '--multi', action='store_true', default=True, help='execute program on mutiple thread')
    return parser.parse_args()

def main():
    args = parse_args()
    ctx = Context(args)
    ctx.cargo_build()
    ctx.execute()


if __name__ == '__main__':
    main()

