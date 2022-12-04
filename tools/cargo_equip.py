#!/bin/env python

import sys, os
import pathlib
import subprocess

file = sys.argv[1]
bin = sys.argv[2]

# create dummy test file to resolve test module while executing cargo equip
os.makedirs('src/tests', exist_ok=True)
pathlib.Path('src/tests/main_test.rs').touch()

# execute cargo equip
cmd = "cargo equip --remove docs --minify libs --exclude-atcoder-crates --no-rustfmt --bin {}".format(bin)
p = subprocess.run(cmd, shell=True, capture_output=True)
if p.returncode != 0:
    print("Failed to execute cargo equip")
    print(p.stderr.decode().replace('\r', ''))
    sys.exit(1)

# overwrite file
with open(file, mode='w') as f:
    f.write(p.stdout.decode('cp932', errors = 'ignore').replace('\r', ''))

