#!/bin/env python
import os
import re
import shutil
import subprocess
import sys

"""
{src}をsrc/bin/template.rsに置き換える.
"""

def lookup_module_project(src):
    return os.path.normpath(os.path.join(os.path.dirname(src), '..', '..', '..', 'template'))

def lookup_template(mod):
    return os.path.normpath(os.path.join(mod, 'src', 'main.rs'))

def cargo_capture(mod, src):
    subprocess.run('cargo capture --module {} --target {}'.format(mod, src), shell=True)

def refresh_src(src):
    mod = lookup_module_project(src)
    template = lookup_template(mod)

    if not os.path.exists(template):
        print("Missing template file: {}".format(template))
        return

    if not os.path.exists(mod):
        print("Missing module project: {}".format(mod))
        return

    shutil.copy(template, src)
    cargo_capture(mod, src)
    print('src={}, template={}, module project={}'.format(src, template, mod))

def main():
    for src in sys.argv[1:]:
        refresh_src(src)

if __name__ == '__main__':
    main()
