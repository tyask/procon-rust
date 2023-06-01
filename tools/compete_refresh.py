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

def lookup_template(src):
    return os.path.normpath(os.path.join(os.path.dirname(src), 'template.rs'))

def cargo_capture(mod, src):
    subprocess.run('cargo capture --module {} --target {}'.format(mod, src), shell=True)

def main():
    src = sys.argv[1]
    template = lookup_template(src)
    mod = lookup_module_project(src)
    print('src={}'.format(src))
    print('template={}'.format(template))
    print('module project={}'.format(mod))

    if not os.path.exists(template):
        print("Missing template file: {}".format(template))
        return

    if not os.path.exists(mod):
        print("Missing module project: {}".format(mod))
        return

    shutil.copy(template, src)
    cargo_capture(mod, src)

if __name__ == '__main__':
    main()
