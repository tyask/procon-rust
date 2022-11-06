#!/bin/bash

CONTEST=$1
OUT=$(cargo compete new $CONTEST 2>&1)
echo "$OUT"

PACKAGE=$(echo $OUT | head -n1 | awk '{print $5}' | sed 's|\\|/|g')
find $PACKAGE -name '*.rs' | while read SRC; do cat $(dirname $0)/../template/src/main.rs > $SRC; done
cp -r $(dirname $0)/../template/.vscode $PACKAGE
code $PACKAGE
