#!/bin/bash

CONTEST=$1
OUT=$(cargo compete new $CONTEST 2>&1)
echo "$OUT"

TEMPLATE=$(dirname $0)/../contest/template
PACKAGE=$(echo $OUT | head -n1 | awk '{print $5}' | sed 's|\\|/|g')
find $PACKAGE -name '*.rs' | while read SRC; do cat $TEMPLATE/src/bin/main.rs > $SRC; done
cp -r $TEMPLATE/.vscode $PACKAGE
