#!/bin/bash

CONTEST=$1
OUT=$(cargo compete new $CONTEST 2>&1)
echo "$OUT"

TEMPLATE=$(dirname $0)/../contest/template
PACKAGE=$(echo "$OUT" | grep 'Created' | awk '{print $5}' | sed 's|\\|/|g')
cp $TEMPLATE/rust-toolchain.toml $PACKAGE

find $PACKAGE -name '*.rs' | while read SRC; do
    cat $TEMPLATE/src/main.rs > $SRC;
    cargo capture --module $TEMPLATE --target $SRC
done


echo '''
[features]
default = ["local"]
local = []''' >> $PACKAGE/Cargo.toml
