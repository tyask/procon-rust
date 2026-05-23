#!/bin/bash

PROBLEM=$1 # a,b
if [ -z "$PROBLEM" ]; then
    cargo run
    exit
fi

if [[ $PROBLEM =~ *\.rs ]]; then
    PROBLEM=$(basename ${SRC_FILE%.*})
fi
SRC_FILE=$(find . -name $PROBLEM.rs)
CONTEST=$(basename $(cd $(dirname $SRC_FILE)/../..; pwd))
BIN=$CONTEST-$PROBLEM

cargo run --bin $BIN
