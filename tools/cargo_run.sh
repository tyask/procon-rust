#!/bin/bash

SRC_FILE=$1
PROBLEM=$(basename ${SRC_FILE%.*})
CONTEST=$(basename $(cd $(dirname $SRC_FILE)/../..; pwd))
BIN=$CONTEST-$PROBLEM

cargo run --bin $BIN
