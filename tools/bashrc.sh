#!/bin/bash

alias cnew="$(dirname ${BASH_SOURCE[0]})/compete_new.sh"
alias t='cargo compete test'
alias r="$(dirname ${BASH_SOURCE[0]})/cargo_run.sh"
alias d="compete_dt"

function compete_dt() {
    if [ -z "$1" ]; then
        echo "ERROR: You must specify source file name"
        echo "Ex) d a"
        return 1
    fi
    $(dirname ${BASH_SOURCE[0]})/compete_dt.py src/bin/$1.rs
}
