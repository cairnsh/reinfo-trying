#!/bin/bash
python -u train-battle-sexes.py \
    > prisoners_dilemma.log &
PID=$!
trap "trap - TERM && kill $PID" INT TERM EXIT
tail -f prisoners_dilemma.log
