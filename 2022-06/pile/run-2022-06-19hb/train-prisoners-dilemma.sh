#!/bin/bash
python -u train-prisoners-dilemma.py \
    > prisoners_dilemma.log &
PID=$!
trap "trap - TERM && kill $PID" INT TERM EXIT
tail -f prisoners_dilemma.log
