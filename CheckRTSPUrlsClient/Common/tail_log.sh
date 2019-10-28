#!/usr/bin/env bash

if [[ $# != 1 ]]; then
    echo
    echo " Usage: ${0} \${FOLDER}"
    echo
    exit
fi

LOG_FILE="$(find ${1} -type f -printf '%T@ %p\n' | grep ".log" | sort -n | tail -1 | cut -f2- -d" ")"
tail -f ${LOG_FILE}
