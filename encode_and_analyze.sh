#!/bin/bash


./encode.sh ${1} ${3}

python analyze.py -dirname ${3} -n ${1} -target_path ${2}

