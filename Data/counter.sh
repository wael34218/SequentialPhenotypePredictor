#!/bin/bash
line_num=1

while read line;do
    printf "$(echo $line | wc -w)\n"
    ((line_num++))
done
