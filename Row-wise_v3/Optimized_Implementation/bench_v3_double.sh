#!/bin/bash

DATA=$(date '+%Y%m%d')

for dir in */; do 
    echo "👉 Processando $dir"
    cd "$dir"

    make clean

    make bench

    ./bench 5000 > "bench_v3_$DATA.txt"

    echo "👉 Bench's feitos :)"

    make clean
    cd ..

done

