#!/bin/bash

DATA=$(date '+%Y%m%d')

for dir in "Row-wise_v3/Optimized_Implementation"*/; do
    for subdir in "$dir"*/; do
        echo "👉 Processando $subdir"
        cd "$subdir"

        sed -i '1s/.*/FORCE_PLATFORM_AARCH64_AES=1/' Makefile

        make clean

        make bench

        ./bench 5000 > "bench_neon_$DATA.txt"

        make clean

        echo "👉 Bench's feitos :)"
        cd ../../..
    done 
done

for dir in "Reference_v3/Optimized_Implementation"*/; do
    for subdir in "$dir"*/; do
        echo "👉 Processando $subdir"
        cd "$subdir"

        sed -i '1s/.*/FORCE_PLATFORM_AARCH64_AES=1/' Makefile

        make clean

        make bench

        ./bench 5000 > "bench_neon_$DATA.txt"

        make clean

        echo "👉 Bench's feitos :)"
        cd ../../..
    done 
done
