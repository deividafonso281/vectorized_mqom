#!/bin/bash

for dir in "NEON/Row-wise"*/; do
    for subdir in "$dir"*/; do
        echo "👉 Processando $subdir"
        cd "$subdir"

        make clean

        make bench

        ./bench 5000 > "bench_neon.txt"

        make clean

        echo "👉 Bench's feitos :)"
        cd ../../..
    done
done

for dir in "NEON/Multiplication_NOT_Transposed"*/; do
    for subdir in "$dir"*/; do
        echo "👉 Processando $subdir"
        cd "$subdir"

        make clean

        make bench

        ./bench 5000 > "bench_neon.txt"

        make clean

        echo "👉 Bench's feitos :)"
        cd ../../..
    done
done

