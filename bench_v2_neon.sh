#!/bin/bash

for dir in "NEON/Row-wise"*/; do

    echo "👉 Processando $dir"
    cd "$dir"

    make clean

    make bench

    ./bench 5000 > "bench_neon.txt"

    make clean

    echo "👉 Bench's feitos :)"
    cd ..

done

for dir in "NEON/Multiplication_NOT_Transposed"*/; do

    echo "👉 Processando $dir"
    cd "$dir"

    make clean

    make bench

    ./bench 5000 > "bench_neon.txt"

    make clean

    echo "👉 Bench's feitos :)"
    cd ..

done

