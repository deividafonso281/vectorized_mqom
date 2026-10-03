#!/bin/bash

DATA=$(date '+%Y%m%d')

for dir in "NEON/Row-wise"*/; do

    echo "👉 Processando $dir"
    cd "$dir"

    make clean

    make bench

    ./bench 5000 > "bench_camera_ready_$DATA.txt"

    make clean

    echo "👉 Bench's feitos :)"
    cd ..

done

for dir in "NEON/Multiplication_NOT_Transposed"*/; do

    echo "👉 Processando $dir"
    cd "$dir"

    make clean

    make bench

    ./bench 5000 > "bench_camera_ready_$DATA.txt"

    make clean

    echo "👉 Bench's feitos :)"
    cd ..

done

