#!/bin/bash

DATA=$(date '+%Y%m%d')

for dir in "AVX/Row-wise"/; do
    for subdir in "$dir"*/; do
        echo "👉 Processando $subdir"
        cd "$subdir"

        sed -i '1s/.*/FORCE_PLATFORM_AVX512_GFNI=1/' Makefile

        make clean

        make bench

        ./bench 5000 > "bench_avx512_gfni_gcc_$DATA.txt"

        sed -i '1s/.*/FORCE_PLATFORM_AVX2=1/' Makefile

        make clean

        make bench

        ./bench 5000 > "bench_avx2_gcc_$DATA.txt"

        make clean

        echo "👉 Bench's feitos :)"
        cd ../..
    done 
done

for dir in "AVX/Reference"*/; do
    for subdir in "$dir"*/; do
        echo "👉 Processando $subdir"
        cd "$subdir"

        sed -i '1s/.*/FORCE_PLATFORM_AVX512_GFNI=1/' Makefile

        make clean

        make bench

        ./bench 5000 > "bench_avx512_gfni_gcc_$DATA.txt"

        sed -i '1s/.*/FORCE_PLATFORM_AVX2=1/' Makefile

        make clean

        make bench

        ./bench 5000 > "bench_avx2_gcc_$DATA.txt"

        make clean

        echo "👉 Bench's feitos :)"
        cd ../..
    done 
done

