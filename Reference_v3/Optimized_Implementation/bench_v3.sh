#!/bin/bash

DATA=$(date '+%Y%m%d')

# for dir in */; do 
#     echo "👉 Processando $dir"
#     cd "$dir"

#     make clean

#     make bench

#     ./bench 5000 > "bench_v3_$DATA.txt"

#     echo "👉 Bench's feitos :)"

#     make clean
#     cd ..

# done


cd Reference_v3/Optimized_Implementation/mqom3_cat1_gf2_shorter_ct

make clean

make bench

./bench 5000 > "bench_v3_20261002.txt"

echo "👉 Bench's feitos :)"

make clean