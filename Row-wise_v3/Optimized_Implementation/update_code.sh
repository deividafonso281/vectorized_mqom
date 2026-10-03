#!/bin/bash

neonfields="mqom3_cat1_gf2_shorter_ct/fields"
neonmkf="mqom3_cat1_gf2_shorter_ct/Makefile"
neonfieldsh="mqom3_cat1_gf2_shorter_ct/fields.h"
neonpiop="mqom3_cat1_gf2_shorter_ct/piop"
neonexpandmq="mqom3_cat1_gf2_shorter_ct/expand_mq.c"
neonexpandmqh="mqom3_cat1_gf2_shorter_ct/expand_mq.h"
neonkeygen="mqom3_cat1_gf2_shorter_ct/keygen.c"
neonbench="mqom3_cat1_gf2_shorter_ct/benchmark"


for dir in */; do
    case "$dir" in mqom2_cat1_gf16_fast_r3)
        echo "⏭️ pulando $dir"
        continue
        ;;
    esac 

    echo "👉 Processando $dir"
    cp -rf "$neonfields" "$dir"
    #cp -rf "$neonmkf" "$dir"
    cp "$neonfieldsh" "$dir"
    cp -rf "$neonpiop" "$dir"
    cp "$neonexpandmq" "$dir"
    cp "$neonexpandmqh" "$dir"
    cp "$neonkeygen" "$dir"
    cp -rf "$neonbench" "$dir"

done


echo "👉 Conjuntos de parametros atualizados"