#!/bin/bash

DATA=$(date '+%Y%m%d')

version=$1

echo "Garantindo que a temperatura esta segura"
TEMP=$(vcgencmd measure_temp)
echo "Temperatura atual: $TEMP"

echo userspace | sudo tee /sys/devices/system/cpu/cpu2/cpufreq/scaling_governor
echo 2400000 | sudo tee /sys/devices/system/cpu/cpu2/cpufreq/scaling_setspeed

echo "Isolando a cpu 2"
cset shield --cpu=2 --kthread=on 

cset shield --exec -- ./bench_${version}.sh > bench_${version}_${DATA}.log
