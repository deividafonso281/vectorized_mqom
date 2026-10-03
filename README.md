# Vectorized MQOM

This repository contains the code of the Paper "Efficient Vectorized Implementation of MQOM" submitted to TCHES 2027. The code was created modifying the reference code available at [MQOM](https://mqom.org/resources.html) and thus should have the same license.

The structure of the repository is as follows:

AVX: holds the code used in the experiments for MQOM version 2.1 on Intel platforms. There are implementations for the Reference code, code using the Row-wise multiplication method and code using the Column-wise multiplication method (only for GF(16) variants). The reference code was modified to have granular measurements of the amount of clock cycles spent in each step of the verification.

NEON: holds the code used in the experiments for MQOM version 2.1 on Arm NEON platforms. There are implementations for the Reference code adapted to use NEON instructions, code using the Row-wise multiplication method and code using the Column-wise multiplication method.

Reference_v3: holds the reference code of MQOM version 3 with modifications in the Makefile's. Specifically, the flag FORCE_PLATFORM is set depending on the experiment and it always use gcc as compiler. The benchmark/timing.c was also modified to set_cpu_affinity(2) instead of set_cpu_affinity(0).

Row-wise_v3: holds the code of MQOM version 3 adapted to use the packed vector-matrix multiplication method proposed in the paper. It has the same modifications in the Makefile's and benchmark/timing.c as Reference_v3 and also the following files were changed:
- fields/fields_arm_neon.h
- fields/fields_avx2.h
- fields/fields_avx512.h
- piop/piop_default.c
- fields.h
- expand_mq.c
- expand_mq.h
- keygen.c

# Platforms Used

We executed the NEON code on a Raspberry Pi 5 with a Cortex-A76 processor and it was necessary to have sudo privileges.

We executed the AVX code on an 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz with HyperThreading and TurboBoost disabled. 

# Instructions for benchmarks

## NEON

First we compiled the kernel module pmu_enable.c and installed the pmu_enable.ko using the command "sudo insmod pmu_enable.ko". Then, for the version 2.1 we executed "sudo ./profile.sh v2_neon" and for version 3 "sudo ./profile.sh v3_neon".

## AVX

For version 2.1 we executed "sudo ./bench_v2_avx.sh". So far we tested the Row-wise method on AVX and it passes the KAT's but we still do not have measures for it. In order to acquire measures one should execute "sudo ./bench_v3_avx.sh".

