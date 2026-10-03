#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>

#include "api.h"

int randombytes(unsigned char* x, unsigned long long xlen) {
    for(unsigned long long j=0; j<xlen; j++)
        x[j] = rand();
    return 0;
}

int main() {

    // Execution
    int ret;
    // Generate the keys
    uint8_t pk[CRYPTO_PUBLICKEYBYTES];
    uint8_t sk[CRYPTO_SECRETKEYBYTES];
    ret = crypto_sign_keypair(pk, sk);
    if(ret) {
        printf("Failure: crypto_sign_keypair\n");
        return 1;
    }

    return 0;
}
