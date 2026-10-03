#include <stdio.h>

int main(void) {
#ifdef __ARM_NEON__
    printf("__ARM_NEON__ definido\n");
#endif
#ifdef __ARM_NEON
    printf("__ARM_NEON definido\n");
#endif
#ifdef __ARM_NEON_FP
    printf("__ARM_NEON_FP definido: %d\n", __ARM_NEON_FP);
#endif
#ifdef __ARM_FEATURE_CRYPTO
    printf("__ARM_FEATURE_CRYPTO definido\n");
#endif
return 0;
}
