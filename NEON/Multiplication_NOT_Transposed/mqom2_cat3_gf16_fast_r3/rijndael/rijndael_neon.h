#ifndef __RIJNDAEL_NEON_H__
#define __RIJNDAEL_NEON_H__

#include "rijndael_common.h"
#include <arm_neon.h>

/* The general Rijndael core context structure */
typedef struct
{
	rijndael_type rtype; /* Type of Rijndael */
        uint32_t Nr; /* Number of rounds */
        uint32_t Nk; /* Number of words in the key */
        uint32_t Nb; /* Number of words in the block*/
        /* Round keys aligned on 4 bytes to avoid unaligned accesses */
        /* NOTE: alignment to allow uint32_t dereference on platforms where unalignment might be an issue */
        __attribute__((aligned(4))) uint8x16_t rk[30];  /* Round keys    */
	__attribute__((aligned(4))) uint32_t bytearray[120];
} rijndael_neon_ctx;


/* ==== Public API ==== */
int aes128_neon_setkey_enc(rijndael_neon_ctx *ctx, const uint8_t key[16]);
int aes256_neon_setkey_enc(rijndael_neon_ctx *ctx, const uint8_t key[32]);
int rijndael256_neon_setkey_enc(rijndael_neon_ctx *ctx, const uint8_t key[32]);
int aes128_neon_enc(const rijndael_neon_ctx *ctx, const uint8_t data_in[16], uint8_t data_out[16]);
int aes256_neon_enc(const rijndael_neon_ctx *ctx, const uint8_t data_in[16], uint8_t data_out[16]);
int rijndael256_neon_enc(const rijndael_neon_ctx *ctx, const uint8_t data_in[32], uint8_t data_out[32]);
/* x2 and x4 encryption APIs */
int aes128_neon_enc_x2(const rijndael_neon_ctx *ctx1, const rijndael_neon_ctx *ctx2, const uint8_t plainText1[16], const uint8_t plainText2[16], uint8_t cipherText1[16], uint8_t cipherText2[16]);
int aes128_neon_enc_x4(const rijndael_neon_ctx *ctx1, const rijndael_neon_ctx *ctx2, const rijndael_neon_ctx *ctx3, const rijndael_neon_ctx *ctx4,
                const uint8_t plainText1[16], const uint8_t plainText2[16], const uint8_t plainText3[16], const uint8_t plainText4[16],
                uint8_t cipherText1[16], uint8_t cipherText2[16], uint8_t cipherText3[16], uint8_t cipherText4[16]);
int aes128_neon_enc_x8(const rijndael_neon_ctx *ctx1, const rijndael_neon_ctx *ctx2, const rijndael_neon_ctx *ctx3, const rijndael_neon_ctx *ctx4,
                  const rijndael_neon_ctx *ctx5, const rijndael_neon_ctx *ctx6, const rijndael_neon_ctx *ctx7, const rijndael_neon_ctx *ctx8,
                const uint8_t plainText1[16], const uint8_t plainText2[16], const uint8_t plainText3[16], const uint8_t plainText4[16],
                const uint8_t plainText5[16], const uint8_t plainText6[16], const uint8_t plainText7[16], const uint8_t plainText8[16],
                uint8_t cipherText1[16], uint8_t cipherText2[16], uint8_t cipherText3[16], uint8_t cipherText4[16],
                uint8_t cipherText5[16], uint8_t cipherText6[16], uint8_t cipherText7[16], uint8_t cipherText8[16]);
int aes256_neon_enc_x2(const rijndael_neon_ctx *ctx1, const rijndael_neon_ctx *ctx2, const uint8_t plainText1[16], const uint8_t plainText2[16], uint8_t cipherText1[16], uint8_t cipherText2[16]);
int aes256_neon_enc_x4(const rijndael_neon_ctx *ctx1, const rijndael_neon_ctx *ctx2, const rijndael_neon_ctx *ctx3, const rijndael_neon_ctx *ctx4,
                const uint8_t plainText1[16], const uint8_t plainText2[16], const uint8_t plainText3[16], const uint8_t plainText4[16],
                uint8_t cipherText1[16], uint8_t cipherText2[16], uint8_t cipherText3[16], uint8_t cipherText4[16]);
int aes256_neon_enc_x8(const rijndael_neon_ctx *ctx1, const rijndael_neon_ctx *ctx2, const rijndael_neon_ctx *ctx3, const rijndael_neon_ctx *ctx4,
                  const rijndael_neon_ctx *ctx5, const rijndael_neon_ctx *ctx6, const rijndael_neon_ctx *ctx7, const rijndael_neon_ctx *ctx8,
                const uint8_t plainText1[16], const uint8_t plainText2[16], const uint8_t plainText3[16], const uint8_t plainText4[16],
                const uint8_t plainText5[16], const uint8_t plainText6[16], const uint8_t plainText7[16], const uint8_t plainText8[16],
                uint8_t cipherText1[16], uint8_t cipherText2[16], uint8_t cipherText3[16], uint8_t cipherText4[16],
                uint8_t cipherText5[16], uint8_t cipherText6[16], uint8_t cipherText7[16], uint8_t cipherText8[16]);
int rijndael256_neon_enc_x2(const rijndael_neon_ctx *ctx1, const rijndael_neon_ctx *ctx2,
                        const uint8_t plainText1[32], const uint8_t plainText2[32],
                        uint8_t cipherText1[32], uint8_t cipherText2[32]);
int rijndael256_neon_enc_x4(const rijndael_neon_ctx *ctx1, const rijndael_neon_ctx *ctx2, const rijndael_neon_ctx *ctx3, const rijndael_neon_ctx *ctx4,
                const uint8_t plainText1[32], const uint8_t plainText2[32], const uint8_t plainText3[32], const uint8_t plainText4[32],
                uint8_t cipherText1[32], uint8_t cipherText2[32], uint8_t cipherText3[32], uint8_t cipherText4[32]);
int rijndael256_neon_enc_x8(const rijndael_neon_ctx *ctx1, const rijndael_neon_ctx *ctx2, const rijndael_neon_ctx *ctx3, const rijndael_neon_ctx *ctx4,
                  const rijndael_neon_ctx *ctx5, const rijndael_neon_ctx *ctx6, const rijndael_neon_ctx *ctx7, const rijndael_neon_ctx *ctx8,
                const uint8_t plainText1[32], const uint8_t plainText2[32], const uint8_t plainText3[32], const uint8_t plainText4[32],
                const uint8_t plainText5[32], const uint8_t plainText6[32], const uint8_t plainText7[32], const uint8_t plainText8[32],
                uint8_t cipherText1[32], uint8_t cipherText2[32], uint8_t cipherText3[32], uint8_t cipherText4[32],
                uint8_t cipherText5[32], uint8_t cipherText6[32], uint8_t cipherText7[32], uint8_t cipherText8[32]);

#endif /* __RIJNDAEL_NEON_H__ */
