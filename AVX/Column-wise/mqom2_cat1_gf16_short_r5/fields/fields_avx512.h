#ifndef __FIELDS_AVX512_H__
#define __FIELDS_AVX512_H__

/* Check if AVX512 is supported */
//#if defined(__AVX512BW__) && defined(__AVX512F__) && defined(__AVX512VL__) && defined(__AVX512VPOPCNTDQ__) && defined(__AVX512VBMI__)

#include "fields_common.h"
#include "fields_ref.h"
/* Needed for memcpy */
#include <string.h>
/* Needed for AVX512 assembly intrinsics */
#include <immintrin.h>

#define TRANSPOSE_MULT_ADVANCED

/* This helper tries to efficiently copy len bytes in the ymm register */
static inline __m512i load_incomplete_m512(const uint8_t *a, uint32_t len)
{
        const __m512i zero = _mm512_setzero_epi32();

	if(len % 4){
		/* In this case, we switch to the slower _mm512_mask_loadu_epi8 */
		__mmask64 mask = ((__mmask64)1 << len) - 1;
		return _mm512_mask_loadu_epi8(zero, mask, (int const*)a);
	}
	else{
		/* Optimize with faster instruction */
		__mmask16 mask = (1 << (len / 4)) - 1;
        	return _mm512_mask_loadu_epi32(zero, mask, (int const*)a);
	}
}

/* This helper tries to efficiently copy len bytes from the ymm register */
static inline void store_incomplete_m512(__m512i in, uint8_t *a, uint32_t len)
{
	if(len % 4){
		/* In this case, we switch to the slower _mm512_mask_storeu_epi8 */
		__mmask64 mask = ((__mmask64)1 << len) - 1;
		_mm512_mask_storeu_epi8((void*)a, mask, in);
	}
	else{
		/* Optimize with faster instruction */
		__mmask16 mask = (1 << (len / 4)) - 1;
        	_mm512_mask_storeu_epi32((void*)a, mask, in);
	}
}

static inline uint8_t parity_avx512(__m512i accu) {
	uint32_t i;
	uint8_t res;
	__attribute__((aligned(64))) uint64_t local_c[8];

	/* We have to compute the parity bit, do it 64 bits per 64 bits */
	accu = _mm512_popcnt_epi64(accu);
	/* Store the result */
	_mm512_storeu_epi64((__m512i*)local_c, accu);
	/* Finish the parity bit computation on the 64-bit parts */
	res = 0;
	for(i = 0; i < 8; i++){
		res ^= (local_c[i] & 1);
	}
	return res;
}

static inline uint8_t sum_uint8_avx512(__m512i accu) {
	uint32_t i;
	uint8_t res;
	__attribute__((aligned(64))) uint8_t local_c[64];

	/* Store the result */
	_mm512_storeu_epi64((__m512i*)local_c, accu);
	/* Finish the xor computation byte pet byte  */
	res = 0;
	for(i = 0; i < 64; i++){
		res ^= local_c[i];
	}

	return res;
}

static inline uint16_t sum_uint16_avx512(__m512i accu) {
	uint32_t i;
	uint16_t res;
	__attribute__((aligned(64))) uint16_t local_c[32];

	/* Store the result */
	_mm512_storeu_epi64((__m512i*)local_c, accu);
	/* Finish the xor computation byte pet byte  */
	res = 0;
	for(i = 0; i < 32; i++){
		res ^= local_c[i];
	}

	return res;
}

/* === GF(2) === */
/* NOTE: for atomic multiplication, using vectorization is suboptimal */
static inline uint8_t gf2_mult_avx512(uint8_t a, uint8_t b)
{
        return gf2_mult_ref(a, b);
}

/*
 * Vector multiplied by a constant in GF(2).
 */
static inline void gf2_constant_vect_mult_avx512(uint8_t b, const uint8_t *a, uint8_t *c, uint32_t len)
{
        gf2_constant_vect_mult_ref(b, a, c, len);

        return;
}

/* XXX: TODO: This is not optimal, since we hardly fill our zmm register to its full capacity */
static inline uint8_t gf2_vect_mult_avx512(const uint8_t *a, const uint8_t *b, uint32_t len_bits)
{
	uint32_t i;
	__m512i accu, _a, _b;
	uint32_t len = (len_bits / 8);

	/* Set the accumulator to 0 */
	accu = _mm512_setzero_epi32();

	for(i = 0; i < len; i += 64){
                if((len-i) < 64){
                        _a = load_incomplete_m512(&a[i], len-i);
                        _b = load_incomplete_m512(&b[i], len-i);
                }
                else{
                        /* Obvious 512-bit */
                        _a = _mm512_loadu_epi64((__m512i*)&a[i]);
                        _b = _mm512_loadu_epi64((__m512i*)&b[i]);
                }
                /* Vectorized AND of inputs and then XOR with the accumulator */
                accu ^= (_a & _b);
	}

	return parity_avx512(accu);
}

/* Matrix and vector multiplication over GF(2) 
 * C = A * X, where X is a vector
 * Matrix is supposed to be square n x n, and vector n x 1
 * The output is a vector n x 1
 * */
/* XXX: TODO: this can be optimized by packing rows in zmm when n <= 256 */
static inline void gf2_mat_mult_avx512(const uint8_t *A, const uint8_t *X, uint8_t *Y, uint32_t n, matrix_type mtype)
{
	GF2_MAT_MULT(A, X, Y, n, mtype, gf2_vect_mult_avx512);
}

/* GF(2) matrix transposition */
static inline void gf2_mat_transpose_avx512(const uint8_t *A, uint8_t *B, uint32_t n, matrix_type mtype)
{
	gf2_mat_transpose_ref(A, B, n, mtype);
}

/* === GF(256) === */
/* NOTE: for atomic multiplication, using vectorization is suboptimal */
static inline uint8_t gf256_mult_avx512(uint8_t x, uint8_t y)
{ 
        return gf256_mult_ref(x, y);
}

static inline __m512i gf256_mult_vectorized_avx512(__m512i _a, __m512i _b)
{
	/* NOTE: when GFNI is detected, we use the accelerated GF(256) Rijndael instruction */
#if defined(__GFNI__) && !defined(NO_GFNI)
	return _mm512_gf2p8mul_epi8(_a, _b);
#else
        /* NOTE: because we do not have a blending based on AVX512, and given the fact that 
         * 64 bits mask extraction is a bit slow, and also given the fact that when using triangular matrices
         * we hit more alignment issues with AVX512, we prefer to fallback to AVX2 on each half of the AVX512
         * The "pure" AVX512 implementation is left in comment below for information, but it is less efficient
         * than the AVX2 fallback (at least on the tested AMD Zen 4 platform) */
        __m256i a_low  = _mm512_castsi512_si256(_a);
        __m256i b_low  = _mm512_castsi512_si256(_b);
        __m256i a_high = _mm512_extracti64x4_epi64(_a, 1);
        __m256i b_high = _mm512_extracti64x4_epi64(_b, 1);
        
        uint32_t j;
        __m256i mask_lsb_low, mask_lsb_high, tmp_low, tmp_high;
        const __m256i red_poly = _mm256_set_epi64x(0x1B1B1B1B1B1B1B1B, 0x1B1B1B1B1B1B1B1B, 0x1B1B1B1B1B1B1B1B, 0x1B1B1B1B1B1B1B1B);
        const __m256i zero     = _mm256_setzero_si256();
        __m256i accu_low = _mm256_setzero_si256();
        __m256i accu_high = _mm256_setzero_si256();

        /* Compute the vectorized multiplication in GF(256) */ 
        for(j = 0; j < 8; j++){
                mask_lsb_low  = _mm256_slli_epi64(b_low, 7-j);
                mask_lsb_high = _mm256_slli_epi64(b_high, 7-j);
                accu_low  ^= _mm256_blendv_epi8(zero, a_low, mask_lsb_low);
                accu_high ^= _mm256_blendv_epi8(zero, a_high, mask_lsb_high);
                tmp_low  = _mm256_add_epi8(a_low, a_low);
                tmp_high = _mm256_add_epi8(a_high, a_high);
                a_low  = _mm256_blendv_epi8(zero, red_poly, a_low)  ^ tmp_low;
                a_high = _mm256_blendv_epi8(zero, red_poly, a_high) ^ tmp_high;
        }
                
        __m512i _accu_low = _mm512_castsi256_si512(accu_low);
        __m512i accu = _mm512_inserti64x4(_accu_low, accu_high, 1);
        
        return accu;
#if 0
        /* Our reduction polynomial */
        const __m512i red_poly = _mm512_set_epi64(0x1B1B1B1B1B1B1B1B, 0x1B1B1B1B1B1B1B1B, 0x1B1B1B1B1B1B1B1B, 0x1B1B1B1B1B1B1B1B,
                                                  0x1B1B1B1B1B1B1B1B, 0x1B1B1B1B1B1B1B1B, 0x1B1B1B1B1B1B1B1B, 0x1B1B1B1B1B1B1B1B);
        const __m512i zero     = _mm512_setzero_epi32();
        __m512i accu = _mm512_setzero_epi32();

        uint32_t j;
        __mmask64 mask;
        __m512i tmp;

        /* Compute the vectorized multiplication in GF(256) */
        for(j = 0; j < 8; j++){
                mask = _mm512_movepi8_mask(_mm512_slli_epi64(_b, 7-j));
                accu ^= _mm512_mask_blend_epi8(mask, zero, _a);
                tmp  = _mm512_add_epi8(_a, _a);
                mask = _mm512_movepi8_mask(_a);
                _a = _mm512_mask_blend_epi8(mask, zero, red_poly) ^ tmp;
        }
        return accu;
#endif
#endif
}

/*              
 * Vector multiplied by a constant in GF(256).
 */                     
static inline void gf256_constant_vect_mult_avx512(uint8_t b, const uint8_t *a, uint8_t *c, uint32_t len)
{
        uint32_t i;
        __m512i _a, _b;

	/* Create a vector _b with the duplicated constant everywhere */
	_b = _mm512_set1_epi8(b);

        for(i = 0; i < len; i += 64){
                if((len-i) < 64){
                        _a = load_incomplete_m512(&a[i], len-i);
	                /* Vectorized multiplication in GF(256) */
        	        store_incomplete_m512(gf256_mult_vectorized_avx512(_a, _b), &c[i], len-i);
                }
                else{
                        /* Obvious 512-bit */
                        _a = _mm512_loadu_epi64((__m512i*)&a[i]);
	                /* Vectorized multiplication in GF(256) */
        	        _mm512_storeu_epi64(&c[i], gf256_mult_vectorized_avx512(_a, _b));
                }
        }

	return;
}

/*
 * Vector to vector multiplication in GF(256).
 * Takes two vectors of length 'len', and returns a byte (element in GF(256))
 */
static inline uint8_t gf256_vect_mult_avx512(const uint8_t *a, const uint8_t *b, uint32_t len)
{
        uint32_t i;
        __m512i accu, _a, _b;

        /* Set the accumulator to 0 */
        accu = _mm512_setzero_epi32();

        for(i = 0; i < len; i += 64){
                if((len-i) < 64){
                        _a = load_incomplete_m512(&a[i], len-i);
                        _b = load_incomplete_m512(&b[i], len-i);
                }
                else{
                        /* Obvious 512-bit */
                        _a = _mm512_loadu_epi64((__m512i*)&a[i]);
                        _b = _mm512_loadu_epi64((__m512i*)&b[i]);
                }
                /* Vectorized multiplication in GF(256) */
                accu ^= gf256_mult_vectorized_avx512(_a, _b);
        }

	return sum_uint8_avx512(accu);
}

static inline void print_m512i_u64(__m512i v) {
    uint64_t tmp[8];
    _mm512_store_si512(tmp, v);
    for (int i = 3; i >= 0; i--) {
        printf("%016llx", (unsigned long long)tmp[i]);
    }
    printf("\n");
    for (int i = 7; i >= 4; i--) {
        printf("%016llx", (unsigned long long)tmp[i]);
    }
    printf("\n");
}

/* Matrix and vector multiplication over GF(256) 
 * C += A * X, where X is a vector
 * Matrix is supposed to be square n x n, and vector n x 1
 * The output is a vector n x 1
 * */
static inline void gf256_mat_mult_avx512(const uint8_t *A, const uint8_t *X, uint8_t *Y_inf, uint8_t *Y_sup, const uint8_t * even_diagonal, uint32_t n)
{
    uint16_t i, j, k, l;
        uint16_t complete_line = (n + 7) & ~7;
	uint16_t batches, remainder;
	uint32_t row; //, ind;
	__m512i accu, _a, dups, _b;
	__m512i acc_inf,prod;

	__m512i zeros, mask;
        __m512i ones = _mm512_set1_epi64(-1);
	zeros =  _mm512_setzero_si512();
        __m128i lanes[4];

        __mmask64 control;

	/* Set the accumulator to 0 */
	batches = n/64;
	remainder = n - batches * 64;
	for (i=0;i<batches;i++) {
                control = -1;
		acc_inf = _mm512_setzero_si512();
		accu = _mm512_setzero_si512();
		row = i*64;
		for (j=0;j<i;j++) {
			_b = _mm512_loadu_si512((__m512i*)&X[64*j]);
                        lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
                        lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
                        lanes[2] = _mm512_extracti32x4_epi32(_b, 2);
                        lanes[3] = _mm512_extracti32x4_epi32(_b, 3);
                        for (l=0;l<4;l++) {
                                for (k=0;k<16;k++) {
                                        dups = _mm512_broadcastb_epi8(lanes[l]);
                                        _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                        accu ^= gf256_mult_vectorized_avx512(_a,dups);
                                        row = row + complete_line;
                                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1); // ideia do chat
                                }
                        }
		}

		_b = _mm512_loadu_si512((__m512i*)&X[64*j]);
                lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
                lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
                lanes[2] = _mm512_extracti32x4_epi32(_b, 2);
                lanes[3] = _mm512_extracti32x4_epi32(_b, 3);
		_a = _mm512_loadu_si512((__m512i*)&even_diagonal[64*i]);
		accu ^= gf256_mult_vectorized_avx512(_a, _b);
                for (l=0;l<4;l++) {
                        for (k=0;k<16;k++) {
                                control = control << 1;
                                mask = _mm512_mask_blend_epi8(control,zeros,ones);
                                dups = _mm512_broadcastb_epi8(lanes[l]);
                                _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                prod = gf256_mult_vectorized_avx512(_a,dups);
                                accu ^= (mask & prod);
                                acc_inf ^= (_mm512_xor_si512(mask, ones) & prod);
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1); // ideia do chat
                        }
                }
                _mm512_storeu_si512((void*)&Y_sup[i*64], accu);

		for (j=i+1;j<batches;j++) {
			_b = _mm512_loadu_si512((__m512i*)&X[64*j]);
                        lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
                        lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
                        lanes[2] = _mm512_extracti32x4_epi32(_b, 2);
                        lanes[3] = _mm512_extracti32x4_epi32(_b, 3);
                        for (l=0;l<4;l++) {
                                for (k=0;k<16;k++) {
                                        dups = _mm512_broadcastb_epi8(lanes[l]);
                                        _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                        acc_inf ^= gf256_mult_vectorized_avx512(_a,dups);
                                        row = row + complete_line;
                                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1); // ideia do chat
                                }
                        }
		}
		if (remainder > 0) {
		_b = load_incomplete_m512(&X[64*j],remainder);
                lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
                lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
                lanes[2] = _mm512_extracti32x4_epi32(_b, 2);
                lanes[3] = _mm512_extracti32x4_epi32(_b, 3);
                for (l=0;l<4;l++) {
                        for (k=0;k<16 && k < remainder-(l*16);k++) {
                                dups = _mm512_broadcastb_epi8(lanes[l]);
                                _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                _a = gf256_mult_vectorized_avx512(_a,dups);
                                acc_inf ^= _a;
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1); // ideia do chat
                        }
                }
		}
                _mm512_storeu_si512((void*)&Y_inf[i*64], acc_inf);
	}

	// Process last block, not my falt -\o/-
	if (remainder > 0) {
	acc_inf = _mm512_setzero_si512();
	accu = _mm512_setzero_si512();
	row = batches*64;
        control = -1;
	for (j=0;j<batches;j++) {
                _b = _mm512_loadu_si512((__m512i*)&X[64*j]);
		lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
                lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
                lanes[2] = _mm512_extracti32x4_epi32(_b, 2);
                lanes[3] = _mm512_extracti32x4_epi32(_b, 3);
                for (l=0;l<4;l++) {
                        for (k=0;k<16;k++) {
                                dups = _mm512_broadcastb_epi8(lanes[l]);
                                _a = load_incomplete_m512(&A[row],remainder);
                                accu ^= gf256_mult_vectorized_avx512(_a,dups);
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1); // ideia do chat
                        }
                }
	}

        zeros = _mm512_setzero_si512();
	_b = load_incomplete_m512(&X[64*j],remainder);
	lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
        lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
        lanes[2] = _mm512_extracti32x4_epi32(_b, 2);
        lanes[3] = _mm512_extracti32x4_epi32(_b, 3);
        _a = load_incomplete_m512(&even_diagonal[64*batches],remainder);
	accu ^= gf256_mult_vectorized_avx512(_a, _b);
	for (l=0;l<4;l++) {
                for (k=0;k<16 && k < remainder-(l*16);k++) {
                        control = control << 1;
                        mask = _mm512_mask_blend_epi8(control,zeros,ones);
                        dups = _mm512_broadcastb_epi8(lanes[l]);
                        _a = load_incomplete_m512(&A[row],remainder);
                        prod = gf256_mult_vectorized_avx512(_a,dups);
                        accu ^= (mask & prod);
                        acc_inf ^= (_mm512_xor_si512(mask, ones) & prod);
                        row = row + complete_line;
                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1); // ideia do chat
                }
        }
	store_incomplete_m512(accu,&Y_sup[batches*64],remainder);
	store_incomplete_m512(acc_inf,&Y_inf[batches*64],remainder);
	// End last block
	}
}

static inline void gf256_mat_mult_single_avx512(const uint8_t *A, const uint8_t *X, uint8_t *Y, uint32_t n)
{
    uint16_t i, j, k, l;
        uint16_t complete_line = (n + 7) & ~7;
	uint16_t batches, remainder;
	uint32_t row; //, ind;
	__m512i _a, dups, _b;
	__m512i acc,prod;

	__m512i zeros, mask;
        __m512i ones = _mm512_set1_epi64(-1);
	zeros =  _mm512_setzero_si512();
        __m128i lanes[4];

        __mmask64 control;

	/* Set the accumulator to 0 */
	batches = n/64;
	remainder = n - batches * 64;
	for (i=0;i<batches;i++) {
                control = -1;
		acc = _mm512_setzero_si512();
		row = (complete_line+1)*i*64;

		_b = _mm512_loadu_si512((__m512i*)&X[64*j]);
                lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
                lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
                lanes[2] = _mm512_extracti32x4_epi32(_b, 2);
                lanes[3] = _mm512_extracti32x4_epi32(_b, 3);
                for (l=0;l<4;l++) {
                        for (k=0;k<16;k++) {
                                control = control << 1;
                                mask = _mm512_mask_blend_epi8(control,zeros,ones);
                                dups = _mm512_broadcastb_epi8(lanes[l]);
                                _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                prod = gf256_mult_vectorized_avx512(_a,dups);
                                acc ^= (_mm512_xor_si512(mask, ones) & prod);
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1); // ideia do chat
                        }
                }

		for (j=i+1;j<batches;j++) {
			_b = _mm512_loadu_si512((__m512i*)&X[64*j]);
                        lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
                        lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
                        lanes[2] = _mm512_extracti32x4_epi32(_b, 2);
                        lanes[3] = _mm512_extracti32x4_epi32(_b, 3);
                        for (l=0;l<4;l++) {
                                for (k=0;k<16;k++) {
                                        dups = _mm512_broadcastb_epi8(lanes[l]);
                                        _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                        acc ^= gf256_mult_vectorized_avx512(_a,dups);
                                        row = row + complete_line;
                                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1); // ideia do chat
                                }
                        }
		}
		if (remainder > 0) {
		_b = load_incomplete_m512(&X[64*j],remainder);
                lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
                lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
                lanes[2] = _mm512_extracti32x4_epi32(_b, 2);
                lanes[3] = _mm512_extracti32x4_epi32(_b, 3);
                for (l=0;l<4;l++) {
                        for (k=0;k<16 && k < remainder-(l*16);k++) {
                                dups = _mm512_broadcastb_epi8(lanes[l]);
                                _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                _a = gf256_mult_vectorized_avx512(_a,dups);
                                acc ^= _a;
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1); // ideia do chat
                        }
                }
		}
                _mm512_storeu_si512((void*)&Y[i*64], acc);
	}

	// Process last block, not my falt -\o/-
	if (remainder > 0) {
	acc = _mm512_setzero_si512();
	row = (complete_line+1)*batches*64;
        control = -1;

        zeros = _mm512_setzero_si512();
	_b = load_incomplete_m512(&X[64*j],remainder);
	lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
        lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
        lanes[2] = _mm512_extracti32x4_epi32(_b, 2);
        lanes[3] = _mm512_extracti32x4_epi32(_b, 3);
	for (l=0;l<4;l++) {
                for (k=0;k<16 && k < remainder-(l*16);k++) {
                        control = control << 1;
                        mask = _mm512_mask_blend_epi8(control,zeros,ones);
                        dups = _mm512_broadcastb_epi8(lanes[l]);
                        _a = load_incomplete_m512(&A[row],remainder);
                        prod = gf256_mult_vectorized_avx512(_a,dups);
                        acc ^= (_mm512_xor_si512(mask, ones) & prod);
                        row = row + complete_line;
                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1); // ideia do chat
                }
        }
	store_incomplete_m512(acc,&Y[batches*64],remainder);
	// End last block
	}
}

/*
 * "Hybrid" multiplication of a constant in GF(2) and a vector in GF(256)
 */
static inline void gf2_gf256_constant_vect_mult_avx512(uint8_t a_gf2, const uint8_t *b_gf256, uint8_t *c_gf256, uint32_t n)
{
        gf2_gf256_constant_vect_mult_ref(a_gf2, b_gf256, c_gf256, n);

        return;
}

/*
 * "Hybrid" multiplication of a constant in GF(256) and a vector in GF(2)
 */
static inline void gf256_gf2_constant_vect_mult_avx512(uint8_t a_gf256, const uint8_t *b_gf2, uint8_t *c_gf256, uint32_t n)
{
        uint32_t i;
        const __m512i zero = _mm512_setzero_epi32();
        const __m128i zero_128 = _mm_setzero_si128();

	/* Load and broadcast the constant */
	__m512i _a = _mm512_set1_epi8(a_gf256);

        for(i = 0; i < n; i += 64){
                /* We use a mask load depending on the mask value in b_gf2 */
                uint32_t len = (n - i) < 64 ? (n - i) : 64;
		uint32_t ceil_len_bits = (len + 7) / 8;
                __m128i mask_128 = _mm_mask_loadu_epi8(zero_128, ((__mmask16)1 << ceil_len_bits) - 1, &b_gf2[(i / 8)]);
                /* Transfer to our 64 bits mask */
                __mmask64 mask = (__mmask64)_mm_movepi64_pi64(mask_128);
		__m512i _c = _mm512_mask_mov_epi8(zero, mask, _a);
		store_incomplete_m512(_c, &c_gf256[i], len);
        }

        return;
}

/*      
 * "Hybrid" scalar multiplication of a vector in GF(2) and a vector in GF(256)
 */
static inline uint8_t gf2_gf256_vect_mult_avx512(const uint8_t *a_gf2, const uint8_t *b_gf256, uint32_t n)
{
        uint32_t i;
        const __m512i zero = _mm512_setzero_epi32();
        const __m128i zero_128 = _mm_setzero_si128();

        /* Set the accumulator to 0 */
        __m512i accu = _mm512_setzero_epi32();

        for(i = 0; i < n; i += 64){
                /* We use a mask load depending on the mask value in a_gf2 */
                uint32_t len = (n - i) < 64 ? (n - i) : 64;
		uint32_t ceil_len_bits = (len + 7) / 8;
                __m128i mask_128 = _mm_mask_loadu_epi8(zero_128, ((__mmask16)1 << ceil_len_bits) - 1, &a_gf2[(i / 8)]);
                /* Transfer to our 64 bits mask */
                __mmask64 mask = (__mmask64)_mm_movepi64_pi64(mask_128);
                accu ^= _mm512_mask_loadu_epi8(zero, mask, (int const*)&b_gf256[i]);
        }

        return sum_uint8_avx512(accu);
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(256) and a vector in GF(2)
 */
static inline uint8_t gf256_gf2_vect_mult_avx512(const uint8_t *a_gf256, const uint8_t *b_gf2, uint32_t n)
{
        return gf2_gf256_vect_mult_avx512(b_gf2, a_gf256, n);
}

/*
 * "Hybrid" matrix multiplication of a matrix in GF(2) and a vector in GF(256), resulting
 *  in a vector in GF(256)
 */
static inline void gf2_gf256_mat_mult_avx512(const uint8_t *A, const uint8_t *X, uint8_t *Y, uint32_t n, matrix_type mtype)
{
        GF2_GF256_MAT_MULT(A, X, Y, n, mtype, gf2_gf256_vect_mult_avx512);
}

/*
 * "Hybrid" matrix multiplication of a matrix in GF(256) and a vector in GF(2), resulting
 *  in a vector in GF(256)
 */
static inline void gf256_gf2_mat_mult_avx512(const uint8_t *A, const uint8_t *X, uint8_t *Y_inf, uint8_t *Y_sup, const uint8_t * even_diagonal, uint32_t n)
{
        /* XXX: NOTE: because of alignment and loading latencies, treating the matrix as "regular" seems always better
         * than using the trianglar shape ... */
        uint16_t i, j, k;
        int l;
        uint16_t complete_line = (n + 7) & ~7;
        uint16_t batches, remainder;
        uint32_t row;
        __m512i accu, _a, dups;
        __m512i acc_inf, mask, prod;
        uint64_t _b;

        const __m128i zero_128 = _mm_setzero_si128();
        __mmask64 control;

        __m512i zeros = _mm512_setzero_si512();;
        __m512i ones = _mm512_set1_epi64(-1);

        batches = n/64;
        remainder = n - batches * 64;
        for (i=0;i<batches;i++) {
                control = -1;
                accu = _mm512_setzero_si512();
                acc_inf = _mm512_setzero_si512();
                row = i*64;
                for (j=0;j<i;j++) {
                        memcpy(&_b, X + 8*j, sizeof(_b));
                        for (k=0;k<64;k++) {
                                dups = _mm512_set1_epi8(-(_b&1));
                                _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                accu = _mm512_xor_si512(_mm512_and_si512(_a,dups),accu);
                                row = row + complete_line;
                                _b = _b >> 1;
                        }
                }
                
                __m128i mask_128 = _mm_mask_loadu_epi8(zero_128, ((__mmask16)1 << 8) - 1, &X[8*j]);
                __mmask64 mask_64 = (__mmask64)_mm_movepi64_pi64(mask_128);
                prod = _mm512_loadu_si512((__m512i*)&even_diagonal[64*i]);
                prod = _mm512_mask_mov_epi8(zeros, mask_64, prod);
                accu = _mm512_xor_si512(prod,accu);

                memcpy(&_b, X + 8*j, sizeof(_b));
                for (k=0;k<64;k++) {
                        control = control << 1;
                        mask = _mm512_mask_blend_epi8(control,zeros,ones);
                        dups = _mm512_set1_epi8(-(_b&1));
                        _a = _mm512_loadu_si512((__m512i*)&A[row]);
                        prod = _mm512_and_si512(_a,dups);
                        accu = _mm512_xor_si512(_mm512_and_si512(mask,prod),accu);
                        //acc_inf = _mm512_xor_si512(_mm512_and_si512(_mm512_xor_si512(mask, ones),prod),acc_inf);
                        acc_inf ^= _mm512_andnot_si512(mask, prod);
                        row = row + complete_line;
                        _b = _b >> 1;
                }
                _mm512_storeu_si512((void*)&Y_sup[i*64], accu);
                

                for (j=i+1;j<batches;j++) {
                        memcpy(&_b, X + 8*j, sizeof(_b));
                        for (k=0;k<64;k++) {
                                dups = _mm512_set1_epi8(-(_b&1));
                                _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                acc_inf = _mm512_xor_si512(_mm512_and_si512(_a,dups),acc_inf);
                                row = row + complete_line;
                                _b = _b >> 1;
                        }
                }
                if (remainder > 0) {
                for (l=((remainder+7)>>3)-1;l>=0;l--) {
                        _b = _b << 8;
                        _b = _b | X[8*j+l];
                }
                for (k=0; k < remainder;k++) {
                        dups = _mm512_set1_epi8(-(_b&1));
                        _a = _mm512_loadu_si512((__m512i*)&A[row]);
                        acc_inf = _mm512_xor_si512(_mm512_and_si512(_a,dups),acc_inf);
                        row = row + complete_line;
                        _b = _b >> 1;
                }
		}
                _mm512_storeu_si512((void*)&Y_inf[i*64], acc_inf);
        }

        // Process last block, not my falt -\o/-
	if (remainder > 0) {
	acc_inf = _mm512_setzero_si512();
	accu = _mm512_setzero_si512();
	row = batches*64;
        control = -1;
        control = control >> (64-remainder);
	for (j=0;j<batches;j++) {
                memcpy(&_b, X + 8*j, sizeof(_b));
                for (k=0;k<64;k++) {
                        dups = _mm512_set1_epi8(-(_b&1));
                        _a = load_incomplete_m512(&A[row],remainder);
                        accu = _mm512_xor_si512(_mm512_and_si512(_a,dups),accu);
                        row = row + complete_line;
                        _b = _b >> 1;
                }
	}

        for (l=((remainder+7)>>3)-1;l>=0;l--) {
                _b = _b << 8;
                _b = _b | X[8*j+l];
        }

        __m128i mask_128 = _mm_mask_loadu_epi8(zero_128, ((__mmask16)1 << ((remainder+7)/8)) - 1, &X[8*j]);
        __mmask64 mask_64 = (__mmask64)_mm_movepi64_pi64(mask_128);
        prod = load_incomplete_m512(&even_diagonal[64*i],remainder);
        prod = _mm512_mask_mov_epi8(zeros, mask_64, prod);
        accu = _mm512_xor_si512(prod,accu);

        for (k=0; k < remainder;k++) {
                control = control << 1;
                mask = _mm512_mask_blend_epi8(control,zeros,ones);
                dups = _mm512_set1_epi8(-(_b&1));
                _a = load_incomplete_m512(&A[row],remainder);
                prod = _mm512_and_si512(_a,dups);
                accu = _mm512_xor_si512(_mm512_and_si512(mask,prod),accu);
                //acc_inf = _mm512_xor_si512(_mm512_and_si512(_mm512_xor_si512(mask, ones),prod),acc_inf);
                acc_inf ^= _mm512_andnot_si512(mask, prod);
                row = row + complete_line;
                _b = _b >> 1;
        }
	store_incomplete_m512(accu,&Y_sup[batches*64],remainder);
	store_incomplete_m512(acc_inf,&Y_inf[batches*64],remainder);
	// End last block
	}	
}

static inline void gf256_gf2_mat_mult_single_avx512(const uint8_t *A, const uint8_t *X, uint8_t *Y, uint32_t n)
{
        /* XXX: NOTE: because of alignment and loading latencies, treating the matrix as "regular" seems always better
         * than using the trianglar shape ... */
        uint16_t i, j, k;
        int l;
        uint16_t complete_line = (n + 7) & ~7;
        uint16_t batches, remainder;
        uint32_t row;
        __m512i _a, dups;
        __m512i acc, mask, prod;
        uint64_t _b;

        __mmask64 control;

        __m512i zeros = _mm512_setzero_si512();;
        __m512i ones = _mm512_set1_epi64(-1);

        batches = n/64;
        remainder = n - batches * 64;
        for (i=0;i<batches;i++) {
                control = -1;
                acc = _mm512_setzero_si512();
                row = (complete_line+1)*i*64;

                
                memcpy(&_b, X + 8*j, sizeof(_b));
                for (k=0;k<64;k++) {
                        control = control << 1;
                        mask = _mm512_mask_blend_epi8(control,zeros,ones);
                        dups = _mm512_set1_epi8(-(_b&1));
                        _a = _mm512_loadu_si512((__m512i*)&A[row]);
                        prod = _mm512_and_si512(_a,dups);
                        //acc = _mm512_xor_si512(_mm512_and_si512(_mm512_xor_si512(mask, ones),prod),acc);
                        acc ^= _mm512_andnot_si512(mask, prod);
                        row = row + complete_line;
                        _b = _b >> 1;
                }
                

                for (j=i+1;j<batches;j++) {  
                        memcpy(&_b, X + 8*j, sizeof(_b));
                        for (k=0;k<64;k++) {
                                dups = _mm512_set1_epi8(-(_b&1));
                                _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                acc = _mm512_xor_si512(_mm512_and_si512(_a,dups),acc);
                                row = row + complete_line;
                                _b = _b >> 1;
                        }
                }
                if (remainder > 0) {
                for (l=((remainder+7)>>3)-1;l>=0;l--) {
                        _b = _b << 8;
                        _b = _b | X[8*j+l];
                }
                for (k=0; k < remainder;k++) {
                        dups = _mm512_set1_epi8(-(_b&1));
                        _a = _mm512_loadu_si512((__m512i*)&A[row]);
                        acc = _mm512_xor_si512(_mm512_and_si512(_a,dups),acc);
                        row = row + complete_line;
                        _b = _b >> 1;
                }
		}
                _mm512_storeu_si512((void*)&Y[i*64], acc);
        }

        // Process last block, not my falt -\o/-
	if (remainder > 0) {
	acc = _mm512_setzero_si512();
	row = (complete_line+1)*batches*64;
        control = -1;
        control = control >> (64-remainder);

        for (l=((remainder+7)>>3)-1;l>=0;l--) {
                _b = _b << 8;
                _b = _b | X[8*j+l];
        }

        for (k=0; k < remainder;k++) {
                control = control << 1;
                mask = _mm512_mask_blend_epi8(control,zeros,ones);
                dups = _mm512_set1_epi8(-(_b&1));
                _a = load_incomplete_m512(&A[row],remainder);
                prod = _mm512_and_si512(_a,dups);
                //acc = _mm512_xor_si512(_mm512_and_si512(_mm512_xor_si512(mask, ones),prod),acc);
                acc ^= _mm512_andnot_si512(mask, prod);
                row = row + complete_line;
                _b = _b >> 1;
        }
	store_incomplete_m512(acc,&Y[batches*64],remainder);
	// End last block
	}	
}

static inline void gf256_transpose_block_avx512(uint8_t *src, uint8_t *dst, uint16_t size_in, uint16_t size_out)
{
    // Cada linha tem 8 bytes → cabe em 64 bits
    __m128i r0 = _mm_loadl_epi64((__m128i*)(src));
    __m128i r1 = _mm_loadl_epi64((__m128i*)(src + 1*size_in));
    __m128i r2 = _mm_loadl_epi64((__m128i*)(src + 2*size_in));
    __m128i r3 = _mm_loadl_epi64((__m128i*)(src + 3*size_in));
    __m128i r4 = _mm_loadl_epi64((__m128i*)(src + 4*size_in));
    __m128i r5 = _mm_loadl_epi64((__m128i*)(src + 5*size_in));
    __m128i r6 = _mm_loadl_epi64((__m128i*)(src + 6*size_in));
    __m128i r7 = _mm_loadl_epi64((__m128i*)(src + 7*size_in));

    // 8-bit → 16-bit
    __m128i t0 = _mm_unpacklo_epi8(r0, r1);
    __m128i t1 = _mm_unpacklo_epi8(r2, r3);
    __m128i t2 = _mm_unpacklo_epi8(r4, r5);
    __m128i t3 = _mm_unpacklo_epi8(r6, r7);

    // 16-bit → 32-bit
    __m128i u0 = _mm_unpacklo_epi16(t0, t1);
    __m128i u1 = _mm_unpackhi_epi16(t0, t1);
    __m128i u2 = _mm_unpacklo_epi16(t2, t3);
    __m128i u3 = _mm_unpackhi_epi16(t2, t3);

    // 32-bit → 64-bit (linhas finais)
    __m128i v0 = _mm_unpacklo_epi32(u0, u2);
    __m128i v1 = _mm_unpackhi_epi32(u0, u2);
    __m128i v2 = _mm_unpacklo_epi32(u1, u3);
    __m128i v3 = _mm_unpackhi_epi32(u1, u3);

    // Cada v* contém 2 linhas transpostas (8 bytes cada)
    _mm_storel_epi64((__m128i*)(dst), v0);
    _mm_storel_epi64((__m128i*)(dst + 1*size_out), _mm_srli_si128(v0, 8));

    _mm_storel_epi64((__m128i*)(dst + 2*size_out), v1);
    _mm_storel_epi64((__m128i*)(dst + 3*size_out), _mm_srli_si128(v1, 8));

    _mm_storel_epi64((__m128i*)(dst + 4*size_out), v2);
    _mm_storel_epi64((__m128i*)(dst + 5*size_out), _mm_srli_si128(v2, 8));

    _mm_storel_epi64((__m128i*)(dst + 6*size_out), v3);
    _mm_storel_epi64((__m128i*)(dst + 7*size_out), _mm_srli_si128(v3, 8));
}

/* GF(256) matrix transposition */
static inline void gf256_mat_transpose_avx512(uint8_t *A, uint16_t len, uint16_t extension)
{
	uint8_t *block1;
	uint8_t *block2;
	uint8_t aux[64] = {0};
	for (int i=0;i<len;i+=8){
		for (int j=0;j<i;j+=8) {
			block1 = A + i*extension + j;
			block2 = A + j*extension + i;
			gf256_transpose_block_avx512(block1,aux,extension,8);
			gf256_transpose_block_avx512(block2,block1,extension,extension);
			for (int k=0;k<8;k++) {
                                memcpy(block2+extension*k,aux+8*k,8);
			}
		}
	}
	for (int i=0;i<len;i+=8){
				block1 = A + i*extension + i;
				gf256_transpose_block_avx512(block1,block1,extension,extension);
	}
}

/*
 * "Hybrid" multiplication of a constant in GF(4) and a vector in GF(256)
 */
static inline void gf4_gf256_constant_vect_mult_avx512(uint8_t a_gf4, const uint8_t *b_gf256, uint8_t *c_gf256, uint32_t n)
{
    gf4_gf256_constant_vect_mult_ref(a_gf4, b_gf256, c_gf256, n);
    return;
}

/*
 * "Hybrid" multiplication of a constant in GF(256) and a vector in GF(4)
 */
static inline void gf256_gf4_constant_vect_mult_avx512(uint8_t a_gf256, const uint8_t *b_gf4, uint8_t *c_gf256, uint32_t n)
{
    gf256_gf4_constant_vect_mult_ref(a_gf256, b_gf4, c_gf256, n);
    return;
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(4) and a vector in GF(256)
 */
static inline uint8_t gf4_gf256_vect_mult_avx512(const uint8_t *a_gf4, const uint8_t *b_gf256, uint32_t n)
{
    return gf4_gf256_vect_mult_ref(a_gf4, b_gf256, n);
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(256) and a vector in GF(4)
 */
static inline uint8_t gf256_gf4_vect_mult_avx512(const uint8_t *a_gf256, const uint8_t *b_gf4, uint32_t n)
{
    return gf4_gf256_vect_mult_avx512(b_gf4, a_gf256, n);
}

/*
 * "Hybrid" matrix multiplication of a matrix in GF(256) and a vector in GF(4), resulting
 *  in a vector in GF(256)
 */
static inline void gf256_gf4_mat_mult_avx512(const uint8_t *A, const uint8_t *X, uint8_t *Y, uint32_t n, matrix_type mtype)
{
    GF256_GF4_MAT_MULT(A, X, Y, n, mtype, gf256_gf4_vect_mult_avx512);
}

static inline void BLC_Compute_Folding(uint8_t* acc, uint8_t* lseed, uint8_t* data_folding, uint32_t nb_vecs, uint32_t row_size, uint32_t bytes_to_add)
{
        uint32_t i;
        uint32_t j;

        uint32_t g2;
        uint32_t diff;
        uint32_t pos;

        __m512i accu[2], folding_element;
        __m512i folding0[2];
        __m512i current_seed;

        /* Set the accumulator to 0 */
        for (i=0; i < 2; i++) {
                accu[i] = _mm512_setzero_si512();
                folding0[i] = _mm512_setzero_si512();
        }

        for (j=0;j<nb_vecs;j+=2) {
                g2 = (j+2 < nb_vecs) ? (j + 2) : 0;
                diff = (j+1) ^ g2;
                diff = 31 - __builtin_clz(diff);
                pos = 0;
                for(i = 0; i < bytes_to_add; i += 64){
                        if((bytes_to_add-i) < 64){
                                current_seed = load_incomplete_m512(&lseed[j*row_size+i], (bytes_to_add-i));
                                accu[pos] = _mm512_xor_si512(current_seed,accu[pos]);
                                folding0[pos] = _mm512_xor_si512(folding0[pos],accu[pos]);
                        }
                        else{
                                /* Obvious 256-bit */
                                current_seed = _mm512_loadu_si512((__m512i*)&lseed[j*row_size+i]);
                                accu[pos] = _mm512_xor_si512(current_seed,accu[pos]);
                                folding0[pos] = _mm512_xor_si512(folding0[pos],accu[pos]);
                        }
                        pos++;
                }
                pos = 0;
                for(i = 0; i < bytes_to_add; i += 64){
                        if((bytes_to_add-i) < 64){
                                current_seed = load_incomplete_m512(&lseed[(j+1)*row_size+i], (bytes_to_add-i));
                                accu[pos] = _mm512_xor_si512(current_seed,accu[pos]);
                                folding_element = load_incomplete_m512(&data_folding[diff*row_size+i], (bytes_to_add-i));
                                folding_element = _mm512_xor_si512(folding_element,accu[pos]);
                                store_incomplete_m512(folding_element, &data_folding[diff*row_size+i], (bytes_to_add-i));
                        }
                        else{
                                /* Obvious 256-bit */
                                current_seed = _mm512_loadu_si512((__m512i*)&lseed[(j+1)*row_size+i]);
                                accu[pos] = _mm512_xor_si512(current_seed,accu[pos]);
                                folding_element = _mm512_loadu_si512((__m512i*)&data_folding[diff*row_size+i]);
                                folding_element = _mm512_xor_si512(folding_element,accu[pos]);
                                _mm512_storeu_si512((void*)&data_folding[diff*row_size+i], folding_element);
                        }
                        pos++;
                }
        }
        pos = 0;
        for(i = 0; i < bytes_to_add; i += 64){
                if((bytes_to_add-i) < 64){
                        store_incomplete_m512(accu[pos], &acc[i], (bytes_to_add-i));
                        store_incomplete_m512(folding0[pos], &data_folding[i], (bytes_to_add-i));
                }
                else {
                        _mm512_storeu_si512((void*)&acc[i], accu[pos]);
                        _mm512_storeu_si512((void*)&data_folding[i], folding0[pos]);
                }
                pos++;
        }
}

/*
 * "Hybrid" multiplication of a constant in GF(16) and a vector in GF(256)
 */
static inline void gf16_gf256_constant_vect_mult_avx512(uint8_t a_gf16, const uint8_t *b_gf256, uint8_t *c_gf256, uint32_t n)
{
    uint8_t a_gf256;
    gf256_vect_lift_from_gf16_ref(&a_gf16, &a_gf256, 1);
    gf256_constant_vect_mult_avx512(a_gf256, b_gf256, c_gf256, n);
    return;
}

/* Vectorized lifting from GF(16) to GF(256) */
static inline void gf256_vect_lift_from_gf16_avx512(const uint8_t *b_gf16, uint8_t *c_gf256, uint32_t len)
{
        uint32_t i;
        __m512i _a_gf16, _a, _b, _c;
	__m256i _tmp;
	__m512i zero = _mm512_setzero_epi32();
        const __m512i shuff_msk_even = _mm512_set_epi8(-1, 31, -1,  30,  -1, 29, -1, 28, -1, 27, -1,  26,  -1, 25, -1, 24,
                                                       -1, 23, -1, 22, -1,  21,  -1, 20, -1, 19, -1, 18, -1,  17,  -1, 16,
						       -1, 15, -1,  14,  -1, 13, -1, 12, -1, 11, -1,  10,  -1, 9, -1, 8,
                                                       -1, 7, -1,  6,  -1, 5, -1, 4, -1, 3, -1,  2,  -1, 1, -1, 0);
        const __m512i shuff_msk_odd  = _mm512_set_epi8(31, -1,  30,  -1, 29, -1, 28, -1, 27, -1,  26,  -1, 25, -1, 24, -1,
                                                       23, -1, 22, -1,  21,  -1, 20, -1, 19, -1, 18, -1,  17,  -1, 16, -1,
                                                       15, -1,  14,  -1, 13, -1, 12, -1, 11, -1,  10,  -1, 9, -1, 8, -1,
                                                       7, -1,  6,  -1, 5, -1, 4, -1, 3, -1,  2,  -1, 1, -1, 0, -1);
        const __m512i lifting_lookup = _mm512_set_epi8(0x0c, 0x0d, 0xec, 0xed, 0x51, 0x50, 0xb1, 0xb0, 0xbc, 0xbd, 0x5c, 0x5d, 0xe1, 0xe0, 0x01, 0x00,
						       0x0c, 0x0d, 0xec, 0xed, 0x51, 0x50, 0xb1, 0xb0, 0xbc, 0xbd, 0x5c, 0x5d, 0xe1, 0xe0, 0x01, 0x00,
						       0x0c, 0x0d, 0xec, 0xed, 0x51, 0x50, 0xb1, 0xb0, 0xbc, 0xbd, 0x5c, 0x5d, 0xe1, 0xe0, 0x01, 0x00,
                                                       0x0c, 0x0d, 0xec, 0xed, 0x51, 0x50, 0xb1, 0xb0, 0xbc, 0xbd, 0x5c, 0x5d, 0xe1, 0xe0, 0x01, 0x00);
        const __m512i nib_mask = _mm512_set1_epi8(0x0f);

        for(i = 0; i < len; i += 64){
                if((len-i) < 64){
                        _a_gf16 = load_incomplete_m512((const uint8_t*)&b_gf16[i/2], (len-i+1)/2);
                }
                else{
                        /* Obvious 256-bit */
                        _a_gf16 = load_incomplete_m512((const uint8_t*)&b_gf16[i/2], 32);
                }
                /* Duplicate lanes */
		_tmp = _mm512_castsi512_si256(_a_gf16);
                _a_gf16 = _mm512_inserti64x4(_a_gf16, _tmp, 1);
                /* Isolate the nibbles in _a_gf16 */
                _a = _a_gf16 & nib_mask;
                _b = _mm512_srli_epi64(_a_gf16, 4) & nib_mask;
                /* Create the nibbles mix using a cross-lane shuffling */
		_a = _mm512_permutex2var_epi8(_a, shuff_msk_even, zero);
		_b = _mm512_permutex2var_epi8(_b, shuff_msk_odd, zero);
                _c = _a | _b;
                /* Lift: since we are on 16 bits, we can perform a vperm lookup inside the register */
                _c = _mm512_shuffle_epi8(lifting_lookup, _c);
                /* Store the result */
                if((len-i) < 64){
                        store_incomplete_m512(_c, &c_gf256[i], (len-i));
                }
                else{
                        _mm512_storeu_epi64((__m512i*)&c_gf256[i], _c);
                }
        }

        return;
}

/* Vectorized lifting from GF(16) to GF(256) */
static inline __m512i gf256_lift64_from_gf16_avx2(__m512i _a_gf16)
{
        __m512i _a, _b, _c;
	__m256i _tmp;
	__m512i zero = _mm512_setzero_epi32();
        const __m512i shuff_msk_even = _mm512_set_epi8(-1, 31, -1,  30,  -1, 29, -1, 28, -1, 27, -1,  26,  -1, 25, -1, 24,
                                                       -1, 23, -1, 22, -1,  21,  -1, 20, -1, 19, -1, 18, -1,  17,  -1, 16,
						       -1, 15, -1,  14,  -1, 13, -1, 12, -1, 11, -1,  10,  -1, 9, -1, 8,
                                                       -1, 7, -1,  6,  -1, 5, -1, 4, -1, 3, -1,  2,  -1, 1, -1, 0);
        const __m512i shuff_msk_odd  = _mm512_set_epi8(31, -1,  30,  -1, 29, -1, 28, -1, 27, -1,  26,  -1, 25, -1, 24, -1,
                                                       23, -1, 22, -1,  21,  -1, 20, -1, 19, -1, 18, -1,  17,  -1, 16, -1,
                                                       15, -1,  14,  -1, 13, -1, 12, -1, 11, -1,  10,  -1, 9, -1, 8, -1,
                                                       7, -1,  6,  -1, 5, -1, 4, -1, 3, -1,  2,  -1, 1, -1, 0, -1);
        const __m512i lifting_lookup = _mm512_set_epi8(0x0c, 0x0d, 0xec, 0xed, 0x51, 0x50, 0xb1, 0xb0, 0xbc, 0xbd, 0x5c, 0x5d, 0xe1, 0xe0, 0x01, 0x00,
						       0x0c, 0x0d, 0xec, 0xed, 0x51, 0x50, 0xb1, 0xb0, 0xbc, 0xbd, 0x5c, 0x5d, 0xe1, 0xe0, 0x01, 0x00,
						       0x0c, 0x0d, 0xec, 0xed, 0x51, 0x50, 0xb1, 0xb0, 0xbc, 0xbd, 0x5c, 0x5d, 0xe1, 0xe0, 0x01, 0x00,
                                                       0x0c, 0x0d, 0xec, 0xed, 0x51, 0x50, 0xb1, 0xb0, 0xbc, 0xbd, 0x5c, 0x5d, 0xe1, 0xe0, 0x01, 0x00);
        const __m512i nib_mask = _mm512_set1_epi8(0x0f);

        /* Duplicate lanes */
        _tmp = _mm512_castsi512_si256(_a_gf16);
        _a_gf16 = _mm512_inserti64x4(_a_gf16, _tmp, 1);
        /* Isolate the nibbles in _a_gf16 */
        _a = _a_gf16 & nib_mask;
        _b = _mm512_srli_epi64(_a_gf16, 4) & nib_mask;
        /* Create the nibbles mix */
        _a = _mm512_permutex2var_epi8(_a, shuff_msk_even, zero);
        _b = _mm512_permutex2var_epi8(_b, shuff_msk_odd, zero);
        _c = _a | _b;
        /* Lift: since we are on 16 bits, we can perform a vperm lookup inside the register */
        _c = _mm512_shuffle_epi8(lifting_lookup, _c);
        /* Store the result */

	return _c;
}

/*
 * "Hybrid" multiplication of a constant in GF(256) and a vector in GF(16)
 */
static inline void gf256_gf16_constant_vect_mult_avx512(uint8_t a_gf256, const uint8_t *b_gf16, uint8_t *c_gf256, uint32_t n)
{
    gf256_vect_lift_from_gf16_avx512(b_gf16, c_gf256, n);
    gf256_constant_vect_mult_avx512(a_gf256, c_gf256, c_gf256, n);
    return;
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(16) and a vector in GF(256)
 */
static inline uint8_t gf16_gf256_vect_mult_avx512(const uint8_t *a_gf16, const uint8_t *b_gf256, uint32_t len)
{
        uint32_t i;
        __m512i accu, _a, _b;
        __m512i _a_gf16;

        /* Set the accumulator to 0 */
        accu = _mm512_setzero_epi32();

        for(i = 0; i < len; i += 64){
                if((len-i) < 64){
                        _a_gf16 = load_incomplete_m512((const uint8_t*)&a_gf16[i/2], (len-i+1)/2);
                        _a = _mm512_setzero_epi32();
                        gf256_vect_lift_from_gf16_avx512((const uint8_t*)&_a_gf16, (uint8_t*)&_a, len-i);
                        _b = load_incomplete_m512(&b_gf256[i], len-i);
                }
                else{
                        /* Obvious 512-bit */
                        _a_gf16 = load_incomplete_m512((const uint8_t*)&a_gf16[i/2], 32);
                        gf256_vect_lift_from_gf16_avx512((const uint8_t*)&_a_gf16, (uint8_t*)&_a, 64);
                        _b = _mm512_loadu_epi64((__m512i*)&b_gf256[i]);
                }
                /* Vectorized multiplication in GF(256) */
                accu ^= gf256_mult_vectorized_avx512(_a, _b);
        }

	return sum_uint8_avx512(accu);
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(256) and a vector in GF(16)
 */
static inline uint8_t gf256_gf16_vect_mult_avx512(const uint8_t *a_gf256, const uint8_t *b_gf16, uint32_t n)
{
    return gf16_gf256_vect_mult_avx512(b_gf16, a_gf256, n);
}

/*
 * "Hybrid" matrix multiplication of a matrix in GF(256) and a vector in GF(16), resulting
 *  in a vector in GF(256)
 */
static inline void gf256_gf16_mat_mult_avx512(const uint8_t *A, const uint8_t *X, uint8_t *Y_inf, uint8_t *Y_sup, const uint8_t * even_diagonal, uint32_t n)
{
    uint16_t i, j, k, l;
        uint16_t complete_line = (n + 7) & ~7;
	uint16_t batches, remainder;
	uint32_t row; //, ind;
	__m512i accu, _a, dups, _b;
	__m512i acc_inf,prod;

	__m512i zeros, mask;
        __m512i ones = _mm512_set1_epi64(-1);
	zeros =  _mm512_setzero_si512();
        __m128i lanes[4];

        __mmask64 control;

	/* Set the accumulator to 0 */
	batches = n/64;
	remainder = n - 64*batches;
	for (i=0;i<batches;i++) {
                control = -1;
		acc_inf = _mm512_setzero_si512();
		accu = _mm512_setzero_si512();
		row = i*64;
		for (j=0;j<i;j++) {
                        _b = load_incomplete_m512(&X[32*j],32);
                        _b = gf256_lift64_from_gf16_avx2(_b);
                        lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
                        lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
                        lanes[2] = _mm512_extracti32x4_epi32(_b, 2);
                        lanes[3] = _mm512_extracti32x4_epi32(_b, 3);
                        for (l=0;l<4;l++) {
                                for (k=0;k<16;k++) {
                                        dups = _mm512_broadcastb_epi8(lanes[l]);
                                        _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                        accu ^= gf256_mult_vectorized_avx512(_a,dups);
                                        row = row + complete_line;
                                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1);
                                }
                        }
		}

		_b = load_incomplete_m512(&X[32*j],32);
		_b = gf256_lift64_from_gf16_avx2(_b);
                lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
                lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
                lanes[2] = _mm512_extracti32x4_epi32(_b, 2);
                lanes[3] = _mm512_extracti32x4_epi32(_b, 3);
		_a = _mm512_loadu_si512((__m512i*)&even_diagonal[64*i]);
		accu ^= gf256_mult_vectorized_avx512(_a, _b);
                for (l=0;l<4;l++) {
                        for (k=0;k<16;k++) {
                                control = control << 1;
                                mask = _mm512_mask_blend_epi8(control,zeros,ones);
                                dups = _mm512_broadcastb_epi8(lanes[l]);
                                _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                prod = gf256_mult_vectorized_avx512(_a,dups);
                                accu ^= (mask & prod);
                                acc_inf ^= (_mm512_xor_si512(mask, ones) & prod);
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1);
                        }
                }
		_mm512_storeu_si512((void*)&Y_sup[i*64], accu);

		for (j=i+1;j<batches;j++) {	
		        _b = load_incomplete_m512(&X[32*j],32);
                        _b = gf256_lift64_from_gf16_avx2(_b);
                        lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
                        lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
                        lanes[2] = _mm512_extracti32x4_epi32(_b, 2);
                        lanes[3] = _mm512_extracti32x4_epi32(_b, 3);
                        for (l=0;l<4;l++) {
                                for (k=0;k<16;k++) {
                                        dups = _mm512_broadcastb_epi8(lanes[l]);
                                        _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                        acc_inf ^= gf256_mult_vectorized_avx512(_a,dups);
                                        row = row + complete_line;
                                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1);
                                }
                        }
		}
		_b = load_incomplete_m512(&X[32*j],remainder/2);
		_b = gf256_lift64_from_gf16_avx2(_b);
                lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
                lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
                lanes[2] = _mm512_extracti32x4_epi32(_b, 2);
                lanes[3] = _mm512_extracti32x4_epi32(_b, 3);
                for (l=0;l<4;l++) {
                        for (k=0;k<16 && k < remainder-(l*16);k++) {
                                dups = _mm512_broadcastb_epi8(lanes[l]);
                                _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                _a = gf256_mult_vectorized_avx512(_a,dups);
                                acc_inf ^= _a;
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1); // ideia do chat
                        }
                }
		_mm512_storeu_si512((void*)&Y_inf[i*64], acc_inf);
	}

	// Process last block, not my falt -\o/-
	acc_inf = _mm512_setzero_si512();
	accu = _mm512_setzero_si512();
	row = batches*64;
        control = -1;
	for (j=0;j<batches;j++) {
		_b = load_incomplete_m512(&X[32*j],32);
                _b = gf256_lift64_from_gf16_avx2(_b);
                lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
                lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
                lanes[2] = _mm512_extracti32x4_epi32(_b, 2);
                lanes[3] = _mm512_extracti32x4_epi32(_b, 3);
		for (l=0;l<4;l++) {
                        for (k=0;k<16;k++) {
                                dups = _mm512_broadcastb_epi8(lanes[l]);
                                _a = load_incomplete_m512(&A[row],remainder);
                                accu ^= gf256_mult_vectorized_avx512(_a,dups);
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1); // ideia do chat
                        }
                }
	}

	zeros = _mm512_setzero_si512();
	_b = load_incomplete_m512(&X[32*j],remainder/2);
	_b = gf256_lift64_from_gf16_avx2(_b);
        lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
        lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
        lanes[2] = _mm512_extracti32x4_epi32(_b, 2);
        lanes[3] = _mm512_extracti32x4_epi32(_b, 3);
        _a = load_incomplete_m512(&even_diagonal[64*batches],remainder);
	accu ^= gf256_mult_vectorized_avx512(_a, _b);
	for (l=0;l<4;l++) {
                for (k=0;k<16 && k < remainder-(l*16);k++) {
                        control = control << 1;
                        mask = _mm512_mask_blend_epi8(control,zeros,ones);
                        dups = _mm512_broadcastb_epi8(lanes[l]);
                        _a = load_incomplete_m512(&A[row],remainder);
                        prod = gf256_mult_vectorized_avx512(_a,dups);
                        accu ^= (mask & prod);
                        acc_inf ^= (_mm512_xor_si512(mask, ones) & prod);
                        row = row + complete_line;
                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1); // ideia do chat
                }
        }
	store_incomplete_m512(accu,&Y_sup[batches*64],remainder);
	store_incomplete_m512(acc_inf,&Y_inf[batches*64],remainder);
	// End last block
}

static inline void gf256_gf16_mat_mult_single_avx512(const uint8_t *A, const uint8_t *X, uint8_t *Y, uint32_t n)
{
    uint16_t i, j, k, l;
        uint16_t complete_line = (n + 7) & ~7;
	uint16_t batches, remainder;
	uint32_t row; //, ind;
	__m512i _a, dups, _b;
	__m512i acc,prod;

	__m512i zeros, mask;
        __m512i ones = _mm512_set1_epi64(-1);
	zeros =  _mm512_setzero_si512();
        __m128i lanes[4];

        __mmask64 control;

	/* Set the accumulator to 0 */
	batches = n/64;
	remainder = n - 64*batches;
	for (i=0;i<batches;i++) {
                control = -1;
		acc = _mm512_setzero_si512();
		row = (complete_line+1)*i*64;

		_b = load_incomplete_m512(&X[32*i],32);
		_b = gf256_lift64_from_gf16_avx2(_b);
                lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
                lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
                lanes[2] = _mm512_extracti32x4_epi32(_b, 2);
                lanes[3] = _mm512_extracti32x4_epi32(_b, 3);
                for (l=0;l<4;l++) {
                        for (k=0;k<16;k++) {
                                control = control << 1;
                                mask = _mm512_mask_blend_epi8(control,zeros,ones);
                                dups = _mm512_broadcastb_epi8(lanes[l]);
                                _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                prod = gf256_mult_vectorized_avx512(_a,dups);
                                acc ^= (_mm512_xor_si512(mask, ones) & prod);
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1);
                        }
                }

		for (j=i+1;j<batches;j++) {	
		        _b = load_incomplete_m512(&X[32*j],32);
                        _b = gf256_lift64_from_gf16_avx2(_b);
                        lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
                        lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
                        lanes[2] = _mm512_extracti32x4_epi32(_b, 2);
                        lanes[3] = _mm512_extracti32x4_epi32(_b, 3);
                        for (l=0;l<4;l++) {
                                for (k=0;k<16;k++) {
                                        dups = _mm512_broadcastb_epi8(lanes[l]);
                                        _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                        acc ^= gf256_mult_vectorized_avx512(_a,dups);
                                        row = row + complete_line;
                                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1);
                                }
                        }
		}
		_b = load_incomplete_m512(&X[32*j],remainder/2);
		_b = gf256_lift64_from_gf16_avx2(_b);
                lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
                lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
                lanes[2] = _mm512_extracti32x4_epi32(_b, 2);
                lanes[3] = _mm512_extracti32x4_epi32(_b, 3);
                for (l=0;l<4;l++) {
                        for (k=0;k<16 && k < remainder-(l*16);k++) {
                                dups = _mm512_broadcastb_epi8(lanes[l]);
                                _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                _a = gf256_mult_vectorized_avx512(_a,dups);
                                acc ^= _a;
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1); // ideia do chat
                        }
                }
		_mm512_storeu_si512((void*)&Y[i*64], acc);
	}

	// Process last block, not my falt -\o/-
	acc = _mm512_setzero_si512();
	row = (complete_line+1)*batches*64;
        control = -1;

	zeros = _mm512_setzero_si512();
	_b = load_incomplete_m512(&X[32*j],remainder/2);
	_b = gf256_lift64_from_gf16_avx2(_b);
        lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
        lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
        lanes[2] = _mm512_extracti32x4_epi32(_b, 2);
        lanes[3] = _mm512_extracti32x4_epi32(_b, 3);
	for (l=0;l<4;l++) {
                for (k=0;k<16 && k < remainder-(l*16);k++) {
                        control = control << 1;
                        mask = _mm512_mask_blend_epi8(control,zeros,ones);
                        dups = _mm512_broadcastb_epi8(lanes[l]);
                        _a = load_incomplete_m512(&A[row],remainder);
                        prod = gf256_mult_vectorized_avx512(_a,dups);
                        acc ^= (_mm512_xor_si512(mask, ones) & prod);
                        row = row + complete_line;
                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1); // ideia do chat
                }
        }
	store_incomplete_m512(acc,&Y[batches*64],remainder);
	// End last block
}

/* === GF(256^2) === */
/* NOTE: for atomic multiplication, using vectorization is suboptimal */
static inline uint16_t gf256to2_mult_avx512(uint16_t x, uint16_t y)
{
        return gf256to2_mult_ref(x, y);
}

/* Vectorize multiplication of _a and _b in GF(256^2): the elements in the field are made of
 * 16 bits each in the lanes of the zmm */
static inline __m512i gf256to2_mult_vectorized_avx512(__m512i _a, __m512i _b)
{
	const __m512i shuff_msk_crossed = _mm512_set_epi8(62, 63, 60, 61, 58, 59, 56, 57, 54, 55, 52, 53,
				 		          50, 51, 48, 49, 46, 47, 44, 45, 42, 43, 40, 41,
						          38, 39, 36, 37, 34, 35, 32, 33, 30, 31, 28, 29,
						          26, 27, 24, 25, 22, 23, 20, 21, 18, 19, 16, 17,
						          14, 15, 12, 13, 10, 11, 8 ,  9,  6,  7,  4,  5,
						          2,  3,  0,  1);
	const __m512i shuff_msk1 = _mm512_set_epi8(62, 62, 60, 60, 58, 58, 56, 56, 54, 54, 52, 52,
                                                   50, 50, 48, 48, 46, 46, 44, 44, 42, 42, 40, 40,
                                                   38, 38, 36, 36, 34, 34, 32, 32, 30, 30, 28, 28,
                                                   26, 26, 24, 24, 22, 22, 20, 20, 18, 18, 16, 16,
                                                   14, 14, 12, 12, 10, 10, 8 ,  8,  6,  6,  4,  4,
                                                   2,  2,  0,  0);
	const __m512i shuff_msk2 = _mm512_set_epi8(63, 63, 61, 61, 59, 59, 57, 57, 55, 55, 53, 53,
                                                   51, 51, 49, 49, 47, 47, 45, 45, 43, 43, 41, 41,
                                                   39, 39, 37, 37, 35, 35, 33, 33, 31, 31, 29, 29,
                                                   27, 27, 25, 25, 23, 23, 21, 21, 19, 19, 17, 17,
                                                   15, 15, 13, 13, 11, 11, 9 ,  9,  7,  7,  5,  5,
                                                   3,  3,  1,  1);
	const __m512i const32 = _mm512_set_epi64(0x0020002000200020, 0x0020002000200020, 0x0020002000200020, 0x0020002000200020,
						 0x0020002000200020, 0x0020002000200020, 0x0020002000200020, 0x0020002000200020);
	const __m512i zero = _mm512_setzero_epi32();
	
	__m512i ab = gf256_mult_vectorized_avx512(_a, _b);
	__m512i a0b0 = _mm512_shuffle_epi8(ab, shuff_msk1);
	__m512i a1b1 = _mm512_shuffle_epi8(ab, shuff_msk2);
	__m512i a1b1_32 = gf256_mult_vectorized_avx512(a1b1, const32);
	/* */
	__m512i a0_xor_a1 = _a ^ _mm512_shuffle_epi8(_a, shuff_msk_crossed);
	__m512i b0_xor_b1 = _mm512_mask_blend_epi8((__mmask64)0xaaaaaaaaaaaaaaaa, zero, _b ^ _mm512_shuffle_epi8(_b, shuff_msk_crossed));
	__m512i mult_ab_xor = gf256_mult_vectorized_avx512(a0_xor_a1, b0_xor_b1);

	/* Compute the result */
	__m512i res = a0b0 ^ a1b1_32 ^ mult_ab_xor;

	return res;
}

/*                      
 * Vector multiplied by a constant in GF(256^2).
 */
static inline void gf256to2_constant_vect_mult_avx512(uint16_t b, const uint16_t *a, uint16_t *c, uint32_t len)
{       
         uint32_t i;
        __m512i _a, _b;

	/* Create a vector _b with the duplicated constant everywhere */
	_b = _mm512_set1_epi16(b);

        for(i = 0; i < (2 * len); i += 64){
                if(((2 * len)-i) < 64){
                        _a = load_incomplete_m512((const uint8_t*)&a[i / 2], (2*len)-i);
	                /* Vectorized multiplication in GF(256^2) */
        	        store_incomplete_m512(gf256to2_mult_vectorized_avx512(_a, _b), (uint8_t*)&c[i / 2], (2*len)-i);
                }
                else{
                        /* Obvious 512-bit */
                        _a = _mm512_loadu_epi64((__m512i*)&a[i / 2]);
	                /* Vectorized multiplication in GF(256^2) */
        	        _mm512_storeu_epi64(&c[i / 2], gf256to2_mult_vectorized_avx512(_a, _b));
                }
        }

        return; 
}

/* Perform a multiplication in GF(256^2) of elements in vectors a an b */
static inline uint16_t gf256to2_vect_mult_avx512(const uint16_t *a, const uint16_t *b, uint32_t len)
{
	uint32_t i;
	__m512i accu, _a, _b;

	/* Set the accumulator to 0 */
	accu = _mm512_setzero_epi32();

	for(i = 0; i < (2 * len); i += 64){
                if(((2 * len)-i) < 64){
                        _a = load_incomplete_m512((const uint8_t*)&a[i / 2], ((2 * len) - i));
                        _b = load_incomplete_m512((const uint8_t*)&b[i / 2], ((2 * len) - i));
                }
                else{
                        /* Obvious 512-bit */
                        _a = _mm512_loadu_epi64((__m512i*)&a[i / 2]);
                        _b = _mm512_loadu_epi64((__m512i*)&b[i / 2]);
                }
                accu ^= gf256to2_mult_vectorized_avx512(_a, _b);
	}

	return sum_uint16_avx512(accu);
}

/*
 * "Hybrid" constant multiplication of a constant in GF(2) and a vector in GF(256^2)
 */
static inline void gf2_gf256to2_constant_vect_mult_avx512(uint8_t a_gf2, const uint16_t *b_gf256to2, uint16_t *c_gf256to2, uint32_t n)
{
        gf2_gf256to2_constant_vect_mult_ref(a_gf2, b_gf256to2, c_gf256to2, n);

        return;
}

/*
 * "Hybrid" constant multiplication of a constant in GF(256^2) and a vector in GF(2)
 */   
static inline void gf256to2_gf2_constant_vect_mult_avx512(uint16_t a_gf256to2, const uint8_t *b_gf2, uint16_t *c_gf256to2, uint32_t n)
{
        uint32_t i;
        const __m512i zero = _mm512_setzero_epi32();
        const __m128i zero_128 = _mm_setzero_si128();

	/* Load and broadcast the constant */
	__m512i _a = _mm512_set1_epi16(a_gf256to2);

        for(i = 0; i < n; i += 32){
                /* We use a mask load depending on the mask value in b_gf2 */
                uint32_t len = (n - i) < 32 ? (n - i) : 32;
		uint32_t ceil_len_bits = (len + 7) / 8;
                __m128i mask_128 = _mm_mask_loadu_epi8(zero_128, ((__mmask16)1 << ceil_len_bits) - 1, &b_gf2[(i / 8)]);
                /* Transfer to our 64 bits mask */
                __mmask64 mask64 = (__mmask64)_mm_movepi64_pi64(mask_128);
		__mmask32 mask = (__mmask32)mask64;
		__m512i _c = _mm512_mask_mov_epi16(zero, mask, _a);
		store_incomplete_m512(_c, (uint8_t*)&c_gf256to2[i], 2 * len);
        }

        return;
}

/*
 * "Hybrid" constant multiplication of a constant in GF(256^2) and a vector in GF(256)
 */
static inline void gf256_gf256to2_constant_vect_mult_avx512(uint8_t a_gf256, const uint16_t *b_gf256to2, uint16_t *c_gf256to2, uint32_t n)
{
        gf256_gf256to2_constant_vect_mult_ref(a_gf256, b_gf256to2, c_gf256to2, n);

        return;
}

/*
 * "Hybrid" constant multiplication of a constant in GF(256^2) and a vector in GF(256)
 */
static inline void gf256to2_gf256_constant_vect_mult_avx512(uint16_t a_gf256to2, const uint8_t *b_gf256, uint16_t *c_gf256to2, uint32_t len)
{
        uint32_t i;
        __m512i _a, _b;                    
        const __m512i shuff_msk = _mm512_set_epi8(31, 31, 30, 30, 29, 29, 28, 28, 27, 27, 26, 26, 25, 25, 24, 24, 
                                                  23, 23, 22, 22, 21, 21, 20, 20, 19, 19, 18, 18, 17, 17, 16, 16, 
                                                  15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8, 
                                                  7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2,  2,  1, 1, 0, 0);

        /* Create a vector _b with the duplicated constant everywhere */
        _a = _mm512_set1_epi16(a_gf256to2);
 
        for(i = 0; i < len; i += 32){
                uint32_t to_load = (len - i) < 32 ? (len - i) : 32;
                _b = load_incomplete_m512((const uint8_t*)&b_gf256[i], to_load);
                /* Duplicate the values in _b */
                _b = _mm512_permutex2var_epi8(_b, shuff_msk, _b);
                /* Vectorized multiplication in GF(256^2) */
                store_incomplete_m512(gf256_mult_vectorized_avx512(_a, _b), (uint8_t*)&c_gf256to2[i], 2 * to_load);
        }
        return;
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(2) and a vector in GF(256^2)
 */
static inline uint16_t gf2_gf256to2_vect_mult_avx512(const uint8_t *a_gf2, const uint16_t *b_gf256to2, uint32_t n)
{
        uint32_t i;
        const __m512i zero = _mm512_setzero_epi32();
        const __m128i zero_128 = _mm_setzero_si128();

        /* Set the accumulator to 0 */
        __m512i accu = _mm512_setzero_epi32();

        for(i = 0; i < n; i += 32){
                /* We use a mask load depending on the mask value in a_gf2 */
                uint32_t len = (n - i) < 32 ? (n - i) : 32;
		uint32_t ceil_len_bits = (len + 7) / 8;
		/* Load 32 bits max */
                __m128i mask_128 = _mm_mask_loadu_epi8(zero_128, ((__mmask16)1 << ceil_len_bits) - 1, &a_gf2[(i / 8)]);
                /* Transfer to our 32 bits mask */
                __mmask64 mask64 = (__mmask64)_mm_movepi64_pi64(mask_128);
		__mmask32 mask = (__mmask32)mask64;
                accu ^= _mm512_mask_loadu_epi16(zero, mask, (int const*)&b_gf256to2[i]);
        }

	return sum_uint16_avx512(accu);
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(256^2) and a vector in GF(256)
 */
static inline uint16_t gf256to2_gf2_vect_mult_avx512(const uint16_t *a_gf256to2, const uint8_t *b_gf2, uint32_t n)
{
        return gf2_gf256to2_vect_mult_avx512(b_gf2, a_gf256to2, n);
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(256) and a vector in GF(256^2)
 */
static inline uint16_t gf256_gf256to2_vect_mult_avx512(const uint8_t *a_gf256, const uint16_t *b_gf256to2, uint32_t len)
{
	/* Note: the multiplication of an element in GF(256) and an element in GF(256^2)
         * simply consists in two multiplications in GF(256) (this is multiplying a constant by a degree 1 polynomial)
         * */
	uint32_t i;
	__m512i accu, _a, _b;

	/* Set the accumulator to 0 */
	accu = _mm512_setzero_epi32();

	for(i = 0; i < (2 * len); i += 64){
		const __m512i shuff_msk = _mm512_set_epi8(31, 31, 30, 30, 29, 29, 28, 28, 27, 27, 26, 26, 25, 25, 24, 24,
							  23, 23, 22, 22, 21, 21, 20, 20, 19, 19, 18, 18, 17, 17, 16, 16,
							  15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8, 
							  7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2,  2,  1, 1, 0, 0);
                if(((2 * len)-i) < 64){
                        _a = load_incomplete_m512((const uint8_t*)&a_gf256[i / 2], ((2 * len)-i) / 2);
                        _b = load_incomplete_m512((const uint8_t*)&b_gf256to2[i / 2], (2 * len)-i);
                }
                else{
                        /* Obvious 512-bit */
                        _a = load_incomplete_m512((const uint8_t*)&a_gf256[i / 2], 32);
                        _b = _mm512_loadu_epi64((__m512i*)&b_gf256to2[i / 2]);
                }
                /* Duplicate the values in _a */
		_a = _mm512_permutex2var_epi8(_a, shuff_msk, _a);
		/* Multiply in GF(256) */		
                accu ^= gf256_mult_vectorized_avx512(_a, _b);
	}

	return sum_uint16_avx512(accu);
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(256^2) and a vector in GF(256)
 */
static inline uint16_t gf256to2_gf256_vect_mult_avx512(const uint16_t *a_gf256to2, const uint8_t *b_gf256, uint32_t n)
{
        return gf256_gf256to2_vect_mult_avx512(b_gf256, a_gf256to2, n);
}

/*
 * "Hybrid" matrix multiplication of a matrix in GF(2) and a vector in GF(256^2), resulting
 *  in a vector in GF(256^2)
 */
static inline void gf2_gf256to2_mat_mult_avx512(const uint8_t *A, const uint16_t *X, uint16_t *Y, uint32_t n, matrix_type mtype)
{
        GF2_GF256to2_MAT_MULT(A, X, Y, n, mtype, gf2_gf256to2_vect_mult_avx512);
}

/*
 * "Hybrid" matrix multiplication of a matrix in GF(256^2) and a vector in GF(2), resulting
 *  in a vector in GF(256^2)
 */
static inline void gf256to2_gf2_mat_mult_avx512(const uint16_t *A, const uint8_t *X, uint16_t *Y_inf, uint16_t *Y_sup, const uint16_t *even_diagonal, uint32_t n)
{
        /* XXX: NOTE: because of alignment and loading latencies, treating the matrix as "regular" seems always better
         * than using the trianglar shape ... */
        uint16_t i, j, k;
                uint16_t complete_line = (n + 7) & ~7;
		uint16_t batches;
		uint32_t row;
		__m512i accu, _a, dups;
		__m512i acc_inf, mask, zeros, prod;

                const __m128i zero_128 = _mm_setzero_si128();
		uint32_t _b;
                __mmask64 control;

		batches = n/32;

		__m512i ones = _mm512_set1_epi64(-1);
                zeros = _mm512_setzero_epi32();

		for (i=0;i<batches;i++) {
                        control = -1;
			accu = _mm512_setzero_si512();
			acc_inf = _mm512_setzero_si512();
			row = i*32;
			for (j=0;j<i;j++) {
                                memcpy(&_b, X + 4*j, sizeof(_b));
				for (k=0;k<32;k++) {
					dups = _mm512_set1_epi8(-(_b&1));
                                        _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                        accu = _mm512_xor_si512(_mm512_and_si512(_a,dups),accu);
					row = row + complete_line;
					_b = _b >> 1;
				}
			}

                        __m128i mask_128 = _mm_mask_loadu_epi8(zero_128, ((__mmask16)1 << 4) - 1, &X[4*j]);
                        __mmask64 mask_64 = (__mmask64)_mm_movepi64_pi64(mask_128);
                        __mmask32 mask_32 = (__mmask32)mask_64;
                        prod = _mm512_loadu_si512((__m512i*)&even_diagonal[32*i]);
                        prod = _mm512_mask_mov_epi16(zeros, mask_32, prod);
                        accu = _mm512_xor_si512(prod,accu);
                        memcpy(&_b, X + 4*j, sizeof(_b));
                        for (k=0;k<32;k++) {
                                control = control << 2;
                                mask = _mm512_mask_blend_epi8(control,zeros,ones);
                                dups = _mm512_set1_epi8(-(_b&1));
                                _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                prod = _mm512_and_si512(_a,dups);
                                accu = _mm512_xor_si512(_mm512_and_si512(mask,prod),accu);
                                //acc_inf = _mm512_xor_si512(_mm512_and_si512(_mm512_xor_si512(mask, ones),prod),acc_inf);
                                acc_inf ^= _mm512_andnot_si512(mask, prod);
                                row = row + complete_line;
                                _b = _b >> 1;
                        }
			_mm512_storeu_si512((void*)&Y_sup[i*32], accu);

			for (j=i+1;j<batches;j++) {
				memcpy(&_b, X + 4*j, sizeof(_b));
				for (k=0;k<32;k++) {
					dups = _mm512_set1_epi8(-(_b&1));
					_a = _mm512_loadu_si512((__m512i*)&A[row]);
					acc_inf = _mm512_xor_si512(_mm512_and_si512(_a,dups),acc_inf);
					row = row + complete_line;
					_b = _b >> 1;
				}
			}
			_mm512_storeu_si512((void*)&Y_inf[i*32], acc_inf);
		}

	
}

static inline void gf256to2_gf2_mat_mult_single_avx512(const uint16_t *A, const uint8_t *X, uint16_t *Y, uint32_t n)
{
        /* XXX: NOTE: because of alignment and loading latencies, treating the matrix as "regular" seems always better
         * than using the trianglar shape ... */
        uint16_t i, j, k;
                uint16_t complete_line = (n + 7) & ~7;
		uint16_t batches;
		uint32_t row;
		__m512i _a, dups;
		__m512i acc, mask, zeros, prod;

		uint32_t _b;
                __mmask64 control;

		batches = n/32;

		__m512i ones = _mm512_set1_epi64(-1);
                zeros = _mm512_setzero_epi32();

		for (i=0;i<batches;i++) {
                        control = -1;
			acc = _mm512_setzero_si512();
			row = (complete_line+1)*i*32;

                        memcpy(&_b, X + 4*j, sizeof(_b));
                        for (k=0;k<32;k++) {
                                control = control << 2;
                                mask = _mm512_mask_blend_epi8(control,zeros,ones);
                                dups = _mm512_set1_epi8(-(_b&1));
                                _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                prod = _mm512_and_si512(_a,dups);
                                //acc = _mm512_xor_si512(_mm512_and_si512(_mm512_xor_si512(mask, ones),prod),acc);
                                acc ^= _mm512_andnot_si512(mask, prod);
                                row = row + complete_line;
                                _b = _b >> 1;
                        }

			for (j=i+1;j<batches;j++) {
				memcpy(&_b, X + 4*j, sizeof(_b));
				for (k=0;k<32;k++) {
					dups = _mm512_set1_epi8(-(_b&1));
					_a = _mm512_loadu_si512((__m512i*)&A[row]);
					acc = _mm512_xor_si512(_mm512_and_si512(_a,dups),acc);
					row = row + complete_line;
					_b = _b >> 1;
				}
			}
			_mm512_storeu_si512((void*)&Y[i*32], acc);
		}

	
}

/*
 * GF(2^16) matrix multiplication
 */
static inline void gf256to2_mat_mult_avx512(const uint16_t *A, const uint16_t *X, uint16_t *Y_inf, uint16_t *Y_sup, const uint16_t *even_diagonal, uint32_t n) 
{
    uint16_t i, j, k, l;
        uint16_t complete_line = (n + 7) & ~7;
	uint16_t batches, remainder;
	uint32_t row; //, ind;
	__m512i accu, _a, dups, _b;
	__m512i acc_inf,prod;

	__m512i zeros, mask;
        __m512i ones = _mm512_set1_epi64(-1);
        zeros =  _mm512_setzero_si512();
        __m128i lanes[4];

        __mmask64 control;

	/* Set the accumulator to 0 */
	batches = n/32;
	remainder = n - batches * 32;
	for (i=0;i<batches;i++) {
                control = -1;
		acc_inf = _mm512_setzero_si512();
		accu = _mm512_setzero_si512();
		row = i*32;
		for (j=0;j<i;j++) {
			_b = _mm512_loadu_si512((__m512i*)&X[32*j]);
                        lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
                        lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
                        lanes[2] = _mm512_extracti32x4_epi32(_b, 2);
                        lanes[3] = _mm512_extracti32x4_epi32(_b, 3);
                        for (l=0;l<4;l++) {
                                for (k=0;k<8;k++) {
                                        dups = _mm512_broadcastw_epi16(lanes[l]);
                                        _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                        accu ^= gf256to2_mult_vectorized_avx512(_a,dups);
                                        row = row + complete_line;
                                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 2); // ideia do chat
                                }
                        }
		}

		_b = _mm512_loadu_si512((__m512i*)&X[32*j]);
                lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
                lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
                lanes[2] = _mm512_extracti32x4_epi32(_b, 2);
                lanes[3] = _mm512_extracti32x4_epi32(_b, 3);
		_a = _mm512_loadu_si512((__m256i*)&even_diagonal[32*i]);
		accu ^= gf256to2_mult_vectorized_avx512(_a,_b);
                for (l=0;l<4;l++) {
                        for (k=0;k<8;k++) {
                                control = control << 2;
                                mask = _mm512_mask_blend_epi8(control,zeros,ones);
                                dups = _mm512_broadcastw_epi16(lanes[l]);
                                _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                prod = gf256to2_mult_vectorized_avx512(_a,dups);
                                accu ^= (mask & prod);
                                acc_inf ^= (_mm512_xor_si512(mask, ones) & prod);
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 2); // ideia do chat
                        }
                }
		_mm512_storeu_si512((void*)&Y_sup[i*32], accu);

		for (j=i+1;j<batches;j++) {
			_b = _mm512_loadu_si512((__m512i*)&X[32*j]);
                        lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
                        lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
                        lanes[2] = _mm512_extracti32x4_epi32(_b, 2);
                        lanes[3] = _mm512_extracti32x4_epi32(_b, 3);
                        for (l=0;l<4;l++) {
                                for (k=0;k<8;k++) {
                                        dups = _mm512_broadcastw_epi16(lanes[l]);
                                        _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                        acc_inf ^= gf256to2_mult_vectorized_avx512(_a,dups);
                                        row = row + complete_line;
                                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 2); // ideia do chat
                                }
                        }
		}
		if (remainder > 0) {
		_b = load_incomplete_m512((uint8_t *) &X[32*j],2*remainder);
                lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
                lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
                lanes[2] = _mm512_extracti32x4_epi32(_b, 2);
                lanes[3] = _mm512_extracti32x4_epi32(_b, 3);
                for (l=0;l<4;l++) {
                        for (k=0;k<8 && k<remainder -(l*8);k++) {
                                dups = _mm512_broadcastw_epi16(lanes[l]);
                                _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                _a = gf256to2_mult_vectorized_avx512(_a,dups);
                                acc_inf ^= _a;
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 2);
                        }
                }
		}
                _mm512_storeu_si512((void*)&Y_inf[i*32],acc_inf);
	}

	// Process last block, not my falt -\o/-
	if (remainder > 0) {
	acc_inf = _mm512_setzero_si512();
	accu = _mm512_setzero_si512();
	row = batches*32;
        control = -1;
	for (j=0;j<batches;j++) {
		_b = _mm512_loadu_si512((__m512i*)&X[32*j]);
		lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
                lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
                lanes[2] = _mm512_extracti32x4_epi32(_b, 2);
                lanes[3] = _mm512_extracti32x4_epi32(_b, 3);
                for (l=0;l<4;l++) {
                        for (k=0;k<8;k++) {
                                dups = _mm512_broadcastw_epi16(lanes[l]);
                                _a = load_incomplete_m512((uint8_t *)&A[row],2*remainder);
                                _a = gf256to2_mult_vectorized_avx512(_a,dups);
                                accu ^= _a;
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 2);
                        }
                }
	}

        zeros = _mm512_setzero_si512();
	_b = load_incomplete_m512((uint8_t *)&X[32*j],2*remainder);
	lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
        lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
        lanes[2] = _mm512_extracti32x4_epi32(_b, 2);
        lanes[3] = _mm512_extracti32x4_epi32(_b, 3);
	_a = load_incomplete_m512((uint8_t *)&even_diagonal[32*batches],2*remainder);
	accu ^= gf256to2_mult_vectorized_avx512(_a,_b);
        for (l=0;l<4;l++) {
                for (k=0;k<8 && k<remainder-(l*8);k++) {
                        control = control << 2;
                        mask = _mm512_mask_blend_epi8(control,zeros,ones);
                        dups = _mm512_broadcastw_epi16(lanes[l]);
                        _a = load_incomplete_m512((uint8_t *)&A[row],2*remainder);
                        prod = gf256to2_mult_vectorized_avx512(_a,dups);
                        accu ^= (mask & prod);
                        acc_inf ^= (_mm512_xor_si512(mask, ones) & prod);
                        row = row + complete_line;
                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 2);
                }
        }
	store_incomplete_m512(accu,(uint8_t *)&Y_sup[batches*32],2*remainder);
	store_incomplete_m512(acc_inf,(uint8_t *)&Y_inf[batches*32],2*remainder);
	// End last block
	}
}

static inline void gf256to2_mat_mult_single_avx512(const uint16_t *A, const uint16_t *X, uint16_t *Y, uint32_t n) 
{
    uint16_t i, j, k, l;
        uint16_t complete_line = (n + 7) & ~7;
	uint16_t batches, remainder;
	uint32_t row; //, ind;
	__m512i _a, dups, _b;
	__m512i acc,prod;

	__m512i zeros, mask;
        __m512i ones = _mm512_set1_epi64(-1);
        zeros =  _mm512_setzero_si512();
        __m128i lanes[4];

        __mmask64 control;

	/* Set the accumulator to 0 */
	batches = n/32;
	remainder = n - batches * 32;
	for (i=0;i<batches;i++) {
                control = -1;
		acc = _mm512_setzero_si512();
		row = (complete_line+1)*i*32;

		_b = _mm512_loadu_si512((__m512i*)&X[32*i]);
                lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
                lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
                lanes[2] = _mm512_extracti32x4_epi32(_b, 2);
                lanes[3] = _mm512_extracti32x4_epi32(_b, 3);
                for (l=0;l<4;l++) {
                        for (k=0;k<8;k++) {
                                control = control << 2;
                                mask = _mm512_mask_blend_epi8(control,zeros,ones);
                                dups = _mm512_broadcastw_epi16(lanes[l]);
                                _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                prod = gf256to2_mult_vectorized_avx512(_a,dups);
                                acc ^= (_mm512_xor_si512(mask, ones) & prod);
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 2); // ideia do chat
                        }
                }

		for (j=i+1;j<batches;j++) {
			_b = _mm512_loadu_si512((__m512i*)&X[32*j]);
                        lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
                        lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
                        lanes[2] = _mm512_extracti32x4_epi32(_b, 2);
                        lanes[3] = _mm512_extracti32x4_epi32(_b, 3);
                        for (l=0;l<4;l++) {
                                for (k=0;k<8;k++) {
                                        dups = _mm512_broadcastw_epi16(lanes[l]);
                                        _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                        acc ^= gf256to2_mult_vectorized_avx512(_a,dups);
                                        row = row + complete_line;
                                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 2); // ideia do chat
                                }
                        }
		}
		if (remainder > 0) {
		_b = load_incomplete_m512((uint8_t *) &X[32*j],2*remainder);
                lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
                lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
                lanes[2] = _mm512_extracti32x4_epi32(_b, 2);
                lanes[3] = _mm512_extracti32x4_epi32(_b, 3);
                for (l=0;l<4;l++) {
                        for (k=0;k<8 && k<remainder -(l*8);k++) {
                                dups = _mm512_broadcastw_epi16(lanes[l]);
                                _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                _a = gf256to2_mult_vectorized_avx512(_a,dups);
                                acc ^= _a;
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 2);
                        }
                }
		}
                _mm512_storeu_si512((void*)&Y[i*32],acc);
	}

	// Process last block, not my falt -\o/-
	if (remainder > 0) {
	acc = _mm512_setzero_si512();
	row = (complete_line+1)*batches*32;
        control = -1;

        zeros = _mm512_setzero_si512();
	_b = load_incomplete_m512((uint8_t *)&X[32*j],2*remainder);
	lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
        lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
        lanes[2] = _mm512_extracti32x4_epi32(_b, 2);
        lanes[3] = _mm512_extracti32x4_epi32(_b, 3);
        for (l=0;l<4;l++) {
                for (k=0;k<8 && k<remainder-(l*8);k++) {
                        control = control << 2;
                        mask = _mm512_mask_blend_epi8(control,zeros,ones);
                        dups = _mm512_broadcastw_epi16(lanes[l]);
                        _a = load_incomplete_m512((uint8_t *)&A[row],2*remainder);
                        prod = gf256to2_mult_vectorized_avx512(_a,dups);
                        acc ^= (_mm512_xor_si512(mask, ones) & prod);
                        row = row + complete_line;
                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 2);
                }
        }
	store_incomplete_m512(acc,(uint8_t *)&Y[batches*32],2*remainder);
	// End last block
	}
}

/*
 * "Hybrid" matrix multiplication of a matrix in GF(256) and a vector in GF(256^2), resulting
 *  in a vector in GF(256^2)
 */
static inline void gf256_gf256to2_mat_mult_avx512(const uint8_t *A, const uint16_t *X, uint16_t *Y, uint32_t n, matrix_type mtype)
{
        GF256to2_MAT_MULT(A, X, Y, n, mtype, gf256_gf256to2_vect_mult_avx512);
}

/*
 * "Hybrid" matrix multiplication of a matrix in GF(256^2) and a vector in GF(256), resulting
 *  in a vector in GF(256^2)
 */
static inline void gf256to2_gf256_mat_mult_avx512(const uint16_t *A, const uint8_t *X, uint16_t *Y_inf, uint16_t *Y_sup, const uint16_t *even_diagonal, uint32_t n)
{
        /* XXX: NOTE: because of alignment and loading latencies, treating the matrix as "regular" seems always better
         * than using the trianglar shape ... */
        uint16_t i, j, k, l;
                uint16_t complete_line = (n + 7) & ~7;
		uint16_t batches, remainder;
		uint32_t row;
		__m512i accu, _a, dups;
		__m512i acc_inf, mask, zeros, prod;
		__m256i _b;

                const __m512i shuff_msk = _mm512_set_epi8(31, 31, 30, 30, 29, 29, 28, 28, 27, 27, 26, 26, 25, 25, 24, 24, 
                                                  23, 23, 22, 22, 21, 21, 20, 20, 19, 19, 18, 18, 17, 17, 16, 16, 
                                                  15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8, 
                                                  7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2,  2,  1, 1, 0, 0);

                __m512i ones = _mm512_set1_epi64(-1);
	        zeros =  _mm512_setzero_si512();
                __m128i lanes[2];

                __mmask64 control;

                batches = n/32;
                remainder = n - batches * 32;
		for (i=0;i<batches;i++) {
                        control = -1;
                        acc_inf = _mm512_setzero_si512();
                        accu = _mm512_setzero_si512();
			row = i*32;
			for (j=0;j<i;j++) {
                                _b = _mm256_lddqu_si256((__m256i*)&X[32*j]);
                                lanes[0] = _mm256_castsi256_si128(_b);
                                lanes[1] = _mm256_extracti128_si256(_b, 1);
                /* Multiply in GF(256) */
                                for (l=0;l<2;l++) {
                                        for (k=0;k<16;k++) {
                                                dups = _mm512_broadcastb_epi8(lanes[l]);
                                                _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                                accu ^= gf256_mult_vectorized_avx512(_a,dups);
                                                row = row + complete_line;
                                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1);
                                        }
                                }
			}

			// preciso arrumar aqui, falta fazer o vdup do prod, tem que ter duas copias consecutivas de cada elemento de b
                        _b = _mm256_lddqu_si256((__m256i*)&X[32*j]);
                        lanes[0] = _mm256_castsi256_si128(_b);
                        lanes[1] = _mm256_extracti128_si256(_b, 1);
                        prod = load_incomplete_m512((const uint8_t*)&X[32*j], 32);
                        prod = _mm512_permutex2var_epi8(prod, shuff_msk, prod);
			_a = _mm512_loadu_si512((__m512i*)&even_diagonal[32*i]);
			accu ^= gf256_mult_vectorized_avx512(_a, prod);
                        for (l=0;l<2;l++) {
                                for (k=0;k<16;k++) {
                                        control = control << 2;
                                        mask = _mm512_mask_blend_epi8(control,zeros,ones);
                                        dups = _mm512_broadcastb_epi8(lanes[l]);
                                        _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                        prod = gf256_mult_vectorized_avx512(_a,dups);
                                        accu ^= (mask & prod);
                                        //acc_inf ^= (_mm512_xor_si512(mask, ones) & prod);
                                        acc_inf ^= _mm512_andnot_si512(mask, prod);
                                        row = row + complete_line;
                                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1);
                                }
                        }
			_mm512_storeu_si512((void*)&Y_sup[i*32], accu);
			// preciso arrumar aqui
			

			for (j=i+1;j<batches;j++) {
				_b = _mm256_lddqu_si256((__m256i*)&X[32*j]);
                                lanes[0] = _mm256_castsi256_si128(_b);
                                lanes[1] = _mm256_extracti128_si256(_b, 1);
                                for (l=0;l<2;l++) {
                                        for (k=0;k<16;k++) {
                                                dups = _mm512_broadcastb_epi8(lanes[l]);
                                                _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                                acc_inf ^= gf256_mult_vectorized_avx512(_a,dups);
                                                row = row + complete_line;
                                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1); 
                                        }
                                }
			}
                        if (remainder > 0) {
                        _a = load_incomplete_m512(&X[32*j],remainder);
                        lanes[0] = _mm512_extracti32x4_epi32(_a, 0);
                        lanes[1] = _mm512_extracti32x4_epi32(_a, 1);
                        for (l=0;l<2;l++) {
                                for (k=0;k<16 && k < remainder-(l*16);k++) {
                                        dups = _mm512_broadcastb_epi8(lanes[l]);
                                        _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                        acc_inf ^= gf256_mult_vectorized_avx512(_a,dups);
                                        row = row + complete_line;
                                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1);
                                }
                        }
                        }
			 _mm512_storeu_si512((void*)&Y_inf[i*32], acc_inf);
		}
        
         // Process last block, not my falt -\o/-
	if (remainder > 0) {
	acc_inf = _mm512_setzero_si512();
	accu = _mm512_setzero_si512();
	row = batches*32;
        control = -1;
	for (j=0;j<batches;j++) {
                _b = _mm256_lddqu_si256((__m256i*)&X[32*j]);
                lanes[0] = _mm256_castsi256_si128(_b);
                lanes[1] = _mm256_extracti128_si256(_b, 1);
                for (l=0;l<2;l++) {
                        for (k=0;k<16;k++) {
                                dups = _mm512_broadcastb_epi8(lanes[l]);
                                _a = load_incomplete_m512((uint8_t *) &A[row],2*remainder);
                                accu ^= gf256_mult_vectorized_avx512(_a,dups);
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1);
                        }
                }
	}

	
        zeros = _mm512_setzero_si512();
	prod = load_incomplete_m512((const uint8_t*)&X[32*j], 2*remainder);
        lanes[0] = _mm512_extracti32x4_epi32(prod, 0);
        lanes[1] = _mm512_extracti32x4_epi32(prod, 1);
	prod = _mm512_permutex2var_epi8(prod, shuff_msk, prod);
        _a = load_incomplete_m512((uint8_t *) &even_diagonal[32*batches],2*remainder);
	accu ^= gf256_mult_vectorized_avx512(_a, prod);
        for (l=0;l<2;l++) {
                for (k=0;k<16 && k<remainder -(l*16);k++) {
                        control = control << 2;
                        mask = _mm512_mask_blend_epi8(control,zeros,ones);
                        dups = _mm512_broadcastb_epi8(lanes[l]);
                        _a = load_incomplete_m512((uint8_t *) &A[row],2*remainder);
                        prod = gf256_mult_vectorized_avx512(_a,dups);
                        accu ^= (mask & prod);
                        acc_inf ^= _mm512_andnot_si512(mask, prod);
                        //acc_inf ^= (_mm512_xor_si512(mask, ones) & prod);
                        row = row + complete_line;
                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1);
                }
        }
	store_incomplete_m512(accu,(uint8_t *) &Y_sup[batches*32],2*remainder);
	store_incomplete_m512(acc_inf,(uint8_t *) &Y_inf[batches*32],2*remainder);
	// End last block
	}
	
}

static inline void gf256to2_transpose_block_avx512(uint16_t *src, uint16_t *dst, uint16_t size_in, uint16_t size_out)
{
    __m128i r0 = _mm_loadu_si128((__m128i*)(src));
    __m128i r1 = _mm_loadu_si128((__m128i*)(src + 1*size_in));
    __m128i r2 = _mm_loadu_si128((__m128i*)(src + 2*size_in));
    __m128i r3 = _mm_loadu_si128((__m128i*)(src + 3*size_in));
    __m128i r4 = _mm_loadu_si128((__m128i*)(src + 4*size_in));
    __m128i r5 = _mm_loadu_si128((__m128i*)(src + 5*size_in));
    __m128i r6 = _mm_loadu_si128((__m128i*)(src + 6*size_in));
    __m128i r7 = _mm_loadu_si128((__m128i*)(src + 7*size_in));

    __m128i t0 = _mm_unpacklo_epi16(r0, r1);
    __m128i t1 = _mm_unpackhi_epi16(r0, r1);
    __m128i t2 = _mm_unpacklo_epi16(r2, r3);
    __m128i t3 = _mm_unpackhi_epi16(r2, r3);
    __m128i t4 = _mm_unpacklo_epi16(r4, r5);
    __m128i t5 = _mm_unpackhi_epi16(r4, r5);
    __m128i t6 = _mm_unpacklo_epi16(r6, r7);
    __m128i t7 = _mm_unpackhi_epi16(r6, r7);

    __m128i u0 = _mm_unpacklo_epi32(t0, t2);
    __m128i u1 = _mm_unpackhi_epi32(t0, t2);
    __m128i u2 = _mm_unpacklo_epi32(t1, t3);
    __m128i u3 = _mm_unpackhi_epi32(t1, t3);
    __m128i u4 = _mm_unpacklo_epi32(t4, t6);
    __m128i u5 = _mm_unpackhi_epi32(t4, t6);
    __m128i u6 = _mm_unpacklo_epi32(t5, t7);
    __m128i u7 = _mm_unpackhi_epi32(t5, t7);

    __m128i v0 = _mm_unpacklo_epi64(u0, u4);
    __m128i v1 = _mm_unpackhi_epi64(u0, u4);
    __m128i v2 = _mm_unpacklo_epi64(u1, u5);
    __m128i v3 = _mm_unpackhi_epi64(u1, u5);
    __m128i v4 = _mm_unpacklo_epi64(u2, u6);
    __m128i v5 = _mm_unpackhi_epi64(u2, u6);
    __m128i v6 = _mm_unpacklo_epi64(u3, u7);
    __m128i v7 = _mm_unpackhi_epi64(u3, u7);

    _mm_storeu_si128((__m128i*)(dst), v0);
    _mm_storeu_si128((__m128i*)(dst + 1*size_out), v1);
    _mm_storeu_si128((__m128i*)(dst + 2*size_out), v2);
    _mm_storeu_si128((__m128i*)(dst + 3*size_out), v3);
    _mm_storeu_si128((__m128i*)(dst + 4*size_out), v4);
    _mm_storeu_si128((__m128i*)(dst + 5*size_out), v5);
    _mm_storeu_si128((__m128i*)(dst + 6*size_out), v6);
    _mm_storeu_si128((__m128i*)(dst + 7*size_out), v7);
}

/* GF(256^2) matrix transposition */
static inline void gf256to2_mat_transpose_avx512(uint16_t *A, uint16_t len, uint16_t extension)
{
	uint16_t *block1;
	uint16_t *block2;
	uint16_t aux[64] = {0};
	__m128i mv;
	for (int i=0;i<len;i+=8){
		for (int j=0;j<i;j+=8) {
			block1 = A + i*extension + j;
			block2 = A + j*extension + i;
			gf256to2_transpose_block_avx512(block1,aux,extension,8);
			gf256to2_transpose_block_avx512(block2,block1,extension,extension);
			for (int k=0;k<8;k++) {
				mv = _mm_load_si128((__m128i*)&aux[8*k]);
                                _mm_storeu_si128((__m128i*)&block2[extension*k],mv);
			}
		}
	}
	for (int i=0;i<len;i+=8){
				block1 = A + i*extension + i;
				gf256to2_transpose_block_avx512(block1,block1,extension,extension);
	}
}

/*
 * "Hybrid" constant multiplication of a constant in GF(4) and a vector in GF(256^2)
 */
static inline void gf4_gf256to2_constant_vect_mult_avx512(uint8_t a_gf4, const uint16_t *b_gf256to2, uint16_t *c_gf256to2, uint32_t n)
{
    gf4_gf256to2_constant_vect_mult_ref(a_gf4, b_gf256to2, c_gf256to2, n);
    return;
}

/*
 * "Hybrid" constant multiplication of a constant in GF(256^2) and a vector in GF(4)
 */
static inline void gf256to2_gf4_constant_vect_mult_avx512(uint16_t a_gf256to2, const uint8_t *b_gf4, uint16_t *c_gf256to2, uint32_t n)
{
    gf256to2_gf4_constant_vect_mult_ref(a_gf256to2, b_gf4, c_gf256to2, n);
    return;
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(4) and a vector in GF(256^2)
 */
static inline uint16_t gf4_gf256to2_vect_mult_avx512(const uint8_t *a_gf4, const uint16_t *b_gf256to2, uint32_t n)
{
    return gf4_gf256to2_vect_mult_ref(a_gf4, b_gf256to2, n);
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(256^2) and a vector in GF(4)
 */
static inline uint16_t gf256to2_gf4_vect_mult_avx512(const uint16_t *a_gf256to2, const uint8_t *b_gf4, uint32_t n)
{
    return gf4_gf256to2_vect_mult_avx512(b_gf4, a_gf256to2, n);
}

/*
 * "Hybrid" matrix multiplication of a matrix in GF(256^2) and a vector in GF(4), resulting
 *  in a vector in GF(256^2)
 */
static inline void gf256to2_gf4_mat_mult_avx512(const uint16_t *A, const uint8_t *X, uint16_t *Y, uint32_t n, matrix_type mtype)
{
    GF256to2_GF4_MAT_MULT(A, X, Y, n, mtype, gf256to2_gf4_vect_mult_avx512);
}

/*
 * "Hybrid" constant multiplication of a constant in GF(16) and a vector in GF(256^2)
 */
static inline void gf16_gf256to2_constant_vect_mult_avx512(uint8_t a_gf16, const uint16_t *b_gf256to2, uint16_t *c_gf256to2, uint32_t n)
{
    uint8_t a_gf256;
    gf256_vect_lift_from_gf16_ref(&a_gf16, &a_gf256, 1);
    gf256_gf256to2_constant_vect_mult_avx512(a_gf256, b_gf256to2, c_gf256to2, n);
    return;
}

/*
 * "Hybrid" constant multiplication of a constant in GF(256^2) and a vector in GF(16)
 */
static inline void gf256to2_gf16_constant_vect_mult_avx512(uint16_t a_gf256to2, const uint8_t *b_gf16, uint16_t *c_gf256to2, uint32_t n)
{
    uint8_t* buf = ((uint8_t*) c_gf256to2) + n;
    gf256_vect_lift_from_gf16_avx512(b_gf16, buf, n);
    gf256to2_gf256_constant_vect_mult_avx512(a_gf256to2, buf, c_gf256to2, n);
    return;
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(16) and a vector in GF(256^2)
 */
static inline uint16_t gf16_gf256to2_vect_mult_avx512(const uint8_t *a_gf16, const uint16_t *b_gf256to2, uint32_t len)
{
        uint32_t i;
        __m512i accu, _a, _b;
        __m512i _a_gf16;

        /* Set the accumulator to 0 */
        accu = _mm512_setzero_epi32();

        for(i = 0; i < (2 * len); i += 64){
                const __m512i shuff_msk = _mm512_set_epi8(31, 31, 30, 30, 29, 29, 28, 28, 27, 27, 26, 26, 25, 25, 24, 24,
                                                          23, 23, 22, 22, 21, 21, 20, 20, 19, 19, 18, 18, 17, 17, 16, 16,
                                                          15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8, 
                                                          7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2,  2,  1, 1, 0, 0);

                if(((2 * len)-i) < 64){
                        _a_gf16 = load_incomplete_m512((const uint8_t*)&a_gf16[i / 4], (((2 * len)-i) + 3) / 4);
                        _a = _mm512_setzero_epi32();
                        gf256_vect_lift_from_gf16_avx512((const uint8_t*)&_a_gf16, (uint8_t*)&_a, ((2 * len) - i + 1)/2);
                        _b = load_incomplete_m512((const uint8_t*)&b_gf256to2[i / 2], ((2 * len) - i));
                }
                else{
                        /* Obvious 512-bit */
                        _a_gf16 = load_incomplete_m512((const uint8_t*)&a_gf16[i/4], 16);
                        gf256_vect_lift_from_gf16_avx512((const uint8_t*)&_a_gf16, (uint8_t*)&_a, 32);
                        _b = _mm512_loadu_epi64((__m512i*)&b_gf256to2[i / 2]);
                }
                /* Duplicate the values in _a */
                _a = _mm512_permutex2var_epi8(_a, shuff_msk, _a);
                /* Multiply in GF(256) */
                accu ^= gf256_mult_vectorized_avx512(_a, _b);
        }

        return sum_uint16_avx512(accu);
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(256^2) and a vector in GF(16)
 */
static inline uint16_t gf256to2_gf16_vect_mult_avx512(const uint16_t *a_gf256to2, const uint8_t *b_gf16, uint32_t n)
{
    return gf16_gf256to2_vect_mult_avx512(b_gf16, a_gf256to2, n);
}

/*
 * "Hybrid" matrix multiplication of a matrix in GF(256^2) and a vector in GF(16), resulting
 *  in a vector in GF(256^2)
 */
static inline void gf256to2_gf16_mat_mult_avx512(const uint16_t *A, const uint8_t *X, uint16_t *Y_inf, uint16_t *Y_sup, const uint16_t *even_diagonal, uint32_t n)
{
    uint16_t i, j, k, l;
	uint16_t batches, remainder;
        uint16_t complete_line;
	uint32_t row;
	__m512i _a;
	__m512i dups, _b;
	__m512i accu, acc_inf,prod;

        const __m512i shuff_msk = _mm512_set_epi8(31, 31, 30, 30, 29, 29, 28, 28, 27, 27, 26, 26, 25, 25, 24, 24, 
                                                  23, 23, 22, 22, 21, 21, 20, 20, 19, 19, 18, 18, 17, 17, 16, 16, 
                                                  15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8, 
                                                  7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2,  2,  1, 1, 0, 0);        

	__m512i zeros, mask;
        __m512i ones = _mm512_set1_epi64(-1);
	zeros =  _mm512_setzero_si512();
        __m128i lanes[2];


        __mmask64 control;

	batches = n/32;
	remainder = n - batches * 32;
        complete_line = (n + 7) & ~7;
	for (i=0;i<batches;i++) {
                control = -1;
                acc_inf = _mm512_setzero_si512();
                accu = _mm512_setzero_si512();
                row = i*32;
		for (j=0;j<i;j++) {
			_b = load_incomplete_m512(&X[16*j],16);
			_b = gf256_lift64_from_gf16_avx2(_b);
                        lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
                        lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
			/* Multiply in GF(256) */
                        for (l=0;l<2;l++) {
                                for (k=0;k<16;k++) {
                                        dups = _mm512_broadcastb_epi8(lanes[l]);
                                        _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                        accu ^= gf256_mult_vectorized_avx512(_a,dups);
                                        row = row + complete_line;
                                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1);
                                }
                        }
		}

		// preciso arrumar aqui, falta fazer o vdup do prod, tem que ter duas copias consecutivas de cada elemento de b
		prod = load_incomplete_m512(&X[16*j],16);
                prod = gf256_lift64_from_gf16_avx2(prod);
                lanes[0] = _mm512_extracti32x4_epi32(prod, 0);
                lanes[1] = _mm512_extracti32x4_epi32(prod, 1);
		prod = _mm512_permutex2var_epi8(prod, shuff_msk, prod);
		_a = _mm512_loadu_si512((__m512i*)&even_diagonal[32*i]);
		accu ^= gf256_mult_vectorized_avx512(_a, prod);
                for (l=0;l<2;l++) {
                        for (k=0;k<16;k++) {
                                control = control << 2;
                                mask = _mm512_mask_blend_epi8(control,zeros,ones);
                                dups = _mm512_broadcastb_epi8(lanes[l]);
                                _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                prod = gf256_mult_vectorized_avx512(_a,dups);
                                accu ^= (mask & prod);
                                acc_inf ^= (_mm512_xor_si512(mask, ones) & prod);
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1);
                        }
                }
		_mm512_storeu_si512((void*)&Y_sup[i*32], accu);
		// preciso arrumar aqui

		for (j=i+1;j<batches;j++) {
			_b = load_incomplete_m512(&X[16*j],16);
			_b = gf256_lift64_from_gf16_avx2(_b);
                        lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
                        lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
                        for (l=0;l<2;l++) {
                                for (k=0;k<16;k++) {
                                        dups = _mm512_broadcastb_epi8(lanes[l]);
                                        _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                        acc_inf ^= gf256_mult_vectorized_avx512(_a,dups);
                                        row = row + complete_line;
                                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1); 
                                }
                        }
		}
		if (remainder > 0) {
		_b = load_incomplete_m512(&X[16*j],(remainder+1)/2);
		_b = gf256_lift64_from_gf16_avx2(_b);
                lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
                lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
                for (l=0;l<2;l++) {
                        for (k=0;k<16 && k<remainder -(l*16);k++) {
                                dups = _mm512_broadcastb_epi8(lanes[l]);
                                _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                acc_inf ^= gf256_mult_vectorized_avx512(_a,dups);
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1);
                        }
                }
		}
		_mm512_storeu_si512((void*)&Y_inf[i*32], acc_inf);
	}

	// Process last block, not my falt -\o/-
	if (remainder > 0) {
	acc_inf = _mm512_setzero_si512();
	accu = _mm512_setzero_si512();
	row = batches*32;
        control = -1;
	for (j=0;j<batches;j++) {
		_b = load_incomplete_m512(&X[16*j],16);
		_b = gf256_lift64_from_gf16_avx2(_b);
                lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
                lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
                for (l=0;l<2;l++) {
                        for (k=0;k<16;k++) {
                                dups = _mm512_broadcastb_epi8(lanes[l]);
                                _a = load_incomplete_m512((uint8_t *) &A[row],2*remainder);
                                accu ^= gf256_mult_vectorized_avx512(_a,dups);
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1);
                        }
                }
	}

	zeros = _mm512_setzero_si512();
	prod = load_incomplete_m512(&X[16*j],(remainder+1)/2);
	prod = gf256_lift64_from_gf16_avx2(prod);
        lanes[0] = _mm512_extracti32x4_epi32(prod, 0);
        lanes[1] = _mm512_extracti32x4_epi32(prod, 1);
	prod = _mm512_permutex2var_epi8(prod, shuff_msk, prod);
	_a = load_incomplete_m512((uint8_t *) &even_diagonal[32*batches],2*remainder);
	accu ^= gf256_mult_vectorized_avx512(_a, prod);
        for (l=0;l<2;l++) {
                for (k=0;k<16 && k<remainder-(l*16);k++) {
                        control = control << 2;
                        mask = _mm512_mask_blend_epi8(control,zeros,ones);
                        dups = _mm512_broadcastb_epi8(lanes[l]);
                        _a = load_incomplete_m512((uint8_t *) &A[row],2*remainder);
                        prod = gf256_mult_vectorized_avx512(_a,dups);
                        accu ^= (mask & prod);
                        acc_inf ^= (_mm512_xor_si512(mask, ones) & prod);
                        row = row + complete_line;
                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1);
                }
        }
	store_incomplete_m512(accu,(uint8_t *) &Y_sup[batches*32],2*remainder);
	store_incomplete_m512(acc_inf,(uint8_t *) &Y_inf[batches*32],2*remainder);
	// End last block
	}
}

static inline void gf256to2_gf16_mat_mult_single_avx512(const uint16_t *A, const uint8_t *X, uint16_t *Y, uint32_t n)
{
    uint16_t i, j, k, l;
	uint16_t batches, remainder;
        uint16_t complete_line;
	uint32_t row;
	__m512i _a;
	__m512i dups, _b;
	__m512i acc,prod;

        __m512i zeros, mask;
        __m512i ones = _mm512_set1_epi64(-1);
	zeros =  _mm512_setzero_si512();
        __m128i lanes[2];

        __mmask64 control;

	batches = n/32;
	remainder = n - batches * 32;
        complete_line = (n + 7) & ~7;
	for (i=0;i<batches;i++) {
                control = -1;
                acc = _mm512_setzero_si512();
                row = (complete_line+1)*i*32;

		// preciso arrumar aqui, falta fazer o vdup do prod, tem que ter duas copias consecutivas de cada elemento de b
		prod = load_incomplete_m512(&X[16*i],16);
                prod = gf256_lift64_from_gf16_avx2(prod);
                lanes[0] = _mm512_extracti32x4_epi32(prod, 0);
                lanes[1] = _mm512_extracti32x4_epi32(prod, 1);
                for (l=0;l<2;l++) {
                        for (k=0;k<16;k++) {
                                control = control << 2;
                                mask = _mm512_mask_blend_epi8(control,zeros,ones);
                                dups = _mm512_broadcastb_epi8(lanes[l]);
                                _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                prod = gf256_mult_vectorized_avx512(_a,dups);
                                acc ^= (_mm512_xor_si512(mask, ones) & prod);
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1);
                        }
                }
		// preciso arrumar aqui

		for (j=i+1;j<batches;j++) {
			_b = load_incomplete_m512(&X[16*j],16);
			_b = gf256_lift64_from_gf16_avx2(_b);
                        lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
                        lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
                        for (l=0;l<2;l++) {
                                for (k=0;k<16;k++) {
                                        dups = _mm512_broadcastb_epi8(lanes[l]);
                                        _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                        acc ^= gf256_mult_vectorized_avx512(_a,dups);
                                        row = row + complete_line;
                                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1); 
                                }
                        }
		}
		if (remainder > 0) {
		_b = load_incomplete_m512(&X[16*j],(remainder+1)/2);
		_b = gf256_lift64_from_gf16_avx2(_b);
                lanes[0] = _mm512_extracti32x4_epi32(_b, 0);
                lanes[1] = _mm512_extracti32x4_epi32(_b, 1);
                for (l=0;l<2;l++) {
                        for (k=0;k<16 && k<remainder -(l*16);k++) {
                                dups = _mm512_broadcastb_epi8(lanes[l]);
                                _a = _mm512_loadu_si512((__m512i*)&A[row]);
                                acc ^= gf256_mult_vectorized_avx512(_a,dups);
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1);
                        }
                }
		}
		_mm512_storeu_si512((void*)&Y[i*32], acc);
	}

	// Process last block, not my falt -\o/-
	if (remainder > 0) {
	acc = _mm512_setzero_si512();
	row = (complete_line+1)*batches*32;
        control = -1;

	zeros = _mm512_setzero_si512();
	prod = load_incomplete_m512(&X[16*j],(remainder+1)/2);
	prod = gf256_lift64_from_gf16_avx2(prod);
        lanes[0] = _mm512_extracti32x4_epi32(prod, 0);
        lanes[1] = _mm512_extracti32x4_epi32(prod, 1);
        for (l=0;l<2;l++) {
                for (k=0;k<16 && k<remainder-(l*16);k++) {
                        control = control << 2;
                        mask = _mm512_mask_blend_epi8(control,zeros,ones);
                        dups = _mm512_broadcastb_epi8(lanes[l]);
                        _a = load_incomplete_m512((uint8_t *) &A[row],2*remainder);
                        prod = gf256_mult_vectorized_avx512(_a,dups);
                        acc ^= (_mm512_xor_si512(mask, ones) & prod);
                        row = row + complete_line;
                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1);
                }
        }
	store_incomplete_m512(acc,(uint8_t *) &Y[batches*32],2*remainder);
	// End last block
	}
}

//#endif /* defined(__AVX512BW__) && defined(__AVX512F__) && defined(__AVX512VL__) && defined(__AVX512VPOPCNTDQ__) */

#endif /* __FIELDS_AVX512_H__ */
