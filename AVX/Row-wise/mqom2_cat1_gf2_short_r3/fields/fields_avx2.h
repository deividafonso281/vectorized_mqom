#ifndef __FIELDS_AVX2_H__
#define __FIELDS_AVX2_H__

/* Check for AVX2 support */
#ifdef __AVX2__

#include "fields_common.h"
#include "fields_ref.h"
/* Needed for memcpy */
#include <string.h>
/* Needed for AVX2 assembly intrinsics */
#include <immintrin.h>

/* === GF(2) === */
/* NOTE: for atomic multiplication, using vectorization is suboptimal */
static inline uint8_t gf2_mult_avx2(uint8_t a, uint8_t b)
{
        return gf2_mult_ref(a, b);
}

#define AVX_MASK_SET 0x80000000

static inline __m256i load_incomplete_m256_aligned32(const uint8_t *a, uint32_t len)
{
        __m256i res;

        if(len == 4){
                /* We only keep one element */
                const __m256i mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, AVX_MASK_SET);
                res = _mm256_maskload_epi32((int const*)a, mask);
		goto out;
        }
        if(len == 8){
                /* We only keep 2 elements */
                const __m256i mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, AVX_MASK_SET, AVX_MASK_SET);
                res = _mm256_maskload_epi32((int const*)a, mask);
		goto out;
        }
        if(len == 12){
                /* We only keep 3 elements */
                const __m256i mask = _mm256_set_epi32(0, 0, 0, 0, 0, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET);
                res = _mm256_maskload_epi32((int const*)a, mask);
		goto out;
        }
        if(len == 16){
                /* We only keep 4 elements */
                const __m256i mask = _mm256_set_epi32(0, 0, 0, 0, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET);
                res = _mm256_maskload_epi32((int const*)a, mask);
		goto out;
        }
        if(len == 20){
                /* We only keep 5 elements */
                const __m256i mask = _mm256_set_epi32(0, 0, 0, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET);
                res = _mm256_maskload_epi32((int const*)a, mask);
		goto out;
        }
        if(len == 24){
                /* We only keep 6 elements */
                const __m256i mask = _mm256_set_epi32(0, 0, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET);
                res = _mm256_maskload_epi32((int const*)a, mask);
		goto out;
        }
        /* We only keep 7 elements */
        const __m256i mask = _mm256_set_epi32(0, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET);
        res = _mm256_maskload_epi32((int const*)a, mask);

out:
        return res;
}

static inline __m256i load_incomplete_m256_unaligned32(const uint8_t *a, uint32_t len)
{
	/* Deal with the 32-bit leftover */
	__m256i res, expanded_leftover;
	uint32_t leftover;
	__m64 _leftover;
	unsigned int i;

	/* Extract the leftover */
	leftover = 0;
	for(i = 0; i < (len % 4); i++){
		leftover |= (a[(4 * (len / 4)) + i] << (8 * i));
	}
	_leftover = _mm_set_pi32(0, leftover);
	expanded_leftover = _mm256_broadcastd_epi32(_mm_movpi64_epi64(_leftover));

	if(len < 4){
		res = expanded_leftover & _mm256_set_epi64x(0, 0, 0, 0xFFFFFF);
		goto out;
	}
	if((len >= 4) && (len < 8)){
		/* We only keep one element */
		const __m256i mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, AVX_MASK_SET);
		res = _mm256_maskload_epi32((int const*)a, mask);
		res = _mm256_blend_epi32(res, expanded_leftover, (1 << 1));
		goto out;
	}
	if((len >= 8) && (len < 12)){
		/* We only keep 2 elements */
		const __m256i mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, AVX_MASK_SET, AVX_MASK_SET);
		res = _mm256_maskload_epi32((int const*)a, mask);
		res = _mm256_blend_epi32(res, expanded_leftover, (1 << 2));
		goto out;
	}
	if((len >= 12) && (len < 16)){
		/* We only keep 3 elements */
		const __m256i mask = _mm256_set_epi32(0, 0, 0, 0, 0, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET);
		res = _mm256_maskload_epi32((int const*)a, mask);
		res = _mm256_blend_epi32(res, expanded_leftover, (1 << 3));
		goto out;
	}
	if((len >= 16) && (len < 20)){
		/* We only keep 4 elements */
		const __m256i mask = _mm256_set_epi32(0, 0, 0, 0, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET);
		res = _mm256_maskload_epi32((int const*)a, mask);
		res = _mm256_blend_epi32(res, expanded_leftover, (1 << 4));
		goto out;
	}
	if((len >= 20) && (len < 24)){
		/* We only keep 5 elements */
		const __m256i mask = _mm256_set_epi32(0, 0, 0, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET);
		res = _mm256_maskload_epi32((int const*)a, mask);
		res = _mm256_blend_epi32(res, expanded_leftover, (1 << 5));
		goto out;
	}
	if((len >= 24) && (len < 28)){
		/* We only keep 6 elements */
		const __m256i mask = _mm256_set_epi32(0, 0, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET);
		res = _mm256_maskload_epi32((int const*)a, mask);
		res = _mm256_blend_epi32(res, expanded_leftover, (1 << 6));
		goto out;
	}
	/* We only keep 7 elements */
	const __m256i mask = _mm256_set_epi32(0, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET);
	res = _mm256_maskload_epi32((int const*)a, mask);
	res = _mm256_blend_epi32(res, expanded_leftover, (1 << 7));

out:
	return res;
}

static inline void store_incomplete_m256_aligned32(__m256i in, uint8_t *a, uint32_t len)
{
        if(len == 4){
                /* We only keep one element */
                __m256i mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, AVX_MASK_SET);
                _mm256_maskstore_epi32((int*)a, mask, in);
		goto out;
        }
        if(len == 8){
                /* We only keep 2 elements */
                __m256i mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, AVX_MASK_SET, AVX_MASK_SET);
                _mm256_maskstore_epi32((int*)a, mask, in);
		goto out;
        }
        if(len == 12){
                /* We only keep 3 elements */
                __m256i mask = _mm256_set_epi32(0, 0, 0, 0, 0, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET);
                _mm256_maskstore_epi32((int*)a, mask, in);
		goto out;
        }
        if(len == 16){
                /* We only keep 4 elements */
                __m256i mask = _mm256_set_epi32(0, 0, 0, 0, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET);
                _mm256_maskstore_epi32((int*)a, mask, in);
		goto out;
        }
        if(len == 20){
                /* We only keep 5 elements */
                __m256i mask = _mm256_set_epi32(0, 0, 0, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET);
                _mm256_maskstore_epi32((int*)a, mask, in);
		goto out;
        }
        if(len == 24){
                /* We only keep 6 elements */
                __m256i mask = _mm256_set_epi32(0, 0, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET);
                _mm256_maskstore_epi32((int*)a, mask, in);
		goto out;
        }
        /* We only keep 7 elements */
        __m256i mask = _mm256_set_epi32(0, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET, AVX_MASK_SET);
        _mm256_maskstore_epi32((int*)a, mask, in);

out:
        return;
}

static inline void store_incomplete_m256_unaligned32(__m256i in, uint8_t *a, uint32_t len)
{
	uint8_t local_a[32];
	_mm256_storeu_si256((__m256i*)&local_a[0], in);
	memcpy(&a[0], &local_a[0], len);

	return;
}

/* This helper tries to efficiently copy len bytes from the ymm register */
static inline void store_incomplete_m256(__m256i in, uint8_t *a, uint32_t len)
{
	if(len == 32){
		_mm256_storeu_si256((__m256i*)a, in);
	}
	else if(len % 4 == 0){
		store_incomplete_m256_aligned32(in, a, len);
	}
	else{
		store_incomplete_m256_unaligned32(in, a, len);
	}
	return;
}


/* This helper tries to efficiently copy len bytes in the ymm register */
static inline __m256i load_incomplete_m256(const uint8_t *a, uint32_t len)
{
	if(len % 4 == 0){
		return load_incomplete_m256_aligned32(a, len);
	}
	else{
		return load_incomplete_m256_unaligned32(a, len);
	}
}

static inline uint8_t parity_avx2(__m256i v) {
	uint8_t res;

	res  = _mm_popcnt_u64(_mm256_extract_epi64(v, 0));
	res ^= _mm_popcnt_u64(_mm256_extract_epi64(v, 1));
	res ^= _mm_popcnt_u64(_mm256_extract_epi64(v, 2));
	res ^= _mm_popcnt_u64(_mm256_extract_epi64(v, 3));

	return (res & 1);
}

static inline uint8_t sum_uint8_avx2(__m256i accu) {
#if 0
        uint32_t i;
        uint8_t res;
        __attribute__((aligned(32))) uint8_t local_c[32];
        /* Store the result */
        _mm256_storeu_si256((__m256i*)local_c, accu);
        /* Finish the xor computation byte pet byte  */
        res = 0;
        for(i = 0; i < 32; i++){
                res ^= local_c[i];
        }

        return res;
#else
	uint32_t i;
	uint8_t res;
	uint64_t a = _mm256_extract_epi64(accu, 0) ^ _mm256_extract_epi64(accu, 1) ^ _mm256_extract_epi64(accu, 2) ^ _mm256_extract_epi64(accu, 3);
	res = 0;
	for(i = 0; i < 8; i++){
		res ^= (a >> (8 * i)) & 0xff;
	}
	return res;
#endif
}

static inline uint64_t sum_uint8x8_avx2(__m256i accu) {
        __m128i a = _mm256_castsi256_si128(accu);
        __m128i b = _mm256_extracti128_si256(accu, 1);
        __m128i c = _mm_blend_epi16(a, b, 0b10101010);
        __m128i d = _mm_blend_epi16(a, b, 0b01010101);
        __m128i mask = _mm_setr_epi8(
                2,3,  0,1,
                6,7,  4,5,
                10,11, 8,9,
                14,15, 12,13
        );

        d = _mm_shuffle_epi8(d, mask);
        a = _mm_xor_si128(c,d);
        mask = _mm_setr_epi8(
                0,2,   4,6,
                8,10,  12,14,
                1,3,   5,7,
                9,11,  13,15
        );
        a = _mm_shuffle_epi8(a, mask);

        return _mm_cvtsi128_si64(a) ^ _mm_extract_epi64(a, 1);
}

static inline uint16_t sum_uint16_avx2(__m256i accu) {
#if 0
        uint32_t i;
        uint16_t res;
        __attribute__((aligned(32))) uint16_t local_c[16]; 
        /* Store the result */
        _mm256_storeu_si256((__m256i*)local_c, accu);
        /* Finish the xor computation byte per uint16_t  */
        res = 0;
        for(i = 0; i < 16; i++){
                res ^= local_c[i];
        }

        return res;
#else
        uint32_t i;
        uint16_t res;
        uint64_t a = _mm256_extract_epi64(accu, 0) ^ _mm256_extract_epi64(accu, 1) ^ _mm256_extract_epi64(accu, 2) ^ _mm256_extract_epi64(accu, 3);
        res = 0;
        for(i = 0; i < 4; i++){
                res ^= (a >> (16 * i)) & 0xffff;
        }
        return res;
#endif
}   

static inline __m128i sum_uint16x8_avx2(__m256i accu) {
        __m128i a = _mm256_castsi256_si128(accu);
        __m128i b = _mm256_extracti128_si256(accu, 1);
        __m128i c = _mm_blend_epi16(a, b, 0b10101010);
        __m128i d = _mm_blend_epi16(a, b, 0b01010101);
        __m128i mask = _mm_setr_epi8(
                2,3,  0,1,
                6,7,  4,5,
                10,11, 8,9,
                14,15, 12,13
        );

        d = _mm_shuffle_epi8(d, mask);
        a = _mm_xor_si128(c,d);

        return a;
}

/*
 * Vector multiplied by a constant in GF(2).
 */
static inline void gf2_constant_vect_mult_avx2(uint8_t b, const uint8_t *a, uint8_t *c, uint32_t len)
{
	gf2_constant_vect_mult_ref(b, a, c, len);

        return;
}

static inline uint8_t gf2_vect_mult_avx2(const uint8_t *a, const uint8_t *b, uint32_t len_bits)
{
	uint32_t i;
	__m256i accu, _a, _b;
	uint32_t len = (len_bits / 8);

	/* Set the accumulator to 0 */
	accu = _mm256_setzero_si256();
	for(i = 0; i < len; i += 32){
		if((len-i) < 32){
			/* Note: if we are here, we are sure that we are 32-bit aligned */
			_a = load_incomplete_m256(&a[i], len-i);
			_b = load_incomplete_m256(&b[i], len-i);
		}
		else{
			/* Obvious 256-bit */
			_a = _mm256_lddqu_si256((__m256i*)&a[i]);
			_b = _mm256_lddqu_si256((__m256i*)&b[i]);
		}
		/* Vectorized AND of inputs and then XOR with the accumulator */
		accu ^= (_a & _b);
	}

	/* Now, we have to compute the parity bit, do it 64 bits per 64 bits */
	return parity_avx2(accu);
}

/* Matrix and vector multiplication over GF(2) 
 * C = A * X, where X is a vector
 * Matrix is supposed to be square n x n, and vector n x 1
 * The output is a vector n x 1
 * */
/* XXX: TODO: this can be optimized by packing rows in zmm when n <= 256 */
static inline void gf2_mat_mult_avx2(const uint8_t *A, const uint8_t *X, uint8_t *Y, uint32_t n, matrix_type mtype)
{
	GF2_MAT_MULT(A, X, Y, n, mtype, gf2_vect_mult_avx2);
}

/* GF(2) matrix transposition */
static inline void gf2_mat_transpose_avx2(const uint8_t *A, uint8_t *B, uint32_t n, matrix_type mtype)
{
        gf2_mat_transpose_ref(A, B, n, mtype);
}

/* === GF(256) === */
/* NOTE: for atomic multiplication, using vectorization is suboptimal */
static inline uint8_t gf256_mult_avx2(uint8_t x, uint8_t y)
{
	return gf256_mult_ref(x, y);
}

static inline __m256i gf256_mult_vectorized_avx2(__m256i _a, __m256i _b)
{
        /* NOTE: when GFNI is detected, we use the accelerated GF(256) Rijndael instruction */
#if defined(__GFNI__) && !defined(NO_GFNI)
	return _mm256_gf2p8mul_epi8(_a, _b);
#else
	/* Fallback to the slower implementation without GFNI */
        /* Our reduction polynomial */
        const __m256i red_poly = _mm256_set_epi64x(0x1B1B1B1B1B1B1B1B, 0x1B1B1B1B1B1B1B1B, 0x1B1B1B1B1B1B1B1B, 0x1B1B1B1B1B1B1B1B);
        const __m256i zero     = _mm256_setzero_si256();
        __m256i accu = _mm256_setzero_si256();

        uint32_t j;
        __m256i mask_lsb, tmp;

        /* Compute the vectorized multiplication in GF(256) */
        for(j = 0; j < 8; j++){
                mask_lsb = _mm256_slli_epi64(_b, 7-j);
                accu ^= _mm256_blendv_epi8(zero, _a, mask_lsb);
                tmp = _mm256_add_epi8(_a, _a);
                _a = _mm256_blendv_epi8(zero, red_poly, _a) ^ tmp;
        }
        return accu;
#endif
}

/*
 * Vector multiplied by a constant in GF(256).
 */
static inline void gf256_constant_vect_mult_avx2(uint8_t b, const uint8_t *a, uint8_t *c, uint32_t len)
{
	uint32_t i;
	__m256i _a, _b;

	/* Load the constant byte b broadcasted in _b */
	_b = _mm256_set1_epi8(b);

        for(i = 0; i < len; i += 32){
                if((len-i) < 32){
                        _a = load_incomplete_m256(&a[i], len-i);
                        /* Vectorized multiplication in GF(256) */
                        store_incomplete_m256(gf256_mult_vectorized_avx2(_a, _b), &c[i], len-i);
                }
                else{
                        /* Obvious 512-bit */
                        _a = _mm256_lddqu_si256((__m256i*)&a[i]);
                        /* Vectorized multiplication in GF(256) */
                        _mm256_storeu_si256((void*)&c[i], gf256_mult_vectorized_avx2(_a, _b));
                }       
        }

        return;
}

/*
 * Vector to vector multiplication in GF(256).
 * Takes two vectors of length 'len', and returns a byte (element in GF(256))
 */
static inline uint8_t gf256_vect_mult_avx2(const uint8_t *a, const uint8_t *b, uint32_t len)
{
	uint32_t i;
	__m256i accu, _a, _b;

	/* Set the accumulator to 0 */
	accu = _mm256_setzero_si256();

	for(i = 0; i < len; i += 32){
		if((len-i) < 32){
			_a = load_incomplete_m256(&a[i], len-i);
			_b = load_incomplete_m256(&b[i], len-i);
		}
		else{
			/* Obvious 256-bit */
			_a = _mm256_lddqu_si256((__m256i*)&a[i]);
			_b = _mm256_lddqu_si256((__m256i*)&b[i]);
		}
		accu ^= gf256_mult_vectorized_avx2(_a, _b); 
	}

	return sum_uint8_avx2(accu);
}

/* Matrix and vector multiplication over GF(256) 
 * C += A * X, where X is a vector
 * Matrix is supposed to be square n x n, and vector n x 1
 * The output is a vector n x 1
 * */
static inline void gf256_mat_mult_avx2(const uint8_t *A, const uint8_t *X, uint8_t *Y_inf, uint8_t *Y_sup, uint32_t n)
{
    uint16_t i, j, k, l;
        uint16_t complete_line = (n + 7) & ~7;
	uint16_t batches, remainder;
	uint32_t row; //, ind;
        uint64_t sumx8;
        uint32_t sumx4;
	__m256i accu, _a, _b;
	__m256i acc_inf;

        __m256i extra_accs[4];

	__m256i zeros, mask;

	/* Set the accumulator to 0 */
	batches = n>>5;
	remainder = n & 31;

	zeros = _mm256_setr_epi64x(0,0,-1,-1);

        k=0;
	for (i=0;i<batches;i++) {
	        mask = _mm256_set1_epi32(-1);
                zeros = _mm256_setr_epi64x(0,0,-1,-1);
                for (l=0;l<2;l++) {
                        for (;k<i*32+l*16+16;k+=4) {
                                row = k*complete_line;
                                extra_accs[0] = _mm256_setzero_si256();
                                extra_accs[1] = _mm256_setzero_si256();
                                extra_accs[2] = _mm256_setzero_si256();
                                extra_accs[3] = _mm256_setzero_si256();
                                for (j=0;j<i;j++) {
                                        _b = _mm256_lddqu_si256((__m256i*)&X[32*j]);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+32*j]);
                                        extra_accs[0] ^= gf256_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+complete_line+32*j]);
                                        extra_accs[1] ^= gf256_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+2*complete_line+32*j]);
                                        extra_accs[2] ^= gf256_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+3*complete_line+32*j]);
                                        extra_accs[3] ^= gf256_mult_vectorized_avx2(_a, _b);
                                }


                                _b = _mm256_lddqu_si256((__m256i*)&X[32*i]);

                                // row k
                                _a = _mm256_lddqu_si256((__m256i*)&A[row+32*i]);
                                accu = gf256_mult_vectorized_avx2(_a, _b);
                                extra_accs[0] ^= _mm256_andnot_si256(mask, accu);
                                accu = (mask & accu);
                                acc_inf = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[0], accu, 0x20),_mm256_permute2x128_si256(extra_accs[0], accu, 0x31));
                                
                                // row k+1
                                mask = _mm256_alignr_epi8(mask,zeros,15);
                                _a = _mm256_lddqu_si256((__m256i*)&A[(row+complete_line)+32*i]);
                                accu = gf256_mult_vectorized_avx2(_a, _b);
                                extra_accs[1] ^= _mm256_andnot_si256(mask, accu);
                                accu = (mask & accu);
                                extra_accs[0] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[1], accu, 0x20),_mm256_permute2x128_si256(extra_accs[1], accu, 0x31));

                                // k xor k+1
                                acc_inf = _mm256_unpacklo_epi64(acc_inf, extra_accs[0]) ^ _mm256_unpackhi_epi64(acc_inf, extra_accs[0]);

                                // row k+2
                                mask = _mm256_alignr_epi8(mask,zeros,15);
                                _a = _mm256_lddqu_si256((__m256i*)&A[(row+2*complete_line)+32*i]);
                                accu = gf256_mult_vectorized_avx2(_a, _b);
                                extra_accs[2] ^= _mm256_andnot_si256(mask,accu);
                                accu = (mask & accu);
                                extra_accs[0] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[2], accu, 0x20),_mm256_permute2x128_si256(extra_accs[2], accu, 0x31));

                                // row k+3
                                mask = _mm256_alignr_epi8(mask,zeros,15);
                                _a = _mm256_lddqu_si256((__m256i*)&A[(row+3*complete_line)+32*i]);
                                accu = gf256_mult_vectorized_avx2(_a, _b);
                                extra_accs[3] ^= _mm256_andnot_si256(mask,accu);
                                accu = (mask & accu);
                                extra_accs[1] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[3], accu, 0x20),_mm256_permute2x128_si256(extra_accs[3], accu, 0x31));
                                
                                // k+3 xor k+2
                                extra_accs[0] = _mm256_unpacklo_epi64(extra_accs[0], extra_accs[1]) ^ _mm256_unpackhi_epi64(extra_accs[0], extra_accs[1]);

                                // (k xor k+1) xor (k+3 xor k+2)
                                extra_accs[1] = _mm256_blend_epi32(acc_inf, extra_accs[0], 0b10101010);
                                extra_accs[2] = _mm256_blend_epi32(acc_inf, extra_accs[0], 0b01010101);
                                extra_accs[2] = _mm256_shuffle_epi32(extra_accs[2], 0b10110001);
                                acc_inf = _mm256_xor_si256(extra_accs[1],extra_accs[2]);


                                extra_accs[0] = _mm256_setzero_si256();
                                extra_accs[1] = _mm256_setzero_si256();
                                extra_accs[2] = _mm256_setzero_si256();
                                extra_accs[3] = _mm256_setzero_si256();
                                for (j=i+1;j<batches;j++) {
                                        _b = _mm256_lddqu_si256((__m256i*)&X[32*j]);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+32*j]);
                                        extra_accs[0] ^= gf256_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+complete_line+32*j]);
                                        extra_accs[1] ^= gf256_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+2*complete_line+32*j]);
                                        extra_accs[2] ^= gf256_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+3*complete_line+32*j]);
                                        extra_accs[3] ^= gf256_mult_vectorized_avx2(_a, _b);
                                }

                                if (remainder > 0) {
                                        _b = load_incomplete_m256(&X[32*j],remainder);
                                        _a = load_incomplete_m256(&A[row+32*batches],remainder);
                                        extra_accs[0] ^= gf256_mult_vectorized_avx2(_a,_b);
                                        _a = load_incomplete_m256(&A[row+complete_line+32*batches],remainder);
                                        extra_accs[1] ^= gf256_mult_vectorized_avx2(_a,_b);
                                        _a = load_incomplete_m256(&A[row+2*complete_line+32*batches],remainder);
                                        extra_accs[2] ^= gf256_mult_vectorized_avx2(_a,_b);
                                        _a = load_incomplete_m256(&A[row+3*complete_line+32*batches],remainder);
                                        extra_accs[3] ^= gf256_mult_vectorized_avx2(_a,_b);
                                }

                                accu = _mm256_setzero_si256();
                                extra_accs[0] = _mm256_xor_si256(_mm256_permute2x128_si256(accu, extra_accs[0], 0x20),_mm256_permute2x128_si256(accu, extra_accs[0], 0x31));
                                extra_accs[1] = _mm256_xor_si256(_mm256_permute2x128_si256(accu, extra_accs[1], 0x20),_mm256_permute2x128_si256(accu, extra_accs[1], 0x31));
                                extra_accs[2] = _mm256_xor_si256(_mm256_permute2x128_si256(accu, extra_accs[2], 0x20),_mm256_permute2x128_si256(accu, extra_accs[2], 0x31));
                                extra_accs[3] = _mm256_xor_si256(_mm256_permute2x128_si256(accu, extra_accs[3], 0x20),_mm256_permute2x128_si256(accu, extra_accs[3], 0x31));
                                extra_accs[0] = _mm256_unpacklo_epi64(extra_accs[0], extra_accs[1]) ^ _mm256_unpackhi_epi64(extra_accs[0], extra_accs[1]);
                                extra_accs[1] = _mm256_unpacklo_epi64(extra_accs[2], extra_accs[3]) ^ _mm256_unpackhi_epi64(extra_accs[2], extra_accs[3]);
                                
                                extra_accs[2] = _mm256_blend_epi32(extra_accs[0], extra_accs[1], 0b10101010);
                                extra_accs[3] = _mm256_blend_epi32(extra_accs[0], extra_accs[1], 0b01010101);
                                extra_accs[3] = _mm256_shuffle_epi32(extra_accs[3], 0b10110001);
                                extra_accs[0] = _mm256_xor_si256(extra_accs[2],extra_accs[3]);
                                acc_inf = _mm256_xor_si256(acc_inf,extra_accs[0]);
                                acc_inf = _mm256_permutevar8x32_epi32(acc_inf,_mm256_setr_epi32(0,1,4,5,2,3,6,7));
                                sumx8 = sum_uint8x8_avx2(acc_inf);
                                sumx4 = (uint32_t) sumx8;
                                if (k == 0) {      
                                        sumx4 = sumx4 >> 8;
                                        memcpy(Y_inf, &sumx4, 3);
                                }
                                else {
                                        memcpy(Y_inf + (k-1), &sumx4, 4);
                                }
                                sumx4 = (uint32_t) (sumx8 >> 32);
                                memcpy(Y_sup + k, &sumx4, 4);
                                mask = _mm256_alignr_epi8(mask,zeros,15);
                        }
                        zeros = _mm256_setzero_si256();
                }
	}

	if (remainder > 0) {
	        mask = _mm256_set1_epi32(-1);
	        zeros = _mm256_setr_epi64x(0,0,-1,-1);
                for (l=0;l<2;l++) {
                        for (;k<n && k < 32*batches+16*(l+1);k+=4) {
	                        row = k*complete_line;
                                extra_accs[0] = _mm256_setzero_si256();
                                extra_accs[1] = _mm256_setzero_si256();
                                extra_accs[2] = _mm256_setzero_si256();
                                extra_accs[3] = _mm256_setzero_si256();
                                for (j=0;j<batches;j++) {
                                        _b = _mm256_lddqu_si256((__m256i*)&X[32*j]);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+32*j]);
                                        extra_accs[0] ^= gf256_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+complete_line+32*j]);
                                        extra_accs[1] ^= gf256_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+2*complete_line+32*j]);
                                        extra_accs[2] ^= gf256_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+3*complete_line+32*j]);
                                        extra_accs[3] ^= gf256_mult_vectorized_avx2(_a, _b);
                                }

                                _b = load_incomplete_m256(&X[32*batches],remainder);

                                // row k
                                _a = load_incomplete_m256(&A[row+32*batches],remainder);
                                accu = gf256_mult_vectorized_avx2(_a, _b);
                                extra_accs[0] ^= _mm256_andnot_si256(mask, accu);
                                accu = (mask & accu);
                                acc_inf = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[0], accu, 0x20),_mm256_permute2x128_si256(extra_accs[0], accu, 0x31));
                                
                                // row k+1
                                mask = _mm256_alignr_epi8(mask,zeros,15);
                                _a = load_incomplete_m256(&A[(row+complete_line)+32*batches],remainder);
                                accu = gf256_mult_vectorized_avx2(_a, _b);
                                extra_accs[1] ^= _mm256_andnot_si256(mask, accu);
                                accu = (mask & accu);
                                extra_accs[0] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[1], accu, 0x20),_mm256_permute2x128_si256(extra_accs[1], accu, 0x31));

                                // k xor k+1
                                acc_inf = _mm256_unpacklo_epi64(acc_inf, extra_accs[0]) ^ _mm256_unpackhi_epi64(acc_inf, extra_accs[0]);

                                // row k+2
                                mask = _mm256_alignr_epi8(mask,zeros,15);
                                _a = load_incomplete_m256(&A[(row+2*complete_line)+32*batches],remainder);
                                accu = gf256_mult_vectorized_avx2(_a, _b);
                                extra_accs[2] ^= _mm256_andnot_si256(mask,accu);
                                accu = (mask & accu);
                                extra_accs[0] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[2], accu, 0x20),_mm256_permute2x128_si256(extra_accs[2], accu, 0x31));

                                // row k+3
                                mask = _mm256_alignr_epi8(mask,zeros,15);
                                _a = load_incomplete_m256(&A[(row+3*complete_line)+32*batches],remainder);
                                accu = gf256_mult_vectorized_avx2(_a, _b);
                                extra_accs[3] ^= _mm256_andnot_si256(mask,accu);
                                accu = (mask & accu);
                                extra_accs[1] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[3], accu, 0x20),_mm256_permute2x128_si256(extra_accs[3], accu, 0x31));
                                
                                // k+3 xor k+2
                                extra_accs[0] = _mm256_unpacklo_epi64(extra_accs[0], extra_accs[1]) ^ _mm256_unpackhi_epi64(extra_accs[0], extra_accs[1]);

                                // (k xor k+1) xor (k+3 xor k+2)
                                extra_accs[1] = _mm256_blend_epi32(acc_inf, extra_accs[0], 0b10101010);
                                extra_accs[2] = _mm256_blend_epi32(acc_inf, extra_accs[0], 0b01010101);
                                extra_accs[2] = _mm256_shuffle_epi32(extra_accs[2], 0b10110001);
                                acc_inf = _mm256_xor_si256(extra_accs[1],extra_accs[2]);
                                acc_inf = _mm256_permutevar8x32_epi32(acc_inf,_mm256_setr_epi32(0,1,4,5,2,3,6,7));

                                sumx8 = sum_uint8x8_avx2(acc_inf);
                                sumx4 = (uint32_t) sumx8;
                                memcpy(Y_inf + (k-1), &sumx4, 4);
                                sumx4 = (uint32_t) (sumx8 >> 32);
                                memcpy(Y_sup + k, &sumx4, 4);
                                mask = _mm256_alignr_epi8(mask,zeros,15);
                        }
                        zeros = _mm256_setzero_si256();
                }
	// End last block
	}

        row = n*complete_line;
        Y_inf[n-1] = gf256_vect_mult_avx2(X,&A[row],n);
}

static inline void gf256_mat_mult_single_avx2(const uint8_t *A, const uint8_t *X, uint8_t *Y, uint32_t n)
{
    GF256_MAT_MULT(A, X, Y, n, TRI_INF, gf256_vect_mult_avx2);
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(2) and a vector in GF(256)
 */
static inline uint8_t gf2_gf256_vect_mult_avx2(const uint8_t *a_gf2, const uint8_t *b_gf256, uint32_t len)
{
        uint32_t i;
	__m256i _a, _b;

        /* Set the accumulator to 0 */
        __m256i accu = _mm256_setzero_si256();

        for(i = 0; i < len; i += 32){
		if((len - i) < 32){
			uint32_t ceil_len = ((len - i) % 8 == 0) ? ((len - i) / 8) : (((len - i) / 8) + 1);
			_a = load_incomplete_m256(&a_gf2[i / 8], ceil_len);
			_b = load_incomplete_m256(&b_gf256[i], len - i);
		}
		else{
			/* Obvious 256-bit */
			_a = load_incomplete_m256(&a_gf2[i / 8], 4);
			_b = _mm256_lddqu_si256((__m256i*)&b_gf256[i]);
		}
		/* Create a selection mask from the bits in _a */
		const __m256i shuff_msk = _mm256_set_epi8(3, 3, 3,  3,  3, 3, 3, 3, 2, 2, 2,  2,  2, 2, 2, 2,
                                                          1, 1, 1,  1,  1, 1, 1, 1, 0, 0, 0,  0,  0, 0, 0, 0);
		const __m256i and_msk = _mm256_set_epi8(0b10000000, 0b01000000, 0b00100000, 0b00010000, 0b00001000, 0b00000100, 0b00000010, 0b00000001,
							0b10000000, 0b01000000, 0b00100000, 0b00010000, 0b00001000, 0b00000100, 0b00000010, 0b00000001,
							0b10000000, 0b01000000, 0b00100000, 0b00010000, 0b00001000, 0b00000100, 0b00000010, 0b00000001,
							0b10000000, 0b01000000, 0b00100000, 0b00010000, 0b00001000, 0b00000100, 0b00000010, 0b00000001);

		/* Copy in the two lanes */
		_a = _mm256_permute4x64_epi64(_a, 0b01000100);
		/* Only keep the selection bits */
		_a = _mm256_shuffle_epi8(_a, shuff_msk) & and_msk;
		/* Transform these bits to either 0 or 0xFF */
		_a = _mm256_cmpeq_epi8(_a, and_msk);
		/* Bytes selection */
                accu ^= (_a & _b);
        }
        return sum_uint8_avx2(accu);
}

/*
 * "Hybrid" multiplication of a constant in GF(2) and a vector in GF(256)
 */
static inline void gf2_gf256_constant_vect_mult_avx2(uint8_t a_gf2, const uint8_t *b_gf256, uint8_t *c_gf256, uint32_t n)
{
	gf2_gf256_constant_vect_mult_ref(a_gf2, b_gf256, c_gf256, n);

        return;
}

/*
 * "Hybrid" multiplication of a constant in GF(256) and a vector in GF(2)
 */
static inline void gf256_gf2_constant_vect_mult_avx2(uint8_t a_gf256, const uint8_t *b_gf2, uint8_t *c_gf256, uint32_t len)
{
	uint32_t i;
	__m256i _a, _b;

        /* Broadcast the constant value */
	_a = _mm256_set1_epi8(a_gf256);

        for(i = 0; i < len; i += 32){
		uint32_t ceil_len;
		if((len - i) < 32){
			ceil_len = ((len - i) % 8 == 0) ? ((len - i) / 8) : (((len - i) / 8) + 1);
			_b = load_incomplete_m256(&b_gf2[i / 8], ceil_len);
		}
		else{
			ceil_len = 4;
			/* Obvious 256-bit */
			_b = load_incomplete_m256(&b_gf2[i / 8], 4);
		}
		/* Create a selection mask from the bits in _a */
		const __m256i shuff_msk = _mm256_set_epi8(3, 3, 3,  3,  3, 3, 3, 3, 2, 2, 2,  2,  2, 2, 2, 2,
                                                          1, 1, 1,  1,  1, 1, 1, 1, 0, 0, 0,  0,  0, 0, 0, 0);
		const __m256i and_msk = _mm256_set_epi8(0b10000000, 0b01000000, 0b00100000, 0b00010000, 0b00001000, 0b00000100, 0b00000010, 0b00000001,
							0b10000000, 0b01000000, 0b00100000, 0b00010000, 0b00001000, 0b00000100, 0b00000010, 0b00000001,
							0b10000000, 0b01000000, 0b00100000, 0b00010000, 0b00001000, 0b00000100, 0b00000010, 0b00000001,
							0b10000000, 0b01000000, 0b00100000, 0b00010000, 0b00001000, 0b00000100, 0b00000010, 0b00000001);

		/* Copy in the two lanes */
		_b = _mm256_permute4x64_epi64(_b, 0b01000100);
		/* Only keep the selection bits */
		_b = _mm256_shuffle_epi8(_b, shuff_msk) & and_msk;
		/* Transform these bits to either 0 or 0xFF */
		_b = _mm256_cmpeq_epi8(_b, and_msk);
		/* Bytes selection */
                __m256i _c = (_a & _b);
		/* Store the result */
		store_incomplete_m256(_c, &c_gf256[i], 8 * ceil_len);
        }

        return;
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(256) and a vector in GF(2)
 */
static inline uint8_t gf256_gf2_vect_mult_avx2(const uint8_t *a_gf256, const uint8_t *b_gf2, uint32_t n)
{
        return gf2_gf256_vect_mult_avx2(b_gf2, a_gf256, n);
}

/* 
 * "Hybrid" matrix multiplication of a matrix in GF(2) and a vector in GF(256), resulting
 *  in a vector in GF(256)
 */                     
static inline void gf2_gf256_mat_mult_avx2(const uint8_t *A, const uint8_t *X, uint8_t *Y, uint32_t n, matrix_type mtype)
{
        /* NOTE: XXX: we force a REG here as it allows for better performance */
        (void)mtype;
        GF2_GF256_MAT_MULT(A, X, Y, n, REG, gf2_gf256_vect_mult_avx2);
}

/*
 * "Hybrid" matrix multiplication of a matrix in GF(256) and a vector in GF(2), resulting
 *  in a vector in GF(256)
 */
static inline void gf256_gf2_mat_mult_avx2(const uint8_t *A, const uint8_t *X, uint8_t *Y_inf, uint8_t *Y_sup, uint32_t n)
{
    uint16_t i, j, k, l;
        uint16_t complete_line = (n + 7) & ~7;
	uint16_t batches, remainder;
	uint32_t row;
        uint64_t sumx8;
        uint32_t sumx4;
	__m256i accu, _a, _b;
	__m256i acc_inf;

        __m256i extra_accs[4];

	__m256i zeros, mask;

	/* Set the accumulator to 0 */
	batches = n>>5;
	remainder = n & 31;

        const __m256i shuff_msk = _mm256_set_epi8(3, 3, 3,  3,  3, 3, 3, 3, 2, 2, 2,  2,  2, 2, 2, 2,
                                                          1, 1, 1,  1,  1, 1, 1, 1, 0, 0, 0,  0,  0, 0, 0, 0);
	const __m256i and_msk = _mm256_set_epi8(0b10000000, 0b01000000, 0b00100000, 0b00010000, 0b00001000, 0b00000100, 0b00000010, 0b00000001,
                                                0b10000000, 0b01000000, 0b00100000, 0b00010000, 0b00001000, 0b00000100, 0b00000010, 0b00000001,
                                                0b10000000, 0b01000000, 0b00100000, 0b00010000, 0b00001000, 0b00000100, 0b00000010, 0b00000001,
                                                0b10000000, 0b01000000, 0b00100000, 0b00010000, 0b00001000, 0b00000100, 0b00000010, 0b00000001);

	k=0;
	for (i=0;i<batches;i++) {
	        mask = _mm256_set1_epi32(-1);
	        zeros = _mm256_setr_epi64x(0,0,-1,-1);
                for (l=0;l<2;l++) {
                        for (;k<i*32+l*16+16;k+=4) {
		                row = k*complete_line;
                                extra_accs[0] = _mm256_setzero_si256();
                                extra_accs[1] = _mm256_setzero_si256();
                                extra_accs[2] = _mm256_setzero_si256();
                                extra_accs[3] = _mm256_setzero_si256();
                                for (j=0;j<i;j++) {
                                        _b = load_incomplete_m256(&X[4*j],4);
                                        _b = _mm256_permute4x64_epi64(_b, 0b01000100);
                                        _b = _mm256_shuffle_epi8(_b, shuff_msk) & and_msk;
                                        _b = _mm256_cmpeq_epi8(_b, and_msk);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+32*j]);
                                        extra_accs[0] ^= (_a & _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+complete_line+32*j]);
                                        extra_accs[1] ^= (_a & _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+2*complete_line+32*j]);
                                        extra_accs[2] ^= (_a & _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+3*complete_line+32*j]);
                                        extra_accs[3] ^= (_a & _b);
                                }

                                _b = load_incomplete_m256(&X[4*i],4);
                                _b = _mm256_permute4x64_epi64(_b, 0b01000100);
                                _b = _mm256_shuffle_epi8(_b, shuff_msk) & and_msk;
                                _b = _mm256_cmpeq_epi8(_b, and_msk);


                                // row k
                                _a = _mm256_lddqu_si256((__m256i*)&A[row+32*i]);
                                accu = (_a & _b);
                                extra_accs[0] ^= _mm256_andnot_si256(mask, accu);
                                accu = (mask & accu);
                                acc_inf = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[0], accu, 0x20),_mm256_permute2x128_si256(extra_accs[0], accu, 0x31));
                                
                                // row k+1
                                mask = _mm256_alignr_epi8(mask,zeros,15);
                                _a = _mm256_lddqu_si256((__m256i*)&A[(row+complete_line)+32*i]);
                                accu = (_a & _b);
                                extra_accs[1] ^= _mm256_andnot_si256(mask, accu);
                                accu = (mask & accu);
                                extra_accs[0] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[1], accu, 0x20),_mm256_permute2x128_si256(extra_accs[1], accu, 0x31));

                                // k xor k+1
                                acc_inf = _mm256_unpacklo_epi64(acc_inf, extra_accs[0]) ^ _mm256_unpackhi_epi64(acc_inf, extra_accs[0]);

                                // row k+2
                                mask = _mm256_alignr_epi8(mask,zeros,15);
                                _a = _mm256_lddqu_si256((__m256i*)&A[(row+2*complete_line)+32*i]);
                                accu = (_a & _b);
                                extra_accs[2] ^= _mm256_andnot_si256(mask,accu);
                                accu = (mask & accu);
                                extra_accs[0] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[2], accu, 0x20),_mm256_permute2x128_si256(extra_accs[2], accu, 0x31));

                                // row k+3
                                mask = _mm256_alignr_epi8(mask,zeros,15);
                                _a = _mm256_lddqu_si256((__m256i*)&A[(row+3*complete_line)+32*i]);
                                accu = (_a & _b);
                                extra_accs[3] ^= _mm256_andnot_si256(mask,accu);
                                accu = (mask & accu);
                                extra_accs[1] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[3], accu, 0x20),_mm256_permute2x128_si256(extra_accs[3], accu, 0x31));
                                
                                // k+3 xor k+2
                                extra_accs[0] = _mm256_unpacklo_epi64(extra_accs[0], extra_accs[1]) ^ _mm256_unpackhi_epi64(extra_accs[0], extra_accs[1]);

                                // (k xor k+1) xor (k+3 xor k+2)
                                extra_accs[1] = _mm256_blend_epi32(acc_inf, extra_accs[0], 0b10101010);
                                extra_accs[2] = _mm256_blend_epi32(acc_inf, extra_accs[0], 0b01010101);
                                extra_accs[2] = _mm256_shuffle_epi32(extra_accs[2], 0b10110001);
                                acc_inf = _mm256_xor_si256(extra_accs[1],extra_accs[2]);


                                extra_accs[0] = _mm256_setzero_si256();
                                extra_accs[1] = _mm256_setzero_si256();
                                extra_accs[2] = _mm256_setzero_si256();
                                extra_accs[3] = _mm256_setzero_si256();
                                for (j=i+1;j<batches;j++) {
                                        _b = load_incomplete_m256(&X[4*j],4);
                                        _b = _mm256_permute4x64_epi64(_b, 0b01000100);
                                        _b = _mm256_shuffle_epi8(_b, shuff_msk) & and_msk;
                                        _b = _mm256_cmpeq_epi8(_b, and_msk);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+32*j]);
                                        extra_accs[0] ^= (_a & _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+complete_line+32*j]);
                                        extra_accs[1] ^= (_a & _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+2*complete_line+32*j]);
                                        extra_accs[2] ^= (_a & _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+3*complete_line+32*j]);
                                        extra_accs[3] ^= (_a & _b);
                                }

                                if (remainder > 0) {
                                        _b = load_incomplete_m256(&X[4*j],remainder/8);
                                        _b = _mm256_permute4x64_epi64(_b, 0b01000100);
                                        _b = _mm256_shuffle_epi8(_b, shuff_msk) & and_msk;
                                        _b = _mm256_cmpeq_epi8(_b, and_msk);
                                        _a = load_incomplete_m256(&A[row+32*batches],remainder);
                                        extra_accs[0] ^= (_a & _b);
                                        _a = load_incomplete_m256(&A[row+complete_line+32*batches],remainder);
                                        extra_accs[1] ^= (_a & _b);
                                        _a = load_incomplete_m256(&A[row+2*complete_line+32*batches],remainder);
                                        extra_accs[2] ^= (_a & _b);
                                        _a = load_incomplete_m256(&A[row+3*complete_line+32*batches],remainder);
                                        extra_accs[3] ^= (_a & _b);
                                }
                                accu = _mm256_setzero_si256();
                                extra_accs[0] = _mm256_xor_si256(_mm256_permute2x128_si256(accu, extra_accs[0], 0x20),_mm256_permute2x128_si256(accu, extra_accs[0], 0x31));
                                extra_accs[1] = _mm256_xor_si256(_mm256_permute2x128_si256(accu, extra_accs[1], 0x20),_mm256_permute2x128_si256(accu, extra_accs[1], 0x31));
                                extra_accs[2] = _mm256_xor_si256(_mm256_permute2x128_si256(accu, extra_accs[2], 0x20),_mm256_permute2x128_si256(accu, extra_accs[2], 0x31));
                                extra_accs[3] = _mm256_xor_si256(_mm256_permute2x128_si256(accu, extra_accs[3], 0x20),_mm256_permute2x128_si256(accu, extra_accs[3], 0x31));
                                extra_accs[0] = _mm256_unpacklo_epi64(extra_accs[0], extra_accs[1]) ^ _mm256_unpackhi_epi64(extra_accs[0], extra_accs[1]);
                                extra_accs[1] = _mm256_unpacklo_epi64(extra_accs[2], extra_accs[3]) ^ _mm256_unpackhi_epi64(extra_accs[2], extra_accs[3]);
                                
                                extra_accs[2] = _mm256_blend_epi32(extra_accs[0], extra_accs[1], 0b10101010);
                                extra_accs[3] = _mm256_blend_epi32(extra_accs[0], extra_accs[1], 0b01010101);
                                extra_accs[3] = _mm256_shuffle_epi32(extra_accs[3], 0b10110001);
                                extra_accs[0] = _mm256_xor_si256(extra_accs[2],extra_accs[3]);
                                acc_inf = _mm256_xor_si256(acc_inf,extra_accs[0]);
                                acc_inf = _mm256_permutevar8x32_epi32(acc_inf,_mm256_setr_epi32(0,1,4,5,2,3,6,7));
                                sumx8 = sum_uint8x8_avx2(acc_inf);
                                sumx4 = (uint32_t) sumx8;
                                if (k == 0) {      
                                        sumx4 = sumx4 >> 8;
                                        memcpy(Y_inf, &sumx4, 3);
                                }
                                else {
                                        memcpy(Y_inf + (k-1), &sumx4, 4);
                                }
                                sumx4 = (uint32_t) (sumx8 >> 32);
                                memcpy(Y_sup + k, &sumx4, 4);
                                mask = _mm256_alignr_epi8(mask,zeros,15);
                        }
                        zeros = _mm256_setzero_si256();
                }
	}

        if (remainder > 0) {
                mask = _mm256_set1_epi32(-1);
                zeros = _mm256_setr_epi64x(0,0,-1,-1);
                for (l=0;l<2;l++) {
                        for (;k<n && k < 32*batches+16*(l+1);k+=4) {
                                row = k*complete_line;
                                extra_accs[0] = _mm256_setzero_si256();
                                extra_accs[1] = _mm256_setzero_si256();
                                extra_accs[2] = _mm256_setzero_si256();
                                extra_accs[3] = _mm256_setzero_si256();
                                for (j=0;j<batches;j++) {
                                        _b = load_incomplete_m256(&X[4*j],4);
                                        _b = _mm256_permute4x64_epi64(_b, 0b01000100);
                                        _b = _mm256_shuffle_epi8(_b, shuff_msk) & and_msk;
                                        _b = _mm256_cmpeq_epi8(_b, and_msk);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+32*j]);
                                        extra_accs[0] ^= (_a & _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+complete_line+32*j]);
                                        extra_accs[1] ^= (_a & _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+2*complete_line+32*j]);
                                        extra_accs[2] ^= (_a & _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+3*complete_line+32*j]);
                                        extra_accs[3] ^= (_a & _b);
                                }
                                _b = load_incomplete_m256(&X[4*j],remainder/8);
                                _b = _mm256_permute4x64_epi64(_b, 0b01000100);
                                _b = _mm256_shuffle_epi8(_b, shuff_msk) & and_msk;
                                _b = _mm256_cmpeq_epi8(_b, and_msk);

                                // row k
                                _a = load_incomplete_m256(&A[row+32*batches],remainder);
                                accu = (_a & _b);
                                extra_accs[0] ^= _mm256_andnot_si256(mask, accu);
                                accu = (mask & accu);
                                acc_inf = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[0], accu, 0x20),_mm256_permute2x128_si256(extra_accs[0], accu, 0x31));
                                
                                // row k+1
                                mask = _mm256_alignr_epi8(mask,zeros,15);
                                _a = load_incomplete_m256(&A[(row+complete_line)+32*batches],remainder);
                                accu = (_a & _b);
                                extra_accs[1] ^= _mm256_andnot_si256(mask, accu);
                                accu = (mask & accu);
                                extra_accs[0] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[1], accu, 0x20),_mm256_permute2x128_si256(extra_accs[1], accu, 0x31));

                                // k xor k+1
                                acc_inf = _mm256_unpacklo_epi64(acc_inf, extra_accs[0]) ^ _mm256_unpackhi_epi64(acc_inf, extra_accs[0]);

                                // row k+2
                                mask = _mm256_alignr_epi8(mask,zeros,15);
                                _a = load_incomplete_m256(&A[(row+2*complete_line)+32*batches],remainder);
                                accu = (_a & _b);
                                extra_accs[2] ^= _mm256_andnot_si256(mask,accu);
                                accu = (mask & accu);
                                extra_accs[0] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[2], accu, 0x20),_mm256_permute2x128_si256(extra_accs[2], accu, 0x31));

                                // row k+3
                                mask = _mm256_alignr_epi8(mask,zeros,15);
                                _a = load_incomplete_m256(&A[(row+3*complete_line)+32*batches],remainder);
                                accu = (_a & _b);
                                extra_accs[3] ^= _mm256_andnot_si256(mask,accu);
                                accu = (mask & accu);
                                extra_accs[1] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[3], accu, 0x20),_mm256_permute2x128_si256(extra_accs[3], accu, 0x31));
                                
                                // k+3 xor k+2
                                extra_accs[0] = _mm256_unpacklo_epi64(extra_accs[0], extra_accs[1]) ^ _mm256_unpackhi_epi64(extra_accs[0], extra_accs[1]);

                                // (k xor k+1) xor (k+3 xor k+2)
                                extra_accs[1] = _mm256_blend_epi32(acc_inf, extra_accs[0], 0b10101010);
                                extra_accs[2] = _mm256_blend_epi32(acc_inf, extra_accs[0], 0b01010101);
                                extra_accs[2] = _mm256_shuffle_epi32(extra_accs[2], 0b10110001);
                                acc_inf = _mm256_xor_si256(extra_accs[1],extra_accs[2]);
                                acc_inf = _mm256_permutevar8x32_epi32(acc_inf,_mm256_setr_epi32(0,1,4,5,2,3,6,7));

                                sumx8 = sum_uint8x8_avx2(acc_inf);
                                sumx4 = (uint32_t) sumx8;
                                memcpy(Y_inf + (k-1), &sumx4, 4);
                                sumx4 = (uint32_t) (sumx8 >> 32);
                                memcpy(Y_sup + k, &sumx4, 4);
                                mask = _mm256_alignr_epi8(mask,zeros,15);
                        }
                        zeros = _mm256_setzero_si256();
                }
        }
	// End last block

        row = n*complete_line;
        Y_inf[n-1] = gf256_gf2_vect_mult_avx2(&A[row],X,n);
}

static inline void gf256_gf2_mat_mult_single_avx2(const uint8_t *A, const uint8_t *X, uint8_t *Y, uint32_t n)
{
        /* NOTE: XXX: we force a REG here as it allows for better performance */
        GF256_GF2_MAT_MULT(A, X, Y, n, REG, gf256_gf2_vect_mult_avx2);
}

static inline void gf256_transpose_block_avx2(uint8_t *src, uint8_t *dst, uint16_t size_in, uint16_t size_out)
{
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

    // 32-bit → 64-bit
    __m128i v0 = _mm_unpacklo_epi32(u0, u2);
    __m128i v1 = _mm_unpackhi_epi32(u0, u2);
    __m128i v2 = _mm_unpacklo_epi32(u1, u3);
    __m128i v3 = _mm_unpackhi_epi32(u1, u3);

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
static inline void gf256_mat_transpose_avx2(uint8_t *A, uint16_t len, uint16_t extension)
{
	uint8_t *block1;
	uint8_t *block2;
	uint8_t aux[64] = {0};
	for (int i=0;i<len;i+=8){
		for (int j=0;j<i;j+=8) {
			block1 = A + i*extension + j;
			block2 = A + j*extension + i;
			gf256_transpose_block_avx2(block1,aux,extension,8);
			gf256_transpose_block_avx2(block2,block1,extension,extension);
			for (int k=0;k<8;k++) {
                                memcpy(block2+extension*k,aux+8*k,8);
			}
		}
	}
	for (int i=0;i<len;i+=8){
				block1 = A + i*extension + i;
				gf256_transpose_block_avx2(block1,block1,extension,extension);
	}
}


/*
 * "Hybrid" multiplication of a constant in GF(4) and a vector in GF(256)
 */
static inline void gf4_gf256_constant_vect_mult_avx2(uint8_t a_gf4, const uint8_t *b_gf256, uint8_t *c_gf256, uint32_t n)
{
    gf4_gf256_constant_vect_mult_ref(a_gf4, b_gf256, c_gf256, n);
    return;
}

/*
 * "Hybrid" multiplication of a constant in GF(256) and a vector in GF(4)
 */
static inline void gf256_gf4_constant_vect_mult_avx2(uint8_t a_gf256, const uint8_t *b_gf4, uint8_t *c_gf256, uint32_t n)
{
    gf256_gf4_constant_vect_mult_ref(a_gf256, b_gf4, c_gf256, n);
    return;
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(4) and a vector in GF(256)
 */
static inline uint8_t gf4_gf256_vect_mult_avx2(const uint8_t *a_gf4, const uint8_t *b_gf256, uint32_t n)
{
    return gf4_gf256_vect_mult_ref(a_gf4, b_gf256, n);
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(256) and a vector in GF(4)
 */
static inline uint8_t gf256_gf4_vect_mult_avx2(const uint8_t *a_gf256, const uint8_t *b_gf4, uint32_t n)
{
	return gf4_gf256_vect_mult_avx2(b_gf4, a_gf256, n);
}

/*
 * "Hybrid" matrix multiplication of a matrix in GF(256) and a vector in GF(4), resulting
 *  in a vector in GF(256)
 */
static inline void gf256_gf4_mat_mult_avx2(const uint8_t *A, const uint8_t *X, uint8_t *Y, uint32_t n, matrix_type mtype)
{
    GF256_GF4_MAT_MULT(A, X, Y, n, mtype, gf256_gf4_vect_mult_avx2);
}

static inline void BLC_Compute_Folding(uint8_t* acc, uint8_t* lseed, uint8_t* data_folding, uint32_t nb_vecs, uint32_t row_size, uint32_t bytes_to_add)
{
        uint32_t i;
        uint32_t j;

        uint32_t g2;
        uint32_t diff;
        uint32_t pos;

        __m256i accu[6], folding_element;
        __m256i current_seed;

        /* Set the accumulator to 0 */
        for (i=0; i < 6; i++) {
                accu[i] = _mm256_setzero_si256();
        }

        for (j=0;j<nb_vecs;j+=2) {
                g2 = (j+2 < nb_vecs) ? (j + 2) : 0;
                diff = (j+1) ^ g2;
                diff = 31 - __builtin_clz(diff);
                pos = 0;
                for(i = 0; i < bytes_to_add; i += 32){
                        if((bytes_to_add-i) < 32){
                                current_seed = load_incomplete_m256(&lseed[j*row_size+i], (bytes_to_add-i));
                                accu[pos] = _mm256_xor_si256(current_seed,accu[pos]);
                                folding_element = load_incomplete_m256(&data_folding[i], (bytes_to_add-i));
                                folding_element = _mm256_xor_si256(folding_element,accu[pos]);
                                store_incomplete_m256(folding_element, &data_folding[i], (bytes_to_add-i));
                        }
                        else{
                                /* Obvious 256-bit */
                                current_seed = _mm256_lddqu_si256((__m256i*)&lseed[j*row_size+i]);
                                accu[pos] = _mm256_xor_si256(current_seed,accu[pos]);
                                folding_element = _mm256_lddqu_si256((__m256i*)&data_folding[i]);
                                folding_element = _mm256_xor_si256(folding_element,accu[pos]);
                                _mm256_storeu_si256((void*)&data_folding[i], folding_element);
                        }
                        pos++;
                }
                pos = 0;
                for(i = 0; i < bytes_to_add; i += 32){
                        if((bytes_to_add-i) < 32){
                                current_seed = load_incomplete_m256(&lseed[(j+1)*row_size+i], (bytes_to_add-i));
                                accu[pos] = _mm256_xor_si256(current_seed,accu[pos]);
                                folding_element = load_incomplete_m256(&data_folding[diff*row_size+i], (bytes_to_add-i));
                                folding_element = _mm256_xor_si256(folding_element,accu[pos]);
                                store_incomplete_m256(folding_element, &data_folding[diff*row_size+i], (bytes_to_add-i));
                        }
                        else{
                                /* Obvious 256-bit */
                                current_seed = _mm256_lddqu_si256((__m256i*)&lseed[(j+1)*row_size+i]);
                                accu[pos] = _mm256_xor_si256(current_seed,accu[pos]);
                                folding_element = _mm256_lddqu_si256((__m256i*)&data_folding[diff*row_size+i]);
                                folding_element = _mm256_xor_si256(folding_element,accu[pos]);
                                _mm256_storeu_si256((void*)&data_folding[diff*row_size+i], folding_element);
                        }
                        pos++;
                }
        }
        pos = 0;
        for(i = 0; i < bytes_to_add; i += 32){
                if((bytes_to_add-i) < 32){
                        store_incomplete_m256(accu[pos], &acc[i], (bytes_to_add-i));
                }
                else {
                        _mm256_storeu_si256((void*)&acc[i], accu[pos]);
                }
                pos++;
        }
}

/*
 * "Hybrid" multiplication of a constant in GF(16) and a vector in GF(256)
 */
static inline void gf16_gf256_constant_vect_mult_avx2(uint8_t a_gf16, const uint8_t *b_gf256, uint8_t *c_gf256, uint32_t n)
{
    uint8_t a_gf256;
    gf256_vect_lift_from_gf16_ref(&a_gf16, &a_gf256, 1);
    gf256_constant_vect_mult_avx2(a_gf256, b_gf256, c_gf256, n);
    return;
}

/* Vectorized lifting from GF(16) to GF(256) */
static inline void gf256_vect_lift_from_gf16_avx2(const uint8_t *b_gf16, uint8_t *c_gf256, uint32_t len)
{
        uint32_t i;
        __m256i _a_gf16, _a, _b, _c;
	const __m256i shuff_msk_even = _mm256_set_epi8(-1, 15, -1,  14,  -1, 13, -1, 12, -1, 11, -1,  10,  -1, 9, -1, 8,
                                                       -1, 7, -1,  6,  -1, 5, -1, 4, -1, 3, -1,  2,  -1, 1, -1, 0);
	const __m256i shuff_msk_odd  = _mm256_set_epi8(15, -1, 14, -1,  13,  -1, 12, -1, 11, -1, 10, -1,  9,  -1, 8, -1,
                                                       7, -1, 6, -1,  5,  -1, 4, -1, 3, -1, 2, -1,  1,  -1, 0, -1);
	const __m256i lifting_lookup = _mm256_set_epi8(0x0c, 0x0d, 0xec, 0xed, 0x51, 0x50, 0xb1, 0xb0, 0xbc, 0xbd, 0x5c, 0x5d, 0xe1, 0xe0, 0x01, 0x00,
  						       0x0c, 0x0d, 0xec, 0xed, 0x51, 0x50, 0xb1, 0xb0, 0xbc, 0xbd, 0x5c, 0x5d, 0xe1, 0xe0, 0x01, 0x00);
	const __m256i nib_mask = _mm256_set1_epi8(0x0f);

	for(i = 0; i < len; i += 32){
                if((len-i) < 32){
                        _a_gf16 = load_incomplete_m256((const uint8_t*)&b_gf16[i/2], (len-i+1)/2);
                }
                else{
                        /* Obvious 256-bit */
                        _a_gf16 = load_incomplete_m256((const uint8_t*)&b_gf16[i/2], 16);
                }

		/* Duplicate lanes */
		_a_gf16 = _mm256_permute4x64_epi64(_a_gf16, 0b01000100);
		/* Isolate the nibbles in _a_gf16 */
		_a = _a_gf16 & nib_mask;
		_b = _mm256_srli_epi64(_a_gf16, 4) & nib_mask;
		/* Create the nibbles mix */
		_c = _mm256_shuffle_epi8(_a, shuff_msk_even) | _mm256_shuffle_epi8(_b, shuff_msk_odd);
		/* Lift: since we are on 16 bits, we can perform a vperm lookup inside the register */
		_c = _mm256_shuffle_epi8(lifting_lookup, _c);
		/* Store the result */
		if((len-i) < 32){
			store_incomplete_m256(_c, &c_gf256[i], (len-i));
		}
		else{
                        _mm256_storeu_si256((__m256i*)&c_gf256[i], _c);
		}
	}

	return;
}

/* Vectorized lifting from GF(16) to GF(256) */
static inline __m256i gf256_lift32_from_gf16_avx2(__m256i _a_gf16)
{
        __m256i _a, _b, _c;
	const __m256i shuff_msk_even = _mm256_set_epi8(-1, 15, -1,  14,  -1, 13, -1, 12, -1, 11, -1,  10,  -1, 9, -1, 8,
                                                       -1, 7, -1,  6,  -1, 5, -1, 4, -1, 3, -1,  2,  -1, 1, -1, 0);
	const __m256i shuff_msk_odd  = _mm256_set_epi8(15, -1, 14, -1,  13,  -1, 12, -1, 11, -1, 10, -1,  9,  -1, 8, -1,
                                                       7, -1, 6, -1,  5,  -1, 4, -1, 3, -1, 2, -1,  1,  -1, 0, -1);
	const __m256i lifting_lookup = _mm256_set_epi8(0x0c, 0x0d, 0xec, 0xed, 0x51, 0x50, 0xb1, 0xb0, 0xbc, 0xbd, 0x5c, 0x5d, 0xe1, 0xe0, 0x01, 0x00,
  						       0x0c, 0x0d, 0xec, 0xed, 0x51, 0x50, 0xb1, 0xb0, 0xbc, 0xbd, 0x5c, 0x5d, 0xe1, 0xe0, 0x01, 0x00);
	const __m256i nib_mask = _mm256_set1_epi8(0x0f);

        /* Duplicate lanes */
        _a_gf16 = _mm256_permute4x64_epi64(_a_gf16, 0b01000100);
        /* Isolate the nibbles in _a_gf16 */
        _a = _a_gf16 & nib_mask;
        _b = _mm256_srli_epi64(_a_gf16, 4) & nib_mask;
        /* Create the nibbles mix */
        _c = _mm256_shuffle_epi8(_a, shuff_msk_even) | _mm256_shuffle_epi8(_b, shuff_msk_odd);
        /* Lift: since we are on 16 bits, we can perform a vperm lookup inside the register */
        _c = _mm256_shuffle_epi8(lifting_lookup, _c);
        /* Store the result */

	return _c;
}

/*
 * "Hybrid" multiplication of a constant in GF(256) and a vector in GF(16)
 */
static inline void gf256_gf16_constant_vect_mult_avx2(uint8_t a_gf256, const uint8_t *b_gf16, uint8_t *c_gf256, uint32_t n)
{
    gf256_vect_lift_from_gf16_avx2(b_gf16, c_gf256, n);
    gf256_constant_vect_mult_avx2(a_gf256, c_gf256, c_gf256, n);
    return;
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(16) and a vector in GF(256)
 */
static inline uint8_t gf16_gf256_vect_mult_avx2(const uint8_t *a_gf16, const uint8_t *b_gf256, uint32_t len)
{
        uint32_t i;
        __m256i accu, _a, _b;
        __m256i _a_gf16;

        /* Set the accumulator to 0 */
        accu = _mm256_setzero_si256();

        for(i = 0; i < len; i += 32){
                if((len-i) < 32){
                        _a_gf16 = load_incomplete_m256((const uint8_t*)&a_gf16[i/2], (len-i+1)/2);
			_a = _mm256_setzero_si256();
                        gf256_vect_lift_from_gf16_avx2((const uint8_t*)&_a_gf16, (uint8_t*)&_a, len-i);
                        _b = load_incomplete_m256((const uint8_t*)&b_gf256[i], len-i);
                }
                else{
                        /* Obvious 256-bit */
                        _a_gf16 = load_incomplete_m256((const uint8_t*)&a_gf16[i/2], 16);
                        gf256_vect_lift_from_gf16_avx2((const uint8_t*)&_a_gf16, (uint8_t*)&_a, 32);
                        _b = _mm256_lddqu_si256((__m256i*)&b_gf256[i]);
                }
                /* Multiply in GF(256) */
                accu ^= gf256_mult_vectorized_avx2(_a, _b);
        }

        return sum_uint8_avx2(accu);
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(256) and a vector in GF(16)
 */
static inline uint8_t gf256_gf16_vect_mult_avx2(const uint8_t *a_gf256, const uint8_t *b_gf16, uint32_t n)
{
    return gf16_gf256_vect_mult_avx2(b_gf16, a_gf256, n);
}

/*
 * "Hybrid" matrix multiplication of a matrix in GF(256) and a vector in GF(16), resulting
 *  in a vector in GF(256)
 */
static inline void gf256_gf16_mat_mult_avx2(const uint8_t *A, const uint8_t *X, uint8_t *Y_inf, uint8_t *Y_sup, uint32_t n)
{
    uint16_t i, j, k, l;
        uint16_t complete_line = (n + 7) & ~7;
	uint16_t batches, remainder;
	uint32_t row;
        uint64_t sumx8;
        uint32_t sumx4;
	__m256i accu, _a, _b;
	__m256i acc_inf;
	__m128i aux;

        __m256i extra_accs[4];

	__m256i zeros, mask;

	/* Set the accumulator to 0 */
	batches = n>>5;
	remainder = n & 31;

	k=0;
	for (i=0;i<batches;i++) {
	        mask = _mm256_set1_epi32(-1);
	        zeros = _mm256_setr_epi64x(0,0,-1,-1);
                for (l=0;l<2;l++) {
                        for (;k<i*32+l*16+16;k+=4) {
		                row = k*complete_line;
                                extra_accs[0] = _mm256_setzero_si256();
                                extra_accs[1] = _mm256_setzero_si256();
                                extra_accs[2] = _mm256_setzero_si256();
                                extra_accs[3] = _mm256_setzero_si256();
                                for (j=0;j<i;j++) {
                                        aux = _mm_load_si128((__m128i*)&X[16*j]);
                                        _b = _mm256_broadcastsi128_si256(aux);
                                        _b = gf256_lift32_from_gf16_avx2(_b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+32*j]);
                                        extra_accs[0] ^= gf256_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+complete_line+32*j]);
                                        extra_accs[1] ^= gf256_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+2*complete_line+32*j]);
                                        extra_accs[2] ^= gf256_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+3*complete_line+32*j]);
                                        extra_accs[3] ^= gf256_mult_vectorized_avx2(_a, _b);
                                }

                                aux = _mm_load_si128((__m128i*)&X[16*j]);
                                _b = _mm256_broadcastsi128_si256(aux);
                                _b = gf256_lift32_from_gf16_avx2(_b);


                                // row k
                                _a = _mm256_lddqu_si256((__m256i*)&A[row+32*i]);
                                accu = gf256_mult_vectorized_avx2(_a, _b);
                                extra_accs[0] ^= _mm256_andnot_si256(mask, accu);
                                accu = (mask & accu);
                                acc_inf = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[0], accu, 0x20),_mm256_permute2x128_si256(extra_accs[0], accu, 0x31));
                                
                                // row k+1
                                mask = _mm256_alignr_epi8(mask,zeros,15);
                                _a = _mm256_lddqu_si256((__m256i*)&A[(row+complete_line)+32*i]);
                                accu = gf256_mult_vectorized_avx2(_a, _b);
                                extra_accs[1] ^= _mm256_andnot_si256(mask, accu);
                                accu = (mask & accu);
                                extra_accs[0] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[1], accu, 0x20),_mm256_permute2x128_si256(extra_accs[1], accu, 0x31));

                                // k xor k+1
                                acc_inf = _mm256_unpacklo_epi64(acc_inf, extra_accs[0]) ^ _mm256_unpackhi_epi64(acc_inf, extra_accs[0]);

                                // row k+2
                                mask = _mm256_alignr_epi8(mask,zeros,15);
                                _a = _mm256_lddqu_si256((__m256i*)&A[(row+2*complete_line)+32*i]);
                                accu = gf256_mult_vectorized_avx2(_a, _b);
                                extra_accs[2] ^= _mm256_andnot_si256(mask,accu);
                                accu = (mask & accu);
                                extra_accs[0] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[2], accu, 0x20),_mm256_permute2x128_si256(extra_accs[2], accu, 0x31));

                                // row k+3
                                mask = _mm256_alignr_epi8(mask,zeros,15);
                                _a = _mm256_lddqu_si256((__m256i*)&A[(row+3*complete_line)+32*i]);
                                accu = gf256_mult_vectorized_avx2(_a, _b);
                                extra_accs[3] ^= _mm256_andnot_si256(mask,accu);
                                accu = (mask & accu);
                                extra_accs[1] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[3], accu, 0x20),_mm256_permute2x128_si256(extra_accs[3], accu, 0x31));
                                
                                // k+3 xor k+2
                                extra_accs[0] = _mm256_unpacklo_epi64(extra_accs[0], extra_accs[1]) ^ _mm256_unpackhi_epi64(extra_accs[0], extra_accs[1]);

                                // (k xor k+1) xor (k+3 xor k+2)
                                extra_accs[1] = _mm256_blend_epi32(acc_inf, extra_accs[0], 0b10101010);
                                extra_accs[2] = _mm256_blend_epi32(acc_inf, extra_accs[0], 0b01010101);
                                extra_accs[2] = _mm256_shuffle_epi32(extra_accs[2], 0b10110001);
                                acc_inf = _mm256_xor_si256(extra_accs[1],extra_accs[2]);


                                extra_accs[0] = _mm256_setzero_si256();
                                extra_accs[1] = _mm256_setzero_si256();
                                extra_accs[2] = _mm256_setzero_si256();
                                extra_accs[3] = _mm256_setzero_si256();
                                for (j=i+1;j<batches;j++) {
                                        aux = _mm_load_si128((__m128i*)&X[16*j]);
                                        _b = _mm256_broadcastsi128_si256(aux);
                                        _b = gf256_lift32_from_gf16_avx2(_b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+32*j]);
                                        extra_accs[0] ^= gf256_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+complete_line+32*j]);
                                        extra_accs[1] ^= gf256_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+2*complete_line+32*j]);
                                        extra_accs[2] ^= gf256_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+3*complete_line+32*j]);
                                        extra_accs[3] ^= gf256_mult_vectorized_avx2(_a, _b);
                                }

                                _b = load_incomplete_m256(&X[16*j],remainder/2);
                                _b = gf256_lift32_from_gf16_avx2(_b);
                                _a = load_incomplete_m256(&A[row+32*batches],remainder);
                                extra_accs[0] ^= gf256_mult_vectorized_avx2(_a,_b);
                                _a = load_incomplete_m256(&A[row+complete_line+32*batches],remainder);
                                extra_accs[1] ^= gf256_mult_vectorized_avx2(_a,_b);
                                _a = load_incomplete_m256(&A[row+2*complete_line+32*batches],remainder);
                                extra_accs[2] ^= gf256_mult_vectorized_avx2(_a,_b);
                                _a = load_incomplete_m256(&A[row+3*complete_line+32*batches],remainder);
                                extra_accs[3] ^= gf256_mult_vectorized_avx2(_a,_b);
                                accu = _mm256_setzero_si256();
                                extra_accs[0] = _mm256_xor_si256(_mm256_permute2x128_si256(accu, extra_accs[0], 0x20),_mm256_permute2x128_si256(accu, extra_accs[0], 0x31));
                                extra_accs[1] = _mm256_xor_si256(_mm256_permute2x128_si256(accu, extra_accs[1], 0x20),_mm256_permute2x128_si256(accu, extra_accs[1], 0x31));
                                extra_accs[2] = _mm256_xor_si256(_mm256_permute2x128_si256(accu, extra_accs[2], 0x20),_mm256_permute2x128_si256(accu, extra_accs[2], 0x31));
                                extra_accs[3] = _mm256_xor_si256(_mm256_permute2x128_si256(accu, extra_accs[3], 0x20),_mm256_permute2x128_si256(accu, extra_accs[3], 0x31));
                                extra_accs[0] = _mm256_unpacklo_epi64(extra_accs[0], extra_accs[1]) ^ _mm256_unpackhi_epi64(extra_accs[0], extra_accs[1]);
                                extra_accs[1] = _mm256_unpacklo_epi64(extra_accs[2], extra_accs[3]) ^ _mm256_unpackhi_epi64(extra_accs[2], extra_accs[3]);
                                
                                extra_accs[2] = _mm256_blend_epi32(extra_accs[0], extra_accs[1], 0b10101010);
                                extra_accs[3] = _mm256_blend_epi32(extra_accs[0], extra_accs[1], 0b01010101);
                                extra_accs[3] = _mm256_shuffle_epi32(extra_accs[3], 0b10110001);
                                extra_accs[0] = _mm256_xor_si256(extra_accs[2],extra_accs[3]);
                                acc_inf = _mm256_xor_si256(acc_inf,extra_accs[0]);
                                acc_inf = _mm256_permutevar8x32_epi32(acc_inf,_mm256_setr_epi32(0,1,4,5,2,3,6,7));
                                sumx8 = sum_uint8x8_avx2(acc_inf);
                                sumx4 = (uint32_t) sumx8;
                                if (k == 0) {      
                                        sumx4 = sumx4 >> 8;
                                        memcpy(Y_inf, &sumx4, 3);
                                }
                                else {
                                        memcpy(Y_inf + (k-1), &sumx4, 4);
                                }
                                sumx4 = (uint32_t) (sumx8 >> 32);
                                memcpy(Y_sup + k, &sumx4, 4);
                                mask = _mm256_alignr_epi8(mask,zeros,15);
                        }
                        zeros = _mm256_setzero_si256();
                }
	}

	mask = _mm256_set1_epi32(-1);
	zeros = _mm256_setr_epi64x(0,0,-1,-1);
        for (l=0;l<2;l++) {
                for (;k<n && k < 32*batches+16*(l+1);k+=4) {
	                row = k*complete_line;
                        extra_accs[0] = _mm256_setzero_si256();
                        extra_accs[1] = _mm256_setzero_si256();
                        extra_accs[2] = _mm256_setzero_si256();
                        extra_accs[3] = _mm256_setzero_si256();
                        for (j=0;j<batches;j++) {
                                aux = _mm_load_si128((__m128i*)&X[16*j]);
                                _b = _mm256_broadcastsi128_si256(aux);
                                _b = gf256_lift32_from_gf16_avx2(_b);
                                _a = _mm256_lddqu_si256((__m256i*)&A[row+32*j]);
                                extra_accs[0] ^= gf256_mult_vectorized_avx2(_a, _b);
                                _a = _mm256_lddqu_si256((__m256i*)&A[row+complete_line+32*j]);
                                extra_accs[1] ^= gf256_mult_vectorized_avx2(_a, _b);
                                _a = _mm256_lddqu_si256((__m256i*)&A[row+2*complete_line+32*j]);
                                extra_accs[2] ^= gf256_mult_vectorized_avx2(_a, _b);
                                _a = _mm256_lddqu_si256((__m256i*)&A[row+3*complete_line+32*j]);
                                extra_accs[3] ^= gf256_mult_vectorized_avx2(_a, _b);
                        }
                        _b = load_incomplete_m256(&X[16*j],remainder/2);
                        _b = gf256_lift32_from_gf16_avx2(_b);

                        // row k
                        _a = load_incomplete_m256(&A[row+32*batches],remainder);
                        accu = gf256_mult_vectorized_avx2(_a, _b);
                        extra_accs[0] ^= _mm256_andnot_si256(mask, accu);
                        accu = (mask & accu);
                        acc_inf = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[0], accu, 0x20),_mm256_permute2x128_si256(extra_accs[0], accu, 0x31));
                        
                        // row k+1
                        mask = _mm256_alignr_epi8(mask,zeros,15);
                        _a = load_incomplete_m256(&A[(row+complete_line)+32*batches],remainder);
                        accu = gf256_mult_vectorized_avx2(_a, _b);
                        extra_accs[1] ^= _mm256_andnot_si256(mask, accu);
                        accu = (mask & accu);
                        extra_accs[0] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[1], accu, 0x20),_mm256_permute2x128_si256(extra_accs[1], accu, 0x31));

                        // k xor k+1
                        acc_inf = _mm256_unpacklo_epi64(acc_inf, extra_accs[0]) ^ _mm256_unpackhi_epi64(acc_inf, extra_accs[0]);

                        // row k+2
                        mask = _mm256_alignr_epi8(mask,zeros,15);
                        _a = load_incomplete_m256(&A[(row+2*complete_line)+32*batches],remainder);
                        accu = gf256_mult_vectorized_avx2(_a, _b);
                        extra_accs[2] ^= _mm256_andnot_si256(mask,accu);
                        accu = (mask & accu);
                        extra_accs[0] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[2], accu, 0x20),_mm256_permute2x128_si256(extra_accs[2], accu, 0x31));

                        // row k+3
                        mask = _mm256_alignr_epi8(mask,zeros,15);
                        _a = load_incomplete_m256(&A[(row+3*complete_line)+32*batches],remainder);
                        accu = gf256_mult_vectorized_avx2(_a, _b);
                        extra_accs[3] ^= _mm256_andnot_si256(mask,accu);
                        accu = (mask & accu);
                        extra_accs[1] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[3], accu, 0x20),_mm256_permute2x128_si256(extra_accs[3], accu, 0x31));
                        
                        // k+3 xor k+2
                        extra_accs[0] = _mm256_unpacklo_epi64(extra_accs[0], extra_accs[1]) ^ _mm256_unpackhi_epi64(extra_accs[0], extra_accs[1]);

                        // (k xor k+1) xor (k+3 xor k+2)
                        extra_accs[1] = _mm256_blend_epi32(acc_inf, extra_accs[0], 0b10101010);
                        extra_accs[2] = _mm256_blend_epi32(acc_inf, extra_accs[0], 0b01010101);
                        extra_accs[2] = _mm256_shuffle_epi32(extra_accs[2], 0b10110001);
                        acc_inf = _mm256_xor_si256(extra_accs[1],extra_accs[2]);
                        acc_inf = _mm256_permutevar8x32_epi32(acc_inf,_mm256_setr_epi32(0,1,4,5,2,3,6,7));

                        sumx8 = sum_uint8x8_avx2(acc_inf);
                        sumx4 = (uint32_t) sumx8;
                        memcpy(Y_inf + (k-1), &sumx4, 4);
                        sumx4 = (uint32_t) (sumx8 >> 32);
                        memcpy(Y_sup + k, &sumx4, 4);
                        mask = _mm256_alignr_epi8(mask,zeros,15);
                }
                zeros = _mm256_setzero_si256();
        } 
	// End last block

        row = n*complete_line;
        Y_inf[n-1] = gf256_gf16_vect_mult_avx2(&A[row],X,n);
}

static inline void gf256_gf16_mat_mult_single_avx2(const uint8_t *A, const uint8_t *X, uint8_t *Y, uint32_t n)
{
    GF256_GF16_MAT_MULT(A, X, Y, n, TRI_INF, gf256_gf16_vect_mult_avx2);
}


/* === GF(256^2) === */
/* NOTE: for atomic multiplication, using vectorization is suboptimal */
static inline uint16_t gf256to2_mult_avx2(uint16_t x, uint16_t y)
{
	return gf256to2_mult_ref(x, y);
}

/* Vectorize multiplication of _a and _b in GF(256^2): the elements in the field are made of
 * 16 bits each in the lanes of the ymm */
static inline __m256i gf256to2_mult_vectorized_avx2(__m256i _a, __m256i _b)
{
        const __m256i shuff_msk_crossed = _mm256_set_epi8(30, 31, 28, 29, 26, 27, 24, 25, 22, 23, 20, 21, 18, 19, 16, 17,
                                                          14, 15, 12, 13, 10, 11, 8 ,  9,  6,  7,  4,  5,  2,  3,  0,  1);
        const __m256i shuff_msk1 = _mm256_set_epi8(30, 30, 28, 28, 26, 26, 24, 24, 22, 22, 20, 20, 18, 18, 16, 16,
                                                   14, 14, 12, 12, 10, 10, 8 ,  8,  6,  6,  4,  4,  2,  2,  0,  0);
        const __m256i shuff_msk2 = _mm256_set_epi8(31, 31, 29, 29, 27, 27, 25, 25, 23, 23, 21, 21, 19, 19, 17, 17,
                                                   15, 15, 13, 13, 11, 11, 9 ,  9,  7,  7,  5,  5,  3,  3,  1,  1);

        const __m256i const32 = _mm256_set_epi64x(0x0020002000200020, 0x0020002000200020, 0x0020002000200020, 0x0020002000200020);

	const __m256i mask_c1 = _mm256_set_epi64x(0x8000800080008000, 0x8000800080008000, 0x8000800080008000, 0x8000800080008000);
        const __m256i zero = _mm256_setzero_si256();

	__m256i ab = gf256_mult_vectorized_avx2(_a, _b);
	__m256i a0b0 = _mm256_shuffle_epi8(ab, shuff_msk1);
	__m256i a1b1 = _mm256_shuffle_epi8(ab, shuff_msk2);
	__m256i a1b1_32 = gf256_mult_vectorized_avx2(a1b1, const32);
	/* */
	__m256i a0_xor_a1 = _a ^ _mm256_shuffle_epi8(_a, shuff_msk_crossed);
	__m256i b0_xor_b1 = _mm256_blendv_epi8(zero, _b ^ _mm256_shuffle_epi8(_b, shuff_msk_crossed), mask_c1);
	__m256i mult_ab_xor = gf256_mult_vectorized_avx2(a0_xor_a1, b0_xor_b1);
	
	/* Compute the result */
	__m256i res = a0b0 ^ a1b1_32 ^ mult_ab_xor;

        return res;
}

/*
 * Vector multiplied by a constant in GF(256^2).
 */
static inline void gf256to2_constant_vect_mult_avx2(uint16_t b, const uint16_t *a, uint16_t *c, uint32_t len)
{
	uint32_t i;
	__m256i _a, _b;

	/* Load the constant byte b broadcasted in _b */
	_b = _mm256_set1_epi16(b);

        for(i = 0; i < (2 * len); i += 32){
                if(((2 * len)-i) < 32){
                        _a = load_incomplete_m256((const uint8_t*)&a[i / 2], ((2 * len) - i));
                        /* Vectorized multiplication in GF(256) */
                        store_incomplete_m256(gf256to2_mult_vectorized_avx2(_a, _b), (uint8_t*)&c[i / 2], (2 * len)-i);
                }
                else{
                        /* Obvious 512-bit */
                        _a = _mm256_lddqu_si256((__m256i*)&a[i / 2]);
                        /* Vectorized multiplication in GF(256) */
                        _mm256_storeu_si256((__m256i*)&c[i / 2], gf256to2_mult_vectorized_avx2(_a, _b));
                }       
        }

        return;
}

/* Perform a multiplication in GF(256^2) of elements in vectors a an b */
static inline uint16_t gf256to2_vect_mult_avx2(const uint16_t *a, const uint16_t *b, uint32_t len)
{
        uint32_t i;
        __m256i accu, _a, _b;

        /* Set the accumulator to 0 */
        accu = _mm256_setzero_si256();

        for(i = 0; i < (2 * len); i += 32){
                if(((2 * len)-i) < 32){
                        _a = load_incomplete_m256((const uint8_t*)&a[i / 2], ((2 * len) - i));
                        _b = load_incomplete_m256((const uint8_t*)&b[i / 2], ((2 * len) - i));
                }
                else{
                        /* Obvious 256-bit */
                        _a = _mm256_lddqu_si256((__m256i*)&a[i / 2]);
                        _b = _mm256_lddqu_si256((__m256i*)&b[i / 2]);
                }
                accu ^= gf256to2_mult_vectorized_avx2(_a, _b);
        }

        return sum_uint16_avx2(accu);
}

/*
 * GF(2^16) matrix multiplication
 */
static inline void gf256to2_mat_mult_avx2(const uint16_t *A, const uint16_t *X, uint16_t *Y_inf, uint16_t *Y_sup, uint32_t n)
{
    uint16_t i, j, k, l;
        uint16_t complete_line = (n + 7) & ~7;
	uint16_t batches, remainder;
	uint32_t row;
        __m128i sumx8;
        uint64_t sumx4;
	__m256i accu, _a, _b;
	__m256i acc_inf;

        __m256i extra_accs[4];

	__m256i zeros, mask;

	/* Set the accumulator to 0 */
	batches = n>>4;
	remainder = n & 15;

        k=0;
	for (i=0;i<batches;i++) {
                mask = _mm256_set1_epi32(-1);
                zeros = _mm256_setr_epi64x(0,0,-1,-1);
                for (l=0;l<2;l++) {
                        for (;k<i*16+l*8+8;k+=4) {        
                                row = k*complete_line;
                                extra_accs[0] = _mm256_setzero_si256();
                                extra_accs[1] = _mm256_setzero_si256();
                                extra_accs[2] = _mm256_setzero_si256();
                                extra_accs[3] = _mm256_setzero_si256();
                                for (j=0;j<i;j++) {
                                        _b = _mm256_lddqu_si256((__m256i*)&X[16*j]);
		                        _a = _mm256_lddqu_si256((__m256i*)&A[row+16*j]);
                                        extra_accs[0] ^= gf256to2_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+complete_line+16*j]);
                                        extra_accs[1] ^= gf256to2_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+2*complete_line+16*j]);
                                        extra_accs[2] ^= gf256to2_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+3*complete_line+16*j]);
                                        extra_accs[3] ^= gf256to2_mult_vectorized_avx2(_a, _b);
                                }

                                _b = _mm256_lddqu_si256((__m256i*)&X[16*j]);
                                
                                // row k
                                _a = _mm256_lddqu_si256((__m256i*)&A[row+16*i]);
                                accu = gf256to2_mult_vectorized_avx2(_a, _b);
                                extra_accs[0] ^= _mm256_andnot_si256(mask, accu);
                                accu = (mask & accu);
                                acc_inf = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[0], accu, 0x20),_mm256_permute2x128_si256(extra_accs[0], accu, 0x31));
                                
                                // row k+1
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                                _a = _mm256_lddqu_si256((__m256i*)&A[(row+complete_line)+16*i]);
                                accu = gf256to2_mult_vectorized_avx2(_a, _b);
                                extra_accs[1] ^= _mm256_andnot_si256(mask, accu);
                                accu = (mask & accu);
                                extra_accs[0] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[1], accu, 0x20),_mm256_permute2x128_si256(extra_accs[1], accu, 0x31));

                                // k xor k+1
                                acc_inf = _mm256_unpacklo_epi64(acc_inf, extra_accs[0]) ^ _mm256_unpackhi_epi64(acc_inf, extra_accs[0]);

                                // row k+2
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                                _a = _mm256_lddqu_si256((__m256i*)&A[(row+2*complete_line)+16*i]);
                                accu = gf256to2_mult_vectorized_avx2(_a, _b);
                                extra_accs[2] ^= _mm256_andnot_si256(mask,accu);
                                accu = (mask & accu);
                                extra_accs[0] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[2], accu, 0x20),_mm256_permute2x128_si256(extra_accs[2], accu, 0x31));

                                // row k+3
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                                _a = _mm256_lddqu_si256((__m256i*)&A[(row+3*complete_line)+16*i]);
                                accu = gf256to2_mult_vectorized_avx2(_a, _b);
                                extra_accs[3] ^= _mm256_andnot_si256(mask,accu);
                                accu = (mask & accu);
                                extra_accs[1] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[3], accu, 0x20),_mm256_permute2x128_si256(extra_accs[3], accu, 0x31));
                                
                                // k+3 xor k+2
                                extra_accs[0] = _mm256_unpacklo_epi64(extra_accs[0], extra_accs[1]) ^ _mm256_unpackhi_epi64(extra_accs[0], extra_accs[1]);

                                // (k xor k+1) xor (k+3 xor k+2)
                                extra_accs[1] = _mm256_blend_epi32(acc_inf, extra_accs[0], 0b10101010);
                                extra_accs[2] = _mm256_blend_epi32(acc_inf, extra_accs[0], 0b01010101);
                                extra_accs[2] = _mm256_shuffle_epi32(extra_accs[2], 0b10110001);
                                acc_inf = _mm256_xor_si256(extra_accs[1],extra_accs[2]);


                                extra_accs[0] = _mm256_setzero_si256();
                                extra_accs[1] = _mm256_setzero_si256();
                                extra_accs[2] = _mm256_setzero_si256();
                                extra_accs[3] = _mm256_setzero_si256();
                                for (j=i+1;j<batches;j++) {
                                        _b = _mm256_lddqu_si256((__m256i*)&X[16*j]);
		                        _a = _mm256_lddqu_si256((__m256i*)&A[row+16*j]);
                                        extra_accs[0] ^= gf256to2_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+complete_line+16*j]);
                                        extra_accs[1] ^= gf256to2_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+2*complete_line+16*j]);
                                        extra_accs[2] ^= gf256to2_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+3*complete_line+16*j]);
                                        extra_accs[3] ^= gf256to2_mult_vectorized_avx2(_a, _b);
                                }

                                if (remainder > 0) {
                                        _b = load_incomplete_m256((uint8_t *) &X[16*j],2*remainder);
                                        _a = load_incomplete_m256((uint8_t *)&A[row+16*batches],2*remainder);
                                        extra_accs[0] ^= gf256to2_mult_vectorized_avx2(_a,_b);
                                        _a = load_incomplete_m256((uint8_t *)&A[row+complete_line+16*batches],2*remainder);
                                        extra_accs[1] ^= gf256to2_mult_vectorized_avx2(_a,_b);
                                        _a = load_incomplete_m256((uint8_t *)&A[row+2*complete_line+16*batches],2*remainder);
                                        extra_accs[2] ^= gf256to2_mult_vectorized_avx2(_a,_b);
                                        _a = load_incomplete_m256((uint8_t *)&A[row+3*complete_line+16*batches],2*remainder);
                                        extra_accs[3] ^= gf256to2_mult_vectorized_avx2(_a,_b);
                                }

                                accu = _mm256_setzero_si256();
                                extra_accs[0] = _mm256_xor_si256(_mm256_permute2x128_si256(accu, extra_accs[0], 0x20),_mm256_permute2x128_si256(accu, extra_accs[0], 0x31));
                                extra_accs[1] = _mm256_xor_si256(_mm256_permute2x128_si256(accu, extra_accs[1], 0x20),_mm256_permute2x128_si256(accu, extra_accs[1], 0x31));
                                extra_accs[2] = _mm256_xor_si256(_mm256_permute2x128_si256(accu, extra_accs[2], 0x20),_mm256_permute2x128_si256(accu, extra_accs[2], 0x31));
                                extra_accs[3] = _mm256_xor_si256(_mm256_permute2x128_si256(accu, extra_accs[3], 0x20),_mm256_permute2x128_si256(accu, extra_accs[3], 0x31));
                                extra_accs[0] = _mm256_unpacklo_epi64(extra_accs[0], extra_accs[1]) ^ _mm256_unpackhi_epi64(extra_accs[0], extra_accs[1]);
                                extra_accs[1] = _mm256_unpacklo_epi64(extra_accs[2], extra_accs[3]) ^ _mm256_unpackhi_epi64(extra_accs[2], extra_accs[3]);
                                
                                extra_accs[2] = _mm256_blend_epi32(extra_accs[0], extra_accs[1], 0b10101010);
                                extra_accs[3] = _mm256_blend_epi32(extra_accs[0], extra_accs[1], 0b01010101);
                                extra_accs[3] = _mm256_shuffle_epi32(extra_accs[3], 0b10110001);
                                extra_accs[0] = _mm256_xor_si256(extra_accs[2],extra_accs[3]);
                                acc_inf = _mm256_xor_si256(acc_inf,extra_accs[0]);
                                acc_inf = _mm256_permutevar8x32_epi32(acc_inf,_mm256_setr_epi32(0,1,4,5,2,3,6,7));
                                sumx8 = sum_uint16x8_avx2(acc_inf);
                                sumx4 = _mm_cvtsi128_si64(sumx8);
                                if (k == 0) {      
                                        sumx4 = sumx4 >> 16;
                                        memcpy(Y_inf, &sumx4, 6);
                                }
                                else {
                                        memcpy(Y_inf + (k-1), &sumx4, 8);
                                }
                                sumx4 = _mm_extract_epi64(sumx8, 1);
                                memcpy(Y_sup + k, &sumx4, 8);
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                        }
                        zeros = _mm256_setzero_si256();
                }
	}

	if (remainder > 0) {
                mask = _mm256_set1_epi32(-1);
	        zeros = _mm256_setr_epi64x(0,0,-1,-1);
                for (l=0;l<2;l++) {
                        for (;k<n && k < 16*batches+8*(l+1);k+=4) {
                                row = k*complete_line;
                                extra_accs[0] = _mm256_setzero_si256();
                                extra_accs[1] = _mm256_setzero_si256();
                                extra_accs[2] = _mm256_setzero_si256();
                                extra_accs[3] = _mm256_setzero_si256();
                                for (j=0;j<batches;j++) {
                                        _b = _mm256_lddqu_si256((__m256i*)&X[16*j]);
		                        _a = _mm256_lddqu_si256((__m256i*)&A[row+16*j]);
                                        extra_accs[0] ^= gf256to2_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+complete_line+16*j]);
                                        extra_accs[1] ^= gf256to2_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+2*complete_line+16*j]);
                                        extra_accs[2] ^= gf256to2_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+3*complete_line+16*j]);
                                        extra_accs[3] ^= gf256to2_mult_vectorized_avx2(_a, _b);
                                }

                                _b = load_incomplete_m256((uint8_t *)&X[16*j],2*remainder);
                                
                                // row k
                                _a = load_incomplete_m256((uint8_t *)&A[row+16*batches],2*remainder);
                                accu = gf256to2_mult_vectorized_avx2(_a, _b);
                                extra_accs[0] ^= _mm256_andnot_si256(mask, accu);
                                accu = (mask & accu);
                                acc_inf = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[0], accu, 0x20),_mm256_permute2x128_si256(extra_accs[0], accu, 0x31));
                                
                                // row k+1
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                                _a = load_incomplete_m256((uint8_t *)&A[(row+complete_line)+16*batches],2*remainder);
                                accu = gf256to2_mult_vectorized_avx2(_a, _b);
                                extra_accs[1] ^= _mm256_andnot_si256(mask, accu);
                                accu = (mask & accu);
                                extra_accs[0] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[1], accu, 0x20),_mm256_permute2x128_si256(extra_accs[1], accu, 0x31));

                                // k xor k+1
                                acc_inf = _mm256_unpacklo_epi64(acc_inf, extra_accs[0]) ^ _mm256_unpackhi_epi64(acc_inf, extra_accs[0]);

                                // row k+2
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                                _a = load_incomplete_m256((uint8_t *)&A[(row+2*complete_line)+16*batches],2*remainder);
                                accu = gf256to2_mult_vectorized_avx2(_a, _b);
                                extra_accs[2] ^= _mm256_andnot_si256(mask,accu);
                                accu = (mask & accu);
                                extra_accs[0] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[2], accu, 0x20),_mm256_permute2x128_si256(extra_accs[2], accu, 0x31));

                                // row k+3
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                                _a = load_incomplete_m256((uint8_t *)&A[(row+3*complete_line)+16*batches],2*remainder);
                                accu = gf256to2_mult_vectorized_avx2(_a, _b);
                                extra_accs[3] ^= _mm256_andnot_si256(mask,accu);
                                accu = (mask & accu);
                                extra_accs[1] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[3], accu, 0x20),_mm256_permute2x128_si256(extra_accs[3], accu, 0x31));
                                
                                // k+3 xor k+2
                                extra_accs[0] = _mm256_unpacklo_epi64(extra_accs[0], extra_accs[1]) ^ _mm256_unpackhi_epi64(extra_accs[0], extra_accs[1]);

                                // (k xor k+1) xor (k+3 xor k+2)
                                extra_accs[1] = _mm256_blend_epi32(acc_inf, extra_accs[0], 0b10101010);
                                extra_accs[2] = _mm256_blend_epi32(acc_inf, extra_accs[0], 0b01010101);
                                extra_accs[2] = _mm256_shuffle_epi32(extra_accs[2], 0b10110001);
                                acc_inf = _mm256_xor_si256(extra_accs[1],extra_accs[2]);
                                acc_inf = _mm256_permutevar8x32_epi32(acc_inf,_mm256_setr_epi32(0,1,4,5,2,3,6,7));

                                sumx8 = sum_uint16x8_avx2(acc_inf);
                                sumx4 = _mm_cvtsi128_si64(sumx8);
                                memcpy(Y_inf + (k-1), &sumx4, 8);
                                sumx4 = _mm_extract_epi64(sumx8, 1);
                                memcpy(Y_sup + k, &sumx4, 8);
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                        }
                        zeros = _mm256_setzero_si256();
                }
	// End last block
	}

        row = n*complete_line;
        Y_inf[n-1] = gf256to2_vect_mult_avx2(X,&A[row],n);
}

static inline void gf256to2_mat_mult_single_avx2(const uint16_t *A, const uint16_t *X, uint16_t *Y, uint32_t n)
{
    GF256to2_MAT_MULT(A, X, Y, n, TRI_INF, gf256to2_vect_mult_avx2);
}

/*
 * "Hybrid" constant multiplication of a constant in GF(2) and a vector in GF(256^2)
 */
static inline void gf2_gf256to2_constant_vect_mult_avx2(uint8_t a_gf2, const uint16_t *b_gf256to2, uint16_t *c_gf256to2, uint32_t n)
{
	gf2_gf256to2_constant_vect_mult_ref(a_gf2, b_gf256to2, c_gf256to2, n);

        return;
}

/*
 * "Hybrid" constant multiplication of a constant in GF(256^2) and a vector in GF(2)
 */
static inline void gf256to2_gf2_constant_vect_mult_avx2(uint16_t a_gf256to2, const uint8_t *b_gf2, uint16_t *c_gf256to2, uint32_t len)
{
        uint32_t i;
	__m256i _a, _b;

        /* Broadcast the constant value */
	_a = _mm256_set1_epi16(a_gf256to2);

        for(i = 0; i < (2*len); i += 32){
		uint32_t ceil_len;
		if(((2*len) - i) < 32){
			ceil_len = (((2*len) - i) % 16 == 0) ? (((2*len) - i) / 16) : ((((2*len) - i) / 16) + 1);
			_b = load_incomplete_m256(&b_gf2[i / 16], ceil_len);
		}
		else{
			/* Obvious 256-bit */
			ceil_len = 2;
			_b = load_incomplete_m256(&b_gf2[i / 16], 2);
		}
		/* Create a selection mask from the bits in _a */
		const __m256i shuff_msk = _mm256_set_epi8(1, 1, 1,  1,  1, 1, 1, 1, 1, 1, 1,  1,  1, 1, 1, 1,
                                                          0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0);
		const __m256i and_msk = _mm256_set_epi8(0b10000000, 0b10000000, 0b01000000, 0b01000000, 0b00100000, 0b00100000, 0b00010000, 0b00010000,
							0b00001000, 0b00001000, 0b00000100, 0b00000100, 0b00000010, 0b00000010, 0b00000001, 0b00000001,
							0b10000000, 0b10000000, 0b01000000, 0b01000000, 0b00100000, 0b00100000, 0b00010000, 0b00010000,
							0b00001000, 0b00001000, 0b00000100, 0b00000100, 0b00000010, 0b00000010, 0b00000001, 0b00000001);

		/* Only keep the selection bits */
		_b = _mm256_permute4x64_epi64(_b, 0b01000100);
		_b = _mm256_shuffle_epi8(_b, shuff_msk);
		_b = _b & and_msk;
		/* Transform these bits to either 0 or 0xFF */
		_b = _mm256_cmpeq_epi8(_b, and_msk);
		/* Bytes selection */
                __m256i _c = (_a & _b);
		/* Store the result */
		store_incomplete_m256(_c, (uint8_t*)&c_gf256to2[i / 2], 16 * ceil_len);
        }
        return;
}

/*
 * "Hybrid" constant multiplication of a constant in GF(256^2) and a vector in GF(256)
 */
static inline void gf256_gf256to2_constant_vect_mult_avx2(uint8_t a_gf256, const uint16_t *b_gf256to2, uint16_t *c_gf256to2, uint32_t n)
{
	gf256_gf256to2_constant_vect_mult_ref(a_gf256, b_gf256to2, c_gf256to2, n);

	return;
}

/*
 * "Hybrid" constant multiplication of a constant in GF(256^2) and a vector in GF(256)
 */
static inline void gf256to2_gf256_constant_vect_mult_avx2(uint16_t a_gf256to2, const uint8_t *b_gf256, uint16_t *c_gf256to2, uint32_t len)
{
	uint32_t i;
	__m256i _a, _b;
        const __m256i shuff_msk = _mm256_set_epi8(15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8,
                                                   7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2,  2,  1, 1, 0, 0);

	/* Load the constant byte b broadcasted in _b */
	_a = _mm256_set1_epi16(a_gf256to2);

        for(i = 0; i < len; i += 16){
		uint32_t to_load = (len - i) < 16 ? (len - i) : 16;
                _b = load_incomplete_m256((const uint8_t*)&b_gf256[i], to_load);
		/* Copy in the two lanes */
		_b = _mm256_permute4x64_epi64(_b, 0b01000100);
                /* Duplicate elements in the register */
                _b = _mm256_shuffle_epi8(_b, shuff_msk);
                /* Vectorized multiplication in GF(256) */
                store_incomplete_m256(gf256_mult_vectorized_avx2(_a, _b), (uint8_t*)&c_gf256to2[i], 2 * to_load);
        }

        return;
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(2) and a vector in GF(256^2)
 */
static inline uint16_t gf2_gf256to2_vect_mult_avx2(const uint8_t *a_gf2, const uint16_t *b_gf256to2, uint32_t len)
{
        uint32_t i;
	__m256i _a, _b;

        /* Set the accumulator to 0 */
        __m256i accu = _mm256_setzero_si256();

        for(i = 0; i < (2*len); i += 32){
		if(((2*len) - i) < 32){
			uint32_t ceil_len = (((2*len) - i) % 16 == 0) ? (((2*len) - i) / 16) : ((((2*len) - i) / 16) + 1);
			_a = load_incomplete_m256(&a_gf2[i / 16], ceil_len);
			_b = load_incomplete_m256((const uint8_t*)&b_gf256to2[i / 2], (2*len) - i);
		}
		else{
			/* Obvious 256-bit */
			_a = load_incomplete_m256(&a_gf2[i / 16], 2);
			_b = _mm256_lddqu_si256((__m256i*)&b_gf256to2[i / 2]);
		}
		/* Create a selection mask from the bits in _a */
		const __m256i shuff_msk = _mm256_set_epi8(1, 1, 1,  1,  1, 1, 1, 1, 1, 1, 1,  1,  1, 1, 1, 1,
                                                          0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0);
		const __m256i and_msk = _mm256_set_epi8(0b10000000, 0b10000000, 0b01000000, 0b01000000, 0b00100000, 0b00100000, 0b00010000, 0b00010000,
							0b00001000, 0b00001000, 0b00000100, 0b00000100, 0b00000010, 0b00000010, 0b00000001, 0b00000001,
							0b10000000, 0b10000000, 0b01000000, 0b01000000, 0b00100000, 0b00100000, 0b00010000, 0b00010000,
							0b00001000, 0b00001000, 0b00000100, 0b00000100, 0b00000010, 0b00000010, 0b00000001, 0b00000001);
		/* Copy in the two lanes */
		_a = _mm256_permute4x64_epi64(_a, 0b01000100);
		/* Only keep the selection bits */
		_a = _mm256_shuffle_epi8(_a, shuff_msk) & and_msk;
		/* Transform these bits to either 0 or 0xFF */
		_a = _mm256_cmpeq_epi8(_a, and_msk);
		/* Bytes selection */
                accu ^= (_a & _b);
        }
        return sum_uint16_avx2(accu);
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(256^2) and a vector in GF(256)
 */
static inline uint16_t gf256to2_gf2_vect_mult_avx2(const uint16_t *a_gf256to2, const uint8_t *b_gf2, uint32_t n)
{
        return gf2_gf256to2_vect_mult_avx2(b_gf2, a_gf256to2, n);
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(256) and a vector in GF(256^2)
 */
static inline uint16_t gf256_gf256to2_vect_mult_avx2(const uint8_t *a_gf256, const uint16_t *b_gf256to2, uint32_t len)
{
        /* Note: the multiplication of an element in GF(256) and an element in GF(256^2)
         * simply consists in two multiplications in GF(256) (this is multiplying a constant by a degree 1 polynomial)
         * */
        uint32_t i;
        __m256i accu, _a, _b;

        /* Set the accumulator to 0 */
        accu = _mm256_setzero_si256();

        for(i = 0; i < (2 * len); i += 32){
                const __m256i shuff_msk = _mm256_set_epi8(15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8,
                                                          7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2,  2,  1, 1, 0, 0);
                if(((2 * len)-i) < 32){
                        _a = load_incomplete_m256((const uint8_t*)&a_gf256[i / 2], ((2 * len)-i) / 2);
                        _b = load_incomplete_m256((const uint8_t*)&b_gf256to2[i / 2], (2 * len)-i);
                }
                else{
                        /* Obvious 256-bit */
                        _a = load_incomplete_m256((const uint8_t*)&a_gf256[i / 2], 16);
                        _b = _mm256_lddqu_si256((__m256i*)&b_gf256to2[i / 2]);
                }
                /* Duplicate the values in _a */
		_a = _mm256_permute4x64_epi64(_a, 0b01000100);
                _a = _mm256_shuffle_epi8(_a, shuff_msk);
                /* Multiply in GF(256) */
                accu ^= gf256_mult_vectorized_avx2(_a, _b);
        }

        return sum_uint16_avx2(accu);
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(256^2) and a vector in GF(256)
 */
static inline uint16_t gf256to2_gf256_vect_mult_avx2(const uint16_t *a_gf256to2, const uint8_t *b_gf256, uint32_t n)
{
        return gf256_gf256to2_vect_mult_avx2(b_gf256, a_gf256to2, n);
}

/*
 * "Hybrid" matrix multiplication of a matrix in GF(2) and a vector in GF(256^2), resulting
 *  in a vector in GF(256^2)
 */
static inline void gf2_gf256to2_mat_mult_avx2(const uint8_t *A, const uint16_t *X, uint16_t *Y, uint32_t n, matrix_type mtype)
{
        GF2_GF256to2_MAT_MULT(A, X, Y, n, mtype, gf2_gf256to2_vect_mult_avx2);
}

/*
 * "Hybrid" matrix multiplication of a matrix in GF(256^2) and a vector in GF(2), resulting
 *  in a vector in GF(256^2)
 */
static inline void gf256to2_gf2_mat_mult_avx2(const uint16_t *A, const uint8_t *X, uint16_t *Y_inf, uint16_t *Y_sup, uint32_t n)
{
    uint16_t i, j, k, l;
	uint16_t batches, remainder;
        uint16_t complete_line;
	uint32_t row;
        __m128i sumx8;
        uint64_t sumx4;
	__m256i _a;
	__m256i _b;
	__m256i accu, acc_inf;
        
        const __m256i shuff_msk = _mm256_set_epi8(1, 1, 1,  1,  1, 1, 1, 1, 1, 1, 1,  1,  1, 1, 1, 1,
                                                        0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0);
        const __m256i and_msk = _mm256_set_epi8(0b10000000, 0b10000000, 0b01000000, 0b01000000, 0b00100000, 0b00100000, 0b00010000, 0b00010000,
                                                0b00001000, 0b00001000, 0b00000100, 0b00000100, 0b00000010, 0b00000010, 0b00000001, 0b00000001,
                                                0b10000000, 0b10000000, 0b01000000, 0b01000000, 0b00100000, 0b00100000, 0b00010000, 0b00010000,
                                                0b00001000, 0b00001000, 0b00000100, 0b00000100, 0b00000010, 0b00000010, 0b00000001, 0b00000001);        

        __m256i extra_accs[4];

	__m256i zeros, mask;

	batches = n>>4;
	remainder = n & 15;
        complete_line = (n + 7) & ~7;

	k=0;
	for (i=0;i<batches;i++) {
                mask = _mm256_set1_epi32(-1);
                zeros = _mm256_setr_epi64x(0,0,-1,-1);
                for (l=0;l<2;l++) {
                        for (;k<i*16+l*8+8;k+=4) {
                                row = k*complete_line;
                                extra_accs[0] = _mm256_setzero_si256();
                                extra_accs[1] = _mm256_setzero_si256();
                                extra_accs[2] = _mm256_setzero_si256();
                                extra_accs[3] = _mm256_setzero_si256();
                                for (j=0;j<i;j++) {
                                        _b = load_incomplete_m256(&X[2*j], 2);
                                        _b = _mm256_permute4x64_epi64(_b, 0b01000100);
                                        _b = _mm256_shuffle_epi8(_b, shuff_msk) & and_msk;
                                        _b = _mm256_cmpeq_epi8(_b, and_msk);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+16*j]);
                                        extra_accs[0] ^= (_a & _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+complete_line+16*j]);
                                        extra_accs[1] ^= (_a & _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+2*complete_line+16*j]);
                                        extra_accs[2] ^= (_a & _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+3*complete_line+16*j]);
                                        extra_accs[3] ^= (_a & _b);
                                }

                                _b = load_incomplete_m256(&X[2*i], 2);
                                _b = _mm256_permute4x64_epi64(_b, 0b01000100);
                                _b = _mm256_shuffle_epi8(_b, shuff_msk) & and_msk;
                                _b = _mm256_cmpeq_epi8(_b, and_msk);
                                 
                                // row k
                                _a = _mm256_lddqu_si256((__m256i*)&A[row+16*i]);
                                accu = (_a & _b);
                                extra_accs[0] ^= _mm256_andnot_si256(mask, accu);
                                accu = (mask & accu);
                                acc_inf = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[0], accu, 0x20),_mm256_permute2x128_si256(extra_accs[0], accu, 0x31));
                                
                                // row k+1
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                                _a = _mm256_lddqu_si256((__m256i*)&A[(row+complete_line)+16*i]);
                                accu = (_a & _b);
                                extra_accs[1] ^= _mm256_andnot_si256(mask, accu);
                                accu = (mask & accu);
                                extra_accs[0] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[1], accu, 0x20),_mm256_permute2x128_si256(extra_accs[1], accu, 0x31));

                                // k xor k+1
                                acc_inf = _mm256_unpacklo_epi64(acc_inf, extra_accs[0]) ^ _mm256_unpackhi_epi64(acc_inf, extra_accs[0]);

                                // row k+2
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                                _a = _mm256_lddqu_si256((__m256i*)&A[(row+2*complete_line)+16*i]);
                                accu = (_a & _b);
                                extra_accs[2] ^= _mm256_andnot_si256(mask,accu);
                                accu = (mask & accu);
                                extra_accs[0] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[2], accu, 0x20),_mm256_permute2x128_si256(extra_accs[2], accu, 0x31));

                                // row k+3
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                                _a = _mm256_lddqu_si256((__m256i*)&A[(row+3*complete_line)+16*i]);
                                accu = (_a & _b);
                                extra_accs[3] ^= _mm256_andnot_si256(mask,accu);
                                accu = (mask & accu);
                                extra_accs[1] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[3], accu, 0x20),_mm256_permute2x128_si256(extra_accs[3], accu, 0x31));
                                
                                // k+3 xor k+2
                                extra_accs[0] = _mm256_unpacklo_epi64(extra_accs[0], extra_accs[1]) ^ _mm256_unpackhi_epi64(extra_accs[0], extra_accs[1]);

                                // (k xor k+1) xor (k+3 xor k+2)
                                extra_accs[1] = _mm256_blend_epi32(acc_inf, extra_accs[0], 0b10101010);
                                extra_accs[2] = _mm256_blend_epi32(acc_inf, extra_accs[0], 0b01010101);
                                extra_accs[2] = _mm256_shuffle_epi32(extra_accs[2], 0b10110001);
                                acc_inf = _mm256_xor_si256(extra_accs[1],extra_accs[2]);


                                extra_accs[0] = _mm256_setzero_si256();
                                extra_accs[1] = _mm256_setzero_si256();
                                extra_accs[2] = _mm256_setzero_si256();
                                extra_accs[3] = _mm256_setzero_si256();
                                for (j=i+1;j<batches;j++) {
                                        _b = load_incomplete_m256(&X[2*j], 2);
                                        _b = _mm256_permute4x64_epi64(_b, 0b01000100);
                                        _b = _mm256_shuffle_epi8(_b, shuff_msk) & and_msk;
                                        _b = _mm256_cmpeq_epi8(_b, and_msk);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+16*j]);
                                        extra_accs[0] ^= (_a & _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+complete_line+16*j]);
                                        extra_accs[1] ^= (_a & _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+2*complete_line+16*j]);
                                        extra_accs[2] ^= (_a & _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+3*complete_line+16*j]);
                                        extra_accs[3] ^= (_a & _b);

                                }

                                if (remainder > 0) {
                                        _b = load_incomplete_m256(&X[2*j],(remainder+7)/8);
                                        _b = _mm256_permute4x64_epi64(_b, 0b01000100);
                                        _b = _mm256_shuffle_epi8(_b, shuff_msk) & and_msk;
                                        _b = _mm256_cmpeq_epi8(_b, and_msk);
                                        _a = load_incomplete_m256((uint8_t *)&A[row+16*batches],2*remainder);
                                        extra_accs[0] ^= (_a & _b);
                                        _a = load_incomplete_m256((uint8_t *)&A[row+complete_line+16*batches],2*remainder);
                                        extra_accs[1] ^= (_a & _b);
                                        _a = load_incomplete_m256((uint8_t *)&A[row+2*complete_line+16*batches],2*remainder);
                                        extra_accs[2] ^= (_a & _b);
                                        _a = load_incomplete_m256((uint8_t *)&A[row+3*complete_line+16*batches],2*remainder);
                                        extra_accs[3] ^= (_a & _b);
                                }
                                
                                accu = _mm256_setzero_si256();
                                extra_accs[0] = _mm256_xor_si256(_mm256_permute2x128_si256(accu, extra_accs[0], 0x20),_mm256_permute2x128_si256(accu, extra_accs[0], 0x31));
                                extra_accs[1] = _mm256_xor_si256(_mm256_permute2x128_si256(accu, extra_accs[1], 0x20),_mm256_permute2x128_si256(accu, extra_accs[1], 0x31));
                                extra_accs[2] = _mm256_xor_si256(_mm256_permute2x128_si256(accu, extra_accs[2], 0x20),_mm256_permute2x128_si256(accu, extra_accs[2], 0x31));
                                extra_accs[3] = _mm256_xor_si256(_mm256_permute2x128_si256(accu, extra_accs[3], 0x20),_mm256_permute2x128_si256(accu, extra_accs[3], 0x31));
                                extra_accs[0] = _mm256_unpacklo_epi64(extra_accs[0], extra_accs[1]) ^ _mm256_unpackhi_epi64(extra_accs[0], extra_accs[1]);
                                extra_accs[1] = _mm256_unpacklo_epi64(extra_accs[2], extra_accs[3]) ^ _mm256_unpackhi_epi64(extra_accs[2], extra_accs[3]);
                                
                                extra_accs[2] = _mm256_blend_epi32(extra_accs[0], extra_accs[1], 0b10101010);
                                extra_accs[3] = _mm256_blend_epi32(extra_accs[0], extra_accs[1], 0b01010101);
                                extra_accs[3] = _mm256_shuffle_epi32(extra_accs[3], 0b10110001);
                                extra_accs[0] = _mm256_xor_si256(extra_accs[2],extra_accs[3]);
                                acc_inf = _mm256_xor_si256(acc_inf,extra_accs[0]);
                                acc_inf = _mm256_permutevar8x32_epi32(acc_inf,_mm256_setr_epi32(0,1,4,5,2,3,6,7));
                                sumx8 = sum_uint16x8_avx2(acc_inf);
                                sumx4 = _mm_cvtsi128_si64(sumx8);
                                if (k == 0) {      
                                        sumx4 = sumx4 >> 16;
                                        memcpy(Y_inf, &sumx4, 6);
                                }
                                else {
                                        memcpy(Y_inf + (k-1), &sumx4, 8);
                                }
                                sumx4 = _mm_extract_epi64(sumx8, 1);
                                memcpy(Y_sup + k, &sumx4, 8);
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                        }
                        zeros = _mm256_setzero_si256();
                }
	}

	if (remainder > 0) {
	        mask = _mm256_set1_epi32(-1);
	        zeros = _mm256_setr_epi64x(0,0,-1,-1);
                for (l=0;l<2;l++) {
                        for (;k<n && k < 16*batches+8*(l+1);k+=4) {
                                row = k*complete_line;
                                extra_accs[0] = _mm256_setzero_si256();
                                extra_accs[1] = _mm256_setzero_si256();
                                extra_accs[2] = _mm256_setzero_si256();
                                extra_accs[3] = _mm256_setzero_si256();
                                for (j=0;j<batches;j++) {
                                        _b = load_incomplete_m256(&X[2*j], 2);
                                        _b = _mm256_permute4x64_epi64(_b, 0b01000100);
                                        _b = _mm256_shuffle_epi8(_b, shuff_msk) & and_msk;
                                        _b = _mm256_cmpeq_epi8(_b, and_msk);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+16*j]);
                                        extra_accs[0] ^= (_a & _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+complete_line+16*j]);
                                        extra_accs[1] ^= (_a & _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+2*complete_line+16*j]);
                                        extra_accs[2] ^= (_a & _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+3*complete_line+16*j]);
                                        extra_accs[3] ^= (_a & _b);
                                }


                                _b = load_incomplete_m256(&X[2*j],(remainder+7)/8);
                                _b = _mm256_permute4x64_epi64(_b, 0b01000100);
                                _b = _mm256_shuffle_epi8(_b, shuff_msk) & and_msk;
                                _b = _mm256_cmpeq_epi8(_b, and_msk);
                                
                                // row k
                                _a = load_incomplete_m256((uint8_t *)&A[row+16*batches],2*remainder);
                                accu = (_a & _b);
                                extra_accs[0] ^= _mm256_andnot_si256(mask, accu);
                                accu = (mask & accu);
                                acc_inf = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[0], accu, 0x20),_mm256_permute2x128_si256(extra_accs[0], accu, 0x31));
                                
                                // row k+1
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                                _a = load_incomplete_m256((uint8_t *)&A[(row+complete_line)+16*batches],2*remainder);
                                accu = (_a & _b);
                                extra_accs[1] ^= _mm256_andnot_si256(mask, accu);
                                accu = (mask & accu);
                                extra_accs[0] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[1], accu, 0x20),_mm256_permute2x128_si256(extra_accs[1], accu, 0x31));

                                // k xor k+1
                                acc_inf = _mm256_unpacklo_epi64(acc_inf, extra_accs[0]) ^ _mm256_unpackhi_epi64(acc_inf, extra_accs[0]);

                                // row k+2
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                                _a = load_incomplete_m256((uint8_t *)&A[(row+2*complete_line)+16*batches],2*remainder);
                                accu = (_a & _b);
                                extra_accs[2] ^= _mm256_andnot_si256(mask,accu);
                                accu = (mask & accu);
                                extra_accs[0] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[2], accu, 0x20),_mm256_permute2x128_si256(extra_accs[2], accu, 0x31));

                                // row k+3
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                                _a = load_incomplete_m256((uint8_t *)&A[(row+3*complete_line)+16*batches],2*remainder);
                                accu = (_a & _b);
                                extra_accs[3] ^= _mm256_andnot_si256(mask,accu);
                                accu = (mask & accu);
                                extra_accs[1] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[3], accu, 0x20),_mm256_permute2x128_si256(extra_accs[3], accu, 0x31));
                                
                                // k+3 xor k+2
                                extra_accs[0] = _mm256_unpacklo_epi64(extra_accs[0], extra_accs[1]) ^ _mm256_unpackhi_epi64(extra_accs[0], extra_accs[1]);

                                // (k xor k+1) xor (k+3 xor k+2)
                                extra_accs[1] = _mm256_blend_epi32(acc_inf, extra_accs[0], 0b10101010);
                                extra_accs[2] = _mm256_blend_epi32(acc_inf, extra_accs[0], 0b01010101);
                                extra_accs[2] = _mm256_shuffle_epi32(extra_accs[2], 0b10110001);
                                acc_inf = _mm256_xor_si256(extra_accs[1],extra_accs[2]);
                                acc_inf = _mm256_permutevar8x32_epi32(acc_inf,_mm256_setr_epi32(0,1,4,5,2,3,6,7));

                                sumx8 = sum_uint16x8_avx2(acc_inf);
                                sumx4 = _mm_cvtsi128_si64(sumx8);
                                memcpy(Y_inf + (k-1), &sumx4, 8);
                                sumx4 = _mm_extract_epi64(sumx8, 1);
                                memcpy(Y_sup + k, &sumx4, 8);
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                        }
                        zeros = _mm256_setzero_si256();
                }
	        // End last block
	}

        row = n*complete_line;
        Y_inf[n-1] = gf256to2_gf2_vect_mult_avx2(&A[row],X,n);
}

static inline void gf256to2_gf2_mat_mult_single_avx2(const uint16_t *A, const uint8_t *X, uint16_t *Y, uint32_t n)
{
        GF256to2_GF2_MAT_MULT(A, X, Y, n, TRI_INF, gf256to2_gf2_vect_mult_avx2);
}

/*
 * "Hybrid" matrix multiplication of a matrix in GF(256) and a vector in GF(256^2), resulting
 *  in a vector in GF(256^2)
 */
static inline void gf256_gf256to2_mat_mult_avx2(const uint8_t *A, const uint16_t *X, uint16_t *Y, uint32_t n, matrix_type mtype)
{
        GF256to2_MAT_MULT(A, X, Y, n, mtype, gf256_gf256to2_vect_mult_avx2);
}

/*
 * "Hybrid" matrix multiplication of a matrix in GF(256^2) and a vector in GF(256), resulting
 *  in a vector in GF(256^2)
 */
static inline void gf256to2_gf256_mat_mult_avx2(const uint16_t *A, const uint8_t *X, uint16_t *Y_inf, uint16_t *Y_sup, uint32_t n)
{
        uint16_t i, j, k, l;
	uint16_t batches, remainder;
        uint16_t complete_line;
	uint32_t row;
        __m128i sumx8;
        uint64_t sumx4;
	__m256i _a;
	__m256i _b;
	__m256i accu, acc_inf;

        const __m256i shuff_msk = _mm256_set_epi8(15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8,
                                                          7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2,  2,  1, 1, 0, 0);                

        __m256i extra_accs[4];

	__m256i zeros, mask;

	batches = n>>4;
	remainder = n & 15;
        complete_line = (n + 7) & ~7;

	k=0;
	for (i=0;i<batches;i++) {
                mask = _mm256_set1_epi32(-1);
                zeros = _mm256_setr_epi64x(0,0,-1,-1);
                for (l=0;l<2;l++) {
                        for (;k<i*16+l*8+8;k+=4) {
                                row = k*complete_line;
                                extra_accs[0] = _mm256_setzero_si256();
                                extra_accs[1] = _mm256_setzero_si256();
                                extra_accs[2] = _mm256_setzero_si256();
                                extra_accs[3] = _mm256_setzero_si256();
                                for (j=0;j<i;j++) {
                                        _b = load_incomplete_m256(&X[16*j], 16);
                                        _b = _mm256_permute4x64_epi64(_b, 0b01000100);
                                        _b = _mm256_shuffle_epi8(_b, shuff_msk);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+16*j]);
                                        extra_accs[0] ^= gf256_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+complete_line+16*j]);
                                        extra_accs[1] ^= gf256_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+2*complete_line+16*j]);
                                        extra_accs[2] ^= gf256_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+3*complete_line+16*j]);
                                        extra_accs[3] ^= gf256_mult_vectorized_avx2(_a, _b);
                                }

                                _b = load_incomplete_m256(&X[16*i], 16);
                                _b = _mm256_permute4x64_epi64(_b, 0b01000100);
                                _b = _mm256_shuffle_epi8(_b, shuff_msk);
                                                        
                                // row k
                                _a = _mm256_lddqu_si256((__m256i*)&A[row+16*i]);
                                accu = gf256_mult_vectorized_avx2(_a, _b);
                                extra_accs[0] ^= _mm256_andnot_si256(mask, accu);
                                accu = (mask & accu);
                                acc_inf = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[0], accu, 0x20),_mm256_permute2x128_si256(extra_accs[0], accu, 0x31));
                                
                                // row k+1
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                                _a = _mm256_lddqu_si256((__m256i*)&A[(row+complete_line)+16*i]);
                                accu = gf256_mult_vectorized_avx2(_a, _b);
                                extra_accs[1] ^= _mm256_andnot_si256(mask, accu);
                                accu = (mask & accu);
                                extra_accs[0] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[1], accu, 0x20),_mm256_permute2x128_si256(extra_accs[1], accu, 0x31));

                                // k xor k+1
                                acc_inf = _mm256_unpacklo_epi64(acc_inf, extra_accs[0]) ^ _mm256_unpackhi_epi64(acc_inf, extra_accs[0]);

                                // row k+2
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                                _a = _mm256_lddqu_si256((__m256i*)&A[(row+2*complete_line)+16*i]);
                                accu = gf256_mult_vectorized_avx2(_a, _b);
                                extra_accs[2] ^= _mm256_andnot_si256(mask,accu);
                                accu = (mask & accu);
                                extra_accs[0] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[2], accu, 0x20),_mm256_permute2x128_si256(extra_accs[2], accu, 0x31));

                                // row k+3
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                                _a = _mm256_lddqu_si256((__m256i*)&A[(row+3*complete_line)+16*i]);
                                accu = gf256_mult_vectorized_avx2(_a, _b);
                                extra_accs[3] ^= _mm256_andnot_si256(mask,accu);
                                accu = (mask & accu);
                                extra_accs[1] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[3], accu, 0x20),_mm256_permute2x128_si256(extra_accs[3], accu, 0x31));
                                
                                // k+3 xor k+2
                                extra_accs[0] = _mm256_unpacklo_epi64(extra_accs[0], extra_accs[1]) ^ _mm256_unpackhi_epi64(extra_accs[0], extra_accs[1]);

                                // (k xor k+1) xor (k+3 xor k+2)
                                extra_accs[1] = _mm256_blend_epi32(acc_inf, extra_accs[0], 0b10101010);
                                extra_accs[2] = _mm256_blend_epi32(acc_inf, extra_accs[0], 0b01010101);
                                extra_accs[2] = _mm256_shuffle_epi32(extra_accs[2], 0b10110001);
                                acc_inf = _mm256_xor_si256(extra_accs[1],extra_accs[2]);


                                extra_accs[0] = _mm256_setzero_si256();
                                extra_accs[1] = _mm256_setzero_si256();
                                extra_accs[2] = _mm256_setzero_si256();
                                extra_accs[3] = _mm256_setzero_si256();
                                for (j=i+1;j<batches;j++) {
                                        _b = load_incomplete_m256(&X[16*j], 16);
                                        _b = _mm256_permute4x64_epi64(_b, 0b01000100);
                                        _b = _mm256_shuffle_epi8(_b, shuff_msk);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+16*j]);
                                        extra_accs[0] ^= gf256_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+complete_line+16*j]);
                                        extra_accs[1] ^= gf256_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+2*complete_line+16*j]);
                                        extra_accs[2] ^= gf256_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+3*complete_line+16*j]);
                                        extra_accs[3] ^= gf256_mult_vectorized_avx2(_a, _b);

                                }

                                if (remainder > 0) {
                                        _b = load_incomplete_m256(&X[16*j],remainder);
                                        _b = _mm256_permute4x64_epi64(_b, 0b01000100);
                                        _b = _mm256_shuffle_epi8(_b, shuff_msk);
                                        _a = load_incomplete_m256((uint8_t *)&A[row+16*batches],2*remainder);
                                        extra_accs[0] ^= gf256_mult_vectorized_avx2(_a,_b);
                                        _a = load_incomplete_m256((uint8_t *)&A[row+complete_line+16*batches],2*remainder);
                                        extra_accs[1] ^= gf256_mult_vectorized_avx2(_a,_b);
                                        _a = load_incomplete_m256((uint8_t *)&A[row+2*complete_line+16*batches],2*remainder);
                                        extra_accs[2] ^= gf256_mult_vectorized_avx2(_a,_b);
                                        _a = load_incomplete_m256((uint8_t *)&A[row+3*complete_line+16*batches],2*remainder);
                                        extra_accs[3] ^= gf256_mult_vectorized_avx2(_a,_b);
                                }
                                
                                accu = _mm256_setzero_si256();
                                extra_accs[0] = _mm256_xor_si256(_mm256_permute2x128_si256(accu, extra_accs[0], 0x20),_mm256_permute2x128_si256(accu, extra_accs[0], 0x31));
                                extra_accs[1] = _mm256_xor_si256(_mm256_permute2x128_si256(accu, extra_accs[1], 0x20),_mm256_permute2x128_si256(accu, extra_accs[1], 0x31));
                                extra_accs[2] = _mm256_xor_si256(_mm256_permute2x128_si256(accu, extra_accs[2], 0x20),_mm256_permute2x128_si256(accu, extra_accs[2], 0x31));
                                extra_accs[3] = _mm256_xor_si256(_mm256_permute2x128_si256(accu, extra_accs[3], 0x20),_mm256_permute2x128_si256(accu, extra_accs[3], 0x31));
                                extra_accs[0] = _mm256_unpacklo_epi64(extra_accs[0], extra_accs[1]) ^ _mm256_unpackhi_epi64(extra_accs[0], extra_accs[1]);
                                extra_accs[1] = _mm256_unpacklo_epi64(extra_accs[2], extra_accs[3]) ^ _mm256_unpackhi_epi64(extra_accs[2], extra_accs[3]);
                                
                                extra_accs[2] = _mm256_blend_epi32(extra_accs[0], extra_accs[1], 0b10101010);
                                extra_accs[3] = _mm256_blend_epi32(extra_accs[0], extra_accs[1], 0b01010101);
                                extra_accs[3] = _mm256_shuffle_epi32(extra_accs[3], 0b10110001);
                                extra_accs[0] = _mm256_xor_si256(extra_accs[2],extra_accs[3]);
                                acc_inf = _mm256_xor_si256(acc_inf,extra_accs[0]);
                                acc_inf = _mm256_permutevar8x32_epi32(acc_inf,_mm256_setr_epi32(0,1,4,5,2,3,6,7));
                                sumx8 = sum_uint16x8_avx2(acc_inf);
                                sumx4 = _mm_cvtsi128_si64(sumx8);
                                if (k == 0) {      
                                        sumx4 = sumx4 >> 16;
                                        memcpy(Y_inf, &sumx4, 6);
                                }
                                else {
                                        memcpy(Y_inf + (k-1), &sumx4, 8);
                                }
                                sumx4 = _mm_extract_epi64(sumx8, 1);
                                memcpy(Y_sup + k, &sumx4, 8);
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                        }
                        zeros = _mm256_setzero_si256();
                }
	}

	if (remainder > 0) {
	        mask = _mm256_set1_epi32(-1);
	        zeros = _mm256_setr_epi64x(0,0,-1,-1);
                for (l=0;l<2;l++) {
                        for (;k<n && k < 16*batches+8*(l+1);k+=4) {
                                row = k*complete_line;
                                extra_accs[0] = _mm256_setzero_si256();
                                extra_accs[1] = _mm256_setzero_si256();
                                extra_accs[2] = _mm256_setzero_si256();
                                extra_accs[3] = _mm256_setzero_si256();
                                for (j=0;j<batches;j++) {
                                        _b = load_incomplete_m256(&X[16*j], 16);
                                        _b = _mm256_permute4x64_epi64(_b, 0b01000100);
                                        _b = _mm256_shuffle_epi8(_b, shuff_msk);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+16*j]);
                                        extra_accs[0] ^= gf256_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+complete_line+16*j]);
                                        extra_accs[1] ^= gf256_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+2*complete_line+16*j]);
                                        extra_accs[2] ^= gf256_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+3*complete_line+16*j]);
                                        extra_accs[3] ^= gf256_mult_vectorized_avx2(_a, _b);
                                }


                                _b = load_incomplete_m256(&X[16*j],remainder);
                                _b = _mm256_permute4x64_epi64(_b, 0b01000100);
                                _b = _mm256_shuffle_epi8(_b, shuff_msk);
                                
                                // row k
                                _a = load_incomplete_m256((uint8_t *)&A[row+16*batches],2*remainder);
                                accu = gf256_mult_vectorized_avx2(_a, _b);
                                extra_accs[0] ^= _mm256_andnot_si256(mask, accu);
                                accu = (mask & accu);
                                acc_inf = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[0], accu, 0x20),_mm256_permute2x128_si256(extra_accs[0], accu, 0x31));
                                
                                // row k+1
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                                _a = load_incomplete_m256((uint8_t *)&A[(row+complete_line)+16*batches],2*remainder);
                                accu = gf256_mult_vectorized_avx2(_a, _b);
                                extra_accs[1] ^= _mm256_andnot_si256(mask, accu);
                                accu = (mask & accu);
                                extra_accs[0] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[1], accu, 0x20),_mm256_permute2x128_si256(extra_accs[1], accu, 0x31));

                                // k xor k+1
                                acc_inf = _mm256_unpacklo_epi64(acc_inf, extra_accs[0]) ^ _mm256_unpackhi_epi64(acc_inf, extra_accs[0]);

                                // row k+2
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                                _a = load_incomplete_m256((uint8_t *)&A[(row+2*complete_line)+16*batches],2*remainder);
                                accu = gf256_mult_vectorized_avx2(_a, _b);
                                extra_accs[2] ^= _mm256_andnot_si256(mask,accu);
                                accu = (mask & accu);
                                extra_accs[0] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[2], accu, 0x20),_mm256_permute2x128_si256(extra_accs[2], accu, 0x31));

                                // row k+3
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                                _a = load_incomplete_m256((uint8_t *)&A[(row+3*complete_line)+16*batches],2*remainder);
                                accu = gf256_mult_vectorized_avx2(_a, _b);
                                extra_accs[3] ^= _mm256_andnot_si256(mask,accu);
                                accu = (mask & accu);
                                extra_accs[1] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[3], accu, 0x20),_mm256_permute2x128_si256(extra_accs[3], accu, 0x31));
                                
                                // k+3 xor k+2
                                extra_accs[0] = _mm256_unpacklo_epi64(extra_accs[0], extra_accs[1]) ^ _mm256_unpackhi_epi64(extra_accs[0], extra_accs[1]);

                                // (k xor k+1) xor (k+3 xor k+2)
                                extra_accs[1] = _mm256_blend_epi32(acc_inf, extra_accs[0], 0b10101010);
                                extra_accs[2] = _mm256_blend_epi32(acc_inf, extra_accs[0], 0b01010101);
                                extra_accs[2] = _mm256_shuffle_epi32(extra_accs[2], 0b10110001);
                                acc_inf = _mm256_xor_si256(extra_accs[1],extra_accs[2]);
                                acc_inf = _mm256_permutevar8x32_epi32(acc_inf,_mm256_setr_epi32(0,1,4,5,2,3,6,7));

                                sumx8 = sum_uint16x8_avx2(acc_inf);
                                sumx4 = _mm_cvtsi128_si64(sumx8);
                                memcpy(Y_inf + (k-1), &sumx4, 8);
                                sumx4 = _mm_extract_epi64(sumx8, 1);
                                memcpy(Y_sup + k, &sumx4, 8);
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                        }
                        zeros = _mm256_setzero_si256();
                }
	        // End last block
	}

        row = n*complete_line;
        Y_inf[n-1] = gf256to2_gf256_vect_mult_avx2(&A[row],X,n);
}

static inline void gf256to2_gf256_mat_mult_single_avx2(const uint16_t *A, const uint8_t *X, uint16_t *Y, uint32_t n)
{
        GF256to2_MAT_MULT(A, X, Y, n, TRI_INF, gf256to2_gf256_vect_mult_avx2);
}

static inline void gf256to2_transpose_block_avx2(uint16_t *src, uint16_t *dst, uint16_t size_in, uint16_t size_out)
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
static inline void gf256to2_mat_transpose_avx2(uint16_t *A, uint16_t len, uint16_t extension)
{
	uint16_t *block1;
	uint16_t *block2;
	uint16_t aux[64] = {0};
	__m128i mv;
	for (int i=0;i<len;i+=8){
		for (int j=0;j<i;j+=8) {
			block1 = A + i*extension + j;
			block2 = A + j*extension + i;
			gf256to2_transpose_block_avx2(block1,aux,extension,8);
			gf256to2_transpose_block_avx2(block2,block1,extension,extension);
			for (int k=0;k<8;k++) {
				mv = _mm_load_si128((__m128i*)&aux[8*k]);
                                _mm_storeu_si128((__m128i*)&block2[extension*k],mv);
			}
		}
	}
	for (int i=0;i<len;i+=8){
				block1 = A + i*extension + i;
				gf256to2_transpose_block_avx2(block1,block1,extension,extension);
	}
}


/*
 * "Hybrid" constant multiplication of a constant in GF(4) and a vector in GF(256^2)
 */
static inline void gf4_gf256to2_constant_vect_mult_avx2(uint8_t a_gf4, const uint16_t *b_gf256to2, uint16_t *c_gf256to2, uint32_t n)
{
    gf4_gf256to2_constant_vect_mult_ref(a_gf4, b_gf256to2, c_gf256to2, n);
    return;
}

/*
 * "Hybrid" constant multiplication of a constant in GF(256^2) and a vector in GF(4)
 */
static inline void gf256to2_gf4_constant_vect_mult_avx2(uint16_t a_gf256to2, const uint8_t *b_gf4, uint16_t *c_gf256to2, uint32_t n)
{
    gf256to2_gf4_constant_vect_mult_ref(a_gf256to2, b_gf4, c_gf256to2, n);
    return;
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(4) and a vector in GF(256^2)
 */
static inline uint16_t gf4_gf256to2_vect_mult_avx2(const uint8_t *a_gf4, const uint16_t *b_gf256to2, uint32_t n)
{
    return gf4_gf256to2_vect_mult_ref(a_gf4, b_gf256to2, n);
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(256^2) and a vector in GF(4)
 */
static inline uint16_t gf256to2_gf4_vect_mult_avx2(const uint16_t *a_gf256to2, const uint8_t *b_gf4, uint32_t n)
{
    return gf4_gf256to2_vect_mult_avx2(b_gf4, a_gf256to2, n);
}

/*
 * "Hybrid" matrix multiplication of a matrix in GF(256^2) and a vector in GF(4), resulting
 *  in a vector in GF(256^2)
 */
static inline void gf256to2_gf4_mat_mult_avx2(const uint16_t *A, const uint8_t *X, uint16_t *Y, uint32_t n, matrix_type mtype)
{
    GF256to2_GF4_MAT_MULT(A, X, Y, n, mtype, gf256to2_gf4_vect_mult_avx2);
}

/*
 * "Hybrid" constant multiplication of a constant in GF(16) and a vector in GF(256^2)
 */
static inline void gf16_gf256to2_constant_vect_mult_avx2(uint8_t a_gf16, const uint16_t *b_gf256to2, uint16_t *c_gf256to2, uint32_t n)
{
    uint8_t a_gf256;
    gf256_vect_lift_from_gf16_ref(&a_gf16, &a_gf256, 1);
    gf256_gf256to2_constant_vect_mult_avx2(a_gf256, b_gf256to2, c_gf256to2, n);
    return;
}

/*
 * "Hybrid" constant multiplication of a constant in GF(256^2) and a vector in GF(16)
 */
static inline void gf256to2_gf16_constant_vect_mult_avx2(uint16_t a_gf256to2, const uint8_t *b_gf16, uint16_t *c_gf256to2, uint32_t n)
{
    uint8_t* buf = ((uint8_t*) c_gf256to2) + n;
    gf256_vect_lift_from_gf16_avx2(b_gf16, buf, n);
    gf256to2_gf256_constant_vect_mult_avx2(a_gf256to2, buf, c_gf256to2, n);
    return;
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(16) and a vector in GF(256^2)
 */
static inline uint16_t gf16_gf256to2_vect_mult_avx2(const uint8_t *a_gf16, const uint16_t *b_gf256to2, uint32_t len)
{
        uint32_t i;
        __m256i accu, _a, _b;
        __m256i _a_gf16;

        /* Set the accumulator to 0 */
        accu = _mm256_setzero_si256();

        for(i = 0; i < (2 * len); i += 32){
                const __m256i shuff_msk = _mm256_set_epi8(15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8,
                                                          7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2,  2,  1, 1, 0, 0);
                if(((2 * len)-i) < 32){
                        _a_gf16 = load_incomplete_m256((const uint8_t*)&a_gf16[i / 4], (((2 * len)-i) + 3)/ 4);
                        _a = _mm256_setzero_si256();
                        gf256_vect_lift_from_gf16_avx2((const uint8_t*)&_a_gf16, (uint8_t*)&_a, ((2 * len)-i+1)/2);
                        _b = load_incomplete_m256((const uint8_t*)&b_gf256to2[i / 2], (2 * len)-i);
                }
                else{
                        /* Obvious 256-bit */
                        _a_gf16 = load_incomplete_m256((const uint8_t*)&a_gf16[i / 4], 8);
                        gf256_vect_lift_from_gf16_avx2((const uint8_t*)&_a_gf16, (uint8_t*)&_a, 16);
                        _b = _mm256_lddqu_si256((__m256i*)&b_gf256to2[i / 2]);
                }
                /* Duplicate the values in _a */
                _a = _mm256_permute4x64_epi64(_a, 0b01000100);
                _a = _mm256_shuffle_epi8(_a, shuff_msk);
                /* Multiply in GF(256) */
                accu ^= gf256_mult_vectorized_avx2(_a, _b);
        }

        return sum_uint16_avx2(accu);
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(256^2) and a vector in GF(16)
 */
static inline uint16_t gf256to2_gf16_vect_mult_avx2(const uint16_t *a_gf256to2, const uint8_t *b_gf16, uint32_t n)
{
    return gf16_gf256to2_vect_mult_avx2(b_gf16, a_gf256to2, n);
}

/*
 * "Hybrid" matrix multiplication of a matrix in GF(256^2) and a vector in GF(16), resulting
 *  in a vector in GF(256^2)
 */
static inline void gf256to2_gf16_mat_mult_avx2(const uint16_t *A, const uint8_t *X, uint16_t *Y_inf, uint16_t *Y_sup, uint32_t n)
{
    uint16_t i, j, k, l;
	uint16_t batches, remainder;
        uint16_t complete_line;
	uint32_t row;
        __m128i sumx8;
        uint64_t sumx4;
	__m256i _a;
	__m256i _b;
	__m256i accu, acc_inf;

        const __m256i shuff_msk = _mm256_set_epi8(15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8,
                                                   7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2,  2,  1, 1, 0, 0);
                

        __m256i extra_accs[4];

	__m256i zeros, mask;

	batches = n>>4;
	remainder = n & 15;
        complete_line = (n + 7) & ~7;

	k=0;
	for (i=0;i<batches;i++) {
                mask = _mm256_set1_epi32(-1);
                zeros = _mm256_setr_epi64x(0,0,-1,-1);
                for (l=0;l<2;l++) {
                        for (;k<i*16+l*8+8;k+=4) {
                                row = k*complete_line;
                                extra_accs[0] = _mm256_setzero_si256();
                                extra_accs[1] = _mm256_setzero_si256();
                                extra_accs[2] = _mm256_setzero_si256();
                                extra_accs[3] = _mm256_setzero_si256();
                                for (j=0;j<i;j++) {
                                        _b = load_incomplete_m256(&X[8*j],8);
                                        _b = gf256_lift32_from_gf16_avx2(_b);
                                        _b = _mm256_permute4x64_epi64(_b, 0b01000100);
                                        _b = _mm256_shuffle_epi8(_b, shuff_msk);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+16*j]);
                                        extra_accs[0] ^= gf256_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+complete_line+16*j]);
                                        extra_accs[1] ^= gf256_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+2*complete_line+16*j]);
                                        extra_accs[2] ^= gf256_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+3*complete_line+16*j]);
                                        extra_accs[3] ^= gf256_mult_vectorized_avx2(_a, _b);
                                }

                                _b = load_incomplete_m256(&X[8*j],8);
                                _b = gf256_lift32_from_gf16_avx2(_b);
                                _b = _mm256_permute4x64_epi64(_b, 0b01000100);
                                _b = _mm256_shuffle_epi8(_b, shuff_msk);
                                 
                                // row k
                                _a = _mm256_lddqu_si256((__m256i*)&A[row+16*i]);
                                accu = gf256_mult_vectorized_avx2(_a, _b);
                                extra_accs[0] ^= _mm256_andnot_si256(mask, accu);
                                accu = (mask & accu);
                                acc_inf = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[0], accu, 0x20),_mm256_permute2x128_si256(extra_accs[0], accu, 0x31));
                                
                                // row k+1
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                                _a = _mm256_lddqu_si256((__m256i*)&A[(row+complete_line)+16*i]);
                                accu = gf256_mult_vectorized_avx2(_a, _b);
                                extra_accs[1] ^= _mm256_andnot_si256(mask, accu);
                                accu = (mask & accu);
                                extra_accs[0] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[1], accu, 0x20),_mm256_permute2x128_si256(extra_accs[1], accu, 0x31));

                                // k xor k+1
                                acc_inf = _mm256_unpacklo_epi64(acc_inf, extra_accs[0]) ^ _mm256_unpackhi_epi64(acc_inf, extra_accs[0]);

                                // row k+2
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                                _a = _mm256_lddqu_si256((__m256i*)&A[(row+2*complete_line)+16*i]);
                                accu = gf256_mult_vectorized_avx2(_a, _b);
                                extra_accs[2] ^= _mm256_andnot_si256(mask,accu);
                                accu = (mask & accu);
                                extra_accs[0] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[2], accu, 0x20),_mm256_permute2x128_si256(extra_accs[2], accu, 0x31));

                                // row k+3
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                                _a = _mm256_lddqu_si256((__m256i*)&A[(row+3*complete_line)+16*i]);
                                accu = gf256_mult_vectorized_avx2(_a, _b);
                                extra_accs[3] ^= _mm256_andnot_si256(mask,accu);
                                accu = (mask & accu);
                                extra_accs[1] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[3], accu, 0x20),_mm256_permute2x128_si256(extra_accs[3], accu, 0x31));
                                
                                // k+3 xor k+2
                                extra_accs[0] = _mm256_unpacklo_epi64(extra_accs[0], extra_accs[1]) ^ _mm256_unpackhi_epi64(extra_accs[0], extra_accs[1]);

                                // (k xor k+1) xor (k+3 xor k+2)
                                extra_accs[1] = _mm256_blend_epi32(acc_inf, extra_accs[0], 0b10101010);
                                extra_accs[2] = _mm256_blend_epi32(acc_inf, extra_accs[0], 0b01010101);
                                extra_accs[2] = _mm256_shuffle_epi32(extra_accs[2], 0b10110001);
                                acc_inf = _mm256_xor_si256(extra_accs[1],extra_accs[2]);


                                extra_accs[0] = _mm256_setzero_si256();
                                extra_accs[1] = _mm256_setzero_si256();
                                extra_accs[2] = _mm256_setzero_si256();
                                extra_accs[3] = _mm256_setzero_si256();
                                for (j=i+1;j<batches;j++) {
                                        _b = load_incomplete_m256(&X[8*j],8);
                                        _b = gf256_lift32_from_gf16_avx2(_b);
                                        _b = _mm256_permute4x64_epi64(_b, 0b01000100);
                                        _b = _mm256_shuffle_epi8(_b, shuff_msk);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+16*j]);
                                        extra_accs[0] ^= gf256_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+complete_line+16*j]);
                                        extra_accs[1] ^= gf256_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+2*complete_line+16*j]);
                                        extra_accs[2] ^= gf256_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+3*complete_line+16*j]);
                                        extra_accs[3] ^= gf256_mult_vectorized_avx2(_a, _b);

                                }

                                if (remainder > 0) {
                                        _b = load_incomplete_m256(&X[8*j],(remainder+1)/2);
                                        _b = gf256_lift32_from_gf16_avx2(_b);
                                        _b = _mm256_permute4x64_epi64(_b, 0b01000100);
                                        _b = _mm256_shuffle_epi8(_b, shuff_msk);
                                        _a = load_incomplete_m256((uint8_t *)&A[row+16*batches],2*remainder);
                                        extra_accs[0] ^= gf256_mult_vectorized_avx2(_a,_b);
                                        _a = load_incomplete_m256((uint8_t *)&A[row+complete_line+16*batches],2*remainder);
                                        extra_accs[1] ^= gf256_mult_vectorized_avx2(_a,_b);
                                        _a = load_incomplete_m256((uint8_t *)&A[row+2*complete_line+16*batches],2*remainder);
                                        extra_accs[2] ^= gf256_mult_vectorized_avx2(_a,_b);
                                        _a = load_incomplete_m256((uint8_t *)&A[row+3*complete_line+16*batches],2*remainder);
                                        extra_accs[3] ^= gf256_mult_vectorized_avx2(_a,_b);
                                }
                                
                                accu = _mm256_setzero_si256();
                                extra_accs[0] = _mm256_xor_si256(_mm256_permute2x128_si256(accu, extra_accs[0], 0x20),_mm256_permute2x128_si256(accu, extra_accs[0], 0x31));
                                extra_accs[1] = _mm256_xor_si256(_mm256_permute2x128_si256(accu, extra_accs[1], 0x20),_mm256_permute2x128_si256(accu, extra_accs[1], 0x31));
                                extra_accs[2] = _mm256_xor_si256(_mm256_permute2x128_si256(accu, extra_accs[2], 0x20),_mm256_permute2x128_si256(accu, extra_accs[2], 0x31));
                                extra_accs[3] = _mm256_xor_si256(_mm256_permute2x128_si256(accu, extra_accs[3], 0x20),_mm256_permute2x128_si256(accu, extra_accs[3], 0x31));
                                extra_accs[0] = _mm256_unpacklo_epi64(extra_accs[0], extra_accs[1]) ^ _mm256_unpackhi_epi64(extra_accs[0], extra_accs[1]);
                                extra_accs[1] = _mm256_unpacklo_epi64(extra_accs[2], extra_accs[3]) ^ _mm256_unpackhi_epi64(extra_accs[2], extra_accs[3]);
                                
                                extra_accs[2] = _mm256_blend_epi32(extra_accs[0], extra_accs[1], 0b10101010);
                                extra_accs[3] = _mm256_blend_epi32(extra_accs[0], extra_accs[1], 0b01010101);
                                extra_accs[3] = _mm256_shuffle_epi32(extra_accs[3], 0b10110001);
                                extra_accs[0] = _mm256_xor_si256(extra_accs[2],extra_accs[3]);
                                acc_inf = _mm256_xor_si256(acc_inf,extra_accs[0]);
                                acc_inf = _mm256_permutevar8x32_epi32(acc_inf,_mm256_setr_epi32(0,1,4,5,2,3,6,7));
                                sumx8 = sum_uint16x8_avx2(acc_inf);
                                sumx4 = _mm_cvtsi128_si64(sumx8);
                                if (k == 0) {      
                                        sumx4 = sumx4 >> 16;
                                        memcpy(Y_inf, &sumx4, 6);
                                }
                                else {
                                        memcpy(Y_inf + (k-1), &sumx4, 8);
                                }
                                sumx4 = _mm_extract_epi64(sumx8, 1);
                                memcpy(Y_sup + k, &sumx4, 8);
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                        }
                        zeros = _mm256_setzero_si256();
                }
	}

	if (remainder > 0) {
	        mask = _mm256_set1_epi32(-1);
	        zeros = _mm256_setr_epi64x(0,0,-1,-1);
                for (l=0;l<2;l++) {
                        for (;k<n && k < 16*batches+8*(l+1);k+=4) {
                                row = k*complete_line;
                                extra_accs[0] = _mm256_setzero_si256();
                                extra_accs[1] = _mm256_setzero_si256();
                                extra_accs[2] = _mm256_setzero_si256();
                                extra_accs[3] = _mm256_setzero_si256();
                                for (j=0;j<batches;j++) {
                                        _b = load_incomplete_m256(&X[8*j],8);
                                        _b = gf256_lift32_from_gf16_avx2(_b);
                                        _b = _mm256_permute4x64_epi64(_b, 0b01000100);
                                        _b = _mm256_shuffle_epi8(_b, shuff_msk);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+16*j]);
                                        extra_accs[0] ^= gf256_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+complete_line+16*j]);
                                        extra_accs[1] ^= gf256_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+2*complete_line+16*j]);
                                        extra_accs[2] ^= gf256_mult_vectorized_avx2(_a, _b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row+3*complete_line+16*j]);
                                        extra_accs[3] ^= gf256_mult_vectorized_avx2(_a, _b);
                                }


                                _b = load_incomplete_m256(&X[8*j],(remainder+1)/2);
                                _b = gf256_lift32_from_gf16_avx2(_b);
                                _b = _mm256_permute4x64_epi64(_b, 0b01000100);
                                _b = _mm256_shuffle_epi8(_b, shuff_msk);
                                
                                // row k
                                _a = load_incomplete_m256((uint8_t *)&A[row+16*batches],2*remainder);
                                accu = gf256_mult_vectorized_avx2(_a, _b);
                                extra_accs[0] ^= _mm256_andnot_si256(mask, accu);
                                accu = (mask & accu);
                                acc_inf = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[0], accu, 0x20),_mm256_permute2x128_si256(extra_accs[0], accu, 0x31));
                                
                                // row k+1
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                                _a = load_incomplete_m256((uint8_t *)&A[(row+complete_line)+16*batches],2*remainder);
                                accu = gf256_mult_vectorized_avx2(_a, _b);
                                extra_accs[1] ^= _mm256_andnot_si256(mask, accu);
                                accu = (mask & accu);
                                extra_accs[0] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[1], accu, 0x20),_mm256_permute2x128_si256(extra_accs[1], accu, 0x31));

                                // k xor k+1
                                acc_inf = _mm256_unpacklo_epi64(acc_inf, extra_accs[0]) ^ _mm256_unpackhi_epi64(acc_inf, extra_accs[0]);

                                // row k+2
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                                _a = load_incomplete_m256((uint8_t *)&A[(row+2*complete_line)+16*batches],2*remainder);
                                accu = gf256_mult_vectorized_avx2(_a, _b);
                                extra_accs[2] ^= _mm256_andnot_si256(mask,accu);
                                accu = (mask & accu);
                                extra_accs[0] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[2], accu, 0x20),_mm256_permute2x128_si256(extra_accs[2], accu, 0x31));

                                // row k+3
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                                _a = load_incomplete_m256((uint8_t *)&A[(row+3*complete_line)+16*batches],2*remainder);
                                accu = gf256_mult_vectorized_avx2(_a, _b);
                                extra_accs[3] ^= _mm256_andnot_si256(mask,accu);
                                accu = (mask & accu);
                                extra_accs[1] = _mm256_xor_si256(_mm256_permute2x128_si256(extra_accs[3], accu, 0x20),_mm256_permute2x128_si256(extra_accs[3], accu, 0x31));
                                
                                // k+3 xor k+2
                                extra_accs[0] = _mm256_unpacklo_epi64(extra_accs[0], extra_accs[1]) ^ _mm256_unpackhi_epi64(extra_accs[0], extra_accs[1]);

                                // (k xor k+1) xor (k+3 xor k+2)
                                extra_accs[1] = _mm256_blend_epi32(acc_inf, extra_accs[0], 0b10101010);
                                extra_accs[2] = _mm256_blend_epi32(acc_inf, extra_accs[0], 0b01010101);
                                extra_accs[2] = _mm256_shuffle_epi32(extra_accs[2], 0b10110001);
                                acc_inf = _mm256_xor_si256(extra_accs[1],extra_accs[2]);
                                acc_inf = _mm256_permutevar8x32_epi32(acc_inf,_mm256_setr_epi32(0,1,4,5,2,3,6,7));

                                sumx8 = sum_uint16x8_avx2(acc_inf);
                                sumx4 = _mm_cvtsi128_si64(sumx8);
                                memcpy(Y_inf + (k-1), &sumx4, 8);
                                sumx4 = _mm_extract_epi64(sumx8, 1);
                                memcpy(Y_sup + k, &sumx4, 8);
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                        }
                        zeros = _mm256_setzero_si256();
                }
	        // End last block
	}

        row = n*complete_line;
        Y_inf[n-1] = gf256to2_gf16_vect_mult_avx2(&A[row],X,n);
}

static inline void gf256to2_gf16_mat_mult_single_avx2(const uint16_t *A, const uint8_t *X, uint16_t *Y, uint32_t n)
{
    GF256_GF16_MAT_MULT(A, X, Y, n, TRI_INF, gf256to2_gf16_vect_mult_avx2);
}

#endif /* __AVX2__ */

#endif /* __FIELDS_AVX2_H__ */
