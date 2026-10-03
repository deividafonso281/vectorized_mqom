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

#define TRANSPOSE_MULT_ADVANCED 1

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


static inline void print_m256i_hex(__m256i v) {
    unsigned long long w0 = _mm256_extract_epi64(v, 0);
    unsigned long long w1 = _mm256_extract_epi64(v, 1);
    unsigned long long w2 = _mm256_extract_epi64(v, 2);
    unsigned long long w3 = _mm256_extract_epi64(v, 3);

    // Cada lane = 16 hex chars. Juntos = 64 chars (256 bits)
    printf("%016llx%016llx%016llx%016llx\n",
           w3, w2, w1, w0);   // ordem invertida para ficar natural (MSB → LSB)
}

/* Matrix and vector multiplication over GF(256) 
 * C += A * X, where X is a vector
 * Matrix is supposed to be square n x n, and vector n x 1
 * The output is a vector n x 1
 * */
static inline void gf256_mat_mult_avx2(const uint8_t *A, const uint8_t *X, uint8_t *Y_inf, uint8_t *Y_sup, const uint8_t * even_diagonal, uint32_t n)
{
    uint16_t i, j, k, l;
        uint16_t complete_line = (n + 7) & ~7;
	uint16_t batches, remainder;
	uint32_t row; //, ind;
	__m256i accu, _a, dups, _b;
	__m256i acc_inf,prod;

	__m256i zeros, mask;
        __m256i ones = _mm256_set1_epi64x(-1);
        __m128i lanes[2];

	/* Set the accumulator to 0 */
	batches = n>>5;
	remainder = n & 31;
	for (i=0;i<batches;i++) {
	        zeros = _mm256_setr_epi64x(0,0,-1,-1);
		acc_inf = _mm256_setzero_si256();
		accu = _mm256_setzero_si256();
		mask = _mm256_set1_epi32(-1);
		row = i<<5;
		for (j=0;j<i;j++) {
			_b = _mm256_lddqu_si256((__m256i*)&X[32*j]);
                        lanes[0] = _mm256_castsi256_si128(_b);
                        lanes[1] = _mm256_extracti128_si256(_b, 1);
                        for (l=0;l<2;l++) {
                                for (k=0;k<16;k++) {
                                        dups = _mm256_broadcastb_epi8(lanes[l]);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                        accu ^= gf256_mult_vectorized_avx2(_a,dups);
                                        row = row + complete_line;
                                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1); // ideia do chat
                                }
                        }
		}

		_b = _mm256_lddqu_si256((__m256i*)&X[32*j]);
                lanes[0] = _mm256_castsi256_si128(_b);
                lanes[1] = _mm256_extracti128_si256(_b, 1);
		_a = _mm256_lddqu_si256((__m256i*)&even_diagonal[32*i]);
		accu ^= gf256_mult_vectorized_avx2(_a, _b);
                for (l=0;l<2;l++) {
                        for (k=0;k<16;k++) {
                                mask = _mm256_alignr_epi8(mask,zeros,15);
                                dups = _mm256_broadcastb_epi8(lanes[l]);
                                _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                prod = gf256_mult_vectorized_avx2(_a,dups);
                                accu ^= (mask & prod);
                                acc_inf ^= (_mm256_xor_si256(mask, ones) & prod);
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1); // ideia do chat
                        }
                        zeros = _mm256_setzero_si256();
                }
                _mm256_storeu_si256((void*)&Y_sup[i*32], accu);

		for (j=i+1;j<batches;j++) {
			_b = _mm256_lddqu_si256((__m256i*)&X[32*j]);
                        lanes[0] = _mm256_castsi256_si128(_b);
                        lanes[1] = _mm256_extracti128_si256(_b, 1);
                        for (l=0;l<2;l++) {
                                for (k=0;k<16;k++) {
                                        dups = _mm256_broadcastb_epi8(lanes[l]);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                        acc_inf ^= gf256_mult_vectorized_avx2(_a,dups);
                                        row = row + complete_line;
                                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1); // ideia do chat
                                }
                        }
		}
		if (remainder > 0) {
		_b = load_incomplete_m256(&X[32*j],remainder);
                lanes[0] = _mm256_castsi256_si128(_b);
                lanes[1] = _mm256_extracti128_si256(_b, 1);
                for (l=0;l<2;l++) {
                        for (k=0;k<16 && k < remainder-(l*16);k++) {
                                dups = _mm256_broadcastb_epi8(lanes[l]);
                                _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                _a = gf256_mult_vectorized_avx2(_a,dups);
                                acc_inf ^= _a;
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1); // ideia do chat
                        }
                }
		}
                _mm256_storeu_si256((void*)&Y_inf[i*32], acc_inf);
	}

	// Process last block, not my falt -\o/-
	if (remainder > 0) {
	acc_inf = _mm256_setzero_si256();
	accu = _mm256_setzero_si256();
	row = batches<<5;
	for (j=0;j<batches;j++) {
                _b = _mm256_lddqu_si256((__m256i*)&X[32*j]);
		lanes[0] = _mm256_castsi256_si128(_b);
                lanes[1] = _mm256_extracti128_si256(_b, 1);
                for (l=0;l<2;l++) {
                        for (k=0;k<16;k++) {
                                dups = _mm256_broadcastb_epi8(lanes[l]);
                                _a = load_incomplete_m256(&A[row],remainder);
                                accu ^= gf256_mult_vectorized_avx2(_a,dups);
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1); // ideia do chat
                        }
                }
	}

	
        mask = _mm256_set1_epi32(-1);
        zeros = _mm256_setr_epi64x(0,0,-1,-1);
	_b = load_incomplete_m256(&X[32*j],remainder);
	lanes[0] = _mm256_castsi256_si128(_b);
        lanes[1] = _mm256_extracti128_si256(_b, 1);
        _a = load_incomplete_m256(&even_diagonal[32*batches],remainder);
	accu ^= gf256_mult_vectorized_avx2(_a, _b);
	for (l=0;l<2;l++) {
                for (k=0;k<16 && k < remainder-(l*16);k++) {
                        mask = _mm256_alignr_epi8(mask,zeros,15);
                        dups = _mm256_broadcastb_epi8(lanes[l]);
                        _a = load_incomplete_m256(&A[row],remainder);
                        prod = gf256_mult_vectorized_avx2(_a,dups);
                        accu ^= (mask & prod);
                        acc_inf ^= (_mm256_xor_si256(mask, ones) & prod);
                        row = row + complete_line;
                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1); // ideia do chat
                }
                zeros = _mm256_setzero_si256();
        }
	store_incomplete_m256(accu,&Y_sup[batches*32],remainder);
	store_incomplete_m256(acc_inf,&Y_inf[batches*32],remainder);
	// End last block
	}
}

static inline void gf256_mat_mult_single_avx2(const uint8_t *A, const uint8_t *X, uint8_t *Y, uint32_t n)
{
    uint16_t i, j, k, l;
        uint16_t complete_line = (n + 7) & ~7;
	uint16_t batches, remainder;
	uint32_t row; //, ind;
	__m256i _a, dups, _b;
	__m256i acc,prod;

	__m256i zeros, mask;
        __m256i ones = _mm256_set1_epi64x(-1);
        __m128i lanes[2];

	/* Set the accumulator to 0 */
	batches = n>>5;
	remainder = n & 31;
	for (i=0;i<batches;i++) {
	        zeros = _mm256_setr_epi64x(0,0,-1,-1);
		acc = _mm256_setzero_si256();
		mask = _mm256_set1_epi32(-1);
		row = ((complete_line+1)*i)<<5;

		_b = _mm256_lddqu_si256((__m256i*)&X[32*i]);
                lanes[0] = _mm256_castsi256_si128(_b);
                lanes[1] = _mm256_extracti128_si256(_b, 1);
                for (l=0;l<2;l++) {
                        for (k=0;k<16;k++) {
                                mask = _mm256_alignr_epi8(mask,zeros,15);
                                dups = _mm256_broadcastb_epi8(lanes[l]);
                                _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                prod = gf256_mult_vectorized_avx2(_a,dups);
                                acc ^= (_mm256_xor_si256(mask, ones) & prod);
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1); // ideia do chat
                        }
                        zeros = _mm256_setzero_si256();
                }

		for (j=i+1;j<batches;j++) {
			_b = _mm256_lddqu_si256((__m256i*)&X[32*j]);
                        lanes[0] = _mm256_castsi256_si128(_b);
                        lanes[1] = _mm256_extracti128_si256(_b, 1);
                        for (l=0;l<2;l++) {
                                for (k=0;k<16;k++) {
                                        dups = _mm256_broadcastb_epi8(lanes[l]);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                        acc ^= gf256_mult_vectorized_avx2(_a,dups);
                                        row = row + complete_line;
                                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1); // ideia do chat
                                }
                        }
		}
		if (remainder > 0) {
		_b = load_incomplete_m256(&X[32*j],remainder);
                lanes[0] = _mm256_castsi256_si128(_b);
                lanes[1] = _mm256_extracti128_si256(_b, 1);
                for (l=0;l<2;l++) {
                        for (k=0;k<16 && k < remainder-(l*16);k++) {
                                dups = _mm256_broadcastb_epi8(lanes[l]);
                                _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                _a = gf256_mult_vectorized_avx2(_a,dups);
                                acc ^= _a;
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1); // ideia do chat
                        }
                }
		}
                _mm256_storeu_si256((void*)&Y[i*32], acc);
	}

	// Process last block, not my falt -\o/-
	if (remainder > 0) {
	acc= _mm256_setzero_si256();
	row = ((complete_line+1)*batches)<<5;
	
        mask = _mm256_set1_epi32(-1);
        zeros = _mm256_setr_epi64x(0,0,-1,-1);
	_b = load_incomplete_m256(&X[32*j],remainder);
	lanes[0] = _mm256_castsi256_si128(_b);
        lanes[1] = _mm256_extracti128_si256(_b, 1);
	for (l=0;l<2;l++) {
                for (k=0;k<16 && k < remainder-(l*16);k++) {
                        mask = _mm256_alignr_epi8(mask,zeros,15);
                        dups = _mm256_broadcastb_epi8(lanes[l]);
                        _a = load_incomplete_m256(&A[row],remainder);
                        prod = gf256_mult_vectorized_avx2(_a,dups);
                        acc ^= (_mm256_xor_si256(mask, ones) & prod);
                        row = row + complete_line;
                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1); // ideia do chat
                }
                zeros = _mm256_setzero_si256();
        }
	store_incomplete_m256(acc,&Y[batches*32],remainder);
	// End last block
	}
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
static inline void gf256_gf2_mat_mult_avx2(const uint8_t *A, const uint8_t *X, uint8_t *Y_inf, uint8_t *Y_sup, const uint8_t * even_diagonal,uint32_t n)
{
        /* XXX: NOTE: because of alignment and loading latencies, treating the matrix as "regular" seems always better
         * than using the trianglar shape ... */
        uint16_t i, j, k, l;
        uint16_t complete_line = (n + 7) & ~7;
        uint16_t batches, remainder;
        uint32_t row;
        __m256i accu, _a, dups;
        __m256i acc_inf, mask, prod;
        uint32_t _b;

        /* Create a selection mask from the bits in _a */
        const __m256i shuff_msk = _mm256_set_epi8(3, 3, 3,  3,  3, 3, 3, 3, 2, 2, 2,  2,  2, 2, 2, 2,
                                                        1, 1, 1,  1,  1, 1, 1, 1, 0, 0, 0,  0,  0, 0, 0, 0);
        const __m256i and_msk = _mm256_set_epi8(0b10000000, 0b01000000, 0b00100000, 0b00010000, 0b00001000, 0b00000100, 0b00000010, 0b00000001,
                                                0b10000000, 0b01000000, 0b00100000, 0b00010000, 0b00001000, 0b00000100, 0b00000010, 0b00000001,
                                                0b10000000, 0b01000000, 0b00100000, 0b00010000, 0b00001000, 0b00000100, 0b00000010, 0b00000001,
                                                0b10000000, 0b01000000, 0b00100000, 0b00010000, 0b00001000, 0b00000100, 0b00000010, 0b00000001);
	
        __m256i zeros;
        __m256i ones = _mm256_set1_epi64x(-1);

        batches = n>>5;
        remainder = n & 31;
        for (i=0;i<batches;i++) {
	        zeros = _mm256_setr_epi64x(0,0,-1,-1);
                mask = _mm256_set1_epi32(-1);
                accu = _mm256_setzero_si256();
                acc_inf = _mm256_setzero_si256();
                row = i<<5;
                for (j=0;j<i;j++) {
                        memcpy(&_b, X + 4*j, sizeof(_b));
                        for (k=0;k<32;k++) {
                                dups = _mm256_set1_epi8(-(_b&1));
                                _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                accu ^= (_a & dups);
                                row = row + complete_line;
                                _b = _b >> 1;
                        }
                }

                _a = load_incomplete_m256(&X[4*j], 4);
                prod = _mm256_lddqu_si256((__m256i*)&even_diagonal[32*i]);
                /* Copy in the two lanes */
		_a = _mm256_permute4x64_epi64(_a, 0b01000100);
		/* Only keep the selection bits */
		_a = _mm256_shuffle_epi8(_a, shuff_msk) & and_msk;
		/* Transform these bits to either 0 or 0xFF */
		_a = _mm256_cmpeq_epi8(_a, and_msk);
                accu ^= (_a & prod);

                memcpy(&_b, X + 4*j, sizeof(_b));
                for (l=0;l<2;l++) {
                        for (k=0;k<16;k++) {
                                mask = _mm256_alignr_epi8(mask,zeros,15);
                                dups = _mm256_set1_epi8(-(_b&1));
                                _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                prod = (_a & dups);
                                accu ^= (mask & prod);
                                acc_inf ^= (_mm256_xor_si256(mask, ones) & prod);
                                row = row + complete_line;
                                _b = _b >> 1;
                        }
                        zeros = _mm256_setzero_si256();
                }
                _mm256_storeu_si256((void*)&Y_sup[i*32], accu);
                

                for (j=i+1;j<batches;j++) {
                        memcpy(&_b, X + 4*j, sizeof(_b));
                        for (k=0;k<32;k++) {
                                dups = _mm256_set1_epi8(-(_b&1));
                                _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                acc_inf ^= (_a & dups);
                                row = row + complete_line;
                                _b = _b >> 1;
                        }
                }
                if (remainder > 0) {
                _b = _b | X[4*j+1];
                _b = _b << 8;
                _b = _b | X[4*j];
                for (k=0; k < remainder;k++) {
                        dups = _mm256_set1_epi8(-(_b&1));
                        _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                        acc_inf ^= (_a & dups);
                        row = row + complete_line;
                        _b = _b >> 1;
                }
		}
                _mm256_storeu_si256((void*)&Y_inf[i*32], acc_inf);
        }

        // Process last block, not my falt -\o/-
	if (remainder > 0) {
	acc_inf = _mm256_setzero_si256();
	accu = _mm256_setzero_si256();
	row = batches<<5;
	for (j=0;j<batches;j++) {
                memcpy(&_b, X + 4*j, sizeof(_b));
                for (k=0;k<32;k++) {
                        dups = _mm256_set1_epi8(-(_b&1));
                        _a = load_incomplete_m256(&A[row],remainder);
                        accu ^= (_a & dups);
                        row = row + complete_line;
                        _b = _b >> 1;
                }
	}

	
        mask = _mm256_set1_epi32(-1);
        zeros = _mm256_setr_epi64x(0,0,-1,-1);
        _b = _b | X[4*j+1];
        _b = _b << 8;
        _b = _b | X[4*j];

        _a = load_incomplete_m256(&X[4*j], 2);
        prod = _mm256_lddqu_si256((__m256i*)&even_diagonal[32*i]);
        /* Copy in the two lanes */
        _a = _mm256_permute4x64_epi64(_a, 0b01000100);
        /* Only keep the selection bits */
        _a = _mm256_shuffle_epi8(_a, shuff_msk) & and_msk;
        /* Transform these bits to either 0 or 0xFF */
        _a = _mm256_cmpeq_epi8(_a, and_msk);
        accu ^= (_a & prod);

        for (k=0; k < remainder;k++) {
                mask = _mm256_alignr_epi8(mask,zeros,15);
                dups = _mm256_set1_epi8(-(_b&1));
                _a = load_incomplete_m256(&A[row],remainder);
                prod = (_a & dups);
                accu ^= (mask & prod);
                acc_inf ^= (_mm256_xor_si256(mask, ones) & prod);
                row = row + complete_line;
                _b = _b >> 1;
        }
	store_incomplete_m256(accu,&Y_sup[batches*32],remainder);
	store_incomplete_m256(acc_inf,&Y_inf[batches*32],remainder);
	// End last block
	}	
}

static inline void gf256_gf2_mat_mult_single_avx2(const uint8_t *A, const uint8_t *X, uint8_t *Y, uint32_t n)
{
        /* XXX: NOTE: because of alignment and loading latencies, treating the matrix as "regular" seems always better
         * than using the trianglar shape ... */
        uint16_t i, j, k, l;
        uint16_t complete_line;
        uint16_t batches, remainder;
        uint32_t row;
        __m256i _a, dups;
        __m256i acc, mask, prod;
        uint32_t _b;


	__m256i zeros;
        __m256i ones = _mm256_set1_epi64x(-1);

        batches = n>>5;
        complete_line = (n + 7) & ~7;

        for (i=0;i<batches;i++) {
	        zeros = _mm256_setr_epi64x(0,0,-1,-1);
                mask = _mm256_set1_epi32(-1);
                acc = _mm256_setzero_si256();
                row = ((complete_line+1)*i)<<5;

                memcpy(&_b, X + 4*j, sizeof(_b));
                for (l=0;l<2;l++) {
                        for (k=0;k<16;k++) {
                                mask = _mm256_alignr_epi8(mask,zeros,15);
                                dups = _mm256_set1_epi8(-(_b&1));
                                _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                prod = (_a & dups);
                                acc ^= (_mm256_xor_si256(mask, ones) & prod);
                                row = row + complete_line;
                                _b = _b >> 1;
                        }
                        zeros = _mm256_setzero_si256();
                }
                

                for (j=i+1;j<batches;j++) {
                        memcpy(&_b, X + 4*j, sizeof(_b));
                        for (k=0;k<32;k++) {
                                dups = _mm256_set1_epi8(-(_b&1));
                                _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                acc ^= (_a & dups);
                                row = row + complete_line;
                                _b = _b >> 1;
                        }
                }
                if (remainder > 0) {
                _b = _b | X[4*j+1];
                _b = _b << 8;
                _b = _b | X[4*j];
                for (k=0; k < remainder;k++) {
                        dups = _mm256_set1_epi8(-(_b&1));
                        _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                        acc ^= (_a & dups);
                        row = row + complete_line;
                        _b = _b >> 1;
                }
		}
                _mm256_storeu_si256((void*)&Y[i*32], acc);
        }

	// Process last block, not my falt -\o/-
	if (remainder > 0) {
	acc = _mm256_setzero_si256();
	row = batches<<5;
	
        mask = _mm256_set1_epi32(-1);
        zeros = _mm256_setr_epi64x(0,0,-1,-1);
        _b = _b | X[4*j+1];
        _b = _b << 8;
        _b = _b | X[4*j];

        for (k=0; k < remainder;k++) {
                mask = _mm256_alignr_epi8(mask,zeros,15);
                dups = _mm256_set1_epi8(-(_b&1));
                _a = load_incomplete_m256(&A[row],remainder);
                prod = (_a & dups);
                acc ^= (_mm256_xor_si256(mask, ones) & prod);
                row = row + complete_line;
                _b = _b >> 1;
        }
	store_incomplete_m256(acc,&Y[batches*32],remainder);
	// End last block
	}
}

static inline void gf256_transpose_block_avx2(uint8_t *src, uint8_t *dst, uint16_t size_in, uint16_t size_out)
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

        __m256i accu[4], folding_element;
        __m256i folding0[4];
        __m256i current_seed;

        /* Set the accumulator to 0 */
        for (i=0; i < 4; i++) {
                accu[i] = _mm256_setzero_si256();
                folding0[i] = _mm256_setzero_si256();
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
                                folding0[pos] = _mm256_xor_si256(folding0[pos],accu[pos]);
                        }
                        else{
                                /* Obvious 256-bit */
                                current_seed = _mm256_lddqu_si256((__m256i*)&lseed[j*row_size+i]);
                                accu[pos] = _mm256_xor_si256(current_seed,accu[pos]);
                                folding0[pos] = _mm256_xor_si256(folding0[pos],accu[pos]);
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
                        store_incomplete_m256(folding0[pos], &data_folding[i], (bytes_to_add-i));
                }
                else {
                        _mm256_storeu_si256((void*)&acc[i], accu[pos]);
                        _mm256_storeu_si256((void*)&data_folding[i], folding0[pos]);
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
static inline void gf256_gf16_mat_mult_avx2(const uint8_t *A, const uint8_t *X, uint8_t *Y_inf, uint8_t *Y_sup, const uint8_t * even_diagonal, uint32_t n)
{
    uint16_t i, j, k, l;
        uint16_t complete_line = (n + 7) & ~7;
	uint16_t batches, remainder;
	uint32_t row; //, ind;
	__m256i accu, _a, dups, _b;
	__m256i acc_inf,prod;
	__m128i aux;

	__m256i zeros, mask;
        __m256i ones = _mm256_set1_epi64x(-1);
        __m128i lanes[2];


	/* Set the accumulator to 0 */
	batches = n>>5;
	remainder = n & 31;
	for (i=0;i<batches;i++) {
	        zeros = _mm256_setr_epi64x(0,0,-1,-1);
		acc_inf = _mm256_setzero_si256();
		accu = _mm256_setzero_si256();
		mask = _mm256_set1_epi32(-1);
		row = i<<5;
		for (j=0;j<i;j++) {
			aux = _mm_load_si128((__m128i*)&X[16*j]);
			_b = _mm256_broadcastsi128_si256(aux);
			_b = gf256_lift32_from_gf16_avx2(_b);
                        lanes[0] = _mm256_castsi256_si128(_b);
                        lanes[1] = _mm256_extracti128_si256(_b, 1);
                        for (l=0;l<2;l++) {
                                for (k=0;k<16;k++) {
                                        dups=_mm256_broadcastb_epi8(lanes[l]);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                        accu ^= gf256_mult_vectorized_avx2(_a,dups);
                                        row = row + complete_line;
                                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1);
                                }
                        }
		}

		aux = _mm_load_si128((__m128i*)&X[16*j]);
		_b = _mm256_broadcastsi128_si256(aux);
		_b = gf256_lift32_from_gf16_avx2(_b);
                lanes[0] = _mm256_castsi256_si128(_b);
                lanes[1] = _mm256_extracti128_si256(_b, 1);
		_a = _mm256_lddqu_si256((__m256i*)&even_diagonal[32*i]);
		accu ^= gf256_mult_vectorized_avx2(_a, _b);
                for (l=0;l<2;l++) {
                        for (k=0;k<16;k++) {
                                mask = _mm256_alignr_epi8(mask,zeros,15);
                                dups = _mm256_broadcastb_epi8(lanes[l]);
                                _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                prod = gf256_mult_vectorized_avx2(_a,dups);
                                accu ^= (mask & prod);
                                acc_inf ^= (_mm256_xor_si256(mask, ones) & prod);
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1);
                        }
                        zeros = _mm256_setzero_si256();
                }
		_mm256_storeu_si256((void*)&Y_sup[i*32], accu);

		for (j=i+1;j<batches;j++) {
			aux = _mm_load_si128((__m128i*)&X[16*j]);
			_b = _mm256_broadcastsi128_si256(aux);
                        _b = gf256_lift32_from_gf16_avx2(_b);
                        lanes[0] = _mm256_castsi256_si128(_b);
                        lanes[1] = _mm256_extracti128_si256(_b, 1);
                        for (l=0;l<2;l++) {
                                for (k=0;k<16;k++) {
                                        dups = _mm256_broadcastb_epi8(lanes[l]);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                        acc_inf ^= gf256_mult_vectorized_avx2(_a,dups);
                                        row = row + complete_line;
                                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1);
                                }
                        }
		}
		_b = load_incomplete_m256(&X[16*j],remainder/2);
		_b = gf256_lift32_from_gf16_avx2(_b);
                lanes[0] = _mm256_castsi256_si128(_b);
                lanes[1] = _mm256_extracti128_si256(_b, 1);
                for (l=0;l<2;l++) {
                        for (k=0;k<16 && k < remainder-(l*16);k++) {
                                dups = _mm256_broadcastb_epi8(lanes[l]);
                                _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                _a = gf256_mult_vectorized_avx2(_a,dups);
                                acc_inf ^= _a;
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1); // ideia do chat
                        }
                }
		_mm256_storeu_si256((void*)&Y_inf[i*32], acc_inf);
	}

	// Process last block, not my falt -\o/-
	acc_inf = _mm256_setzero_si256();
	accu = _mm256_setzero_si256();
	row = batches<<5;
	for (j=0;j<batches;j++) {
		aux = _mm_load_si128((__m128i*)&X[16*j]);
                _b = _mm256_broadcastsi128_si256(aux);
                _b = gf256_lift32_from_gf16_avx2(_b);
                lanes[0] = _mm256_castsi256_si128(_b);
                lanes[1] = _mm256_extracti128_si256(_b, 1);
		for (l=0;l<2;l++) {
                        for (k=0;k<16;k++) {
                                dups = _mm256_broadcastb_epi8(lanes[l]);
                                _a = load_incomplete_m256(&A[row],remainder);
                                accu ^= gf256_mult_vectorized_avx2(_a,dups);
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1); // ideia do chat
                        }
                }
	}

	mask = _mm256_set1_epi32(-1);
	zeros = _mm256_setr_epi64x(0,0,-1,-1);
	_b = load_incomplete_m256(&X[16*j],remainder/2);
	_b = gf256_lift32_from_gf16_avx2(_b);
        lanes[0] = _mm256_castsi256_si128(_b);
        lanes[1] = _mm256_extracti128_si256(_b, 1);
        _a = load_incomplete_m256(&even_diagonal[32*batches],remainder);
	accu ^= gf256_mult_vectorized_avx2(_a, _b);
	for (l=0;l<2;l++) {
                for (k=0;k<16 && k < remainder-(l*16);k++) {
                        mask = _mm256_alignr_epi8(mask,zeros,15);
                        dups = _mm256_broadcastb_epi8(lanes[l]);
                        _a = load_incomplete_m256(&A[row],remainder);
                        prod = gf256_mult_vectorized_avx2(_a,dups);
                        accu ^= (mask & prod);
                        acc_inf ^= (_mm256_xor_si256(mask, ones) & prod);
                        row = row + complete_line;
                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1); // ideia do chat
                }
                zeros = _mm256_setzero_si256();
        }
	store_incomplete_m256(accu,&Y_sup[batches*32],remainder);
	store_incomplete_m256(acc_inf,&Y_inf[batches*32],remainder);
	// End last block
}

static inline void gf256_gf16_mat_mult_single_avx2(const uint8_t *A, const uint8_t *X, uint8_t *Y, uint32_t n)
{
    uint16_t i, j, k, l;
        uint16_t complete_line = (n + 7) & ~7;
	uint16_t batches, remainder;
	uint32_t row; //, ind;
	__m256i _a, dups, _b;
	__m256i acc,prod;
	__m128i aux;

	__m256i zeros, mask;
        __m256i ones = _mm256_set1_epi64x(-1);
        __m128i lanes[2];


	/* Set the accumulator to 0 */
	batches = n>>5;
	remainder = n & 31;
	for (i=0;i<batches;i++) {
	        zeros = _mm256_setr_epi64x(0,0,-1,-1);
		acc = _mm256_setzero_si256();
		mask = _mm256_set1_epi32(-1);
		row = ((complete_line+1)*i)<<5;

		aux = _mm_load_si128((__m128i*)&X[16*i]);
		_b = _mm256_broadcastsi128_si256(aux);
		_b = gf256_lift32_from_gf16_avx2(_b);
                lanes[0] = _mm256_castsi256_si128(_b);
                lanes[1] = _mm256_extracti128_si256(_b, 1);
                for (l=0;l<2;l++) {
                        for (k=0;k<16;k++) {
                                mask = _mm256_alignr_epi8(mask,zeros,15);
                                dups = _mm256_broadcastb_epi8(lanes[l]);
                                _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                prod = gf256_mult_vectorized_avx2(_a,dups);
                                acc ^= (_mm256_xor_si256(mask, ones) & prod);
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1);
                        }
                        zeros = _mm256_setzero_si256();
                }

		for (j=i+1;j<batches;j++) {
			aux = _mm_load_si128((__m128i*)&X[16*j]);
			_b = _mm256_broadcastsi128_si256(aux);
                        _b = gf256_lift32_from_gf16_avx2(_b);
                        lanes[0] = _mm256_castsi256_si128(_b);
                        lanes[1] = _mm256_extracti128_si256(_b, 1);
                        for (l=0;l<2;l++) {
                                for (k=0;k<16;k++) {
                                        dups = _mm256_broadcastb_epi8(lanes[l]);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                        acc ^= gf256_mult_vectorized_avx2(_a,dups);
                                        row = row + complete_line;
                                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1);
                                }
                        }
		}
		_b = load_incomplete_m256(&X[16*j],remainder/2);
		_b = gf256_lift32_from_gf16_avx2(_b);
                lanes[0] = _mm256_castsi256_si128(_b);
                lanes[1] = _mm256_extracti128_si256(_b, 1);
                for (l=0;l<2;l++) {
                        for (k=0;k<16 && k < remainder-(l<<4);k++) {
                                dups = _mm256_broadcastb_epi8(lanes[l]);
                                _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                _a = gf256_mult_vectorized_avx2(_a,dups);
                                acc ^= _a;
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1); // ideia do chat
                        }
                }
		_mm256_storeu_si256((void*)&Y[i*32], acc);
	}

	// Process last block, not my falt -\o/-
	acc = _mm256_setzero_si256();
	row = ((complete_line+1)*batches)<<5;

	mask = _mm256_set1_epi32(-1);
	zeros = _mm256_setr_epi64x(0,0,-1,-1);
	_b = load_incomplete_m256(&X[16*j],remainder/2);
	_b = gf256_lift32_from_gf16_avx2(_b);
        lanes[0] = _mm256_castsi256_si128(_b);
        lanes[1] = _mm256_extracti128_si256(_b, 1);
	for (l=0;l<2;l++) {
                for (k=0;k<16 && k < remainder-(l<<4);k++) {
                        mask = _mm256_alignr_epi8(mask,zeros,15);
                        dups = _mm256_broadcastb_epi8(lanes[l]);
                        _a = load_incomplete_m256(&A[row],remainder);
                        prod = gf256_mult_vectorized_avx2(_a,dups);
                        acc ^= (_mm256_xor_si256(mask, ones) & prod);
                        row = row + complete_line;
                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 1); // ideia do chat
                }
                zeros = _mm256_setzero_si256();
        }
	store_incomplete_m256(acc,&Y[batches*32],remainder);
	// End last block
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
static inline void gf256to2_mat_mult_avx2(const uint16_t *A, const uint16_t *X, uint16_t *Y_inf, uint16_t *Y_sup, const uint16_t *even_diagonal, uint32_t n)
{
    uint16_t i, j, k, l;
        uint16_t complete_line = (n + 7) & ~7;
	uint16_t batches, remainder;
	uint32_t row; //, ind;
	__m256i accu, _a, dups, _b;
	__m256i acc_inf,prod;

	__m256i zeros, mask;
        __m256i ones = _mm256_set1_epi64x(-1);
        __m128i lanes[2];

	/* Set the accumulator to 0 */
	batches = n>>4;
	remainder = n & 15;
	for (i=0;i<batches;i++) {
	        zeros = _mm256_setr_epi64x(0,0,-1,-1);
		acc_inf = _mm256_setzero_si256();
		accu = _mm256_setzero_si256();
		mask = _mm256_set1_epi32(-1);
		row = i<<4;
		for (j=0;j<i;j++) {
			_b = _mm256_lddqu_si256((__m256i*)&X[16*j]);
                        lanes[0] = _mm256_castsi256_si128(_b);
                        lanes[1] = _mm256_extracti128_si256(_b, 1);
                        for (l=0;l<2;l++) {
                                for (k=0;k<8;k++) {
                                        dups = _mm256_broadcastw_epi16(lanes[l]);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                        accu ^= gf256to2_mult_vectorized_avx2(_a,dups);
                                        row = row + complete_line;
                                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 2); // ideia do chat
                                }
                        }
		}

		_b = _mm256_lddqu_si256((__m256i*)&X[16*j]);
                lanes[0] = _mm256_castsi256_si128(_b);
                lanes[1] = _mm256_extracti128_si256(_b, 1);
		_a = _mm256_lddqu_si256((__m256i*)&even_diagonal[16*i]);
		accu ^= gf256to2_mult_vectorized_avx2(_a,_b);
                for (l=0;l<2;l++) {
                        for (k=0;k<8;k++) {
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                                dups = _mm256_broadcastw_epi16(lanes[l]);
                                _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                prod = gf256to2_mult_vectorized_avx2(_a,dups);
                                accu ^= (mask & prod);
                                acc_inf ^= (_mm256_xor_si256(mask, ones) & prod);
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 2); // ideia do chat
                        }
                        zeros = _mm256_setzero_si256();
                }
		_mm256_storeu_si256((void*)&Y_sup[i*16], accu);

		for (j=i+1;j<batches;j++) {
			_b = _mm256_lddqu_si256((__m256i*)&X[16*j]);
                        lanes[0] = _mm256_castsi256_si128(_b);
                        lanes[1] = _mm256_extracti128_si256(_b, 1);
                        for (l=0;l<2;l++) {
                                for (k=0;k<8;k++) {
                                        dups = _mm256_broadcastw_epi16(lanes[l]);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                        acc_inf ^= gf256to2_mult_vectorized_avx2(_a,dups);
                                        row = row + complete_line;
                                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 2); // ideia do chat
                                }
                        }
		}
		if (remainder > 0) {
		_b = load_incomplete_m256((uint8_t *) &X[16*j],2*remainder);
                lanes[0] = _mm256_castsi256_si128(_b);
                lanes[1] = _mm256_extracti128_si256(_b, 1);
                for (l=0;l<2;l++) {
                        for (k=0;k<8 && k<remainder -(l<<3);k++) {
                                dups = _mm256_broadcastw_epi16(lanes[l]);
                                _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                _a = gf256to2_mult_vectorized_avx2(_a,dups);
                                acc_inf ^= _a;
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 2);
                        }
                }
		}
                _mm256_storeu_si256((void*)&Y_inf[i*16],acc_inf);
	}

	// Process last block, not my falt -\o/-
	if (remainder > 0) {
	acc_inf = _mm256_setzero_si256();
	accu = _mm256_setzero_si256();
	row = batches<<4;
	for (j=0;j<batches;j++) {
		_b = _mm256_lddqu_si256((__m256i*)&X[16*j]);
                lanes[0] = _mm256_castsi256_si128(_b);
                lanes[1] = _mm256_extracti128_si256(_b, 1);
                for (l=0;l<2;l++) {
                        for (k=0;k<8;k++) {
                                dups = _mm256_broadcastw_epi16(lanes[l]);
                                _a = load_incomplete_m256((uint8_t *)&A[row],2*remainder);
                                _a = gf256to2_mult_vectorized_avx2(_a,dups);
                                accu ^= _a;
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 2);
                        }
                }
	}

	mask = _mm256_set1_epi32(-1);
	zeros = _mm256_setr_epi64x(0,0,-1,-1);
	_b = load_incomplete_m256((uint8_t *)&X[16*j],2*remainder);
        lanes[0] = _mm256_castsi256_si128(_b);
        lanes[1] = _mm256_extracti128_si256(_b, 1);
	_a = load_incomplete_m256((uint8_t *)&even_diagonal[16*batches],2*remainder);
	accu ^= gf256to2_mult_vectorized_avx2(_a,_b);
        for (l=0;l<2;l++) {
                for (k=0;k<8 && k<remainder-(l<<3);k++) {
                        mask = _mm256_alignr_epi8(mask,zeros,14);
                        dups = _mm256_broadcastw_epi16(lanes[l]);
                        _a = load_incomplete_m256((uint8_t *)&A[row],2*remainder);
                        prod = gf256to2_mult_vectorized_avx2(_a,dups);
                        accu ^= (mask & prod);
                        acc_inf ^= (_mm256_xor_si256(mask, ones) & prod);
                        row = row + complete_line;
                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 2);
                }
                zeros = _mm256_setzero_si256();
        }
	store_incomplete_m256(accu,(uint8_t *)&Y_sup[batches*16],2*remainder);
	store_incomplete_m256(acc_inf,(uint8_t *)&Y_inf[batches*16],2*remainder);
	// End last block
	}
}

static inline void gf256to2_mat_mult_single_avx2(const uint16_t *A, const uint16_t *X, uint16_t *Y, uint32_t n)
{
    uint16_t i, j, k, l;
        uint16_t complete_line = (n + 7) & ~7;
	uint16_t batches, remainder;
	uint32_t row; //, ind;
	__m256i _a, dups, _b;
	__m256i acc,prod;

	__m256i zeros, mask;
        __m256i ones = _mm256_set1_epi64x(-1);
        __m128i lanes[2];

	/* Set the accumulator to 0 */
	batches = n>>4;
	remainder = n & 15;
	for (i=0;i<batches;i++) {
	        zeros = _mm256_setr_epi64x(0,0,-1,-1);
		acc = _mm256_setzero_si256();
		mask = _mm256_set1_epi32(-1);
		row = ((complete_line+1)*i)<<4;

		_b = _mm256_lddqu_si256((__m256i*)&X[16*i]);
                lanes[0] = _mm256_castsi256_si128(_b);
                lanes[1] = _mm256_extracti128_si256(_b, 1);
                for (l=0;l<2;l++) {
                        for (k=0;k<8;k++) {
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                                dups = _mm256_broadcastw_epi16(lanes[l]);
                                _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                prod = gf256to2_mult_vectorized_avx2(_a,dups);
                                acc ^= (_mm256_xor_si256(mask, ones) & prod);
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 2); // ideia do chat
                        }
                        zeros = _mm256_setzero_si256();
                }

		for (j=i+1;j<batches;j++) {
			_b = _mm256_lddqu_si256((__m256i*)&X[16*j]);
                        lanes[0] = _mm256_castsi256_si128(_b);
                        lanes[1] = _mm256_extracti128_si256(_b, 1);
                        for (l=0;l<2;l++) {
                                for (k=0;k<8;k++) {
                                        dups = _mm256_broadcastw_epi16(lanes[l]);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                        acc ^= gf256to2_mult_vectorized_avx2(_a,dups);
                                        row = row + complete_line;
                                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 2); // ideia do chat
                                }
                        }
		}
		if (remainder > 0) {
		_b = load_incomplete_m256((uint8_t *) &X[16*j],2*remainder);
                lanes[0] = _mm256_castsi256_si128(_b);
                lanes[1] = _mm256_extracti128_si256(_b, 1);
                for (l=0;l<2;l++) {
                        for (k=0;k<8 && k<remainder -(l<<3);k++) {
                                dups = _mm256_broadcastw_epi16(lanes[l]);
                                _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                _a = gf256to2_mult_vectorized_avx2(_a,dups);
                                acc ^= _a;
                                row = row + complete_line;
                                lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 2);
                        }
                }
		}
                _mm256_storeu_si256((void*)&Y[i*16],acc);
	}

	// Process last block, not my falt -\o/-
	if (remainder > 0) {
	acc = _mm256_setzero_si256();
	row = ((complete_line+1)*batches)<<4;

	mask = _mm256_set1_epi32(-1);
	zeros = _mm256_setr_epi64x(0,0,-1,-1);
	_b = load_incomplete_m256((uint8_t *)&X[16*j],2*remainder);
        lanes[0] = _mm256_castsi256_si128(_b);
        lanes[1] = _mm256_extracti128_si256(_b, 1);
        for (l=0;l<2;l++) {
                for (k=0;k<8 && k<remainder-(l<<3);k++) {
                        mask = _mm256_alignr_epi8(mask,zeros,14);
                        dups = _mm256_broadcastw_epi16(lanes[l]);
                        _a = load_incomplete_m256((uint8_t *)&A[row],2*remainder);
                        prod = gf256to2_mult_vectorized_avx2(_a,dups);
                        acc ^= (_mm256_xor_si256(mask, ones) & prod);
                        row = row + complete_line;
                        lanes[l] = _mm_alignr_epi8(lanes[l], lanes[l], 2);
                }
                zeros = _mm256_setzero_si256();
        }
	store_incomplete_m256(acc,(uint8_t *)&Y[batches*16],2*remainder);
	// End last block
	}
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
static inline void gf256to2_gf2_mat_mult_avx2(const uint16_t *A, const uint8_t *X, uint16_t *Y_inf, uint16_t *Y_sup, const uint16_t *even_diagonal, uint32_t n)
{
        /* XXX: NOTE: because of alignment and loading latencies, treating the matrix as "regular" seems always better
         * than using the trianglar shape ... */
        uint16_t i, j, k, l;
                uint16_t complete_line = (n + 7) & ~7;
		uint16_t batches;
		uint32_t row;
		__m256i accu, _a, dups;
		__m256i acc_inf, mask, zeros, prod;
		const __m256i shuff_msk = _mm256_set_epi8(1, 1, 1,  1,  1, 1, 1, 1, 1, 1, 1,  1,  1, 1, 1, 1,
                                                          0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0);
		const __m256i and_msk = _mm256_set_epi8(0b10000000, 0b10000000, 0b01000000, 0b01000000, 0b00100000, 0b00100000, 0b00010000, 0b00010000,
							0b00001000, 0b00001000, 0b00000100, 0b00000100, 0b00000010, 0b00000010, 0b00000001, 0b00000001,
							0b10000000, 0b10000000, 0b01000000, 0b01000000, 0b00100000, 0b00100000, 0b00010000, 0b00010000,
							0b00001000, 0b00001000, 0b00000100, 0b00000100, 0b00000010, 0b00000010, 0b00000001, 0b00000001);
                uint16_t _b;

		batches = n>>4;

		__m256i ones = _mm256_set1_epi64x(-1);

		for (i=0;i<batches;i++) {
                        zeros = _mm256_setr_epi64x(0,0,-1,-1);
			mask = _mm256_set1_epi32(-1);
			accu = _mm256_setzero_si256();
			acc_inf = _mm256_setzero_si256();
			row = i<<4;
			for (j=0;j<i;j++) {
                                memcpy(&_b, X + 2*j, sizeof(_b));
				for (k=0;k<16;k++) {
					dups = _mm256_set1_epi8(-(_b&1));
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row]);
					accu ^= (_a & dups);
					row = row + complete_line;
					_b = _b >> 1;
				}
			}

			_a = load_incomplete_m256(&X[2*j], 2);
                        prod = _mm256_lddqu_si256((__m256i*)&even_diagonal[16*i]);
			_a = _mm256_permute4x64_epi64(_a, 0b01000100);
                        _a = _mm256_shuffle_epi8(_a, shuff_msk);
                        _a = _a & and_msk;
                        _a = _mm256_cmpeq_epi8(_a, and_msk);
                        accu ^= (_a & prod);
                        memcpy(&_b, X + 2*j, sizeof(_b));
                        for (l=0;l<2;l++) {
                                for (k=0;k<8;k++) {
                                        mask = _mm256_alignr_epi8(mask,zeros,14);
                                        dups = _mm256_set1_epi8(-(_b&1));
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                        prod = (_a & dups);
                                        accu ^= (mask & prod);
                                        acc_inf ^= (_mm256_xor_si256(mask, ones) & prod);
                                        row = row + complete_line;
                                        _b = _b >> 1;
                                }
                                zeros = _mm256_setzero_si256();
                        }
			_mm256_storeu_si256((void*)&Y_sup[i*16], accu);
			

			for (j=i+1;j<batches;j++) {
				memcpy(&_b, X + 2*j, sizeof(_b));
				for (k=0;k<16;k++) {
					dups = _mm256_set1_epi8(-(_b&1));
					_a = _mm256_lddqu_si256((__m256i*)&A[row]);
					acc_inf ^= (_a & dups);
					row = row + complete_line;
					_b = _b >> 1;
				}
			}
			_mm256_storeu_si256((void*)&Y_inf[i*16], acc_inf);
		}

	
}

static inline void gf256to2_gf2_mat_mult_single_avx2(const uint16_t *A, const uint8_t *X, uint16_t *Y, uint32_t n)
{
        /* XXX: NOTE: because of alignment and loading latencies, treating the matrix as "regular" seems always better
         * than using the trianglar shape ... */
        uint16_t i, j, k, l;
                uint16_t complete_line = (n + 7) & ~7;
		uint16_t batches;
		uint32_t row;
		__m256i _a, dups;
		__m256i acc, mask, zeros, prod;
		uint16_t _b;

		batches = n>>4;

		__m256i ones = _mm256_set1_epi64x(-1);

		for (i=0;i<batches;i++) {
                        zeros = _mm256_setr_epi64x(0,0,-1,-1);
			mask = _mm256_set1_epi32(-1);
			acc = _mm256_setzero_si256();
			row = ((complete_line+1)*i)<<4;

                        memcpy(&_b, X + 2*i, sizeof(_b));
                        for (l=0;l<2;l++) {
                                for (k=0;k<8;k++) {
                                        mask = _mm256_alignr_epi8(mask,zeros,14);
                                        dups = _mm256_set1_epi8(-(_b&1));
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                        prod = (_a & dups);
                                        acc ^= (_mm256_xor_si256(mask, ones) & prod);
                                        row = row + complete_line;
                                        _b = _b >> 1;
                                }
                                zeros = _mm256_setzero_si256();
                        }

			for (j=i+1;j<batches;j++) {
				memcpy(&_b, X + 2*j, sizeof(_b));
				for (k=0;k<16;k++) {
					dups = _mm256_set1_epi8(-(_b&1));
					_a = _mm256_lddqu_si256((__m256i*)&A[row]);
					acc ^= (_a & dups);
					row = row + complete_line;
					_b = _b >> 1;
				}
			}
			_mm256_storeu_si256((void*)&Y[i*16], acc);
		}

	
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
static inline void gf256to2_gf256_mat_mult_avx2(const uint16_t *A, const uint8_t *X, uint16_t *Y_inf, uint16_t *Y_sup, const uint16_t *even_diagonal, uint32_t n)
{
        /* XXX: NOTE: because of alignment and loading latencies, treating the matrix as "regular" seems always better
         * than using the trianglar shape ... */
        uint16_t i, j, k, l;
                uint16_t complete_line = (n + 7) & ~7;
		uint16_t batches, remainder;
		uint32_t row;
		__m256i accu, _a, dups;
		__m256i acc_inf, mask, zeros, prod;
		__m128i _b;

                const __m256i shuff_msk = _mm256_set_epi8(15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8,
                                                           7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2,  2,  1, 1, 0, 0);

                __m256i ones = _mm256_set1_epi64x(-1);

                batches = n>>4;
                remainder = n & 15;
		for (i=0;i<batches;i++) {
                        zeros = _mm256_setr_epi64x(0,0,-1,-1);
                        acc_inf = _mm256_setzero_si256();
                        accu = _mm256_setzero_si256();
                        mask = _mm256_set1_epi32(-1);
			row = i<<4;
			for (j=0;j<i;j++) {
				_b = _mm_load_si128((__m128i*)&X[16*j]);
                /* Multiply in GF(256) */
				for (k=0;k<16;k++) {
					dups=_mm256_broadcastb_epi8(_b);
					_a = _mm256_lddqu_si256((__m256i*)&A[row]);
					accu ^= gf256_mult_vectorized_avx2(_a,dups);
					row = row + complete_line;
					_b = _mm_alignr_epi8(_b, _b, 1);
				}
			}

			// preciso arrumar aqui, falta fazer o vdup do prod, tem que ter duas copias consecutivas de cada elemento de b
                        _b = _mm_load_si128((__m128i*)&X[16*j]);
                        prod = _mm256_castsi128_si256(_b);
                        prod = _mm256_permute4x64_epi64(prod, 0b01000100);
                        prod = _mm256_shuffle_epi8(prod, shuff_msk);
			_a = _mm256_lddqu_si256((__m256i*)&even_diagonal[16*i]); 
			accu ^= gf256_mult_vectorized_avx2(_a, prod);
                        for (l=0;l<2;l++) {
                                for (k=0;k<8;k++) {
                                        mask = _mm256_alignr_epi8(mask,zeros,14);
                                        dups=_mm256_broadcastb_epi8(_b);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                        prod = gf256_mult_vectorized_avx2(_a, dups);
                                        accu ^= (mask & prod);
                                        acc_inf ^= (_mm256_xor_si256(mask, ones) & prod);
                                        row = row + complete_line;
                                        _b = _mm_alignr_epi8(_b,_b,1);
                                }
                                zeros = _mm256_setzero_si256();
                        }
			_mm256_storeu_si256((void*)&Y_sup[i*16], accu);
			// preciso arrumar aqui
			

			for (j=i+1;j<batches;j++) {
				_b = _mm_load_si128((__m128i*)&X[16*j]);
				for (k=0;k<16;k++) {
					dups=_mm256_broadcastb_epi8(_b);
					_a = _mm256_lddqu_si256((__m256i*)&A[row]);
					acc_inf ^= gf256_mult_vectorized_avx2(_a, dups);
					row = row + complete_line;
					_b = _mm_alignr_epi8(_b,_b,1);
				}
			}
                        if (remainder > 0) {
                        _a = load_incomplete_m256(&X[16*j],remainder);
                        _b = _mm256_castsi256_si128(_a);
                        for (k=0;k < remainder;k++) {
                                dups = _mm256_broadcastb_epi8(_b);
                                _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                acc_inf ^= gf256_mult_vectorized_avx2(_a,dups);
                                row = row + complete_line;
                                _b = _mm_alignr_epi8(_b, _b, 1); // ideia do chat
                        }
                        }
			 _mm256_storeu_si256((void*)&Y_inf[i*16], acc_inf);
		}
        
         // Process last block, not my falt -\o/-
	if (remainder > 0) {
	acc_inf = _mm256_setzero_si256();
	accu = _mm256_setzero_si256();
	row = batches<<4;
	for (j=0;j<batches;j++) {
                _b = _mm_load_si128((__m128i*)&X[16*j]);
                for (k=0;k<16;k++) {
                        dups = _mm256_broadcastb_epi8(_b);
                        _a = load_incomplete_m256((uint8_t *) &A[row],2*remainder);
                        accu ^= gf256_mult_vectorized_avx2(_a,dups);
                        row = row + complete_line;
                        _b = _mm_alignr_epi8(_b, _b, 1); // ideia do chat
                }
	}

	
        mask = _mm256_set1_epi32(-1);
        zeros = _mm256_setr_epi64x(0,0,-1,-1);
	prod = load_incomplete_m256(&X[16*j],remainder);
	_b = _mm256_castsi256_si128(prod);
        prod = _mm256_permute4x64_epi64(prod, 0b01000100);
        prod = _mm256_shuffle_epi8(prod, shuff_msk);
        _a = load_incomplete_m256((uint8_t *) &even_diagonal[16*batches],2*remainder);
	accu ^= gf256_mult_vectorized_avx2(_a, prod);
        for (l=0;l<2;l++) {
                for (k=0;k<8 && k<remainder -(l<<3);k++) {
                        mask = _mm256_alignr_epi8(mask,zeros,14);
                        dups = _mm256_broadcastb_epi8(_b);
                        _a = load_incomplete_m256((uint8_t *) &A[row],2*remainder);
                        prod = gf256_mult_vectorized_avx2(_a,dups);
                        accu ^= (mask & prod);
                        acc_inf ^= (_mm256_xor_si256(mask, ones) & prod);
                        row = row + complete_line;
                        _b = _mm_alignr_epi8(_b, _b, 1); // ideia do chat
                }
                zeros = _mm256_setzero_si256();
        }
	store_incomplete_m256(accu,(uint8_t *) &Y_sup[batches*16],2*remainder);
	store_incomplete_m256(acc_inf,(uint8_t *) &Y_inf[batches*16],2*remainder);
	// End last block
	}
	
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
static inline void gf256to2_gf16_mat_mult_avx2(const uint16_t *A, const uint8_t *X, uint16_t *Y_inf, uint16_t *Y_sup, const uint16_t *even_diagonal, uint32_t n)
{
    uint16_t i, j, k, l;
	uint16_t batches, remainder;
        uint16_t complete_line;
	uint32_t row;
	__m256i _a;
	__m256i dups, _b;
	__m256i accu, acc_inf,prod;

        const __m256i shuff_msk = _mm256_set_epi8(15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8,
                                                   7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2,  2,  1, 1, 0, 0);
                

	__m256i zeros, mask;
        __m256i ones = _mm256_set1_epi64x(-1);
        __m128i lanes[2];

	batches = n>>4;
	remainder = n & 15;
        complete_line = (n + 7) & ~7;
	for (i=0;i<batches;i++) {
                zeros = _mm256_setr_epi64x(0,0,-1,-1);
                acc_inf = _mm256_setzero_si256();
		accu = _mm256_setzero_si256();
		mask = _mm256_set1_epi32(-1);
		row = i<<4;
		for (j=0;j<i;j++) {
			_b = load_incomplete_m256(&X[8*j],8);
			_b = gf256_lift32_from_gf16_avx2(_b);
                        lanes[0] = _mm256_castsi256_si128(_b);
			/* Multiply in GF(256) */
                        for (k=0;k<16;k++) {
                                dups= _mm256_broadcastb_epi8(lanes[0]);
                                _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                accu ^= gf256_mult_vectorized_avx2(_a,dups);
                                row = row + complete_line;
                                lanes[0] = _mm_alignr_epi8(lanes[0], lanes[0], 1);
                        }
		}

		// preciso arrumar aqui, falta fazer o vdup do prod, tem que ter duas copias consecutivas de cada elemento de b
		_b = load_incomplete_m256(&X[8*j],8);
		_b = gf256_lift32_from_gf16_avx2(_b);
                lanes[0] = _mm256_castsi256_si128(_b);
		prod = _mm256_permute4x64_epi64(_b, 0b01000100);
		prod = _mm256_shuffle_epi8(prod, shuff_msk);
		_a = _mm256_lddqu_si256((__m256i*)&even_diagonal[16*i]);
		prod = gf256_mult_vectorized_avx2(_a, prod);
		accu ^= prod;
                for (l=0;l<2;l++) {
                        for (k=0;k<8;k++) {
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                                dups = _mm256_broadcastb_epi8(lanes[0]);
                                _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                prod = gf256_mult_vectorized_avx2(_a,dups);
                                accu ^= (mask & prod);
                                acc_inf ^= (_mm256_xor_si256(mask, ones) & prod);
                                row = row + complete_line;
                                lanes[0] = _mm_alignr_epi8(lanes[0], lanes[0], 1);
                        }
                        zeros = _mm256_setzero_si256();
                }
		_mm256_storeu_si256((void*)&Y_sup[i*16], accu);
		// preciso arrumar aqui

		for (j=i+1;j<batches;j++) {
			_b = load_incomplete_m256(&X[8*j],8);
			_b = gf256_lift32_from_gf16_avx2(_b);
                        lanes[0] = _mm256_castsi256_si128(_b);
                        for (l=0;l<2;l++) {
                                for (k=0;k<8;k++) {
                                        dups = _mm256_broadcastb_epi8(lanes[0]);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                        acc_inf ^= gf256_mult_vectorized_avx2(_a,dups);
                                        row = row + complete_line;
                                        lanes[0] = _mm_alignr_epi8(lanes[0], lanes[0], 1);
                                }
                        }
		}
		if (remainder > 0) {
		_b = load_incomplete_m256(&X[8*j],(remainder+1)/2);
		_b = gf256_lift32_from_gf16_avx2(_b);
                lanes[0] = _mm256_castsi256_si128(_b);
                for (l=0;l<2;l++) {
                        for (k=0;k<8 && k<remainder -(l<<3);k++) {
                                dups = _mm256_broadcastb_epi8(lanes[0]);
                                _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                acc_inf ^= gf256_mult_vectorized_avx2(_a,dups);
                                row = row + complete_line;
                                lanes[0] = _mm_alignr_epi8(lanes[0], lanes[0], 1);
                        }
                }
		}
		_mm256_storeu_si256((void*)&Y_inf[i*16], acc_inf);
	}

	// Process last block, not my falt -\o/-
	if (remainder > 0) {
	acc_inf = _mm256_setzero_si256();
	accu = _mm256_setzero_si256();
	row = batches<<4;
	for (j=0;j<batches;j++) {
		_b = load_incomplete_m256(&X[8*j],8);
		_b = gf256_lift32_from_gf16_avx2(_b);
                lanes[0] = _mm256_castsi256_si128(_b);
                for (l=0;l<2;l++) {
                        for (k=0;k<8;k++) {
                                dups = _mm256_broadcastb_epi8(lanes[0]);
                                _a = load_incomplete_m256((uint8_t *)&A[row],2*remainder);
                                accu ^= gf256_mult_vectorized_avx2(_a,dups);
                                row = row + complete_line;
                                lanes[0] = _mm_alignr_epi8(lanes[0], lanes[0], 1);
                        }
                }
	}

	mask = _mm256_set1_epi32(-1);
	zeros = _mm256_setr_epi64x(0,0,-1,-1);
	_b = load_incomplete_m256(&X[8*j],(remainder+1)/2);
	_b = gf256_lift32_from_gf16_avx2(_b);
        lanes[0] = _mm256_castsi256_si128(_b);
	prod = _mm256_permute4x64_epi64(_b, 0b01000100);
	prod = _mm256_shuffle_epi8(prod, shuff_msk);
	_a = load_incomplete_m256((uint8_t *)&even_diagonal[16*batches],2*remainder);
	prod = gf256_mult_vectorized_avx2(_a, prod);
	accu ^= prod;
        for (l=0;l<2;l++) {
                for (k=0;k<8 && k<remainder-(l<<3);k++) {
                        mask = _mm256_alignr_epi8(mask,zeros,14);
                        dups = _mm256_broadcastb_epi8(lanes[0]);
                        _a = load_incomplete_m256((uint8_t *)&A[row],2*remainder);
                        prod = gf256_mult_vectorized_avx2(_a,dups);
                        accu ^= (mask & prod);
                        acc_inf ^= (_mm256_xor_si256(mask, ones) & prod);
                        row = row + complete_line;
                        lanes[0] = _mm_alignr_epi8(lanes[0], lanes[0], 1);
                }
                zeros = _mm256_setzero_si256();
        }
	store_incomplete_m256(accu,(uint8_t *)&Y_sup[batches*16],2*remainder);
	store_incomplete_m256(acc_inf,(uint8_t *)&Y_inf[batches*16],2*remainder);
	// End last block
	}
}

static inline void gf256to2_gf16_mat_mult_single_avx2(const uint16_t *A, const uint8_t *X, uint16_t *Y, uint32_t n)
{
    uint16_t i, j, k, l;
        uint16_t complete_line = (n + 7) & ~7;
	uint16_t batches, remainder;
	uint32_t row;
	__m256i _a;
	__m256i dups, _b;
	__m256i acc,prod;

	__m256i zeros, mask;
        __m256i ones = _mm256_set1_epi64x(-1);
        __m128i lanes[2];

	batches = n>>4;
	remainder = n & 15;
	for (i=0;i<batches;i++) {
                zeros = _mm256_setr_epi64x(0,0,-1,-1);
                acc = _mm256_setzero_si256();
		mask = _mm256_set1_epi32(-1);
		row = ((complete_line+1)*i)<<4;

		// preciso arrumar aqui, falta fazer o vdup do prod, tem que ter duas copias consecutivas de cada elemento de b
		_b = load_incomplete_m256(&X[8*i],8);
		_b = gf256_lift32_from_gf16_avx2(_b);
                lanes[0] = _mm256_castsi256_si128(_b);
                for (l=0;l<2;l++) {
                        for (k=0;k<8;k++) {
                                mask = _mm256_alignr_epi8(mask,zeros,14);
                                dups = _mm256_broadcastb_epi8(lanes[0]);
                                _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                prod = gf256_mult_vectorized_avx2(_a,dups);
                                acc ^= (_mm256_xor_si256(mask, ones) & prod);
                                row = row + complete_line;
                                lanes[0] = _mm_alignr_epi8(lanes[0], lanes[0], 1);
                        }
                        zeros = _mm256_setzero_si256();
                }
		// preciso arrumar aqui

		for (j=i+1;j<batches;j++) {
			_b = load_incomplete_m256(&X[8*j],8);
			_b = gf256_lift32_from_gf16_avx2(_b);
                        lanes[0] = _mm256_castsi256_si128(_b);
                        for (l=0;l<2;l++) {
                                for (k=0;k<8;k++) {
                                        dups = _mm256_broadcastb_epi8(lanes[0]);
                                        _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                        acc ^= gf256_mult_vectorized_avx2(_a,dups);
                                        row = row + complete_line;
                                        lanes[0] = _mm_alignr_epi8(lanes[0], lanes[0], 1);
                                }
                        }
		}
		if (remainder > 0) {
		_b = load_incomplete_m256(&X[8*j],(remainder+1)/2);
		_b = gf256_lift32_from_gf16_avx2(_b);
                lanes[0] = _mm256_castsi256_si128(_b);
                for (l=0;l<2;l++) {
                        for (k=0;k<8 && k<remainder -(l<<3);k++) {
                                dups = _mm256_broadcastb_epi8(lanes[0]);
                                _a = _mm256_lddqu_si256((__m256i*)&A[row]);
                                acc ^= gf256_mult_vectorized_avx2(_a,dups);
                                row = row + complete_line;
                                lanes[0] = _mm_alignr_epi8(lanes[0], lanes[0], 1);
                        }
                }
		}
		_mm256_storeu_si256((void*)&Y[i*16], acc);
	}

	// Process last block, not my falt -\o/-
	if (remainder > 0) {
	acc = _mm256_setzero_si256();
	row = ((complete_line+1)*batches)<<4;

	mask = _mm256_set1_epi32(-1);
	zeros = _mm256_setr_epi64x(0,0,-1,-1);
	_b = load_incomplete_m256(&X[8*j],(remainder+1)/2);
	_b = gf256_lift32_from_gf16_avx2(_b);
        lanes[0] = _mm256_castsi256_si128(_b);
        for (l=0;l<2;l++) {
                for (k=0;k<8 && k<remainder-(l<<3);k++) {
                        mask = _mm256_alignr_epi8(mask,zeros,14);
                        dups = _mm256_broadcastb_epi8(lanes[0]);
                        _a = load_incomplete_m256((uint8_t *)&A[row],2*remainder);
                        prod = gf256_mult_vectorized_avx2(_a,dups);
                        acc ^= (_mm256_xor_si256(mask, ones) & prod);
                        row = row + complete_line;
                        lanes[0] = _mm_alignr_epi8(lanes[0], lanes[0], 1);
                }
                zeros = _mm256_setzero_si256();
        }
	store_incomplete_m256(acc,(uint8_t *)&Y[batches*16],2*remainder);
	// End last block
	}
}

#endif /* __AVX2__ */

#endif /* __FIELDS_AVX2_H__ */
