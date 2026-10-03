#ifndef __FIELDS_NEON_H__
#define __FIELDS_NEON_H__

/* Check for NEON support */
#ifdef __ARM_NEON

#include "fields_common.h"
#include "fields_ref.h"
/* Needed for memcpy */
#include <string.h>
/* Needed for NEON assembly intrinsics */
#include <arm_neon.h>

#include <stdio.h>

/* === GF(2) === */
/* NOTE: for atomic multiplication, using vectorization is suboptimal */
static inline uint8_t gf2_mult_neon(uint8_t a, uint8_t b)
{
        return gf2_mult_ref(a, b);
}

#define NEON_MASK_SET 0xFFFFFFFF


/* This helper tries to efficiently copy len bytes from the ymm register 
*/
static inline void store_incomplete_m128(uint8x16_t in, uint8_t *a, uint32_t len)
{
	uint8_t output[16];
	vst1q_u8(output,in);
	for (uint32_t i=0;i<len;i++) {
		a[i] = output[i];
	}
        return;
}


/* This helper tries to efficiently copy len bytes in the ymm register
uint32x4_t 
*/
static inline uint8x16_t load_incomplete_m128(const uint8_t *a, uint32_t len)
{
	uint8x16_t res;

	uint8_t output[16] = {0};

	for (uint32_t i=0;i<len;i++) {
		output[i] = a[i];
	}
	res = vld1q_u8(output);

	return res;
}


static inline uint8_t parity_neon(uint8x16_t v) {
	uint8_t res;
	res = (vaddvq_u8(vcntq_u8(v)) & 1);

	return res;
}

static inline uint8_t sum_uint8_neon(uint8x16_t accu) {
        uint8_t res;

		uint8x8_t x8 = veor_u8(vget_low_u8(accu), vget_high_u8(accu));
		x8 = veor_u8(x8, vext_u8(x8, x8, 4));
		x8 = veor_u8(x8, vext_u8(x8, x8, 2));
		x8 = veor_u8(x8, vext_u8(x8, x8, 1));
		res = vget_lane_u8(x8, 0);

        return res;     
}

static inline uint16_t sum_uint16_neon(uint8x16_t accu) {
        uint16_t res;

		uint16x8_t accu16 = vreinterpretq_u16_u8(accu);

		uint16x4_t x16 = veor_u16(vget_low_u16(accu16), vget_high_u16(accu16));
		x16 = veor_u16(x16, vext_u16(x16, x16, 2));
		x16 = veor_u16(x16, vext_u16(x16, x16, 1));
		res = vget_lane_u16(x16, 0);

        return res;     
}  

/*
 * Vector multiplied by a constant in GF(2).
 */
static inline void gf2_constant_vect_mult_neon(uint8_t b, const uint8_t *a, uint8_t *c, uint32_t len)
{
	gf2_constant_vect_mult_ref(b, a, c, len);

        return;
}

static inline uint8_t gf2_vect_mult_neon(const uint8_t *a, const uint8_t *b, uint32_t len_bits)
{
	uint32_t i;
	uint8x16_t accu, _a, _b;
	uint32_t len = (len_bits / 8);

	/* Set the accumulator to 0 */
	accu = vdupq_n_u8(0);

	for(i = 0; i < len; i += 16){
		if((len-i) < 16){
			/* Note: if we are here, we are sure that we are 32-bit aligned */
			_a = load_incomplete_m128(&a[i], len-i);
			_b = load_incomplete_m128(&b[i], len-i);
		}
		else{
			/* Obvious 256-bit */
			_a = vld1q_u8(&a[i]);
			_b = vld1q_u8(&b[i]);
		}
		/* Vectorized AND of inputs and then XOR with the accumulator */
		accu = veorq_u8(vandq_u8(_a,_b),accu);
	}

	/* Now, we have to compute the parity bit, do it 64 bits per 64 bits */
	return parity_neon(accu);
}

/* Matrix and vector multiplication over GF(2) 
 * C = A * X, where X is a vector
 * Matrix is supposed to be square n x n, and vector n x 1
 * The output is a vector n x 1
 * */
/* XXX: TODO: this can be optimized by packing rows in zmm when n <= 256 
*/
static inline void gf2_mat_mult_neon(const uint8_t *A, const uint8_t *X, uint8_t *Y, uint32_t n, matrix_type mtype)
{
	GF2_MAT_MULT(A, X, Y, n, mtype, gf2_vect_mult_neon);
}

/* GF(2) matrix transposition 
*/
static inline void gf2_mat_transpose_neon(const uint8_t *A, uint8_t *B, uint32_t n, matrix_type mtype)
{
        gf2_mat_transpose_ref(A, B, n, mtype);
}

/* === GF(256) === */
/* NOTE: for atomic multiplication, using vectorization is suboptimal
*/
static inline uint8_t gf256_mult_neon(uint8_t x, uint8_t y)
{
	return gf256_mult_ref(x, y);
}

static inline uint8x16_t gf256_mult_vectorized_neon(uint8x16_t _a, uint8x16_t _b)
{
	poly8x16_t a = vreinterpretq_p8_u8(_a); 
	poly8x16_t b = vreinterpretq_p8_u8(_b);

    poly16x8_t d = vmull_p8(vget_low_p8(a),vget_low_p8(b));
    poly16x8_t c = vmull_high_p8(a,b);
    poly8x16_t dprime = vreinterpretq_p8_p16(d); 
    poly8x16_t cprime = vreinterpretq_p8_p16(c);
    poly8x16_t aprime = vuzp1q_p8(dprime,cprime);
    poly8x16_t bprime = vuzp2q_p8(dprime,cprime);
    
    uint8x16_t h=vreinterpretq_u8_p8(bprime);
    uint8x16_t h1=vshrq_n_u8(h,4);

    uint8x16_t h0 = vandq_u8(h, vdupq_n_u8(0x0f)); 
    uint8x16_t t1 = veorq_u8(h1, vshrq_n_u8(h1, 1));
    h0 = veorq_u8(h0, t1);
    poly8x16_t prime = vmulq_p8(vreinterpretq_p8_u8(h0),vdupq_n_p8(0x1B));
    bprime = vmulq_p8(vreinterpretq_p8_u8(h1),vdupq_n_p8(0x1B));

    uint8x16_t j=vreinterpretq_u8_p8(bprime);
    uint8x16_t e=vreinterpretq_u8_p8(prime);
    h0 = veorq_u8(vreinterpretq_u8_p8(aprime), veorq_u8(e, vshlq_n_u8(j, 4)));

    return h0;
}

/*
 * Vector multiplied by a constant in GF(256).
 */
static inline void gf256_constant_vect_mult_neon(uint8_t b, const uint8_t *a, uint8_t *c, uint32_t len)
{
	uint32_t i;
	uint8x16_t _a, _b;

	/* Load the constant byte b broadcasted in _b */
	_b = vdupq_n_u8(b);

        for(i = 0; i < len; i += 16){
                if((len-i) < 16){
						_a = load_incomplete_m128(&a[i], len-i);
						store_incomplete_m128(gf256_mult_vectorized_neon(_a, _b), &c[i], len-i);
                }
                else{
                        /* Obvious 512-bit */
                        _a = vld1q_u8(&a[i]);
                        /* Vectorized multiplication in GF(256) */
                        vst1q_u8((void*)&c[i], gf256_mult_vectorized_neon(_a, _b));
                }       
        }

        return;
}

/*
 * Vector to vector multiplication in GF(256).
 * Takes two vectors of length 'len', and returns a byte (element in GF(256))
 */
static inline uint8_t gf256_vect_mult_neon(const uint8_t *a, const uint8_t *b, uint32_t len)
{
	uint32_t i;
	uint8x16_t accu, _a, _b;

	/* Set the accumulator to 0 */
	accu = vdupq_n_u8(0);

	for(i = 0; i < len; i += 16){
		if((len-i) < 16){
			_a = load_incomplete_m128(&a[i], len-i);
			_b = load_incomplete_m128(&b[i], len-i);
		}
		else{
			/* Obvious 256-bit */
			_a = vld1q_u8(&a[i]);
			_b = vld1q_u8(&b[i]);
		}
		accu = veorq_u8(accu, gf256_mult_vectorized_neon(_a, _b));
	}

	return sum_uint8_neon(accu);
}

static inline void gf256_mat_mult_neon(const uint8_t *A, const uint8_t *X, uint8_t *Y_inf, uint8_t *Y_sup, uint32_t n)
{
    uint16_t i, j, k;
	uint16_t complete_line = (n + 7) & ~7;
	uint16_t batches, remainder;
	uint32_t row;
	uint64_t sumx8;
    uint32_t sumx4;
	uint8x16_t accu, _a, _b;
	uint8x16_t acc_inf;

	uint8x16_t extra_accs[4];

	uint8x16_t zeros, mask;

	zeros = vdupq_n_u8(0);

	/* Set the accumulator to 0 */
	batches = n/16;
	remainder = n - batches * 16;
	
	k=0;
	for (i=0;i<batches;i++) {
		mask = vdupq_n_u8(255);
		for (;k<i*16+16;k+=4) {
			row = k*complete_line;
			extra_accs[0] = vdupq_n_u8(0);
			extra_accs[1] = vdupq_n_u8(0);
			extra_accs[2] = vdupq_n_u8(0);
			extra_accs[3] = vdupq_n_u8(0); 
			for (j=0;j<i;j++) {
				_b = vld1q_u8(&X[16*j]);
				_a = vld1q_u8(&A[row+16*j]);
				extra_accs[0] ^= gf256_mult_vectorized_neon(_a, _b);
				_a = vld1q_u8(&A[row+complete_line+16*j]);
				extra_accs[1] ^= gf256_mult_vectorized_neon(_a, _b);
				_a = vld1q_u8(&A[row+2*complete_line+16*j]);
				extra_accs[2] ^= gf256_mult_vectorized_neon(_a, _b);
				_a = vld1q_u8(&A[row+3*complete_line+16*j]);
				extra_accs[3] ^= gf256_mult_vectorized_neon(_a, _b);
			}

			_b = vld1q_u8(&X[16*i]);

			// row k
			_a = vld1q_u8(&A[row+16*i]);
			accu = gf256_mult_vectorized_neon(_a, _b);
			extra_accs[0] = veorq_u8(extra_accs[0], vbicq_u8(accu, mask));
			accu = vandq_u8(mask, accu);
			extra_accs[0] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(extra_accs[0]),vreinterpretq_u64_u8(accu))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(extra_accs[0]),vreinterpretq_u64_u8(accu))));
			
			// row k+1
			mask = vextq_u8(zeros,mask,15);
			_a = vld1q_u8(&A[row+complete_line+16*i]);
			accu = gf256_mult_vectorized_neon(_a, _b);
			extra_accs[1] = veorq_u8(extra_accs[1], vbicq_u8(accu, mask));
			accu = vandq_u8(mask, accu);
			extra_accs[1] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(extra_accs[1]),vreinterpretq_u64_u8(accu))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(extra_accs[1]),vreinterpretq_u64_u8(accu))));

			// row k+2
			mask = vextq_u8(zeros,mask,15);
			_a = vld1q_u8(&A[row+2*complete_line+16*i]);
			accu = gf256_mult_vectorized_neon(_a, _b);
			extra_accs[2] = veorq_u8(extra_accs[2], vbicq_u8(accu, mask));
			accu = vandq_u8(mask, accu);
			extra_accs[2] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(extra_accs[2]),vreinterpretq_u64_u8(accu))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(extra_accs[2]),vreinterpretq_u64_u8(accu))));

			// row k+3
			mask = vextq_u8(zeros,mask,15);
			_a = vld1q_u8(&A[row+3*complete_line+16*i]);
			accu = gf256_mult_vectorized_neon(_a, _b);
			extra_accs[3] = veorq_u8(extra_accs[3], vbicq_u8(accu, mask));
			accu = vandq_u8(mask, accu);
			extra_accs[3] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(extra_accs[3]),vreinterpretq_u64_u8(accu))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(extra_accs[3]),vreinterpretq_u64_u8(accu))));
			
			// k xor k+2
			acc_inf = veorq_u8(vreinterpretq_u8_u32(vtrn1q_u32(vreinterpretq_u32_u8(extra_accs[0]),vreinterpretq_u32_u8(extra_accs[2]))), vreinterpretq_u8_u32(vtrn2q_u32(vreinterpretq_u32_u8(extra_accs[0]),vreinterpretq_u32_u8(extra_accs[2]))));

			// k+1 xor k+3
			extra_accs[0] = veorq_u8(vreinterpretq_u8_u32(vtrn1q_u32(vreinterpretq_u32_u8(extra_accs[1]),vreinterpretq_u32_u8(extra_accs[3]))), vreinterpretq_u8_u32(vtrn2q_u32(vreinterpretq_u32_u8(extra_accs[1]),vreinterpretq_u32_u8(extra_accs[3]))));

			// (k xor k+2) xor (k+1 xor k+3)
			acc_inf = veorq_u8(vreinterpretq_u8_u16(vtrn1q_u16(vreinterpretq_u16_u8(acc_inf),vreinterpretq_u16_u8(extra_accs[0]))), vreinterpretq_u8_u16(vtrn2q_u16(vreinterpretq_u16_u8(acc_inf),vreinterpretq_u16_u8(extra_accs[0])))); 

			extra_accs[0] = vdupq_n_u8(0);
			extra_accs[1] = vdupq_n_u8(0);
			extra_accs[2] = vdupq_n_u8(0);
			extra_accs[3] = vdupq_n_u8(0);
			for (j=i+1;j<batches;j++) {
				_b = vld1q_u8(&X[16*j]);
				_a = vld1q_u8(&A[row+16*j]);
				extra_accs[0] ^= gf256_mult_vectorized_neon(_a, _b);
				_a = vld1q_u8(&A[row+complete_line+16*j]);
				extra_accs[1] ^= gf256_mult_vectorized_neon(_a, _b);
				_a = vld1q_u8(&A[row+2*complete_line+16*j]);
				extra_accs[2] ^= gf256_mult_vectorized_neon(_a, _b);
				_a = vld1q_u8(&A[row+3*complete_line+16*j]);
				extra_accs[3] ^= gf256_mult_vectorized_neon(_a, _b);
			}
			if (remainder > 0) {
			_b = load_incomplete_m128(&X[16*j],remainder);
			_a = load_incomplete_m128(&A[row+16*batches],remainder);
			extra_accs[0] ^= gf256_mult_vectorized_neon(_a,_b);
			_a = load_incomplete_m128(&A[row+complete_line+16*batches],remainder);
			extra_accs[1] ^= gf256_mult_vectorized_neon(_a,_b);
			_a = load_incomplete_m128(&A[row+2*complete_line+16*batches],remainder);
			extra_accs[2] ^= gf256_mult_vectorized_neon(_a,_b);
			_a = load_incomplete_m128(&A[row+3*complete_line+16*batches],remainder);
			extra_accs[3] ^= gf256_mult_vectorized_neon(_a,_b);
			}

			accu = vdupq_n_u8(0);
			extra_accs[0] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(accu),vreinterpretq_u64_u8(extra_accs[0]))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(accu),vreinterpretq_u64_u8(extra_accs[0]))));
			extra_accs[1] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(accu),vreinterpretq_u64_u8(extra_accs[1]))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(accu),vreinterpretq_u64_u8(extra_accs[1]))));
			extra_accs[2] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(accu),vreinterpretq_u64_u8(extra_accs[2]))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(accu),vreinterpretq_u64_u8(extra_accs[2]))));
			extra_accs[3] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(accu),vreinterpretq_u64_u8(extra_accs[3]))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(accu),vreinterpretq_u64_u8(extra_accs[3]))));
			extra_accs[0] = veorq_u8(vreinterpretq_u8_u32(vtrn1q_u32(vreinterpretq_u32_u8(extra_accs[0]),vreinterpretq_u32_u8(extra_accs[2]))), vreinterpretq_u8_u32(vtrn2q_u32(vreinterpretq_u32_u8(extra_accs[0]),vreinterpretq_u32_u8(extra_accs[2]))));
			extra_accs[1] = veorq_u8(vreinterpretq_u8_u32(vtrn1q_u32(vreinterpretq_u32_u8(extra_accs[1]),vreinterpretq_u32_u8(extra_accs[3]))), vreinterpretq_u8_u32(vtrn2q_u32(vreinterpretq_u32_u8(extra_accs[1]),vreinterpretq_u32_u8(extra_accs[3]))));
			extra_accs[0] = veorq_u8(vreinterpretq_u8_u16(vtrn1q_u16(vreinterpretq_u16_u8(extra_accs[0]),vreinterpretq_u16_u8(extra_accs[1]))), vreinterpretq_u8_u16(vtrn2q_u16(vreinterpretq_u16_u8(extra_accs[0]),vreinterpretq_u16_u8(extra_accs[1])))); 
			acc_inf = veorq_u8(acc_inf,extra_accs[0]);
			acc_inf = veorq_u8(vuzp1q_u8(acc_inf,acc_inf),vuzp2q_u8(acc_inf,acc_inf));
			sumx8 = vgetq_lane_u64(vreinterpretq_u64_u8(acc_inf), 0);
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
			mask = vextq_u8(zeros,mask,15);
		}
	}

	if (remainder > 0) {
		mask = vdupq_n_u8(255);
		for (k=16*batches;k<n;k+=4) {
			row = k*complete_line;
			extra_accs[0] = vdupq_n_u8(0);
			extra_accs[1] = vdupq_n_u8(0);
			extra_accs[2] = vdupq_n_u8(0);
			extra_accs[3] = vdupq_n_u8(0);
			for (j=0;j<batches;j++) {
				_b = vld1q_u8(&X[16*j]);
				_a = vld1q_u8(&A[row+16*j]);
				extra_accs[0] ^= gf256_mult_vectorized_neon(_a, _b);
				_a = vld1q_u8(&A[row+complete_line+16*j]);
				extra_accs[1] ^= gf256_mult_vectorized_neon(_a, _b);
				_a = vld1q_u8(&A[row+2*complete_line+16*j]);
				extra_accs[2] ^= gf256_mult_vectorized_neon(_a, _b);
				_a = vld1q_u8(&A[row+3*complete_line+16*j]);
				extra_accs[3] ^= gf256_mult_vectorized_neon(_a, _b);
			}

			_b = load_incomplete_m128(&X[16*batches],remainder);

			// row k
			_a = load_incomplete_m128(&A[row+16*batches],remainder);
			accu = gf256_mult_vectorized_neon(_a, _b);
			extra_accs[0] = veorq_u8(extra_accs[0], vbicq_u8(accu, mask));
			accu = vandq_u8(mask, accu);
			extra_accs[0] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(extra_accs[0]),vreinterpretq_u64_u8(accu))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(extra_accs[0]),vreinterpretq_u64_u8(accu))));
			
			// row k+1
			mask = vextq_u8(zeros,mask,15);
			_a = load_incomplete_m128(&A[row+complete_line+16*batches],remainder);
			accu = gf256_mult_vectorized_neon(_a, _b);
			extra_accs[1] = veorq_u8(extra_accs[1], vbicq_u8(accu, mask));
			accu = vandq_u8(mask, accu);
			extra_accs[1] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(extra_accs[1]),vreinterpretq_u64_u8(accu))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(extra_accs[1]),vreinterpretq_u64_u8(accu))));

			// row k+2
			mask = vextq_u8(zeros,mask,15);
			_a = load_incomplete_m128(&A[row+2*complete_line+16*batches],remainder);
			accu = gf256_mult_vectorized_neon(_a, _b);
			extra_accs[2] = veorq_u8(extra_accs[2], vbicq_u8(accu, mask));
			accu = vandq_u8(mask, accu);
			extra_accs[2] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(extra_accs[2]),vreinterpretq_u64_u8(accu))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(extra_accs[2]),vreinterpretq_u64_u8(accu))));

			// row k+3
			mask = vextq_u8(zeros,mask,15);
			_a = load_incomplete_m128(&A[row+3*complete_line+16*batches],remainder);
			accu = gf256_mult_vectorized_neon(_a, _b);
			extra_accs[3] = veorq_u8(extra_accs[3], vbicq_u8(accu, mask));
			accu = vandq_u8(mask, accu);
			extra_accs[3] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(extra_accs[3]),vreinterpretq_u64_u8(accu))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(extra_accs[3]),vreinterpretq_u64_u8(accu))));
			
			// k xor k+2
			acc_inf = veorq_u8(vreinterpretq_u8_u32(vtrn1q_u32(vreinterpretq_u32_u8(extra_accs[0]),vreinterpretq_u32_u8(extra_accs[2]))), vreinterpretq_u8_u32(vtrn2q_u32(vreinterpretq_u32_u8(extra_accs[0]),vreinterpretq_u32_u8(extra_accs[2]))));

			// k+1 xor k+3
			extra_accs[0] = veorq_u8(vreinterpretq_u8_u32(vtrn1q_u32(vreinterpretq_u32_u8(extra_accs[1]),vreinterpretq_u32_u8(extra_accs[3]))), vreinterpretq_u8_u32(vtrn2q_u32(vreinterpretq_u32_u8(extra_accs[1]),vreinterpretq_u32_u8(extra_accs[3]))));

			// (k xor k+2) xor (k+1 xor k+3)
			acc_inf = veorq_u8(vreinterpretq_u8_u16(vtrn1q_u16(vreinterpretq_u16_u8(acc_inf),vreinterpretq_u16_u8(extra_accs[0]))), vreinterpretq_u8_u16(vtrn2q_u16(vreinterpretq_u16_u8(acc_inf),vreinterpretq_u16_u8(extra_accs[0])))); 

			acc_inf = veorq_u8(vuzp1q_u8(acc_inf,acc_inf),vuzp2q_u8(acc_inf,acc_inf));
			sumx8 = vgetq_lane_u64(vreinterpretq_u64_u8(acc_inf), 0);
			sumx4 = (uint32_t) sumx8;
			memcpy(Y_inf + (k-1), &sumx4, 4);
			sumx4 = (uint32_t) (sumx8 >> 32);
			memcpy(Y_sup + k, &sumx4, 4);
			mask = vextq_u8(zeros,mask,15);
		}
	// End last block
	}

	/*Last element of Y_inf*/
	row = n*complete_line;
    Y_inf[n-1] = gf256_vect_mult_neon(X,&A[row],n);
}

static inline void gf256_mat_mult_single_neon(const uint8_t *A, const uint8_t *X, uint8_t *Y, uint32_t n)
{
    GF256_MAT_MULT(A, X, Y, n, TRI_INF, gf256_vect_mult_neon);
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(2) and a vector in GF(256) -- NOK
 */
static inline uint8_t gf2_gf256_vect_mult_neon(const uint8_t *a_gf2, const uint8_t *b_gf256, uint32_t len)
{
        uint32_t i;
        const uint8x16_t zero = vdupq_n_u8(0);
		uint8x16_t _a, _b;
		uint8x8_t _alow, _ahigh;

        /* Set the accumulator to 0 */
        uint8x16_t accu = vdupq_n_u8(0);
		uint32_t ceil_len;

        for(i = 0; i < len; i += 16){
		if((len - i) < 16){
			ceil_len = ((len - i) % 8 == 0) ? ((len - i) / 8) : (((len - i) / 8) + 1);
			_b = load_incomplete_m128(&b_gf256[i], len - i);
		}
		else{
			/* Obvious 256-bit */
			ceil_len = 2;
			_b = vld1q_u8(&b_gf256[i]);
		}
		if (ceil_len == 2) {
			_ahigh = vdup_n_u8(a_gf2[i/8 + 1]);
		}
		else {
			_ahigh = vdup_n_u8(0);
		}
		_alow = vdup_n_u8(a_gf2[i/8]);
	
		_a = vcombine_u8(_alow,_ahigh);
	
		/* Create a selection mask from the bits in _a */
		const uint8x16_t and_msk = {0b00000001, 0b00000010, 0b00000100, 0b00001000, 0b00010000, 0b00100000, 0b01000000, 0b10000000,
							0b00000001, 0b00000010, 0b00000100, 0b00001000, 0b00010000, 0b00100000, 0b01000000, 0b10000000};
		/* Only keep the selection bits */
		_a = vandq_u8(_a,and_msk);
		/* Transform these bits to either 0 or 0xFF and keep 0x80 */
		_a = vcgtq_u8(_a,zero);
		/* Use blending for bytes selection */
				accu ^= vbslq_u8(_a, _b, zero);
        }
		return sum_uint8_neon(accu);
}

/*
 * "Hybrid" multiplication of a constant in GF(2) and a vector in GF(256) --OK
 */
static inline void gf2_gf256_constant_vect_mult_neon(uint8_t a_gf2, const uint8_t *b_gf256, uint8_t *c_gf256, uint32_t n)
{
	gf2_gf256_constant_vect_mult_ref(a_gf2, b_gf256, c_gf256, n);

        return;
}

/*
 * "Hybrid" multiplication of a constant in GF(256) and a vector in GF(2) --NOK
 */
static inline void gf256_gf2_constant_vect_mult_neon(uint8_t a_gf256, const uint8_t *b_gf2, uint8_t *c_gf256, uint32_t len)
{
	uint32_t i;
    const uint8x16_t zero = vdupq_n_u8(0);
	uint8x16_t _a, _b;
	uint8x8_t _blow, _bhigh;

        /* Broadcast the constant value */
	_a = vdupq_n_u8(a_gf256);
	uint32_t ceil_len;

        for(i = 0; i < len; i += 16){
		if (len-i<16) {
			ceil_len = ((len - i) % 8 == 0) ? ((len - i) / 8) : (((len - i) / 8) + 1);
		}
		else {
			ceil_len = 2;
		}
		if(ceil_len==2){
			_bhigh = vdup_n_u8(b_gf2[i/8 + 1]);
		}
		else{
			_bhigh = vdup_n_u8(0);
		}
			_blow = vdup_n_u8(b_gf2[i/8]);

			_b = vcombine_u8(_blow,_bhigh);

			/* Create a selection mask from the bits in _a */
			const uint8x16_t and_msk = {0b00000001, 0b00000010, 0b00000100, 0b00001000, 0b00010000, 0b00100000, 0b01000000, 0b10000000,
								0b00000001, 0b00000010, 0b00000100, 0b00001000, 0b00010000, 0b00100000, 0b01000000, 0b10000000};
			/* Only keep the selection bits */
			_b = vandq_u8(_b,and_msk);
			/* Transform these bits to either 0 or 0xFF and keep 0x80 */
			_b = vcgtq_u8(_b,zero);
			/* Use blending for bytes selection */
					uint8x16_t _c = vbslq_u8(_b, _a, zero);
			/* Store the result */

			vst1q_u8(&c_gf256[i],_c);
        }

        return;
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(256) and a vector in GF(2) --OK
 */
static inline uint8_t gf256_gf2_vect_mult_neon(const uint8_t *a_gf256, const uint8_t *b_gf2, uint32_t n)
{
        return gf2_gf256_vect_mult_neon(b_gf2, a_gf256, n);
}

/* 
 * "Hybrid" matrix multiplication of a matrix in GF(2) and a vector in GF(256), resulting
 *  in a vector in GF(256) --OK
 */                     
static inline void gf2_gf256_mat_mult_neon(const uint8_t *A, const uint8_t *X, uint8_t *Y, uint32_t n, matrix_type mtype)
{                       
        /* NOTE: XXX: we force a REG here as it allows for better performance */
        (void)mtype;
        GF2_GF256_MAT_MULT(A, X, Y, n, REG, gf2_gf256_vect_mult_neon);
}

/*
 * "Hybrid" matrix multiplication of a matrix in GF(256) and a vector in GF(2), resulting
 *  in a vector in GF(256) 
 */
static inline void gf256_gf2_mat_mult_neon(const uint8_t *A, const uint8_t *X, uint8_t *Y_inf, uint8_t *Y_sup, uint32_t n)
{
        uint16_t i, j, k;
	uint16_t complete_line = (n + 7) & ~7;
	uint16_t batches;
	uint32_t row; 
	uint64_t sumx8;
    uint32_t sumx4;
	uint8x16_t accu, _a, _b;
	uint8x16_t acc_inf;
	const uint8x16_t and_msk = {0b00000001, 0b00000010, 0b00000100, 0b00001000, 0b00010000, 0b00100000, 0b01000000, 0b10000000,
						0b00000001, 0b00000010, 0b00000100, 0b00001000, 0b00010000, 0b00100000, 0b01000000, 0b10000000};
	uint8x8_t low; 
	uint8x8_t high;

	uint8x16_t extra_accs[4];

	uint8x16_t zeros, mask;

	zeros = vdupq_n_u8(0);

	/* Set the accumulator to 0 */
	batches = n/16;

	k=0;
	for (i=0;i<batches;i++) {
		mask = vdupq_n_u8(255);
		for (;k<i*16+16;k+=4) {
			row = k*complete_line;
			extra_accs[0] = vdupq_n_u8(0);
			extra_accs[1] = vdupq_n_u8(0);
			extra_accs[2] = vdupq_n_u8(0);
			extra_accs[3] = vdupq_n_u8(0); 
			for (j=0;j<i;j++) {
				low = vdup_n_u8(X[2*j]); 
				high = vdup_n_u8(X[2*j+1]);
				_b = vcombine_u8(low,high);
				_b = vandq_u8(_b,and_msk);
				_b = vcgtq_u8(_b,zeros);
				_a = vld1q_u8(&A[row+16*j]);
				extra_accs[0] ^= vbslq_u8(_b, _a, zeros);
				_a = vld1q_u8(&A[row+complete_line+16*j]);
				extra_accs[1] ^= vbslq_u8(_b, _a, zeros);
				_a = vld1q_u8(&A[row+2*complete_line+16*j]);
				extra_accs[2] ^= vbslq_u8(_b, _a, zeros);
				_a = vld1q_u8(&A[row+3*complete_line+16*j]);
				extra_accs[3] ^= vbslq_u8(_b, _a, zeros);
			}

			low = vdup_n_u8(X[2*j]); 
			high = vdup_n_u8(X[2*j+1]);
			_b = vcombine_u8(low,high);
			_b = vandq_u8(_b,and_msk);
			_b = vcgtq_u8(_b,zeros);

			// row k
			_a = vld1q_u8(&A[row+16*i]);
			accu = vbslq_u8(_b, _a, zeros);
			extra_accs[0] = veorq_u8(extra_accs[0], vbicq_u8(accu, mask));
			accu = vandq_u8(mask, accu);
			extra_accs[0] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(extra_accs[0]),vreinterpretq_u64_u8(accu))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(extra_accs[0]),vreinterpretq_u64_u8(accu))));
			
			// row k+1
			mask = vextq_u8(zeros,mask,15);
			_a = vld1q_u8(&A[row+complete_line+16*i]);
			accu = vbslq_u8(_b, _a, zeros);
			extra_accs[1] = veorq_u8(extra_accs[1], vbicq_u8(accu, mask));
			accu = vandq_u8(mask, accu);
			extra_accs[1] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(extra_accs[1]),vreinterpretq_u64_u8(accu))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(extra_accs[1]),vreinterpretq_u64_u8(accu))));

			// row k+2
			mask = vextq_u8(zeros,mask,15);
			_a = vld1q_u8(&A[row+2*complete_line+16*i]);
			accu = vbslq_u8(_b, _a, zeros);
			extra_accs[2] = veorq_u8(extra_accs[2], vbicq_u8(accu, mask));
			accu = vandq_u8(mask, accu);
			extra_accs[2] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(extra_accs[2]),vreinterpretq_u64_u8(accu))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(extra_accs[2]),vreinterpretq_u64_u8(accu))));

			// row k+3
			mask = vextq_u8(zeros,mask,15);
			_a = vld1q_u8(&A[row+3*complete_line+16*i]);
			accu = vbslq_u8(_b, _a, zeros);
			extra_accs[3] = veorq_u8(extra_accs[3], vbicq_u8(accu, mask));
			accu = vandq_u8(mask, accu);
			extra_accs[3] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(extra_accs[3]),vreinterpretq_u64_u8(accu))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(extra_accs[3]),vreinterpretq_u64_u8(accu))));
			
			// k xor k+2
			acc_inf = veorq_u8(vreinterpretq_u8_u32(vtrn1q_u32(vreinterpretq_u32_u8(extra_accs[0]),vreinterpretq_u32_u8(extra_accs[2]))), vreinterpretq_u8_u32(vtrn2q_u32(vreinterpretq_u32_u8(extra_accs[0]),vreinterpretq_u32_u8(extra_accs[2]))));

			// k+1 xor k+3
			extra_accs[0] = veorq_u8(vreinterpretq_u8_u32(vtrn1q_u32(vreinterpretq_u32_u8(extra_accs[1]),vreinterpretq_u32_u8(extra_accs[3]))), vreinterpretq_u8_u32(vtrn2q_u32(vreinterpretq_u32_u8(extra_accs[1]),vreinterpretq_u32_u8(extra_accs[3]))));

			// (k xor k+2) xor (k+1 xor k+3)
			acc_inf = veorq_u8(vreinterpretq_u8_u16(vtrn1q_u16(vreinterpretq_u16_u8(acc_inf),vreinterpretq_u16_u8(extra_accs[0]))), vreinterpretq_u8_u16(vtrn2q_u16(vreinterpretq_u16_u8(acc_inf),vreinterpretq_u16_u8(extra_accs[0])))); 

			extra_accs[0] = vdupq_n_u8(0);
			extra_accs[1] = vdupq_n_u8(0);
			extra_accs[2] = vdupq_n_u8(0);
			extra_accs[3] = vdupq_n_u8(0);
			for (j=i+1;j<batches;j++) {
				low = vdup_n_u8(X[2*j]); 
				high = vdup_n_u8(X[2*j+1]);
				_b = vcombine_u8(low,high);
				_b = vandq_u8(_b,and_msk);
				_b = vcgtq_u8(_b,zeros);
				_a = vld1q_u8(&A[row+16*j]);
				extra_accs[0] ^= vbslq_u8(_b, _a, zeros);
				_a = vld1q_u8(&A[row+complete_line+16*j]);
				extra_accs[1] ^= vbslq_u8(_b, _a, zeros);
				_a = vld1q_u8(&A[row+2*complete_line+16*j]);
				extra_accs[2] ^= vbslq_u8(_b, _a, zeros);
				_a = vld1q_u8(&A[row+3*complete_line+16*j]);
				extra_accs[3] ^= vbslq_u8(_b, _a, zeros);
			}
			
			accu = vdupq_n_u8(0);
			extra_accs[0] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(accu),vreinterpretq_u64_u8(extra_accs[0]))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(accu),vreinterpretq_u64_u8(extra_accs[0]))));
			extra_accs[1] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(accu),vreinterpretq_u64_u8(extra_accs[1]))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(accu),vreinterpretq_u64_u8(extra_accs[1]))));
			extra_accs[2] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(accu),vreinterpretq_u64_u8(extra_accs[2]))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(accu),vreinterpretq_u64_u8(extra_accs[2]))));
			extra_accs[3] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(accu),vreinterpretq_u64_u8(extra_accs[3]))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(accu),vreinterpretq_u64_u8(extra_accs[3]))));
			extra_accs[0] = veorq_u8(vreinterpretq_u8_u32(vtrn1q_u32(vreinterpretq_u32_u8(extra_accs[0]),vreinterpretq_u32_u8(extra_accs[2]))), vreinterpretq_u8_u32(vtrn2q_u32(vreinterpretq_u32_u8(extra_accs[0]),vreinterpretq_u32_u8(extra_accs[2]))));
			extra_accs[1] = veorq_u8(vreinterpretq_u8_u32(vtrn1q_u32(vreinterpretq_u32_u8(extra_accs[1]),vreinterpretq_u32_u8(extra_accs[3]))), vreinterpretq_u8_u32(vtrn2q_u32(vreinterpretq_u32_u8(extra_accs[1]),vreinterpretq_u32_u8(extra_accs[3]))));
			extra_accs[0] = veorq_u8(vreinterpretq_u8_u16(vtrn1q_u16(vreinterpretq_u16_u8(extra_accs[0]),vreinterpretq_u16_u8(extra_accs[1]))), vreinterpretq_u8_u16(vtrn2q_u16(vreinterpretq_u16_u8(extra_accs[0]),vreinterpretq_u16_u8(extra_accs[1])))); 
			acc_inf = veorq_u8(acc_inf,extra_accs[0]);
			acc_inf = veorq_u8(vuzp1q_u8(acc_inf,acc_inf),vuzp2q_u8(acc_inf,acc_inf));
			sumx8 = vgetq_lane_u64(vreinterpretq_u64_u8(acc_inf), 0);
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
			mask = vextq_u8(zeros,mask,15);
		}
	}
	// End last block

	/*Last element of Y_inf*/
	row = n*complete_line;
	Y_inf[n-1] = gf256_gf2_vect_mult_neon(&A[row],X,n);
}

static inline void gf256_transpose_block_neon(uint8_t *src, uint8_t *dst, uint16_t size_in, uint16_t size_out)
{
    uint8x8_t r0 = vld1_u8(src);
    uint8x8_t r1 = vld1_u8(src + (1)*size_in);
    uint8x8_t r2 = vld1_u8(src + (2)*size_in);
    uint8x8_t r3 = vld1_u8(src + (3)*size_in);
    uint8x8_t r4 = vld1_u8(src + (4)*size_in);
    uint8x8_t r5 = vld1_u8(src + (5)*size_in);
    uint8x8_t r6 = vld1_u8(src + (6)*size_in);
    uint8x8_t r7 = vld1_u8(src + (7)*size_in);

    uint8x8x2_t t01 = vtrn_u8(r0, r1);
    uint8x8x2_t t02 = vtrn_u8(r2, r3);

    uint16x4x2_t s01 = vtrn_u16(
        vreinterpret_u16_u8(t01.val[0]),
        vreinterpret_u16_u8(t02.val[0])
    );

    uint16x4x2_t s02 = vtrn_u16(
        vreinterpret_u16_u8(t01.val[1]),
        vreinterpret_u16_u8(t02.val[1])
    );

    uint32x2_t a0 = vreinterpret_u32_u16(s01.val[0]);
    uint32x2_t a1 = vreinterpret_u32_u16(s01.val[1]);
    uint32x2_t a2 = vreinterpret_u32_u16(s02.val[0]);
    uint32x2_t a3 = vreinterpret_u32_u16(s02.val[1]);

    t01 = vtrn_u8(r4, r5);
    t02 = vtrn_u8(r6, r7);

    s01 = vtrn_u16(
        vreinterpret_u16_u8(t01.val[0]),
        vreinterpret_u16_u8(t02.val[0])
    );

    s02 = vtrn_u16(
        vreinterpret_u16_u8(t01.val[1]),
        vreinterpret_u16_u8(t02.val[1])
    );

    uint32x2_t a4 = vreinterpret_u32_u16(s01.val[0]);
    uint32x2_t a5 = vreinterpret_u32_u16(s01.val[1]);
    uint32x2_t a6 = vreinterpret_u32_u16(s02.val[0]);
    uint32x2_t a7 = vreinterpret_u32_u16(s02.val[1]);

    uint32x2x2_t final = vtrn_u32(a0, a4);

    uint8x8_t out0 = vreinterpret_u8_u32(final.val[0]);
    uint8x8_t out4 = vreinterpret_u8_u32(final.val[1]);

    final = vtrn_u32(a1,a5);

    uint8x8_t out2 = vreinterpret_u8_u32(final.val[0]);
    uint8x8_t out6 = vreinterpret_u8_u32(final.val[1]);

    final = vtrn_u32(a2, a6);

    uint8x8_t out1 = vreinterpret_u8_u32(final.val[0]);
    uint8x8_t out5 = vreinterpret_u8_u32(final.val[1]);

    final = vtrn_u32(a3, a7);

    uint8x8_t out3 = vreinterpret_u8_u32(final.val[0]);
    uint8x8_t out7 = vreinterpret_u8_u32(final.val[1]);

    vst1_u8(dst, out0);
    vst1_u8(dst + (1)*size_out , out1);
    vst1_u8(dst + (2)*size_out , out2);
    vst1_u8(dst + (3)*size_out , out3);
    vst1_u8(dst + (4)*size_out , out4);
    vst1_u8(dst + (5)*size_out , out5);
    vst1_u8(dst + (6)*size_out , out6);
    vst1_u8(dst + (7)*size_out , out7);
}


/* GF(256) matrix transposition */
static inline void gf256_mat_transpose_neon(uint8_t *A, uint16_t len, uint16_t extension)
{
	uint8_t *block1;
	uint8_t *block2;
	uint8_t aux[64] = {0};
	uint8x8_t mv;
	for (int i=0;i<len;i+=8){
		for (int j=0;j<i;j+=8) {
			block1 = A + i*extension + j;
			block2 = A + j*extension + i;
			gf256_transpose_block_neon(block1,aux,extension,8);
			gf256_transpose_block_neon(block2,block1,extension,extension);
			for (int k=0;k<8;k++) {
				mv = vld1_u8(aux + 8*k);
				vst1_u8(block2+extension*k,mv);
			}
		}
	}
	for (int i=0;i<len;i+=8){
				block1 = A + i*extension + i;
				gf256_transpose_block_neon(block1,block1,extension,extension);
	}
}

static inline void BLC_Compute_Folding(uint8_t* acc, uint8_t* lseed, uint8_t* data_folding, uint32_t nb_vecs, uint32_t row_size, uint32_t bytes_to_add)
{
        uint32_t i;
        uint32_t j;

        uint32_t g2;
        uint32_t diff;
        uint32_t pos;

        uint8x16_t accu[12], folding_element;
        uint8x16_t folding0[12];
        uint8x16_t current_seed;

        /* Set the accumulator to 0 */
        for (i=0; i < 12; i++) {
                accu[i] = vdupq_n_u8(0);
                folding0[i] = vdupq_n_u8(0);
        }

        for (j=0;j<nb_vecs;j+=2) {
                g2 = (j+2 < nb_vecs) ? (j + 2) : 0;
                diff = (j+1) ^ g2;
                diff = 31 - __builtin_clz(diff);
                pos = 0;
                for(i = 0; i < bytes_to_add; i += 16){
                        if((bytes_to_add-i) < 16){
                                current_seed = load_incomplete_m128(&lseed[j*row_size+i], (bytes_to_add-i));
                                accu[pos] = veorq_u8(current_seed,accu[pos]);
                                folding0[pos] = veorq_u8(folding0[pos],accu[pos]);
                        }
                        else{
                                /* Obvious 256-bit */
                                current_seed = vld1q_u8(&lseed[j*row_size+i]);
                                accu[pos] = veorq_u8(current_seed,accu[pos]);
                                folding0[pos] = veorq_u8(folding0[pos],accu[pos]);
                        }
                        pos++;
                }
                pos = 0;
                for(i = 0; i < bytes_to_add; i += 16){
                        if((bytes_to_add-i) < 16){
                                current_seed = load_incomplete_m128(&lseed[(j+1)*row_size+i], (bytes_to_add-i));
                                accu[pos] = veorq_u8(current_seed,accu[pos]);
                                folding_element = load_incomplete_m128(&data_folding[diff*row_size+i], (bytes_to_add-i));
                                folding_element = veorq_u8(folding_element,accu[pos]);
                                store_incomplete_m128(folding_element, &data_folding[diff*row_size+i], (bytes_to_add-i));
                        }
                        else{
                                /* Obvious 256-bit */
                                current_seed = vld1q_u8(&lseed[(j+1)*row_size+i]);
                                accu[pos] = veorq_u8(current_seed,accu[pos]);
                                folding_element = vld1q_u8(&data_folding[diff*row_size+i]);
                                folding_element = veorq_u8(folding_element,accu[pos]);
                                vst1q_u8(&data_folding[diff*row_size+i], folding_element);
                        }
                        pos++;
                }
        }
        pos = 0;
        for(i = 0; i < bytes_to_add; i += 16){
                if((bytes_to_add-i) < 16){
                        store_incomplete_m128(accu[pos], &acc[i], (bytes_to_add-i));
                        store_incomplete_m128(folding0[pos], &data_folding[i], (bytes_to_add-i));
                }
                else {
                        vst1q_u8(&acc[i], accu[pos]);
                        vst1q_u8(&data_folding[i], folding0[pos]);
                }
                pos++;
        }
}

/* 
 * "Hybrid" multiplication of a constant in GF(16) and a vector in GF(256)
 */
static inline void gf16_gf256_constant_vect_mult_neon(uint8_t a_gf16, const uint8_t *b_gf256, uint8_t *c_gf256, uint32_t n)
{
    uint8_t a_gf256;
    gf256_vect_lift_from_gf16_ref(&a_gf16, &a_gf256, 1);
    gf256_constant_vect_mult_neon(a_gf256, b_gf256, c_gf256, n);
    return;
}

/* Vectorized lifting from GF(16) to GF(256) */
static inline void gf256_vect_lift_from_gf16_neon(const uint8_t *b_gf16, uint8_t *c_gf256, uint32_t len)
{
        uint32_t i;
        uint8x16_t _a_gf16, _c, _a, _b;
	const uint8x16_t lifting_lookup = {0x00, 0x01, 0xe0, 0xe1, 0x5d, 0x5c, 0xbd, 0xbc, 0xb0, 0xb1, 0x50, 0x51, 0xed, 0xec, 0x0d, 0x0c};

	for(i = 0; i < len; i += 16){
                if((len-i) < 16){
                        _a_gf16 = load_incomplete_m128((const uint8_t*)&b_gf16[i/2], (len-i+1)/2);
                }
                else{
                        /* Obvious 256-bit */
                        _a_gf16 = load_incomplete_m128((const uint8_t*)&b_gf16[i/2], 8);
                }

		/* Duplicate lanes */
		_a = vshrq_n_u8(vandq_u8(_a_gf16,vdupq_n_u8(0xf0)),4);
		_b = vandq_u8(_a_gf16,vdupq_n_u8(0x0f));
		_a_gf16 = vzip1q_u8(_b,_a);
		/* Lift: since we are on 16 bits, we can perform a vperm lookup inside the register */
		_c = vqtbl1q_u8(lifting_lookup, _a_gf16);
		/* Store the result */
		if((len-i) < 16){
			store_incomplete_m128(_c, &c_gf256[i], (len-i));
		}
		else{
                        vst1q_u8(&c_gf256[i], _c);
		}
	}

	return;
}

/* lift from 16 elements GF(16) to GF(256) */
static inline uint8x16_t gf256_lift16_from_gf16_neon(const uint8x16_t _a_gf16)
{
        uint8x16_t _c, _a, _b;
	const uint8x16_t lifting_lookup = {0x00, 0x01, 0xe0, 0xe1, 0x5d, 0x5c, 0xbd, 0xbc, 0xb0, 0xb1, 0x50, 0x51, 0xed, 0xec, 0x0d, 0x0c};

	/* Duplicate lanes */
	_a = vshrq_n_u8(vandq_u8(_a_gf16,vdupq_n_u8(0xf0)),4);
	_b = vandq_u8(_a_gf16,vdupq_n_u8(0x0f));
	_a = vzip1q_u8(_b,_a);
	/* Lift: since we are on 16 bits, we can perform a vperm lookup inside the register */
	_c = vqtbl1q_u8(lifting_lookup, _a);
	/* Store the result */

	return _c;
}

/*
 * "Hybrid" multiplication of a constant in GF(256) and a vector in GF(16)
 */
static inline void gf256_gf16_constant_vect_mult_neon(uint8_t a_gf256, const uint8_t *b_gf16, uint8_t *c_gf256, uint32_t n)
{
    gf256_vect_lift_from_gf16_neon(b_gf16, c_gf256, n);
    gf256_constant_vect_mult_neon(a_gf256, c_gf256, c_gf256, n);
    return;
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(16) and a vector in GF(256)
 */
static inline uint8_t gf16_gf256_vect_mult_neon(const uint8_t *a_gf16, const uint8_t *b_gf256, uint32_t len)
{
        uint32_t i;
        uint8x16_t accu, _a, _b;
        uint8x16_t _a_gf16;

        /* Set the accumulator to 0 */
        accu = vdupq_n_u8(0);

        for(i = 0; i < len; i += 16){
                if((len-i) < 16){
                        _a_gf16 = load_incomplete_m128((const uint8_t*)&a_gf16[i/2], (len-i+1)/2);
						_a = vdupq_n_u8(0);
                        gf256_vect_lift_from_gf16_neon((const uint8_t*)&_a_gf16, (uint8_t*)&_a, len-i);
                        _b = load_incomplete_m128((const uint8_t*)&b_gf256[i], len-i);
                }
                else{
                        /* Obvious 256-bit */
                        _a_gf16 = load_incomplete_m128((const uint8_t*)&a_gf16[i/2], 8);
						_a = vdupq_n_u8(0);
                        gf256_vect_lift_from_gf16_neon((const uint8_t*)&_a_gf16, (uint8_t*)&_a, 16);
                        _b = vld1q_u8(&b_gf256[i]);
                }
                /* Multiply in GF(256) */
                accu ^= gf256_mult_vectorized_neon(_a, _b);
        }

        return sum_uint8_neon(accu);
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(256) and a vector in GF(16)
 */
static inline uint8_t gf256_gf16_vect_mult_neon(const uint8_t *a_gf256, const uint8_t *b_gf16, uint32_t n)
{
    return gf16_gf256_vect_mult_neon(b_gf16, a_gf256, n);
}

/*
 * "Hybrid" matrix multiplication of a matrix in GF(256) and a vector in GF(16), resulting
 *  in a vector in GF(256)
 */
static inline void gf256_gf16_mat_mult_neon(const uint8_t *A, const uint8_t *X, uint8_t *Y_inf, uint8_t *Y_sup, uint32_t n)
{
    uint16_t i, j, k;
	uint16_t complete_line = (n + 7) & ~7;
	uint16_t batches, remainder;
	uint32_t row; 
	uint64_t sumx8;
    uint32_t sumx4;
	uint8x16_t accu, _a, _b;
	uint8x16_t acc_inf;
	uint8x8_t aux;

	uint8x16_t extra_accs[4];

	uint8x16_t zeros, mask;

	zeros = vdupq_n_u8(0);

	/* Set the accumulator to 0 */
	batches = n/16;
	remainder = n - 16*batches;

	k=0;
	for (i=0;i<batches;i++) {
		mask = vdupq_n_u8(255);
		for (;k<i*16+16;k+=4) {
			row = k*complete_line;
			extra_accs[0] = vdupq_n_u8(0);
			extra_accs[1] = vdupq_n_u8(0);
			extra_accs[2] = vdupq_n_u8(0);
			extra_accs[3] = vdupq_n_u8(0); 
			for (j=0;j<i;j++) {
				aux = vld1_u8(&X[8*j]);
				_b = vcombine_u8(aux,aux);
				_b = gf256_lift16_from_gf16_neon(_b);
				_a = vld1q_u8(&A[row+16*j]);
				extra_accs[0] ^= gf256_mult_vectorized_neon(_a, _b);
				_a = vld1q_u8(&A[row+complete_line+16*j]);
				extra_accs[1] ^= gf256_mult_vectorized_neon(_a, _b);
				_a = vld1q_u8(&A[row+2*complete_line+16*j]);
				extra_accs[2] ^= gf256_mult_vectorized_neon(_a, _b);
				_a = vld1q_u8(&A[row+3*complete_line+16*j]);
				extra_accs[3] ^= gf256_mult_vectorized_neon(_a, _b);
			}

			aux = vld1_u8(&X[8*j]);
			_b = vcombine_u8(aux,aux);
			_b = gf256_lift16_from_gf16_neon(_b);

			// row k
			_a = vld1q_u8(&A[row+16*i]);
			accu = gf256_mult_vectorized_neon(_a, _b);
			extra_accs[0] = veorq_u8(extra_accs[0], vbicq_u8(accu, mask));
			accu = vandq_u8(mask, accu);
			extra_accs[0] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(extra_accs[0]),vreinterpretq_u64_u8(accu))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(extra_accs[0]),vreinterpretq_u64_u8(accu))));
			
			// row k+1
			mask = vextq_u8(zeros,mask,15);
			_a = vld1q_u8(&A[row+complete_line+16*i]);
			accu = gf256_mult_vectorized_neon(_a, _b);
			extra_accs[1] = veorq_u8(extra_accs[1], vbicq_u8(accu, mask));
			accu = vandq_u8(mask, accu);
			extra_accs[1] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(extra_accs[1]),vreinterpretq_u64_u8(accu))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(extra_accs[1]),vreinterpretq_u64_u8(accu))));

			// row k+2
			mask = vextq_u8(zeros,mask,15);
			_a = vld1q_u8(&A[row+2*complete_line+16*i]);
			accu = gf256_mult_vectorized_neon(_a, _b);
			extra_accs[2] = veorq_u8(extra_accs[2], vbicq_u8(accu, mask));
			accu = vandq_u8(mask, accu);
			extra_accs[2] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(extra_accs[2]),vreinterpretq_u64_u8(accu))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(extra_accs[2]),vreinterpretq_u64_u8(accu))));

			// row k+3
			mask = vextq_u8(zeros,mask,15);
			_a = vld1q_u8(&A[row+3*complete_line+16*i]);
			accu = gf256_mult_vectorized_neon(_a, _b);
			extra_accs[3] = veorq_u8(extra_accs[3], vbicq_u8(accu, mask));
			accu = vandq_u8(mask, accu);
			extra_accs[3] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(extra_accs[3]),vreinterpretq_u64_u8(accu))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(extra_accs[3]),vreinterpretq_u64_u8(accu))));
			
			// k xor k+2
			acc_inf = veorq_u8(vreinterpretq_u8_u32(vtrn1q_u32(vreinterpretq_u32_u8(extra_accs[0]),vreinterpretq_u32_u8(extra_accs[2]))), vreinterpretq_u8_u32(vtrn2q_u32(vreinterpretq_u32_u8(extra_accs[0]),vreinterpretq_u32_u8(extra_accs[2]))));

			// k+1 xor k+3
			extra_accs[0] = veorq_u8(vreinterpretq_u8_u32(vtrn1q_u32(vreinterpretq_u32_u8(extra_accs[1]),vreinterpretq_u32_u8(extra_accs[3]))), vreinterpretq_u8_u32(vtrn2q_u32(vreinterpretq_u32_u8(extra_accs[1]),vreinterpretq_u32_u8(extra_accs[3]))));

			// (k xor k+2) xor (k+1 xor k+3)
			acc_inf = veorq_u8(vreinterpretq_u8_u16(vtrn1q_u16(vreinterpretq_u16_u8(acc_inf),vreinterpretq_u16_u8(extra_accs[0]))), vreinterpretq_u8_u16(vtrn2q_u16(vreinterpretq_u16_u8(acc_inf),vreinterpretq_u16_u8(extra_accs[0])))); 

			extra_accs[0] = vdupq_n_u8(0);
			extra_accs[1] = vdupq_n_u8(0);
			extra_accs[2] = vdupq_n_u8(0);
			extra_accs[3] = vdupq_n_u8(0);
			for (j=i+1;j<batches;j++) {
				aux = vld1_u8(&X[8*j]);
				_b = vcombine_u8(aux,aux);
				_b = gf256_lift16_from_gf16_neon(_b);
				_a = vld1q_u8(&A[row+16*j]);
				extra_accs[0] ^= gf256_mult_vectorized_neon(_a, _b);
				_a = vld1q_u8(&A[row+complete_line+16*j]);
				extra_accs[1] ^= gf256_mult_vectorized_neon(_a, _b);
				_a = vld1q_u8(&A[row+2*complete_line+16*j]);
				extra_accs[2] ^= gf256_mult_vectorized_neon(_a, _b);
				_a = vld1q_u8(&A[row+3*complete_line+16*j]);
				extra_accs[3] ^= gf256_mult_vectorized_neon(_a, _b);
			}
			_b = load_incomplete_m128(&X[8*j],remainder/2);
			_b = gf256_lift16_from_gf16_neon(_b);
			_a = load_incomplete_m128(&A[row+16*batches],remainder);
			extra_accs[0] ^= gf256_mult_vectorized_neon(_a,_b);
			_a = load_incomplete_m128(&A[row+complete_line+16*batches],remainder);
			extra_accs[1] ^= gf256_mult_vectorized_neon(_a,_b);
			_a = load_incomplete_m128(&A[row+2*complete_line+16*batches],remainder);
			extra_accs[2] ^= gf256_mult_vectorized_neon(_a,_b);
			_a = load_incomplete_m128(&A[row+3*complete_line+16*batches],remainder);
			extra_accs[3] ^= gf256_mult_vectorized_neon(_a,_b);
			
			accu = vdupq_n_u8(0);
			extra_accs[0] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(accu),vreinterpretq_u64_u8(extra_accs[0]))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(accu),vreinterpretq_u64_u8(extra_accs[0]))));
			extra_accs[1] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(accu),vreinterpretq_u64_u8(extra_accs[1]))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(accu),vreinterpretq_u64_u8(extra_accs[1]))));
			extra_accs[2] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(accu),vreinterpretq_u64_u8(extra_accs[2]))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(accu),vreinterpretq_u64_u8(extra_accs[2]))));
			extra_accs[3] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(accu),vreinterpretq_u64_u8(extra_accs[3]))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(accu),vreinterpretq_u64_u8(extra_accs[3]))));
			extra_accs[0] = veorq_u8(vreinterpretq_u8_u32(vtrn1q_u32(vreinterpretq_u32_u8(extra_accs[0]),vreinterpretq_u32_u8(extra_accs[2]))), vreinterpretq_u8_u32(vtrn2q_u32(vreinterpretq_u32_u8(extra_accs[0]),vreinterpretq_u32_u8(extra_accs[2]))));
			extra_accs[1] = veorq_u8(vreinterpretq_u8_u32(vtrn1q_u32(vreinterpretq_u32_u8(extra_accs[1]),vreinterpretq_u32_u8(extra_accs[3]))), vreinterpretq_u8_u32(vtrn2q_u32(vreinterpretq_u32_u8(extra_accs[1]),vreinterpretq_u32_u8(extra_accs[3]))));
			extra_accs[0] = veorq_u8(vreinterpretq_u8_u16(vtrn1q_u16(vreinterpretq_u16_u8(extra_accs[0]),vreinterpretq_u16_u8(extra_accs[1]))), vreinterpretq_u8_u16(vtrn2q_u16(vreinterpretq_u16_u8(extra_accs[0]),vreinterpretq_u16_u8(extra_accs[1])))); 
			acc_inf = veorq_u8(acc_inf,extra_accs[0]);
			acc_inf = veorq_u8(vuzp1q_u8(acc_inf,acc_inf),vuzp2q_u8(acc_inf,acc_inf));
			sumx8 = vgetq_lane_u64(vreinterpretq_u64_u8(acc_inf), 0);
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
			mask = vextq_u8(zeros,mask,15);
		}
	}

	mask = vdupq_n_u8(255);
	for (k=16*batches;k<n;k+=4) {
		row = k*complete_line;
		extra_accs[0] = vdupq_n_u8(0);
		extra_accs[1] = vdupq_n_u8(0);
		extra_accs[2] = vdupq_n_u8(0);
		extra_accs[3] = vdupq_n_u8(0);
		for (j=0;j<batches;j++) {
			aux = vld1_u8(&X[8*j]);
			_b = vcombine_u8(aux,aux);
			_b = gf256_lift16_from_gf16_neon(_b);
			_a = vld1q_u8(&A[row+16*j]);
			extra_accs[0] ^= gf256_mult_vectorized_neon(_a, _b);
			_a = vld1q_u8(&A[row+complete_line+16*j]);
			extra_accs[1] ^= gf256_mult_vectorized_neon(_a, _b);
			_a = vld1q_u8(&A[row+2*complete_line+16*j]);
			extra_accs[2] ^= gf256_mult_vectorized_neon(_a, _b);
			_a = vld1q_u8(&A[row+3*complete_line+16*j]);
			extra_accs[3] ^= gf256_mult_vectorized_neon(_a, _b);
		}

		_b = load_incomplete_m128(&X[8*j],remainder/2);
		_b = gf256_lift16_from_gf16_neon(_b);

		// row k
		_a = load_incomplete_m128(&A[row+16*batches],remainder);
		accu = gf256_mult_vectorized_neon(_a, _b);
		extra_accs[0] = veorq_u8(extra_accs[0], vbicq_u8(accu, mask));
		accu = vandq_u8(mask, accu);
		extra_accs[0] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(extra_accs[0]),vreinterpretq_u64_u8(accu))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(extra_accs[0]),vreinterpretq_u64_u8(accu))));
		
		// row k+1
		mask = vextq_u8(zeros,mask,15);
		_a = load_incomplete_m128(&A[row+complete_line+16*batches],remainder);
		accu = gf256_mult_vectorized_neon(_a, _b);
		extra_accs[1] = veorq_u8(extra_accs[1], vbicq_u8(accu, mask));
		accu = vandq_u8(mask, accu);
		extra_accs[1] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(extra_accs[1]),vreinterpretq_u64_u8(accu))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(extra_accs[1]),vreinterpretq_u64_u8(accu))));

		// row k+2
		mask = vextq_u8(zeros,mask,15);
		_a = load_incomplete_m128(&A[row+2*complete_line+16*batches],remainder);
		accu = gf256_mult_vectorized_neon(_a, _b);
		extra_accs[2] = veorq_u8(extra_accs[2], vbicq_u8(accu, mask));
		accu = vandq_u8(mask, accu);
		extra_accs[2] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(extra_accs[2]),vreinterpretq_u64_u8(accu))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(extra_accs[2]),vreinterpretq_u64_u8(accu))));

		// row k+3
		mask = vextq_u8(zeros,mask,15);
		_a = load_incomplete_m128(&A[row+3*complete_line+16*batches],remainder);
		accu = gf256_mult_vectorized_neon(_a, _b);
		extra_accs[3] = veorq_u8(extra_accs[3], vbicq_u8(accu, mask));
		accu = vandq_u8(mask, accu);
		extra_accs[3] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(extra_accs[3]),vreinterpretq_u64_u8(accu))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(extra_accs[3]),vreinterpretq_u64_u8(accu))));
		
		// k xor k+2
		acc_inf = veorq_u8(vreinterpretq_u8_u32(vtrn1q_u32(vreinterpretq_u32_u8(extra_accs[0]),vreinterpretq_u32_u8(extra_accs[2]))), vreinterpretq_u8_u32(vtrn2q_u32(vreinterpretq_u32_u8(extra_accs[0]),vreinterpretq_u32_u8(extra_accs[2]))));

		// k+1 xor k+3
		extra_accs[0] = veorq_u8(vreinterpretq_u8_u32(vtrn1q_u32(vreinterpretq_u32_u8(extra_accs[1]),vreinterpretq_u32_u8(extra_accs[3]))), vreinterpretq_u8_u32(vtrn2q_u32(vreinterpretq_u32_u8(extra_accs[1]),vreinterpretq_u32_u8(extra_accs[3]))));

		// (k xor k+2) xor (k+1 xor k+3)
		acc_inf = veorq_u8(vreinterpretq_u8_u16(vtrn1q_u16(vreinterpretq_u16_u8(acc_inf),vreinterpretq_u16_u8(extra_accs[0]))), vreinterpretq_u8_u16(vtrn2q_u16(vreinterpretq_u16_u8(acc_inf),vreinterpretq_u16_u8(extra_accs[0])))); 

		acc_inf = veorq_u8(vuzp1q_u8(acc_inf,acc_inf),vuzp2q_u8(acc_inf,acc_inf));
		sumx8 = vgetq_lane_u64(vreinterpretq_u64_u8(acc_inf), 0);
		sumx4 = (uint32_t) sumx8;
		memcpy(Y_inf + (k-1), &sumx4, 4);
		sumx4 = (uint32_t) (sumx8 >> 32);
		memcpy(Y_sup + k, &sumx4, 4);
		mask = vextq_u8(zeros,mask,15);
	}
	// End last block

	/*Last element of Y_inf*/
	row = n*complete_line;
	Y_inf[n-1] = gf256_gf16_vect_mult_neon(&A[row],X,n);
}

static inline void gf256_gf16_mat_mult_single_neon(const uint8_t *A, const uint8_t *X, uint8_t *Y, uint32_t n)
{
    GF256_GF16_MAT_MULT(A, X, Y, n, TRI_INF, gf256_gf16_vect_mult_neon);
}

/* === GF(256^2) === */
/* NOTE: for atomic multiplication, using vectorization is suboptimal --OK */
static inline uint16_t gf256to2_mult_neon(uint16_t x, uint16_t y)
{
	return gf256to2_mult_ref(x, y);
}

/* Vectorize multiplication of _a and _b in GF(256^2): the elements in the field are made of
 * 16 bits each in the lanes of the ymm --NOK */
static inline uint8x16_t gf256to2_mult_vectorized_neon(uint8x16_t _a, uint8x16_t _b)
{

		const uint8x16_t const32 = {0x20, 0x00, 0x20, 0x00, 0x20, 0x00, 0x20, 0x00, 0x20, 0x00, 0x20, 0x00, 0x20, 0x00, 0x20, 0x00};

	const uint8x16_t mask_c1 = {0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF};
        const uint8x16_t zero = vdupq_n_u8(0);

	uint8x16_t ab = gf256_mult_vectorized_neon(_a, _b);
	uint8x16_t a0b0 = vuzp1q_u8(ab, ab);
	a0b0 = vzip1q_u8(a0b0,a0b0);
	uint8x16_t a1b1 = vuzp2q_u8(ab, ab);
	a1b1 = vzip1q_u8(a1b1,a1b1);
	uint8x16_t a1b1_32 = gf256_mult_vectorized_neon(a1b1, const32);
	/* */
	uint8x16_t a0_xor_a1 = veorq_u8(_a, vrev16q_u8(_a));
	uint8x16_t b0_xor_b1 = vbslq_u8(mask_c1, veorq_u8(_b, vrev16q_u8(_b)), zero);
	uint8x16_t mult_ab_xor = gf256_mult_vectorized_neon(a0_xor_a1, b0_xor_b1);
	
	/* Compute the result */
	uint8x16_t res = veorq_u8(veorq_u8(a0b0, a1b1_32), mult_ab_xor);

        return res;
}

/*
 * Vector multiplied by a constant in GF(256^2). --OK
 */
static inline void gf256to2_constant_vect_mult_neon(uint16_t b, const uint16_t *a, uint16_t *c, uint32_t len)
{
	uint32_t i;
	uint8x16_t _a, _b;

	/* Load the constant byte b broadcasted in _b */
	_b = vreinterpretq_u8_u16(vdupq_n_u16(b));

        for(i = 0; i < (2 * len); i += 16){
                if(((2 * len)-i) < 16){
                        _a = load_incomplete_m128((const uint8_t*)&a[i / 2], ((2 * len) - i));
                        /* Vectorized multiplication in GF(256) */
                        store_incomplete_m128(gf256to2_mult_vectorized_neon(_a, _b), (uint8_t*)&c[i / 2], (2 * len)-i);
                }
                else{
                        /* Obvious 512-bit */
						_a = vreinterpretq_u8_u16(vld1q_u16(&a[i / 2]));
                        /* Vectorized multiplication in GF(256) */
                        vst1q_u8((uint8_t*)&c[i / 2], gf256to2_mult_vectorized_neon(_a, _b));
                }       
        }

        return;
}

/* Perform a multiplication in GF(256^2) of elements in vectors a an b --OK */
static inline uint16_t gf256to2_vect_mult_neon(const uint16_t *a, const uint16_t *b, uint32_t len)
{
        uint32_t i;
        uint8x16_t accu, _a, _b;

        /* Set the accumulator to 0 */
        accu = vdupq_n_u8(0);

        for(i = 0; i < (2 * len); i += 16){
                if(((2 * len)-i) < 16){
						_a = load_incomplete_m128((const uint8_t*)&a[i / 2], ((2 * len) - i));
                        _b = load_incomplete_m128((const uint8_t*)&b[i / 2], ((2 * len) - i));
                }
                else{
                        /* Obvious 256-bit */
						_a = vld1q_u8((const uint8_t*)&a[i/2]);
						_b = vld1q_u8((const uint8_t*)&b[i/2]);
                }
                accu ^= gf256to2_mult_vectorized_neon(_a, _b);
        }

        return sum_uint16_neon(accu);
}

/*
 * GF(2^16) matrix multiplication 
 */
static inline void gf256to2_mat_mult_neon(const uint16_t *A, const uint16_t *X, uint16_t *Y_inf, uint16_t *Y_sup, uint32_t n)
{
    uint16_t i, j, k;
	uint16_t complete_line = (n + 7) & ~7;
	uint16_t batches, remainder;
	uint32_t row; 
	uint64_t sumx4;
	uint16x8_t accu, _a, _b;
	uint16x8_t acc_inf;

	uint16x8_t extra_accs[4];

	uint16x8_t zeros, mask;

	zeros = vdupq_n_u16(0);

	/* Set the accumulator to 0 */
	batches = n/8;
	remainder = n - batches * 8;

	k=0;
	for (i=0;i<batches;i++) {
		mask = vdupq_n_u16(-1);
		for (;k<i*8+8;k+=4) {
			row = k*complete_line;
			extra_accs[0] = vdupq_n_u16(0);
			extra_accs[1] = vdupq_n_u16(0);
			extra_accs[2] = vdupq_n_u16(0);
			extra_accs[3] = vdupq_n_u16(0); 
			for (j=0;j<i;j++) {
				_b = vld1q_u16(&X[8*j]);
				_a = vld1q_u16(&A[row+8*j]);
				extra_accs[0] ^= vreinterpretq_u16_u8(gf256to2_mult_vectorized_neon(vreinterpretq_u8_u16(_a),vreinterpretq_u8_u16(_b)));
				_a = vld1q_u16(&A[row+complete_line+8*j]);
				extra_accs[1] ^= vreinterpretq_u16_u8(gf256to2_mult_vectorized_neon(vreinterpretq_u8_u16(_a),vreinterpretq_u8_u16(_b)));
				_a = vld1q_u16(&A[row+2*complete_line+8*j]);
				extra_accs[2] ^= vreinterpretq_u16_u8(gf256to2_mult_vectorized_neon(vreinterpretq_u8_u16(_a),vreinterpretq_u8_u16(_b)));
				_a = vld1q_u16(&A[row+3*complete_line+8*j]);
				extra_accs[3] ^= vreinterpretq_u16_u8(gf256to2_mult_vectorized_neon(vreinterpretq_u8_u16(_a),vreinterpretq_u8_u16(_b)));
			}

			_b = vld1q_u16(&X[8*i]);

			// row k
			_a = vld1q_u16(&A[row+8*i]);
			accu = vreinterpretq_u16_u8(gf256to2_mult_vectorized_neon(vreinterpretq_u8_u16(_a),vreinterpretq_u8_u16(_b)));
			extra_accs[0] = veorq_u16(extra_accs[0], vbicq_u16(accu, mask));
			accu = vandq_u16(mask, accu);
			extra_accs[0] = veorq_u16(vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u16(extra_accs[0]),vreinterpretq_u64_u16(accu))), vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u16(extra_accs[0]),vreinterpretq_u64_u16(accu))));
			
			// row k+1
			mask = vextq_u16(zeros,mask,7);
			_a = vld1q_u16(&A[row+complete_line+8*i]);
			accu = vreinterpretq_u16_u8(gf256to2_mult_vectorized_neon(vreinterpretq_u8_u16(_a),vreinterpretq_u8_u16(_b)));
			extra_accs[1] = veorq_u16(extra_accs[1], vbicq_u16(accu, mask));
			accu = vandq_u16(mask, accu);
			extra_accs[1] = veorq_u16(vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u16(extra_accs[1]),vreinterpretq_u64_u16(accu))), vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u16(extra_accs[1]),vreinterpretq_u64_u16(accu))));

			// row k+2
			mask = vextq_u16(zeros,mask,7);
			_a = vld1q_u16(&A[row+2*complete_line+8*i]);
			accu = vreinterpretq_u16_u8(gf256to2_mult_vectorized_neon(vreinterpretq_u8_u16(_a),vreinterpretq_u8_u16(_b)));
			extra_accs[2] = veorq_u16(extra_accs[2], vbicq_u16(accu, mask));
			accu = vandq_u16(mask, accu);
			extra_accs[2] = veorq_u16(vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u16(extra_accs[2]),vreinterpretq_u64_u16(accu))), vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u16(extra_accs[2]),vreinterpretq_u64_u16(accu))));

			// row k+3
			mask = vextq_u16(zeros,mask,7);
			_a = vld1q_u16(&A[row+3*complete_line+8*i]);
			accu = vreinterpretq_u16_u8(gf256to2_mult_vectorized_neon(vreinterpretq_u8_u16(_a),vreinterpretq_u8_u16(_b)));
			extra_accs[3] = veorq_u16(extra_accs[3], vbicq_u16(accu, mask));
			accu = vandq_u16(mask, accu);
			extra_accs[3] = veorq_u16(vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u16(extra_accs[3]),vreinterpretq_u64_u16(accu))), vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u16(extra_accs[3]),vreinterpretq_u64_u16(accu))));
			
			// k xor k+2
			acc_inf = veorq_u16(vreinterpretq_u16_u32(vtrn1q_u32(vreinterpretq_u32_u16(extra_accs[0]),vreinterpretq_u32_u16(extra_accs[2]))), vreinterpretq_u16_u32(vtrn2q_u32(vreinterpretq_u32_u16(extra_accs[0]),vreinterpretq_u32_u16(extra_accs[2]))));

			// k+1 xor k+3
			extra_accs[0] = veorq_u16(vreinterpretq_u16_u32(vtrn1q_u32(vreinterpretq_u32_u16(extra_accs[1]),vreinterpretq_u32_u16(extra_accs[3]))), vreinterpretq_u16_u32(vtrn2q_u32(vreinterpretq_u32_u16(extra_accs[1]),vreinterpretq_u32_u16(extra_accs[3]))));

			// (k xor k+2) xor (k+1 xor k+3)
			acc_inf = veorq_u16(vtrn1q_u16(acc_inf,extra_accs[0]), vtrn2q_u16(acc_inf,extra_accs[0])); 

			extra_accs[0] = vdupq_n_u16(0);
			extra_accs[1] = vdupq_n_u16(0);
			extra_accs[2] = vdupq_n_u16(0);
			extra_accs[3] = vdupq_n_u16(0);
			for (j=i+1;j<batches;j++) {
				_b = vld1q_u16(&X[8*j]);
				_a = vld1q_u16(&A[row+8*j]);
				extra_accs[0] ^= vreinterpretq_u16_u8(gf256to2_mult_vectorized_neon(vreinterpretq_u8_u16(_a),vreinterpretq_u8_u16(_b)));
				_a = vld1q_u16(&A[row+complete_line+8*j]);
				extra_accs[1] ^= vreinterpretq_u16_u8(gf256to2_mult_vectorized_neon(vreinterpretq_u8_u16(_a),vreinterpretq_u8_u16(_b)));
				_a = vld1q_u16(&A[row+2*complete_line+8*j]);
				extra_accs[2] ^= vreinterpretq_u16_u8(gf256to2_mult_vectorized_neon(vreinterpretq_u8_u16(_a),vreinterpretq_u8_u16(_b)));
				_a = vld1q_u16(&A[row+3*complete_line+8*j]);
				extra_accs[3] ^= vreinterpretq_u16_u8(gf256to2_mult_vectorized_neon(vreinterpretq_u8_u16(_a),vreinterpretq_u8_u16(_b)));
			}
			if (remainder > 0) {
			_b = vreinterpretq_u16_u8(load_incomplete_m128((uint8_t *) &X[8*j],2*remainder));
			_a = vreinterpretq_u16_u8(load_incomplete_m128((uint8_t *)&A[row+8*batches],2*remainder));
			extra_accs[0] ^= vreinterpretq_u16_u8(gf256to2_mult_vectorized_neon(vreinterpretq_u8_u16(_a),vreinterpretq_u8_u16(_b)));
			_a = vreinterpretq_u16_u8(load_incomplete_m128((uint8_t *)&A[row+complete_line+8*batches],2*remainder));
			extra_accs[1] ^= vreinterpretq_u16_u8(gf256to2_mult_vectorized_neon(vreinterpretq_u8_u16(_a),vreinterpretq_u8_u16(_b)));
			_a = vreinterpretq_u16_u8(load_incomplete_m128((uint8_t *)&A[row+2*complete_line+8*batches],2*remainder));
			extra_accs[2] ^= vreinterpretq_u16_u8(gf256to2_mult_vectorized_neon(vreinterpretq_u8_u16(_a),vreinterpretq_u8_u16(_b)));
			_a = vreinterpretq_u16_u8(load_incomplete_m128((uint8_t *)&A[row+3*complete_line+8*batches],2*remainder));
			extra_accs[3] ^= vreinterpretq_u16_u8(gf256to2_mult_vectorized_neon(vreinterpretq_u8_u16(_a),vreinterpretq_u8_u16(_b)));
			}

			accu = vdupq_n_u16(0);
			extra_accs[0] = veorq_u16(vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u16(accu),vreinterpretq_u64_u16(extra_accs[0]))), vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u16(accu),vreinterpretq_u64_u16(extra_accs[0]))));
			extra_accs[1] = veorq_u16(vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u16(accu),vreinterpretq_u64_u16(extra_accs[1]))), vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u16(accu),vreinterpretq_u64_u16(extra_accs[1]))));
			extra_accs[2] = veorq_u16(vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u16(accu),vreinterpretq_u64_u16(extra_accs[2]))), vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u16(accu),vreinterpretq_u64_u16(extra_accs[2]))));
			extra_accs[3] = veorq_u16(vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u16(accu),vreinterpretq_u64_u16(extra_accs[3]))), vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u16(accu),vreinterpretq_u64_u16(extra_accs[3]))));
			extra_accs[0] = veorq_u16(vreinterpretq_u16_u32(vtrn1q_u32(vreinterpretq_u32_u16(extra_accs[0]),vreinterpretq_u32_u16(extra_accs[2]))), vreinterpretq_u16_u32(vtrn2q_u32(vreinterpretq_u32_u16(extra_accs[0]),vreinterpretq_u32_u16(extra_accs[2]))));
			extra_accs[1] = veorq_u16(vreinterpretq_u16_u32(vtrn1q_u32(vreinterpretq_u32_u16(extra_accs[1]),vreinterpretq_u32_u16(extra_accs[3]))), vreinterpretq_u16_u32(vtrn2q_u32(vreinterpretq_u32_u16(extra_accs[1]),vreinterpretq_u32_u16(extra_accs[3]))));
			extra_accs[0] = veorq_u16(vtrn1q_u16(extra_accs[0],extra_accs[1]), vtrn2q_u16(extra_accs[0],extra_accs[1])); 
			acc_inf = veorq_u16(acc_inf,extra_accs[0]);
			sumx4 = vgetq_lane_u64(vreinterpretq_u64_u16(acc_inf), 0);
			if (k == 0) {      
					sumx4 = sumx4 >> 16;
					memcpy(Y_inf, &sumx4, 6);
			}
			else {
					memcpy(Y_inf + (k-1), &sumx4, 8);
			}
			sumx4 = vgetq_lane_u64(vreinterpretq_u64_u16(acc_inf), 1);
			memcpy(Y_sup + k, &sumx4, 8);
			mask = vextq_u16(zeros,mask,7);
		}
	}

	if (remainder > 0) {
		mask = vdupq_n_u16(-1);
		for (k=8*batches;k<n;k+=4) {
			row = k*complete_line;
			extra_accs[0] = vdupq_n_u16(0);
			extra_accs[1] = vdupq_n_u16(0);
			extra_accs[2] = vdupq_n_u16(0);
			extra_accs[3] = vdupq_n_u16(0); 
			for (j=0;j<batches;j++) {
				_b = vld1q_u16(&X[8*j]);
				_a = vld1q_u16(&A[row+8*j]);
				extra_accs[0] ^= vreinterpretq_u16_u8(gf256to2_mult_vectorized_neon(vreinterpretq_u8_u16(_a),vreinterpretq_u8_u16(_b)));
				_a = vld1q_u16(&A[row+complete_line+8*j]);
				extra_accs[1] ^= vreinterpretq_u16_u8(gf256to2_mult_vectorized_neon(vreinterpretq_u8_u16(_a),vreinterpretq_u8_u16(_b)));
				_a = vld1q_u16(&A[row+2*complete_line+8*j]);
				extra_accs[2] ^= vreinterpretq_u16_u8(gf256to2_mult_vectorized_neon(vreinterpretq_u8_u16(_a),vreinterpretq_u8_u16(_b)));
				_a = vld1q_u16(&A[row+3*complete_line+8*j]);
				extra_accs[3] ^= vreinterpretq_u16_u8(gf256to2_mult_vectorized_neon(vreinterpretq_u8_u16(_a),vreinterpretq_u8_u16(_b)));
			
			}

			_b = vreinterpretq_u16_u8(load_incomplete_m128((uint8_t *)&X[8*batches],2*remainder));

			// row k
			_a = vreinterpretq_u16_u8(load_incomplete_m128((uint8_t *)&A[row+8*batches],2*remainder));
			accu = vreinterpretq_u16_u8(gf256to2_mult_vectorized_neon(vreinterpretq_u8_u16(_a),vreinterpretq_u8_u16(_b)));
			extra_accs[0] = veorq_u16(extra_accs[0], vbicq_u16(accu, mask));
			accu = vandq_u16(mask, accu);
			extra_accs[0] = veorq_u16(vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u16(extra_accs[0]),vreinterpretq_u64_u16(accu))), vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u16(extra_accs[0]),vreinterpretq_u64_u16(accu))));
			
			// row k+1
			mask = vextq_u16(zeros,mask,7);
			_a = vreinterpretq_u16_u8(load_incomplete_m128((uint8_t *)&A[row+complete_line+8*batches],2*remainder));
			accu = vreinterpretq_u16_u8(gf256to2_mult_vectorized_neon(vreinterpretq_u8_u16(_a),vreinterpretq_u8_u16(_b)));
			extra_accs[1] = veorq_u16(extra_accs[1], vbicq_u16(accu, mask));
			accu = vandq_u16(mask, accu);
			extra_accs[1] = veorq_u16(vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u16(extra_accs[1]),vreinterpretq_u64_u16(accu))), vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u16(extra_accs[1]),vreinterpretq_u64_u16(accu))));

			// row k+2
			mask = vextq_u16(zeros,mask,7);
			_a = vreinterpretq_u16_u8(load_incomplete_m128((uint8_t *)&A[row+2*complete_line+8*batches],2*remainder));
			accu = vreinterpretq_u16_u8(gf256to2_mult_vectorized_neon(vreinterpretq_u8_u16(_a),vreinterpretq_u8_u16(_b)));
			extra_accs[2] = veorq_u16(extra_accs[2], vbicq_u16(accu, mask));
			accu = vandq_u16(mask, accu);
			extra_accs[2] = veorq_u16(vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u16(extra_accs[2]),vreinterpretq_u64_u16(accu))), vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u16(extra_accs[2]),vreinterpretq_u64_u16(accu))));

			// row k+3
			mask = vextq_u16(zeros,mask,7);
			_a = vreinterpretq_u16_u8(load_incomplete_m128((uint8_t *)&A[row+3*complete_line+8*batches],2*remainder));
			accu = vreinterpretq_u16_u8(gf256to2_mult_vectorized_neon(vreinterpretq_u8_u16(_a),vreinterpretq_u8_u16(_b)));
			extra_accs[3] = veorq_u16(extra_accs[3], vbicq_u16(accu, mask));
			accu = vandq_u16(mask, accu);
			extra_accs[3] = veorq_u16(vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u16(extra_accs[3]),vreinterpretq_u64_u16(accu))), vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u16(extra_accs[3]),vreinterpretq_u64_u16(accu))));
			
			// k xor k+2
			acc_inf = veorq_u16(vreinterpretq_u16_u32(vtrn1q_u32(vreinterpretq_u32_u16(extra_accs[0]),vreinterpretq_u32_u16(extra_accs[2]))), vreinterpretq_u16_u32(vtrn2q_u32(vreinterpretq_u32_u16(extra_accs[0]),vreinterpretq_u32_u16(extra_accs[2]))));

			// k+1 xor k+3
			extra_accs[0] = veorq_u16(vreinterpretq_u16_u32(vtrn1q_u32(vreinterpretq_u32_u16(extra_accs[1]),vreinterpretq_u32_u16(extra_accs[3]))), vreinterpretq_u16_u32(vtrn2q_u32(vreinterpretq_u32_u16(extra_accs[1]),vreinterpretq_u32_u16(extra_accs[3]))));

			// (k xor k+2) xor (k+1 xor k+3)
			acc_inf = veorq_u16(vtrn1q_u16(acc_inf,extra_accs[0]), vtrn2q_u16(acc_inf,extra_accs[0])); 

			sumx4 = vgetq_lane_u64(vreinterpretq_u64_u16(acc_inf), 0);
			memcpy(Y_inf + (k-1), &sumx4, 8);
			sumx4 = vgetq_lane_u64(vreinterpretq_u64_u16(acc_inf), 1);
			memcpy(Y_sup + k, &sumx4, 8);
			mask = vextq_u16(zeros,mask,7);
		}
	// End last block
	}

	/*Last element of Y_inf*/
	row = n*complete_line;
	Y_inf[n-1] = gf256to2_vect_mult_neon(X,&A[row],n);
}

static inline void gf256to2_mat_mult_single_neon(const uint16_t *A, const uint16_t *X, uint16_t *Y, uint32_t n)
{
        GF256to2_MAT_MULT(A, X, Y, n, TRI_INF, gf256to2_vect_mult_neon);
}

/*
 * "Hybrid" constant multiplication of a constant in GF(2) and a vector in GF(256^2) 
 */
static inline void gf2_gf256to2_constant_vect_mult_neon(uint8_t a_gf2, const uint16_t *b_gf256to2, uint16_t *c_gf256to2, uint32_t n)
{
	gf2_gf256to2_constant_vect_mult_ref(a_gf2, b_gf256to2, c_gf256to2, n);

        return;
}

/*
 * "Hybrid" constant multiplication of a constant in GF(256^2) and a vector in GF(2)
 */
static inline void gf256to2_gf2_constant_vect_mult_neon(uint16_t a_gf256to2, const uint8_t *b_gf2, uint16_t *c_gf256to2, uint32_t len)
{
       uint32_t i;
        const uint8x16_t zero = vdupq_n_u8(0);
	uint8x16_t _a, _b;

        /* Broadcast the constant value */
	_a = vreinterpretq_u8_u16(vdupq_n_u16(a_gf256to2));

        for(i = 0; i < (2*len); i += 16){
		if(((2*len) - i) < 16){
			_b = vdupq_n_u8(b_gf2[i / 16]);
		}
		else{
			/* Obvious 256-bit */
			_b = vdupq_n_u8(b_gf2[i / 16]);
		}
		/* Create a selection mask from the bits in _a */
		const uint8x16_t and_msk = {0b00000001, 0b00000001, 0b00000010, 0b00000010, 0b00000100, 0b00000100, 0b00001000, 0b00001000,
								0b00010000, 0b00010000, 0b00100000, 0b00100000, 0b01000000, 0b01000000, 0b10000000, 0b10000000};
		/* Only keep the selection bits */
			_b = vandq_u8(_b,and_msk);
			/* Transform these bits to either 0 or 0xFF and keep 0x80 */
			_b = vcgtq_u8(_b,zero);
			/* Use blending for bytes selection */
					uint8x16_t _c = vbslq_u8(_b, _a, zero);
			/* Store the result */

			vst1q_u16(&c_gf256to2[i/2],vreinterpretq_u16_u8(_c));
        }
        return;
}

/*
 * "Hybrid" constant multiplication of a constant in GF(256^2) and a vector in GF(256)
 */
static inline void gf256_gf256to2_constant_vect_mult_neon(uint8_t a_gf256, const uint16_t *b_gf256to2, uint16_t *c_gf256to2, uint32_t n)
{
	gf256_gf256to2_constant_vect_mult_ref(a_gf256, b_gf256to2, c_gf256to2, n);

	return;
}

/*
 * "Hybrid" constant multiplication of a constant in GF(256^2) and a vector in GF(256) --NOK
 */
static inline void gf256to2_gf256_constant_vect_mult_neon(uint16_t a_gf256to2, const uint8_t *b_gf256, uint16_t *c_gf256to2, uint32_t len)
{
	uint32_t i;
	uint8x16_t _a, _b;

	/* Load the constant byte b broadcasted in _b */
	_a = vreinterpretq_u8_u16(vdupq_n_u16(a_gf256to2));

        for(i = 0; i < len; i += 8){
		uint32_t to_load = (len - i) < 8 ? (len - i) : 8;
                _b = load_incomplete_m128((const uint8_t*)&b_gf256[i], to_load);
		/* Copy in the two lanes */
		_b = vzip1q_u8(_b,_b);
                /* Vectorized multiplication in GF(256) */
                store_incomplete_m128(gf256_mult_vectorized_neon(_a, _b), (uint8_t*)&c_gf256to2[i], 2 * to_load);
        }

        return;
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(2) and a vector in GF(256^2) --NOK
 */
static inline uint16_t gf2_gf256to2_vect_mult_neon(const uint8_t *a_gf2, const uint16_t *b_gf256to2, uint32_t len)
{
	    uint32_t i;
        const uint8x16_t zero = vdupq_n_u8(0);
	uint8x16_t _a, _b;

        /* Set the accumulator to 0 */
        uint8x16_t accu = vdupq_n_u8(0);

        for(i = 0; i < (2*len); i += 16){
		if(((2*len) - i) < 16){
			_a = vdupq_n_u8(a_gf2[i / 16]);
			_b = load_incomplete_m128((const uint8_t*)&b_gf256to2[i / 2], (2*len) - i);
		}
		else{
			/* Obvious 256-bit */
			_a = vdupq_n_u8(a_gf2[i / 16]);
			_b = vreinterpretq_u8_u16(vld1q_u16(&b_gf256to2[i / 2]));
		}
		/* Create a selection mask from the bits in _a */
		const uint8x16_t and_msk = {0b00000001, 0b00000001, 0b00000010, 0b00000010, 0b00000100, 0b00000100, 0b00001000, 0b00001000,
								0b00010000, 0b00010000, 0b00100000, 0b00100000, 0b01000000, 0b01000000, 0b10000000, 0b10000000};
		_a = vandq_u8(_a,and_msk);
		/* Transform these bits to either 0 or 0xFF and keep 0x80 */
		_a = vcgtq_u8(_a,zero);
		/* Use blending for bytes selection */
                accu ^= vbslq_u8(_a, _b, zero);
        }
        return sum_uint16_neon(accu);
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(256^2) and a vector in GF(256) --OK
 */
static inline uint16_t gf256to2_gf2_vect_mult_neon(const uint16_t *a_gf256to2, const uint8_t *b_gf2, uint32_t n)
{
        return gf2_gf256to2_vect_mult_neon(b_gf2, a_gf256to2, n);
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(256) and a vector in GF(256^2) --NOK
 */
static inline uint16_t gf256_gf256to2_vect_mult_neon(const uint8_t *a_gf256, const uint16_t *b_gf256to2, uint32_t len)
{
        /* Note: the multiplication of an element in GF(256) and an element in GF(256^2)
         * simply consists in two multiplications in GF(256) (this is multiplying a constant by a degree 1 polynomial)
         * */
		uint32_t i;
        uint8x16_t accu, _a, _b;

        /* Set the accumulator to 0 */
        accu = vdupq_n_u8(0);

        for(i = 0; i < (2 * len); i += 16){
                if(((2 * len)-i) < 16){
                        _a = load_incomplete_m128((const uint8_t*)&a_gf256[i / 2], ((2 * len)-i) / 2);
                        _b = load_incomplete_m128((const uint8_t*)&b_gf256to2[i / 2], (2 * len)-i);
                }
                else{
                        /* Obvious 256-bit */
                        _a = load_incomplete_m128((const uint8_t*)&a_gf256[i / 2], 8);
                        _b = vreinterpretq_u8_u16(vld1q_u16(&b_gf256to2[i / 2]));
                }
				_a = vzip1q_u8(_a,_a);
                /* Multiply in GF(256) */
                accu ^= gf256_mult_vectorized_neon(_a, _b);
        }

        return sum_uint16_neon(accu);
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(256^2) and a vector in GF(256) 
 */
static inline uint16_t gf256to2_gf256_vect_mult_neon(const uint16_t *a_gf256to2, const uint8_t *b_gf256, uint32_t n)
{
        return gf256_gf256to2_vect_mult_neon(b_gf256, a_gf256to2, n);
}

/*
 * "Hybrid" matrix multiplication of a matrix in GF(2) and a vector in GF(256^2), resulting
 *  in a vector in GF(256^2) 
 */
static inline void gf2_gf256to2_mat_mult_neon(const uint8_t *A, const uint16_t *X, uint16_t *Y, uint32_t n, matrix_type mtype)
{
        GF2_GF256to2_MAT_MULT(A, X, Y, n, mtype, gf2_gf256to2_vect_mult_neon);
}

/*
 * "Hybrid" matrix multiplication of a matrix in GF(256^2) and a vector in GF(2), resulting
 *  in a vector in GF(256^2) 
 */
static inline void gf256to2_gf2_mat_mult_neon(const uint16_t *A, const uint8_t *X, uint16_t *Y_inf, uint16_t *Y_sup, uint32_t n)
{
    uint16_t i, j, k;
	uint16_t complete_line = (n + 7) & ~7;
	uint16_t batches;
	uint32_t row;
	uint64_t sumx4;
	uint16x8_t _a;
	uint16x8_t accu, acc_inf, prod;
	const uint16x8_t and_msk = {0b00000001, 0b00000010, 0b00000100, 0b00001000, 0b00010000, 0b00100000, 0b01000000, 0b10000000};
	uint16_t _b;

	uint16x8_t zeros, mask;

	uint16x8_t extra_accs[4];

	zeros = vdupq_n_u16(0);

	batches = n/8;

	k=0;
	for (i=0;i<batches;i++) {
		mask = vdupq_n_u16(-1);
		for (;k<i*8+8;k+=4) {
			row = k*complete_line;
			extra_accs[0] = vdupq_n_u16(0);
			extra_accs[1] = vdupq_n_u16(0);
			extra_accs[2] = vdupq_n_u16(0);
			extra_accs[3] = vdupq_n_u16(0); 
			for (j=0;j<i;j++) {
				_b = X[j];
				prod = vdupq_n_u16(_b);
				prod = vandq_u16(prod,and_msk);
				prod = vcgtq_u16(prod,zeros);
				_a = vld1q_u16(&A[row+8*j]);
				extra_accs[0] ^= vbslq_u16(prod, _a, zeros);
				_a = vld1q_u16(&A[row+complete_line+8*j]);
				extra_accs[1] ^= vbslq_u16(prod, _a, zeros);
				_a = vld1q_u16(&A[row+2*complete_line+8*j]);
				extra_accs[2] ^= vbslq_u16(prod, _a, zeros);
				_a = vld1q_u16(&A[row+3*complete_line+8*j]);
				extra_accs[3] ^= vbslq_u16(prod, _a, zeros);
			}

			_b = X[i];
			prod = vdupq_n_u16(_b);
			prod = vandq_u16(prod,and_msk);
			prod = vcgtq_u16(prod,zeros);

			// row k
			_a = vld1q_u16(&A[row+8*i]);
			accu = vbslq_u16(prod, _a, zeros);
			extra_accs[0] = veorq_u16(extra_accs[0], vbicq_u16(accu, mask));
			accu = vandq_u16(mask, accu);
			extra_accs[0] = veorq_u16(vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u16(extra_accs[0]),vreinterpretq_u64_u16(accu))), vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u16(extra_accs[0]),vreinterpretq_u64_u16(accu))));
			
			// row k+1
			mask = vextq_u16(zeros,mask,7);
			_a = vld1q_u16(&A[row+complete_line+8*i]);
			accu = vbslq_u16(prod, _a, zeros);
			extra_accs[1] = veorq_u16(extra_accs[1], vbicq_u16(accu, mask));
			accu = vandq_u16(mask, accu);
			extra_accs[1] = veorq_u16(vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u16(extra_accs[1]),vreinterpretq_u64_u16(accu))), vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u16(extra_accs[1]),vreinterpretq_u64_u16(accu))));

			// row k+2
			mask = vextq_u16(zeros,mask,7);
			_a = vld1q_u16(&A[row+2*complete_line+8*i]);
			accu = vbslq_u16(prod, _a, zeros);
			extra_accs[2] = veorq_u16(extra_accs[2], vbicq_u16(accu, mask));
			accu = vandq_u16(mask, accu);
			extra_accs[2] = veorq_u16(vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u16(extra_accs[2]),vreinterpretq_u64_u16(accu))), vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u16(extra_accs[2]),vreinterpretq_u64_u16(accu))));

			// row k+3
			mask = vextq_u16(zeros,mask,7);
			_a = vld1q_u16(&A[row+3*complete_line+8*i]);
			accu = vbslq_u16(prod, _a, zeros);
			extra_accs[3] = veorq_u16(extra_accs[3], vbicq_u16(accu, mask));
			accu = vandq_u16(mask, accu);
			extra_accs[3] = veorq_u16(vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u16(extra_accs[3]),vreinterpretq_u64_u16(accu))), vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u16(extra_accs[3]),vreinterpretq_u64_u16(accu))));
			
			// k xor k+2
			acc_inf = veorq_u16(vreinterpretq_u16_u32(vtrn1q_u32(vreinterpretq_u32_u16(extra_accs[0]),vreinterpretq_u32_u16(extra_accs[2]))), vreinterpretq_u16_u32(vtrn2q_u32(vreinterpretq_u32_u16(extra_accs[0]),vreinterpretq_u32_u16(extra_accs[2]))));

			// k+1 xor k+3
			extra_accs[0] = veorq_u16(vreinterpretq_u16_u32(vtrn1q_u32(vreinterpretq_u32_u16(extra_accs[1]),vreinterpretq_u32_u16(extra_accs[3]))), vreinterpretq_u16_u32(vtrn2q_u32(vreinterpretq_u32_u16(extra_accs[1]),vreinterpretq_u32_u16(extra_accs[3]))));

			// (k xor k+2) xor (k+1 xor k+3)
			acc_inf = veorq_u16(vtrn1q_u16(acc_inf,extra_accs[0]), vtrn2q_u16(acc_inf,extra_accs[0])); 

			extra_accs[0] = vdupq_n_u16(0);
			extra_accs[1] = vdupq_n_u16(0);
			extra_accs[2] = vdupq_n_u16(0);
			extra_accs[3] = vdupq_n_u16(0);
			for (j=i+1;j<batches;j++) {
				_b = X[j];
				prod = vdupq_n_u16(_b);
				prod = vandq_u16(prod,and_msk);
				prod = vcgtq_u16(prod,zeros);
				_a = vld1q_u16(&A[row+8*j]);
				extra_accs[0] ^= vbslq_u16(prod, _a, zeros);
				_a = vld1q_u16(&A[row+complete_line+8*j]);
				extra_accs[1] ^= vbslq_u16(prod, _a, zeros);
				_a = vld1q_u16(&A[row+2*complete_line+8*j]);
				extra_accs[2] ^= vbslq_u16(prod, _a, zeros);
				_a = vld1q_u16(&A[row+3*complete_line+8*j]);
				extra_accs[3] ^= vbslq_u16(prod, _a, zeros);
			}
			accu = vdupq_n_u16(0);
			extra_accs[0] = veorq_u16(vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u16(accu),vreinterpretq_u64_u16(extra_accs[0]))), vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u16(accu),vreinterpretq_u64_u16(extra_accs[0]))));
			extra_accs[1] = veorq_u16(vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u16(accu),vreinterpretq_u64_u16(extra_accs[1]))), vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u16(accu),vreinterpretq_u64_u16(extra_accs[1]))));
			extra_accs[2] = veorq_u16(vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u16(accu),vreinterpretq_u64_u16(extra_accs[2]))), vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u16(accu),vreinterpretq_u64_u16(extra_accs[2]))));
			extra_accs[3] = veorq_u16(vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u16(accu),vreinterpretq_u64_u16(extra_accs[3]))), vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u16(accu),vreinterpretq_u64_u16(extra_accs[3]))));
			extra_accs[0] = veorq_u16(vreinterpretq_u16_u32(vtrn1q_u32(vreinterpretq_u32_u16(extra_accs[0]),vreinterpretq_u32_u16(extra_accs[2]))), vreinterpretq_u16_u32(vtrn2q_u32(vreinterpretq_u32_u16(extra_accs[0]),vreinterpretq_u32_u16(extra_accs[2]))));
			extra_accs[1] = veorq_u16(vreinterpretq_u16_u32(vtrn1q_u32(vreinterpretq_u32_u16(extra_accs[1]),vreinterpretq_u32_u16(extra_accs[3]))), vreinterpretq_u16_u32(vtrn2q_u32(vreinterpretq_u32_u16(extra_accs[1]),vreinterpretq_u32_u16(extra_accs[3]))));
			extra_accs[0] = veorq_u16(vtrn1q_u16(extra_accs[0],extra_accs[1]), vtrn2q_u16(extra_accs[0],extra_accs[1])); 
			acc_inf = veorq_u16(acc_inf,extra_accs[0]);
			sumx4 = vgetq_lane_u64(vreinterpretq_u64_u16(acc_inf), 0);
			if (k == 0) {      
					sumx4 = sumx4 >> 16;
					memcpy(Y_inf, &sumx4, 6);
			}
			else {
					memcpy(Y_inf + (k-1), &sumx4, 8);
			}
			sumx4 = vgetq_lane_u64(vreinterpretq_u64_u16(acc_inf), 1);
			memcpy(Y_sup + k, &sumx4, 8);
			mask = vextq_u16(zeros,mask,7);
		}
	}
	// End last block

	/*Last element of Y_inf*/
	row = n*complete_line;
	Y_inf[n-1] = gf2_gf256to2_vect_mult_neon(X,&A[row],n);
}

static inline void gf256to2_gf2_mat_mult_single_neon(const uint16_t *A, const uint8_t *X, uint16_t *Y, uint32_t n)
{
    GF256to2_GF2_MAT_MULT(A, X, Y, n, TRI_INF, gf256to2_gf2_vect_mult_neon);
}

/*
 * "Hybrid" matrix multiplication of a matrix in GF(256) and a vector in GF(256^2), resulting
 *  in a vector in GF(256^2) 
 */
static inline void gf256_gf256to2_mat_mult_neon(const uint8_t *A, const uint16_t *X, uint16_t *Y, uint32_t n, matrix_type mtype)
{
	/* XXX: NOTE: because of alignment and loading latencies, treating the matrix as "regular" seems always better
		* than using the trianglar shape ... */
	(void)mtype;
	GF256to2_MAT_MULT(A, X, Y, n, REG, gf256_gf256to2_vect_mult_neon);
}

static inline void gf256to2_gf256_mat_mult_neon(const uint16_t *A, const uint8_t *X, uint16_t *Y_inf, uint16_t *Y_sup, uint32_t n)
{
	/* XXX: NOTE: because of alignment and loading latencies, treating the matrix as "regular" seems always better
		* than using the trianglar shape ... */
	uint16_t i, j, k;
	uint16_t complete_line = (n + 7) & ~7;
	uint16_t batches;
	uint32_t row;
	uint64_t sumx4;
	uint8x16_t accu, _a, prod;
	uint8x16_t acc_inf, mask, zeros;
	uint8x8_t _b;

	uint8x16_t extra_accs[4];

	batches = n/8;

	zeros = vdupq_n_u8(0);

	k=0;
	for (i=0;i<batches;i++) {
		mask = vdupq_n_u8(-1);
		for (;k<i*8+8;k+=4) {
			row = k*complete_line;
			extra_accs[0] = vdupq_n_u8(0);
			extra_accs[1] = vdupq_n_u8(0);
			extra_accs[2] = vdupq_n_u8(0);
			extra_accs[3] = vdupq_n_u8(0); 
			for (j=0;j<i;j++) {
				_b = vld1_u8(&X[8*j]);
				prod = vcombine_u8(vzip1_u8(_b,_b),vzip2_u8(_b,_b));
				_a = vreinterpretq_u8_u16(vld1q_u16(&A[row+8*j]));
				extra_accs[0] ^= gf256_mult_vectorized_neon(_a,prod);
				_a = vreinterpretq_u8_u16(vld1q_u16(&A[row+complete_line+8*j]));
				extra_accs[1] ^= gf256_mult_vectorized_neon(_a,prod);
				_a = vreinterpretq_u8_u16(vld1q_u16(&A[row+2*complete_line+8*j]));
				extra_accs[2] ^= gf256_mult_vectorized_neon(_a,prod);
				_a = vreinterpretq_u8_u16(vld1q_u16(&A[row+3*complete_line+8*j]));
				extra_accs[3] ^= gf256_mult_vectorized_neon(_a,prod);
			}

			_b = vld1_u8(&X[8*i]);
			prod = vcombine_u8(vzip1_u8(_b,_b),vzip2_u8(_b,_b));
				
			// row k
			_a = vreinterpretq_u8_u16(vld1q_u16(&A[row+8*i]));
			accu = gf256_mult_vectorized_neon(_a,prod);
			extra_accs[0] = veorq_u8(extra_accs[0], vbicq_u8(accu, mask));
			accu = vandq_u8(mask, accu);
			extra_accs[0] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(extra_accs[0]),vreinterpretq_u64_u8(accu))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(extra_accs[0]),vreinterpretq_u64_u8(accu))));
			
			// row k+1
			mask = vextq_u8(zeros,mask,14);
			_a = vreinterpretq_u8_u16(vld1q_u16(&A[row+complete_line+8*i]));
			accu = gf256_mult_vectorized_neon(_a,prod);
			extra_accs[1] = veorq_u8(extra_accs[1], vbicq_u8(accu, mask));
			accu = vandq_u8(mask, accu);
			extra_accs[1] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(extra_accs[1]),vreinterpretq_u64_u8(accu))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(extra_accs[1]),vreinterpretq_u64_u8(accu))));

			//18/06/2026 -- continuar aqui

			// row k+2
			mask = vextq_u8(zeros,mask,14);
			_a = vreinterpretq_u8_u16(vld1q_u16(&A[row+2*complete_line+8*i]));
			accu = gf256_mult_vectorized_neon(_a,prod);
			extra_accs[2] = veorq_u8(extra_accs[2], vbicq_u8(accu, mask));
			accu = vandq_u8(mask, accu);
			extra_accs[2] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(extra_accs[2]),vreinterpretq_u64_u8(accu))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(extra_accs[2]),vreinterpretq_u64_u8(accu))));

			// row k+3
			mask = vextq_u8(zeros,mask,14);
			_a = vreinterpretq_u8_u16(vld1q_u16(&A[row+3*complete_line+8*i]));
			accu = gf256_mult_vectorized_neon(_a,prod);
			extra_accs[3] = veorq_u8(extra_accs[3], vbicq_u8(accu, mask));
			accu = vandq_u8(mask, accu);
			extra_accs[3] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(extra_accs[3]),vreinterpretq_u64_u8(accu))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(extra_accs[3]),vreinterpretq_u64_u8(accu))));
			
			// k xor k+2
			acc_inf = veorq_u8(vreinterpretq_u8_u32(vtrn1q_u32(vreinterpretq_u32_u8(extra_accs[0]),vreinterpretq_u32_u8(extra_accs[2]))), vreinterpretq_u8_u32(vtrn2q_u32(vreinterpretq_u32_u8(extra_accs[0]),vreinterpretq_u32_u8(extra_accs[2]))));

			// k+1 xor k+3
			extra_accs[0] = veorq_u8(vreinterpretq_u8_u32(vtrn1q_u32(vreinterpretq_u32_u8(extra_accs[1]),vreinterpretq_u32_u8(extra_accs[3]))), vreinterpretq_u8_u32(vtrn2q_u32(vreinterpretq_u32_u8(extra_accs[1]),vreinterpretq_u32_u8(extra_accs[3]))));

			// (k xor k+2) xor (k+1 xor k+3)
			acc_inf = veorq_u8(vreinterpretq_u8_u16(vtrn1q_u16(vreinterpretq_u16_u8(acc_inf),vreinterpretq_u16_u8(extra_accs[0]))), vreinterpretq_u8_u16(vtrn2q_u16(vreinterpretq_u16_u8(acc_inf),vreinterpretq_u16_u8(extra_accs[0])))); 

			extra_accs[0] = vdupq_n_u8(0);
			extra_accs[1] = vdupq_n_u8(0);
			extra_accs[2] = vdupq_n_u8(0);
			extra_accs[3] = vdupq_n_u8(0);
			for (j=i+1;j<batches;j++) {
				_b = vld1_u8(&X[8*j]);
				prod = vcombine_u8(vzip1_u8(_b,_b),vzip2_u8(_b,_b));
				_a = vreinterpretq_u8_u16(vld1q_u16(&A[row+8*j]));
				extra_accs[0] ^= gf256_mult_vectorized_neon(_a,prod);
				_a = vreinterpretq_u8_u16(vld1q_u16(&A[row+complete_line+8*j]));
				extra_accs[1] ^= gf256_mult_vectorized_neon(_a,prod);
				_a = vreinterpretq_u8_u16(vld1q_u16(&A[row+2*complete_line+8*j]));
				extra_accs[2] ^= gf256_mult_vectorized_neon(_a,prod);
				_a = vreinterpretq_u8_u16(vld1q_u16(&A[row+3*complete_line+8*j]));
				extra_accs[3] ^= gf256_mult_vectorized_neon(_a,prod);
			}

			accu = vdupq_n_u8(0);
			extra_accs[0] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(accu),vreinterpretq_u64_u8(extra_accs[0]))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(accu),vreinterpretq_u64_u8(extra_accs[0]))));
			extra_accs[1] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(accu),vreinterpretq_u64_u8(extra_accs[1]))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(accu),vreinterpretq_u64_u8(extra_accs[1]))));
			extra_accs[2] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(accu),vreinterpretq_u64_u8(extra_accs[2]))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(accu),vreinterpretq_u64_u8(extra_accs[2]))));
			extra_accs[3] = veorq_u8(vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u8(accu),vreinterpretq_u64_u8(extra_accs[3]))), vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u8(accu),vreinterpretq_u64_u8(extra_accs[3]))));
			extra_accs[0] = veorq_u8(vreinterpretq_u8_u32(vtrn1q_u32(vreinterpretq_u32_u8(extra_accs[0]),vreinterpretq_u32_u8(extra_accs[2]))), vreinterpretq_u8_u32(vtrn2q_u32(vreinterpretq_u32_u8(extra_accs[0]),vreinterpretq_u32_u8(extra_accs[2]))));
			extra_accs[1] = veorq_u8(vreinterpretq_u8_u32(vtrn1q_u32(vreinterpretq_u32_u8(extra_accs[1]),vreinterpretq_u32_u8(extra_accs[3]))), vreinterpretq_u8_u32(vtrn2q_u32(vreinterpretq_u32_u8(extra_accs[1]),vreinterpretq_u32_u8(extra_accs[3]))));
			extra_accs[0] = veorq_u8(vreinterpretq_u8_u16(vtrn1q_u16(vreinterpretq_u16_u8(extra_accs[0]),vreinterpretq_u16_u8(extra_accs[1]))), vreinterpretq_u8_u16(vtrn2q_u16(vreinterpretq_u16_u8(extra_accs[0]),vreinterpretq_u16_u8(extra_accs[1])))); 
			acc_inf = veorq_u8(acc_inf,extra_accs[0]);
			sumx4 = vgetq_lane_u64(vreinterpretq_u64_u8(acc_inf), 0);
			if (k == 0) {      
					sumx4 = sumx4 >> 16;
					memcpy(Y_inf, &sumx4, 6);
			}
			else {
					memcpy(Y_inf + (k-1), &sumx4, 8);
			}
			sumx4 = vgetq_lane_u64(vreinterpretq_u64_u8(acc_inf), 1);
			memcpy(Y_sup + k, &sumx4, 8);
			mask = vextq_u8(zeros,mask,14);
		}
	}
	// End last block

	/*Last element of Y_inf*/
	row = n*complete_line;
	Y_inf[n-1] = gf256_gf256to2_vect_mult_neon(X,&A[row],n);
	
}

static inline void gf256to2_transpose_block_neon(uint16_t *src, uint16_t *dst, uint16_t size_in, uint16_t size_out)
{
    uint16x8_t r0 = vld1q_u16(src);
    uint16x8_t r1 = vld1q_u16(src + (1)*size_in);
    uint16x8_t r2 = vld1q_u16(src + (2)*size_in);
    uint16x8_t r3 = vld1q_u16(src + (3)*size_in);
    uint16x8_t r4 = vld1q_u16(src + (4)*size_in);
    uint16x8_t r5 = vld1q_u16(src + (5)*size_in);
    uint16x8_t r6 = vld1q_u16(src + (6)*size_in);
    uint16x8_t r7 = vld1q_u16(src + (7)*size_in);

    uint16x8x2_t t01 = vtrnq_u16(r0, r1);
    uint16x8x2_t t23 = vtrnq_u16(r2, r3);
    uint16x8x2_t t45 = vtrnq_u16(r4, r5);
    uint16x8x2_t t67 = vtrnq_u16(r6, r7);

    uint32x4x2_t s02 = vtrnq_u32(
        vreinterpretq_u32_u16(t01.val[0]),
        vreinterpretq_u32_u16(t23.val[0])
    );

    uint32x4x2_t s13 = vtrnq_u32(
        vreinterpretq_u32_u16(t01.val[1]),
        vreinterpretq_u32_u16(t23.val[1])
    );

    uint32x4x2_t s46 = vtrnq_u32(
        vreinterpretq_u32_u16(t45.val[0]),
        vreinterpretq_u32_u16(t67.val[0])
    );

    uint32x4x2_t s57 = vtrnq_u32(
        vreinterpretq_u32_u16(t45.val[1]),
        vreinterpretq_u32_u16(t67.val[1])
    );

    uint64x2_t a0 = vreinterpretq_u64_u32(s02.val[0]);
    uint64x2_t a1 = vreinterpretq_u64_u32(s02.val[1]);
    uint64x2_t a2 = vreinterpretq_u64_u32(s13.val[0]);
    uint64x2_t a3 = vreinterpretq_u64_u32(s13.val[1]);
    uint64x2_t a4 = vreinterpretq_u64_u32(s46.val[0]);
    uint64x2_t a5 = vreinterpretq_u64_u32(s46.val[1]);
    uint64x2_t a6 = vreinterpretq_u64_u32(s57.val[0]);
    uint64x2_t a7 = vreinterpretq_u64_u32(s57.val[1]);

    uint16x8_t out0 = vreinterpretq_u16_u64(vzip1q_u64(a0, a4));
    uint16x8_t out4 = vreinterpretq_u16_u64(vzip2q_u64(a0, a4));

    uint16x8_t out2 = vreinterpretq_u16_u64(vzip1q_u64(a1, a5));
    uint16x8_t out6 = vreinterpretq_u16_u64(vzip2q_u64(a1, a5));

    uint16x8_t out1 = vreinterpretq_u16_u64(vzip1q_u64(a2, a6));
    uint16x8_t out5 = vreinterpretq_u16_u64(vzip2q_u64(a2, a6));

    uint16x8_t out3 = vreinterpretq_u16_u64(vzip1q_u64(a3, a7));
    uint16x8_t out7 = vreinterpretq_u16_u64(vzip2q_u64(a3, a7));

    // store final matrix
    vst1q_u16(dst, out0);
    vst1q_u16(dst + (1)*size_out , out1);
    vst1q_u16(dst + (2)*size_out , out2);
    vst1q_u16(dst + (3)*size_out , out3);
    vst1q_u16(dst + (4)*size_out , out4);
    vst1q_u16(dst + (5)*size_out , out5);
    vst1q_u16(dst + (6)*size_out , out6);
    vst1q_u16(dst + (7)*size_out , out7);
}

/* GF(256^2) matrix transposition  */
static inline void gf256to2_mat_transpose_neon(uint16_t *A, uint16_t len, uint16_t extension)
{
	uint16_t *block1;
	uint16_t *block2;
	uint16_t aux[64] = {0};
	uint16x8_t mv;
	for (int i=0;i<len;i+=8){
		for (int j=0;j<i;j+=8) {
			block1 = A + i*extension + j;
			block2 = A + j*extension + i;
			gf256to2_transpose_block_neon(block1,aux,extension,8);
			gf256to2_transpose_block_neon(block2,block1,extension,extension);
			for (int k=0;k<8;k++) {
				mv = vld1q_u16(aux + 8*k);
				vst1q_u16(block2+extension*k,mv);
			}
		}
	}
	for (int i=0;i<len;i+=8){
				block1 = A + i*extension + i;
				gf256to2_transpose_block_neon(block1,block1,extension,extension);
	}
}

/*
 * "Hybrid" constant multiplication of a constant in GF(16) and a vector in GF(256^2)
 */
static inline void gf16_gf256to2_constant_vect_mult_neon(uint8_t a_gf16, const uint16_t *b_gf256to2, uint16_t *c_gf256to2, uint32_t n)
{
    uint8_t a_gf256;
    gf256_vect_lift_from_gf16_ref(&a_gf16, &a_gf256, 1);
    gf256_gf256to2_constant_vect_mult_neon(a_gf256, b_gf256to2, c_gf256to2, n);
    return;
}

/*
 * "Hybrid" constant multiplication of a constant in GF(256^2) and a vector in GF(16)
 */
static inline void gf256to2_gf16_constant_vect_mult_neon(uint16_t a_gf256to2, const uint8_t *b_gf16, uint16_t *c_gf256to2, uint32_t n)
{
    uint8_t* buf = ((uint8_t*) c_gf256to2) + n;
    gf256_vect_lift_from_gf16_neon(b_gf16, buf, n);
    gf256to2_gf256_constant_vect_mult_neon(a_gf256to2, buf, c_gf256to2, n);
    return;
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(16) and a vector in GF(256^2)
 */
static inline uint16_t gf16_gf256to2_vect_mult_neon(const uint8_t *a_gf16, const uint16_t *b_gf256to2, uint32_t len)
{
        uint32_t i;
        uint8x16_t accu, _a, _b;
        uint8x16_t _a_gf16;

        /* Set the accumulator to 0 */
        accu = vdupq_n_u8(0);

        for(i = 0; i < (2 * len); i += 16){
                if(((2 * len)-i) < 16){
                        _a_gf16 = load_incomplete_m128((const uint8_t*)&a_gf16[i / 4], (((2 * len)-i) + 3)/ 4);
                        _a = vdupq_n_u8(0);
                        gf256_vect_lift_from_gf16_neon((const uint8_t*)&_a_gf16, (uint8_t*)&_a, ((2 * len)-i+1)/2);
                        _b = load_incomplete_m128((const uint8_t*)&b_gf256to2[i / 2], (2 * len)-i);
                }
                else{
                        /* Obvious 256-bit */
                        _a_gf16 = load_incomplete_m128((const uint8_t*)&a_gf16[i / 4], 4);
						_a = vdupq_n_u8(0);
                        gf256_vect_lift_from_gf16_neon((const uint8_t*)&_a_gf16, (uint8_t*)&_a, 8);
                        _b = vreinterpretq_u8_u16(vld1q_u16(&b_gf256to2[i / 2]));
                }
                /* Duplicate the values in _a */
                _a = vzip1q_u8(_a,_a);
                /* Multiply in GF(256) */
                accu ^= gf256_mult_vectorized_neon(_a, _b);
        }

        return sum_uint16_neon(accu);
}

/*
 * "Hybrid" scalar multiplication of a vector in GF(256^2) and a vector in GF(16)
 */
static inline uint16_t gf256to2_gf16_vect_mult_neon(const uint16_t *a_gf256to2, const uint8_t *b_gf16, uint32_t n)
{
    return gf16_gf256to2_vect_mult_neon(b_gf16, a_gf256to2, n);
}

/*
 * "Hybrid" matrix multiplication of a matrix in GF(256^2) and a vector in GF(16), resulting
 *  in a vector in GF(256^2)
 */
static inline void gf256to2_gf16_mat_mult_neon(const uint16_t *A, const uint8_t *X, uint16_t *Y_inf, uint16_t *Y_sup, uint32_t n)
{
    uint16_t i, j, k;
	uint16_t complete_line = (n + 7) & ~7;
	uint16_t batches, remainder;
	uint32_t row;
	uint64_t sumx4;
	uint16x8_t _a;
	uint8x16_t _b;
	uint16x8_t accu, acc_inf;
	uint8x8_t aux;

	uint16x8_t zeros, mask;

	uint16x8_t extra_accs[4];

	zeros = vdupq_n_u16(0);

	batches = n/8;
	remainder = n - batches * 8;

	k=0;
	for (i=0;i<batches;i++) {
		mask = vdupq_n_u16(-1);
		for (;k<i*8+8;k+=4) {
			row = k*complete_line;
			extra_accs[0] = vdupq_n_u16(0);
			extra_accs[1] = vdupq_n_u16(0);
			extra_accs[2] = vdupq_n_u16(0);
			extra_accs[3] = vdupq_n_u16(0); 
			for (j=0;j<i;j++) {
				_b = load_incomplete_m128(&X[4*j],4);
				_b = gf256_lift16_from_gf16_neon(_b);
				aux = vget_low_u8(_b);
				_b = vcombine_u8(vzip1_u8(aux,aux),vzip2_u8(aux,aux));
				_a = vld1q_u16(&A[row+8*j]);
				extra_accs[0] ^= vreinterpretq_u16_u8(gf256_mult_vectorized_neon(vreinterpretq_u8_u16(_a),_b));
				_a = vld1q_u16(&A[row+complete_line+8*j]);
				extra_accs[1] ^= vreinterpretq_u16_u8(gf256_mult_vectorized_neon(vreinterpretq_u8_u16(_a),_b));
				_a = vld1q_u16(&A[row+2*complete_line+8*j]);
				extra_accs[2] ^= vreinterpretq_u16_u8(gf256_mult_vectorized_neon(vreinterpretq_u8_u16(_a),_b));
				_a = vld1q_u16(&A[row+3*complete_line+8*j]);
				extra_accs[3] ^= vreinterpretq_u16_u8(gf256_mult_vectorized_neon(vreinterpretq_u8_u16(_a),_b));
			}

			_b = load_incomplete_m128(&X[4*j],4);
			_b = gf256_lift16_from_gf16_neon(_b);
			aux = vget_low_u8(_b);
			_b = vcombine_u8(vzip1_u8(aux,aux),vzip2_u8(aux,aux));

			// row k
			_a = vld1q_u16(&A[row+8*i]);
			accu = vreinterpretq_u16_u8(gf256_mult_vectorized_neon(vreinterpretq_u8_u16(_a),_b));
			extra_accs[0] = veorq_u16(extra_accs[0], vbicq_u16(accu, mask));
			accu = vandq_u16(mask, accu);
			extra_accs[0] = veorq_u16(vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u16(extra_accs[0]),vreinterpretq_u64_u16(accu))), vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u16(extra_accs[0]),vreinterpretq_u64_u16(accu))));
			
			// row k+1
			mask = vextq_u16(zeros,mask,7);
			_a = vld1q_u16(&A[row+complete_line+8*i]);
			accu = vreinterpretq_u16_u8(gf256_mult_vectorized_neon(vreinterpretq_u8_u16(_a),_b));
			extra_accs[1] = veorq_u16(extra_accs[1], vbicq_u16(accu, mask));
			accu = vandq_u16(mask, accu);
			extra_accs[1] = veorq_u16(vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u16(extra_accs[1]),vreinterpretq_u64_u16(accu))), vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u16(extra_accs[1]),vreinterpretq_u64_u16(accu))));

			// row k+2
			mask = vextq_u16(zeros,mask,7);
			_a = vld1q_u16(&A[row+2*complete_line+8*i]);
			accu = vreinterpretq_u16_u8(gf256_mult_vectorized_neon(vreinterpretq_u8_u16(_a),_b));
			extra_accs[2] = veorq_u16(extra_accs[2], vbicq_u16(accu, mask));
			accu = vandq_u16(mask, accu);
			extra_accs[2] = veorq_u16(vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u16(extra_accs[2]),vreinterpretq_u64_u16(accu))), vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u16(extra_accs[2]),vreinterpretq_u64_u16(accu))));

			// row k+3
			mask = vextq_u16(zeros,mask,7);
			_a = vld1q_u16(&A[row+3*complete_line+8*i]);
			accu = vreinterpretq_u16_u8(gf256_mult_vectorized_neon(vreinterpretq_u8_u16(_a),_b));
			extra_accs[3] = veorq_u16(extra_accs[3], vbicq_u16(accu, mask));
			accu = vandq_u16(mask, accu);
			extra_accs[3] = veorq_u16(vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u16(extra_accs[3]),vreinterpretq_u64_u16(accu))), vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u16(extra_accs[3]),vreinterpretq_u64_u16(accu))));
			
			// k xor k+2
			acc_inf = veorq_u16(vreinterpretq_u16_u32(vtrn1q_u32(vreinterpretq_u32_u16(extra_accs[0]),vreinterpretq_u32_u16(extra_accs[2]))), vreinterpretq_u16_u32(vtrn2q_u32(vreinterpretq_u32_u16(extra_accs[0]),vreinterpretq_u32_u16(extra_accs[2]))));

			// k+1 xor k+3
			extra_accs[0] = veorq_u16(vreinterpretq_u16_u32(vtrn1q_u32(vreinterpretq_u32_u16(extra_accs[1]),vreinterpretq_u32_u16(extra_accs[3]))), vreinterpretq_u16_u32(vtrn2q_u32(vreinterpretq_u32_u16(extra_accs[1]),vreinterpretq_u32_u16(extra_accs[3]))));

			// (k xor k+2) xor (k+1 xor k+3)
			acc_inf = veorq_u16(vtrn1q_u16(acc_inf,extra_accs[0]), vtrn2q_u16(acc_inf,extra_accs[0])); 

			extra_accs[0] = vdupq_n_u16(0);
			extra_accs[1] = vdupq_n_u16(0);
			extra_accs[2] = vdupq_n_u16(0);
			extra_accs[3] = vdupq_n_u16(0);
			for (j=i+1;j<batches;j++) {
				_b = load_incomplete_m128(&X[4*j],4);
				_b = gf256_lift16_from_gf16_neon(_b);
				aux = vget_low_u8(_b);
				_b = vcombine_u8(vzip1_u8(aux,aux),vzip2_u8(aux,aux));
				_a = vld1q_u16(&A[row+8*j]);
				extra_accs[0] ^= vreinterpretq_u16_u8(gf256_mult_vectorized_neon(vreinterpretq_u8_u16(_a),_b));
				_a = vld1q_u16(&A[row+complete_line+8*j]);
				extra_accs[1] ^= vreinterpretq_u16_u8(gf256_mult_vectorized_neon(vreinterpretq_u8_u16(_a),_b));
				_a = vld1q_u16(&A[row+2*complete_line+8*j]);
				extra_accs[2] ^= vreinterpretq_u16_u8(gf256_mult_vectorized_neon(vreinterpretq_u8_u16(_a),_b));
				_a = vld1q_u16(&A[row+3*complete_line+8*j]);
				extra_accs[3] ^= vreinterpretq_u16_u8(gf256_mult_vectorized_neon(vreinterpretq_u8_u16(_a),_b));
			}
			if (remainder > 0) {
				_b = load_incomplete_m128(&X[4*j],(remainder+1)/2);
				_b = gf256_lift16_from_gf16_neon(_b);
				aux = vget_low_u8(_b);
				_b = vcombine_u8(vzip1_u8(aux,aux),vzip2_u8(aux,aux));
				_a = vreinterpretq_u16_u8(load_incomplete_m128((uint8_t *)&A[row+8*batches],2*remainder));
				extra_accs[0] ^= vreinterpretq_u16_u8(gf256_mult_vectorized_neon(vreinterpretq_u8_u16(_a),_b));
				_a = vreinterpretq_u16_u8(load_incomplete_m128((uint8_t *)&A[row+complete_line+8*batches],2*remainder));
				extra_accs[1] ^= vreinterpretq_u16_u8(gf256_mult_vectorized_neon(vreinterpretq_u8_u16(_a),_b));
				_a = vreinterpretq_u16_u8(load_incomplete_m128((uint8_t *)&A[row+2*complete_line+8*batches],2*remainder));
				extra_accs[2] ^= vreinterpretq_u16_u8(gf256_mult_vectorized_neon(vreinterpretq_u8_u16(_a),_b));
				_a = vreinterpretq_u16_u8(load_incomplete_m128((uint8_t *)&A[row+3*complete_line+8*batches],2*remainder));
				extra_accs[3] ^= vreinterpretq_u16_u8(gf256_mult_vectorized_neon(vreinterpretq_u8_u16(_a),_b));
			}
			accu = vdupq_n_u16(0);
			extra_accs[0] = veorq_u16(vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u16(accu),vreinterpretq_u64_u16(extra_accs[0]))), vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u16(accu),vreinterpretq_u64_u16(extra_accs[0]))));
			extra_accs[1] = veorq_u16(vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u16(accu),vreinterpretq_u64_u16(extra_accs[1]))), vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u16(accu),vreinterpretq_u64_u16(extra_accs[1]))));
			extra_accs[2] = veorq_u16(vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u16(accu),vreinterpretq_u64_u16(extra_accs[2]))), vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u16(accu),vreinterpretq_u64_u16(extra_accs[2]))));
			extra_accs[3] = veorq_u16(vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u16(accu),vreinterpretq_u64_u16(extra_accs[3]))), vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u16(accu),vreinterpretq_u64_u16(extra_accs[3]))));
			extra_accs[0] = veorq_u16(vreinterpretq_u16_u32(vtrn1q_u32(vreinterpretq_u32_u16(extra_accs[0]),vreinterpretq_u32_u16(extra_accs[2]))), vreinterpretq_u16_u32(vtrn2q_u32(vreinterpretq_u32_u16(extra_accs[0]),vreinterpretq_u32_u16(extra_accs[2]))));
			extra_accs[1] = veorq_u16(vreinterpretq_u16_u32(vtrn1q_u32(vreinterpretq_u32_u16(extra_accs[1]),vreinterpretq_u32_u16(extra_accs[3]))), vreinterpretq_u16_u32(vtrn2q_u32(vreinterpretq_u32_u16(extra_accs[1]),vreinterpretq_u32_u16(extra_accs[3]))));
			extra_accs[0] = veorq_u16(vtrn1q_u16(extra_accs[0],extra_accs[1]), vtrn2q_u16(extra_accs[0],extra_accs[1])); 
			acc_inf = veorq_u16(acc_inf,extra_accs[0]);
			sumx4 = vgetq_lane_u64(vreinterpretq_u64_u16(acc_inf), 0);
			if (k == 0) {      
					sumx4 = sumx4 >> 16;
					memcpy(Y_inf, &sumx4, 6);
			}
			else {
					memcpy(Y_inf + (k-1), &sumx4, 8);
			}
			sumx4 = vgetq_lane_u64(vreinterpretq_u64_u16(acc_inf), 1);
			memcpy(Y_sup + k, &sumx4, 8);
			mask = vextq_u16(zeros,mask,7);
		}
	}

	if (remainder > 0) {
		mask = vdupq_n_u16(-1);
		for (k=8*batches;k<n;k+=4) {
			row = k*complete_line;
			extra_accs[0] = vdupq_n_u16(0);
			extra_accs[1] = vdupq_n_u16(0);
			extra_accs[2] = vdupq_n_u16(0);
			extra_accs[3] = vdupq_n_u16(0); 
			for (j=0;j<batches;j++) {
				_b = load_incomplete_m128(&X[4*j],4);
				_b = gf256_lift16_from_gf16_neon(_b);
				aux = vget_low_u8(_b);
				_b = vcombine_u8(vzip1_u8(aux,aux),vzip2_u8(aux,aux));
				_a = vld1q_u16(&A[row+8*j]);
				extra_accs[0] ^= vreinterpretq_u16_u8(gf256_mult_vectorized_neon(vreinterpretq_u8_u16(_a),_b));
				_a = vld1q_u16(&A[row+complete_line+8*j]);
				extra_accs[1] ^= vreinterpretq_u16_u8(gf256_mult_vectorized_neon(vreinterpretq_u8_u16(_a),_b));
				_a = vld1q_u16(&A[row+2*complete_line+8*j]);
				extra_accs[2] ^= vreinterpretq_u16_u8(gf256_mult_vectorized_neon(vreinterpretq_u8_u16(_a),_b));
				_a = vld1q_u16(&A[row+3*complete_line+8*j]);
				extra_accs[3] ^= vreinterpretq_u16_u8(gf256_mult_vectorized_neon(vreinterpretq_u8_u16(_a),_b));
			}

			_b = load_incomplete_m128(&X[4*batches],(remainder+1)/2);
			_b = gf256_lift16_from_gf16_neon(_b);
			aux = vget_low_u8(_b);
			_b = vcombine_u8(vzip1_u8(aux,aux),vzip2_u8(aux,aux));

			// row k
			_a = vreinterpretq_u16_u8(load_incomplete_m128((uint8_t *)&A[row+8*batches],2*remainder));
			accu = vreinterpretq_u16_u8(gf256_mult_vectorized_neon(vreinterpretq_u8_u16(_a),_b));
			extra_accs[0] = veorq_u16(extra_accs[0], vbicq_u16(accu, mask));
			accu = vandq_u16(mask, accu);
			extra_accs[0] = veorq_u16(vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u16(extra_accs[0]),vreinterpretq_u64_u16(accu))), vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u16(extra_accs[0]),vreinterpretq_u64_u16(accu))));
			
			// row k+1
			mask = vextq_u16(zeros,mask,7);
			_a = vreinterpretq_u16_u8(load_incomplete_m128((uint8_t *)&A[row+complete_line+8*batches],2*remainder));
			accu = vreinterpretq_u16_u8(gf256_mult_vectorized_neon(vreinterpretq_u8_u16(_a),_b));
			extra_accs[1] = veorq_u16(extra_accs[1], vbicq_u16(accu, mask));
			accu = vandq_u16(mask, accu);
			extra_accs[1] = veorq_u16(vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u16(extra_accs[1]),vreinterpretq_u64_u16(accu))), vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u16(extra_accs[1]),vreinterpretq_u64_u16(accu))));

			// row k+2
			mask = vextq_u16(zeros,mask,7);
			_a = vreinterpretq_u16_u8(load_incomplete_m128((uint8_t *)&A[row+2*complete_line+8*batches],2*remainder));
			accu = vreinterpretq_u16_u8(gf256_mult_vectorized_neon(vreinterpretq_u8_u16(_a),_b));
			extra_accs[2] = veorq_u16(extra_accs[2], vbicq_u16(accu, mask));
			accu = vandq_u16(mask, accu);
			extra_accs[2] = veorq_u16(vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u16(extra_accs[2]),vreinterpretq_u64_u16(accu))), vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u16(extra_accs[2]),vreinterpretq_u64_u16(accu))));

			// row k+3
			mask = vextq_u16(zeros,mask,7);
			_a = vreinterpretq_u16_u8(load_incomplete_m128((uint8_t *)&A[row+3*complete_line+8*batches],2*remainder));
			accu = vreinterpretq_u16_u8(gf256_mult_vectorized_neon(vreinterpretq_u8_u16(_a),_b));
			extra_accs[3] = veorq_u16(extra_accs[3], vbicq_u16(accu, mask));
			accu = vandq_u16(mask, accu);
			extra_accs[3] = veorq_u16(vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u16(extra_accs[3]),vreinterpretq_u64_u16(accu))), vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u16(extra_accs[3]),vreinterpretq_u64_u16(accu))));
			
			// k xor k+2
			acc_inf = veorq_u16(vreinterpretq_u16_u32(vtrn1q_u32(vreinterpretq_u32_u16(extra_accs[0]),vreinterpretq_u32_u16(extra_accs[2]))), vreinterpretq_u16_u32(vtrn2q_u32(vreinterpretq_u32_u16(extra_accs[0]),vreinterpretq_u32_u16(extra_accs[2]))));

			// k+1 xor k+3
			extra_accs[0] = veorq_u16(vreinterpretq_u16_u32(vtrn1q_u32(vreinterpretq_u32_u16(extra_accs[1]),vreinterpretq_u32_u16(extra_accs[3]))), vreinterpretq_u16_u32(vtrn2q_u32(vreinterpretq_u32_u16(extra_accs[1]),vreinterpretq_u32_u16(extra_accs[3]))));

			// (k xor k+2) xor (k+1 xor k+3)
			acc_inf = veorq_u16(vtrn1q_u16(acc_inf,extra_accs[0]), vtrn2q_u16(acc_inf,extra_accs[0])); 

			sumx4 = vgetq_lane_u64(vreinterpretq_u64_u16(acc_inf), 0);
			memcpy(Y_inf + (k-1), &sumx4, 8);
			sumx4 = vgetq_lane_u64(vreinterpretq_u64_u16(acc_inf), 1);
			memcpy(Y_sup + k, &sumx4, 8);
			mask = vextq_u16(zeros,mask,7);
		}
	}
	// End last block

	/*Last element of Y_inf*/
	row = n*complete_line;
	Y_inf[n-1] = gf16_gf256to2_vect_mult_neon(X,&A[row],n);
}

static inline void gf256to2_gf16_mat_mult_single_neon(const uint16_t *A, const uint8_t *X, uint16_t *Y, uint32_t n)
{
    GF256to2_GF16_MAT_MULT(A, X, Y, n, TRI_INF, gf256to2_gf16_vect_mult_neon);
}

#endif /* __ARM_NEON */

#endif /* __FIELDS_NEON_H__ */
