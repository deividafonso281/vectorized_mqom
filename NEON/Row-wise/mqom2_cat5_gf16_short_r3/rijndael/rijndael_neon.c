#include "rijndael_platform.h"

#if defined(RIJNDAEL_NEON)
#include "rijndael_neon.h"

/* Version using instructions in the NEON SIMD*/

static const uint8_t rcon[256] = {
    0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a,
    0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39,
    0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a,
    0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8,
    0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef,
    0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc,
    0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b,
    0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3,
    0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94,
    0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20,
    0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35,
    0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f,
    0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04,
    0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63,
    0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd,
    0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d
};

/* Rijndael primitives *******************************************************/
/************************************************************************/
/* Multiplication and squaring over the Rijndael Galois field */
#define RIJNDAEL_MODULUS 0x1B /* The Rijndael field GF(2^8) modulus */
static inline uint8_t gmul(uint8_t x, uint8_t y){
	uint8_t res;

	res = (-(y >> 7) & x);
	res = (-(y >> 6 & 1) & x) ^ (-(res >> 7) & RIJNDAEL_MODULUS) ^ (res << 1);
	res = (-(y >> 5 & 1) & x) ^ (-(res >> 7) & RIJNDAEL_MODULUS) ^ (res << 1);
	res = (-(y >> 4 & 1) & x) ^ (-(res >> 7) & RIJNDAEL_MODULUS) ^ (res << 1);
	res = (-(y >> 3 & 1) & x) ^ (-(res >> 7) & RIJNDAEL_MODULUS) ^ (res << 1);
	res = (-(y >> 2 & 1) & x) ^ (-(res >> 7) & RIJNDAEL_MODULUS) ^ (res << 1);
	res = (-(y >> 1 & 1) & x) ^ (-(res >> 7) & RIJNDAEL_MODULUS) ^ (res << 1);
	res = (-(y      & 1) & x) ^ (-(res >> 7) & RIJNDAEL_MODULUS) ^ (res << 1);

	return res;
}
static inline uint8_t gsquare(uint8_t x){
	return gmul(x, x);
}

/* Sbox computed as a circuit for constant time */
#define SBOX_BIT_EXTRACT(s, a, b, c, d, e) (((s >> a) & 1) ^ ((s >> b) & 1) ^ ((s >> c) & 1) ^ ((s >> d) & 1) ^ ((s >> e) & 1))
static inline uint8_t sbox(uint8_t s)
{
	uint8_t out;
	/* First, compute the inverse of s as s^254  */
	uint8_t s2   = gsquare(s);     /* s^2 */
	uint8_t s3   = gmul(s, s2);    /* s^3 = s * s^2*/
	uint8_t s5   = gmul(s3, s2);   /* s^5 = s^3 * s^2 */
	uint8_t s7   = gmul(s5, s2);   /* s^7 = s^5 * s^2 */
	uint8_t s14  = gsquare(s7);    /* s^14 = (s^7)^2 */
	uint8_t s28  = gsquare(s14);   /* s^28 = (s^14)^2 */
	uint8_t s56  = gsquare(s28);   /* s^56 = (s^28)^2 */
	uint8_t s63  = gmul(s56, s7);  /* s^63 = s^56 * s^7 */
	uint8_t s126 = gsquare(s63);   /* s^126 = (s^63)^2 */
	uint8_t s252 = gsquare(s126);  /* s^252 = (s^126)^2 */
	uint8_t sinv = gmul(s252, s2); /* This is s^254 = s^-1 in GF(2^8) */
	/* Secondly, compute the affine part of the SBox: A * sinv */
	uint8_t out0 = SBOX_BIT_EXTRACT(sinv, 0, 4, 5, 6, 7);
	uint8_t out1 = SBOX_BIT_EXTRACT(sinv, 0, 1, 5, 6, 7);
	uint8_t out2 = SBOX_BIT_EXTRACT(sinv, 0, 1, 2, 6, 7);
	uint8_t out3 = SBOX_BIT_EXTRACT(sinv, 0, 1, 2, 3, 7);
	uint8_t out4 = SBOX_BIT_EXTRACT(sinv, 0, 1, 2, 3, 4);
	uint8_t out5 = SBOX_BIT_EXTRACT(sinv, 1, 2, 3, 4, 5);
	uint8_t out6 = SBOX_BIT_EXTRACT(sinv, 2, 3, 4, 5, 6);
	uint8_t out7 = SBOX_BIT_EXTRACT(sinv, 3, 4, 5, 6, 7);
	/* Put all the bits in out and add the constant part */
	out = (out0 | (out1 << 1) | (out2 << 2) | (out3 << 3) | (out4 << 4) | (out5 << 5) | (out6 << 6) | (out7 << 7)) ^ 0x63;
	/* Return the value */
	return out;
}

/* Optimized gmul for MixColumns constants: thanks to the mixcolumns constants,
 * we can simplify gmul
 */
#define xtime(x) ((uint8_t)(((x)<<1) ^ ((((x)>>7) & 1) * 0x1b)))
static inline uint8_t gmul_mc(uint8_t x, uint8_t y){
	return ((uint8_t)(((y & 1) * x) ^ (((y>>1) & 1) * xtime(x)) ^ (((y>>2) & 1) * xtime(xtime(x))) ^ (((y>>3) & 1) * xtime(xtime(xtime(x)))) ^ (((y>>4) & 1) * xtime(xtime(xtime(xtime(x))))))) & 0xff;
}

static inline void mix_columns(const rijndael_neon_ctx *ctx, uint8_t *state){
	uint8_t s[32];
	memcpy(s, state, 4 * ctx->Nb);

	state[0]  = gmul_mc(s[0], 2) ^ gmul_mc(s[3], 1) ^ gmul_mc(s[2], 1) ^ gmul_mc(s[1], 3);
	state[1]  = gmul_mc(s[1], 2) ^ gmul_mc(s[0], 1) ^ gmul_mc(s[3], 1) ^ gmul_mc(s[2], 3);
	state[2]  = gmul_mc(s[2], 2) ^ gmul_mc(s[1], 1) ^ gmul_mc(s[0], 1) ^ gmul_mc(s[3], 3);
	state[3]  = gmul_mc(s[3], 2) ^ gmul_mc(s[2], 1) ^ gmul_mc(s[1], 1) ^ gmul_mc(s[0], 3);
	/**/
	state[4]  = gmul_mc(s[4], 2) ^ gmul_mc(s[7], 1) ^ gmul_mc(s[6], 1) ^ gmul_mc(s[5], 3);
	state[5]  = gmul_mc(s[5], 2) ^ gmul_mc(s[4], 1) ^ gmul_mc(s[7], 1) ^ gmul_mc(s[6], 3);
	state[6]  = gmul_mc(s[6], 2) ^ gmul_mc(s[5], 1) ^ gmul_mc(s[4], 1) ^ gmul_mc(s[7], 3);
	state[7]  = gmul_mc(s[7], 2) ^ gmul_mc(s[6], 1) ^ gmul_mc(s[5], 1) ^ gmul_mc(s[4], 3);
	/**/
	state[8]  = gmul_mc(s[8], 2) ^ gmul_mc(s[11], 1) ^ gmul_mc(s[10], 1) ^ gmul_mc(s[9], 3);
	state[9]  = gmul_mc(s[9], 2) ^ gmul_mc(s[8], 1) ^ gmul_mc(s[11], 1) ^ gmul_mc(s[10], 3);
	state[10] = gmul_mc(s[10], 2) ^ gmul_mc(s[9], 1) ^ gmul_mc(s[8], 1) ^ gmul_mc(s[11], 3);
	state[11] = gmul_mc(s[11], 2) ^ gmul_mc(s[10], 1) ^ gmul_mc(s[9], 1) ^ gmul_mc(s[8], 3);
	/**/
	state[12] = gmul_mc(s[12], 2) ^ gmul_mc(s[15], 1) ^ gmul_mc(s[14], 1) ^ gmul_mc(s[13], 3);
	state[13] = gmul_mc(s[13], 2) ^ gmul_mc(s[12], 1) ^ gmul_mc(s[15], 1) ^ gmul_mc(s[14], 3);
	state[14] = gmul_mc(s[14], 2) ^ gmul_mc(s[13], 1) ^ gmul_mc(s[12], 1) ^ gmul_mc(s[15], 3);
	state[15] = gmul_mc(s[15], 2) ^ gmul_mc(s[14], 1) ^ gmul_mc(s[13], 1) ^ gmul_mc(s[12], 3);
	/**/
	if(ctx->Nb == 8){
		state[16 + 0]  = gmul_mc(s[16 + 0], 2) ^ gmul_mc(s[16 + 3], 1) ^ gmul_mc(s[16 + 2], 1) ^ gmul_mc(s[16 + 1], 3);
		state[16 + 1]  = gmul_mc(s[16 + 1], 2) ^ gmul_mc(s[16 + 0], 1) ^ gmul_mc(s[16 + 3], 1) ^ gmul_mc(s[16 + 2], 3);
		state[16 + 2]  = gmul_mc(s[16 + 2], 2) ^ gmul_mc(s[16 + 1], 1) ^ gmul_mc(s[16 + 0], 1) ^ gmul_mc(s[16 + 3], 3);
		state[16 + 3]  = gmul_mc(s[16 + 3], 2) ^ gmul_mc(s[16 + 2], 1) ^ gmul_mc(s[16 + 1], 1) ^ gmul_mc(s[16 + 0], 3);
		/**/
		state[16 + 4]  = gmul_mc(s[16 + 4], 2) ^ gmul_mc(s[16 + 7], 1) ^ gmul_mc(s[16 + 6], 1) ^ gmul_mc(s[16 + 5], 3);
		state[16 + 5]  = gmul_mc(s[16 + 5], 2) ^ gmul_mc(s[16 + 4], 1) ^ gmul_mc(s[16 + 7], 1) ^ gmul_mc(s[16 + 6], 3);
		state[16 + 6]  = gmul_mc(s[16 + 6], 2) ^ gmul_mc(s[16 + 5], 1) ^ gmul_mc(s[16 + 4], 1) ^ gmul_mc(s[16 + 7], 3);
		state[16 + 7]  = gmul_mc(s[16 + 7], 2) ^ gmul_mc(s[16 + 6], 1) ^ gmul_mc(s[16 + 5], 1) ^ gmul_mc(s[16 + 4], 3);
		/**/
		state[16 + 8]  = gmul_mc(s[16 + 8], 2) ^ gmul_mc(s[16 + 11], 1) ^ gmul_mc(s[16 + 10], 1) ^ gmul_mc(s[16 + 9], 3);
		state[16 + 9]  = gmul_mc(s[16 + 9], 2) ^ gmul_mc(s[16 + 8], 1) ^ gmul_mc(s[16 + 11], 1) ^ gmul_mc(s[16 + 10], 3);
		state[16 + 10] = gmul_mc(s[16 + 10], 2) ^ gmul_mc(s[16 + 9], 1) ^ gmul_mc(s[16 + 8], 1) ^ gmul_mc(s[16 + 11], 3);
		state[16 + 11] = gmul_mc(s[16 + 11], 2) ^ gmul_mc(s[16 + 10], 1) ^ gmul_mc(s[16 + 9], 1) ^ gmul_mc(s[16 + 8], 3);
		/**/
		state[16 + 12] = gmul_mc(s[16 + 12], 2) ^ gmul_mc(s[16 + 15], 1) ^ gmul_mc(s[16 + 14], 1) ^ gmul_mc(s[16 + 13], 3);
		state[16 + 13] = gmul_mc(s[16 + 13], 2) ^ gmul_mc(s[16 + 12], 1) ^ gmul_mc(s[16 + 15], 1) ^ gmul_mc(s[16 + 14], 3);
		state[16 + 14] = gmul_mc(s[16 + 14], 2) ^ gmul_mc(s[16 + 13], 1) ^ gmul_mc(s[16 + 12], 1) ^ gmul_mc(s[16 + 15], 3);
		state[16 + 15] = gmul_mc(s[16 + 15], 2) ^ gmul_mc(s[16 + 14], 1) ^ gmul_mc(s[16 + 13], 1) ^ gmul_mc(s[16 + 12], 3);
	}
}

static inline void add_rkey(const rijndael_neon_ctx *ctx, uint8_t *state, const uint8_t *rkey){
	uint32_t i;
	for(i = 0; i < (4 * ctx->Nb); i++){
		state[i] ^= rkey[i];
	}
}

static inline void sched(uint8_t *in, uint8_t n){
	/* Rotate word, apply sbox and rcon */
	uint8_t t = in[0];
	in[0] = sbox(in[1]) ^ rcon[n];
	in[1] = sbox(in[2]);
	in[2] = sbox(in[3]);
	in[3] = sbox(t);

	return;
}

#define ROTR32(x, n) ((x << (32 - n)) | (x >> n))

static inline uint32_t AES_sbox_x4(uint32_t in) {
  uint8x16_t sbox_val = vreinterpretq_u8_u32(vdupq_n_u32(in));
  sbox_val = vaeseq_u8(sbox_val, vdupq_n_u8(0));

  return vgetq_lane_u32(vreinterpretq_u32_u8(sbox_val), 0);
}

/* Encryption key schedule */
static int rijndael_setkey_enc(rijndael_neon_ctx *ctx, const uint8_t *key, rijndael_type rtype)
{

	uint32_t s;
	uint32_t i;
	int ret = -1;

	if((ctx == NULL) || (key == NULL)){
		goto err;
	}
	switch(rtype){
		case AES128:{
			ctx->Nr = 10;
			ctx->Nk = 4;
			ctx->Nb = 4;
			break;
		}
		case AES256:{
			ctx->Nr = 14;
			ctx->Nk = 8;
			ctx->Nb = 4;
			break;
		}
		case RIJNDAEL_256_256:{
			ctx->Nr = 14;
			ctx->Nk = 8;
			ctx->Nb = 8;
			break;
		}
		default:{
			ret = -1;
			goto err;
		}
	}
	ctx->rtype = rtype;

	/* Perform the key schedule */
	memcpy(&ctx->bytearray, key, 4 * ctx->Nk);
	uint8_t rcon = 1;
	uint32_t aux;
	uint8_t total_it = (ctx->Nr+1)*(ctx->Nb);
	uint8_t start_it = (ctx->Nk);
	uint32x4_t helper;
	for (i=start_it;i<total_it; i+= 4) {
		if (i%(ctx->Nk) == 0) {
			aux = ctx->bytearray[i-1];
			s = AES_sbox_x4(aux);
			aux = ctx->bytearray[i-1*(ctx->Nk)];
			aux = ROTR32(s, 8) ^ rcon ^ aux;
			ctx->bytearray[i] = aux;
			rcon = (rcon << 1) ^ ((rcon >> 7) * 0x11b);
		}
		else if ((ctx->Nk)>6 && (i%(ctx->Nk)== 4)) {
			aux = ctx->bytearray[i-1];
			s = AES_sbox_x4(aux);
			aux = ctx->bytearray[i-1*(ctx->Nk)];
			aux = s ^ aux;
			ctx->bytearray[i] = aux;
		}
		ctx->bytearray[i+1] = ctx->bytearray[i] ^ ctx->bytearray[i+(1-ctx->Nk)];
		ctx->bytearray[i+2] = ctx->bytearray[i+1] ^ ctx->bytearray[i+(2-ctx->Nk)];
		ctx->bytearray[i+3] = ctx->bytearray[i+2] ^ ctx->bytearray[i+(3-ctx->Nk)];
	}
	for (i=0;i<total_it;i+=4) {
		helper = vld1q_u32(ctx->bytearray+i);
		ctx->rk[i>>2] = vreinterpretq_u8_u32(helper);
	}

	ret = 0;

err:
	return ret;
}

/* Encryption primitive */
static int rinjdael_enc(const rijndael_neon_ctx *ctx, const uint8_t *data_in, uint8_t *data_out)
{
	uint32_t r;
	/* Our local state (maximum 16 bytes)*/
	uint8x16_t state[2];
	state[1] = vdupq_n_u8(0);
	int ret = -1;


	uint8x16_t msk_half = {0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0, 0,
							0xFF, 0xFF, 0, 0, 0xFF, 0, 0, 0};
	
	uint8x16_t tbl = {0,1,6,7,4,5,10,11,8,9,14,15,12,13,2,3};
	uint8x16_t swp_halfs;

	if((ctx == NULL) || (data_in == NULL) || (data_out == NULL)){
		goto err;
	}
	/* Sanity check for array access */
	if((4 * ctx->Nb * (ctx->Nr + 1)) > sizeof(ctx->rk)){
		goto err;
	}
	state[0] = vld1q_u8(data_in);
	if (ctx->Nb == 8) {
		state[1] = vld1q_u8(data_in+16);
	}

	/* All our rounds except the last one */
	for(r = 0; r < (ctx->Nr) - 1; r++){
		if (ctx->Nb != 8) {
			state[0] = vaeseq_u8(state[0],ctx->rk[r]);
			state[0] = vaesmcq_u8(state[0]);
		}
		else if (ctx->Nb == 8) {
			state[0] = vaeseq_u8(state[0],ctx->rk[2*r]);
			state[1] = vaeseq_u8(state[1],ctx->rk[2*r+1]);
			state[0] = vqtbl1q_u8(state[0],tbl);
			state[1] = vqtbl1q_u8(state[1],tbl);
			swp_halfs = vbslq_u8(msk_half,state[1],state[0]);
			state[0] = vbslq_u8(msk_half,state[0],state[1]);
			state[1] = swp_halfs;
			state[0] = vaesmcq_u8(state[0]);
			state[1] = vaesmcq_u8(state[1]);
		}
	}
	/* Last round without mixcolumns */
	if (ctx->Nb != 8) {
		state[0] = vaeseq_u8(state[0],ctx->rk[(ctx->Nr)-1]);
		state[0] = veorq_u8(state[0],ctx->rk[ctx->Nr]);
	}
	else if (ctx->Nb == 8) {
		state[0] = vaeseq_u8(state[0],ctx->rk[2*(ctx->Nr)-2]);
		state[1] = vaeseq_u8(state[1],ctx->rk[2*(ctx->Nr)-1]);
		
		state[0] = vqtbl1q_u8(state[0],tbl);
		state[1] = vqtbl1q_u8(state[1],tbl);
		swp_halfs = vbslq_u8(msk_half,state[1],state[0]);
		state[0] = vbslq_u8(msk_half,state[0],state[1]);
		state[1] = swp_halfs;

		state[0] = veorq_u8(state[0],ctx->rk[2*(ctx->Nr)]);
		state[1] = veorq_u8(state[1],ctx->rk[2*(ctx->Nr)+1]);

	}

	/* Copy back state to ciphertext */
	vst1q_u8(data_out,state[0]);
	if (ctx->Nb == 8) {
		vst1q_u8(data_out+16,state[1]);
	}

	ret = 0;

err:
	return ret;
}

/* ==== Public APIs ===== */

WEAK int aes128_neon_setkey_enc(rijndael_neon_ctx *ctx, const uint8_t key[16])
{
	return rijndael_setkey_enc(ctx, key, AES128);
}

WEAK int aes256_neon_setkey_enc(rijndael_neon_ctx *ctx, const uint8_t key[32])
{
	return rijndael_setkey_enc(ctx, key, AES256);
}

WEAK int rijndael256_neon_setkey_enc(rijndael_neon_ctx *ctx, const uint8_t key[32])
{
	return rijndael_setkey_enc(ctx, key, RIJNDAEL_256_256);
}

WEAK int aes128_neon_enc(const rijndael_neon_ctx *ctx, const uint8_t data_in[16], uint8_t data_out[16])
{
	int ret = -1;

	if(ctx->rtype != AES128){
		goto err;
	}

	ret = rinjdael_enc(ctx, data_in, data_out);
	
err:
	return ret;
}

WEAK int aes128_neon_enc_x2(const rijndael_neon_ctx *ctx1, const rijndael_neon_ctx *ctx2, const uint8_t plainText1[16], const uint8_t plainText2[16], uint8_t cipherText1[16], uint8_t cipherText2[16])
{
	int ret;

	ret  = aes128_neon_enc(ctx1, plainText1, cipherText1);
	ret |= aes128_neon_enc(ctx2, plainText2, cipherText2);

	return ret;
}

WEAK int aes128_neon_enc_x4(const rijndael_neon_ctx *ctx1, const rijndael_neon_ctx *ctx2, const rijndael_neon_ctx *ctx3, const rijndael_neon_ctx *ctx4,
                const uint8_t plainText1[16], const uint8_t plainText2[16], const uint8_t plainText3[16], const uint8_t plainText4[16],
                uint8_t cipherText1[16], uint8_t cipherText2[16], uint8_t cipherText3[16], uint8_t cipherText4[16])
{
	int ret;

	ret  = aes128_neon_enc(ctx1, plainText1, cipherText1);
	ret |= aes128_neon_enc(ctx2, plainText2, cipherText2);
	ret |= aes128_neon_enc(ctx3, plainText3, cipherText3);
	ret |= aes128_neon_enc(ctx4, plainText4, cipherText4);

	return ret;
}

WEAK int aes128_neon_enc_x8(const rijndael_neon_ctx *ctx1, const rijndael_neon_ctx *ctx2, const rijndael_neon_ctx *ctx3, const rijndael_neon_ctx *ctx4,
                  const rijndael_neon_ctx *ctx5, const rijndael_neon_ctx *ctx6, const rijndael_neon_ctx *ctx7, const rijndael_neon_ctx *ctx8,
                const uint8_t plainText1[16], const uint8_t plainText2[16], const uint8_t plainText3[16], const uint8_t plainText4[16],
                const uint8_t plainText5[16], const uint8_t plainText6[16], const uint8_t plainText7[16], const uint8_t plainText8[16],
                uint8_t cipherText1[16], uint8_t cipherText2[16], uint8_t cipherText3[16], uint8_t cipherText4[16],
                uint8_t cipherText5[16], uint8_t cipherText6[16], uint8_t cipherText7[16], uint8_t cipherText8[16])
{
	int ret = 0;
        ret |= aes128_neon_enc_x4(ctx1, ctx2, ctx3, ctx4, plainText1, plainText2, plainText3, plainText4, cipherText1, cipherText2, cipherText3, cipherText4);
        ret |= aes128_neon_enc_x4(ctx5, ctx6, ctx7, ctx8, plainText5, plainText6, plainText7, plainText8, cipherText5, cipherText6, cipherText7, cipherText8);
	return ret;
}

WEAK int aes256_neon_enc(const rijndael_neon_ctx *ctx, const uint8_t data_in[16], uint8_t data_out[16])
{
	int ret = -1;

	if(ctx->rtype != AES256){
		goto err;
	}

	ret = rinjdael_enc(ctx, data_in, data_out);
	
err:
	return ret;
}

WEAK int aes256_neon_enc_x2(const rijndael_neon_ctx *ctx1, const rijndael_neon_ctx *ctx2, const uint8_t plainText1[16], const uint8_t plainText2[16], uint8_t cipherText1[16], uint8_t cipherText2[16])
{
	int ret;

	ret  = aes256_neon_enc(ctx1, plainText1, cipherText1);
	ret |= aes256_neon_enc(ctx2, plainText2, cipherText2);

	return ret;
}


WEAK int aes256_neon_enc_x4(const rijndael_neon_ctx *ctx1, const rijndael_neon_ctx *ctx2, const rijndael_neon_ctx *ctx3, const rijndael_neon_ctx *ctx4,
                const uint8_t plainText1[16], const uint8_t plainText2[16], const uint8_t plainText3[16], const uint8_t plainText4[16],
                uint8_t cipherText1[16], uint8_t cipherText2[16], uint8_t cipherText3[16], uint8_t cipherText4[16])
{
	int ret;

	ret  = aes256_neon_enc(ctx1, plainText1, cipherText1);
	ret |= aes256_neon_enc(ctx2, plainText2, cipherText2);
	ret |= aes256_neon_enc(ctx3, plainText3, cipherText3);
	ret |= aes256_neon_enc(ctx4, plainText4, cipherText4);

	return ret;
}

WEAK int aes256_neon_enc_x8(const rijndael_neon_ctx *ctx1, const rijndael_neon_ctx *ctx2, const rijndael_neon_ctx *ctx3, const rijndael_neon_ctx *ctx4,
                  const rijndael_neon_ctx *ctx5, const rijndael_neon_ctx *ctx6, const rijndael_neon_ctx *ctx7, const rijndael_neon_ctx *ctx8,
                const uint8_t plainText1[16], const uint8_t plainText2[16], const uint8_t plainText3[16], const uint8_t plainText4[16],
                const uint8_t plainText5[16], const uint8_t plainText6[16], const uint8_t plainText7[16], const uint8_t plainText8[16],
                uint8_t cipherText1[16], uint8_t cipherText2[16], uint8_t cipherText3[16], uint8_t cipherText4[16],
                uint8_t cipherText5[16], uint8_t cipherText6[16], uint8_t cipherText7[16], uint8_t cipherText8[16])
{       
	int ret = 0;
        ret |= aes256_neon_enc_x4(ctx1, ctx2, ctx3, ctx4, plainText1, plainText2, plainText3, plainText4, cipherText1, cipherText2, cipherText3, cipherText4);
        ret |= aes256_neon_enc_x4(ctx5, ctx6, ctx7, ctx8, plainText5, plainText6, plainText7, plainText8, cipherText5, cipherText6, cipherText7, cipherText8);
	return ret;
}

WEAK int rijndael256_neon_enc(const rijndael_neon_ctx *ctx, const uint8_t data_in[32], uint8_t data_out[32])
{
	int ret = -1;

	if(ctx->rtype != RIJNDAEL_256_256){
		goto err;
	}

	ret = rinjdael_enc(ctx, data_in, data_out);
	
err:
	return ret;
}

WEAK int rijndael256_neon_enc_x2(const rijndael_neon_ctx *ctx1, const rijndael_neon_ctx *ctx2,
                        const uint8_t plainText1[32], const uint8_t plainText2[32],
                        uint8_t cipherText1[32], uint8_t cipherText2[32])
{
	int ret;

	ret  = rijndael256_neon_enc(ctx1, plainText1, cipherText1);
	ret |= rijndael256_neon_enc(ctx2, plainText2, cipherText2);

	return ret;

}


WEAK int rijndael256_neon_enc_x4(const rijndael_neon_ctx *ctx1, const rijndael_neon_ctx *ctx2, const rijndael_neon_ctx *ctx3, const rijndael_neon_ctx *ctx4,
                const uint8_t plainText1[32], const uint8_t plainText2[32], const uint8_t plainText3[32], const uint8_t plainText4[32],
                uint8_t cipherText1[32], uint8_t cipherText2[32], uint8_t cipherText3[32], uint8_t cipherText4[32])
{
	int ret;

	ret  = rijndael256_neon_enc(ctx1, plainText1, cipherText1);
	ret |= rijndael256_neon_enc(ctx2, plainText2, cipherText2);
	ret |= rijndael256_neon_enc(ctx3, plainText3, cipherText3);
	ret |= rijndael256_neon_enc(ctx4, plainText4, cipherText4);

	return ret;
}

WEAK int rijndael256_neon_enc_x8(const rijndael_neon_ctx *ctx1, const rijndael_neon_ctx *ctx2, const rijndael_neon_ctx *ctx3, const rijndael_neon_ctx *ctx4,
                  const rijndael_neon_ctx *ctx5, const rijndael_neon_ctx *ctx6, const rijndael_neon_ctx *ctx7, const rijndael_neon_ctx *ctx8,
                const uint8_t plainText1[32], const uint8_t plainText2[32], const uint8_t plainText3[32], const uint8_t plainText4[32],
                const uint8_t plainText5[32], const uint8_t plainText6[32], const uint8_t plainText7[32], const uint8_t plainText8[32],
                uint8_t cipherText1[32], uint8_t cipherText2[32], uint8_t cipherText3[32], uint8_t cipherText4[32],
                uint8_t cipherText5[32], uint8_t cipherText6[32], uint8_t cipherText7[32], uint8_t cipherText8[32])
{
	int ret = 0;
        ret |= rijndael256_neon_enc_x4(ctx1, ctx2, ctx3, ctx4, plainText1, plainText2, plainText3, plainText4, cipherText1, cipherText2, cipherText3, cipherText4);
        ret |= rijndael256_neon_enc_x4(ctx5, ctx6, ctx7, ctx8, plainText5, plainText6, plainText7, plainText8, cipherText5, cipherText6, cipherText7, cipherText8);
	return ret;
}

#else /* !RIJNDAEL_CONSTANT_TIME_REF, */
/*
 * Dummy definition to avoid the empty translation unit ISO C warning
 */
typedef int dummy;
#endif
