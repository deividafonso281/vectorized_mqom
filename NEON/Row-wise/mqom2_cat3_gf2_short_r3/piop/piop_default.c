#include "piop.h"
#if MQOM2_PARAM_WITH_STATISTICAL_BATCHING == 1
#include "xof.h"
#endif
#include "piop_cache.h"
#include "benchmark.h"
#include "expand_mq.h"

/* Some useful types definition */
/* NOTE: we use multi-dimensional array types to ease usage of indices.
 * While we can use pure local variables, these become too large to fit the stack
 * and heap allocation is needed. */
typedef field_ext_elt (*MatrixSetMQ)[MQOM2_PARAM_MQ_N_times8+1][FIELD_EXT_PACKING(MQOM2_PARAM_MQ_N_times8)];
typedef field_ext_elt (*VectorSetMQ)[FIELD_EXT_PACKING(MQOM2_PARAM_MQ_N)];

static inline void compute_t1(const field_ext_elt A_hat[MQOM2_PARAM_MQ_M/MQOM2_PARAM_MU][MQOM2_PARAM_MQ_N_times8+1][FIELD_EXT_PACKING(MQOM2_PARAM_MQ_N_times8)], const field_base_elt x[FIELD_BASE_PACKING(MQOM2_PARAM_MQ_N)], const field_ext_elt b_hat[MQOM2_PARAM_MQ_M/MQOM2_PARAM_MU][FIELD_EXT_PACKING(MQOM2_PARAM_MQ_N)], field_ext_elt t1[2][FIELD_EXT_PACKING(MQOM2_PARAM_MQ_N)], uint32_t i, piop_cache *cache)
{	
	if(is_entry_active_piop_cache(cache, i)){
		get_entry_piop_cache(cache, i, t1[0]);
	}
	else{
		field_ext_base_mat_mult((field_ext_elt*)A_hat[i], x, (field_ext_elt*)t1[0], (field_ext_elt*)t1[1], MQOM2_PARAM_MQ_N);
		field_ext_vect_add(t1[0], b_hat[i], t1[0], MQOM2_PARAM_MQ_N);
        field_ext_vect_add(t1[1], b_hat[i+1], t1[1], MQOM2_PARAM_MQ_N);
		set_entry_piop_cache(cache, i, t1[0]);
	}

	return;
}

static int ComputePz(const field_ext_elt x0_e[FIELD_EXT_PACKING(MQOM2_PARAM_MQ_N)], const field_base_elt x[FIELD_BASE_PACKING(MQOM2_PARAM_MQ_N)], const field_ext_elt A_hat[MQOM2_PARAM_MQ_M/MQOM2_PARAM_MU][MQOM2_PARAM_MQ_N_times8+1][FIELD_EXT_PACKING(MQOM2_PARAM_MQ_N_times8)], const field_ext_elt b_hat[MQOM2_PARAM_MQ_M/MQOM2_PARAM_MU][FIELD_EXT_PACKING(MQOM2_PARAM_MQ_N)],field_ext_elt z0[FIELD_EXT_PACKING(MQOM2_PARAM_MQ_M/MQOM2_PARAM_MU)], field_ext_elt z1[FIELD_EXT_PACKING(MQOM2_PARAM_MQ_M/MQOM2_PARAM_MU)], piop_cache *cache) {
    int ret = -1;
    uint32_t i;
    
    field_ext_elt t0[2][FIELD_EXT_PACKING(MQOM2_PARAM_MQ_N)];
    field_ext_elt t1[2][FIELD_EXT_PACKING(MQOM2_PARAM_MQ_N)];
    field_ext_elt z_0i[2], z_1i[2];

    for(i = 0; i < MQOM2_PARAM_MQ_M/(2*MQOM2_PARAM_MU); i++) {
        __BENCHMARK_START__(BS_PIOP_MAT_MUL_EXT);
        field_ext_mat_mult((field_ext_elt*)A_hat[i], x0_e, t0[0],t0[1], MQOM2_PARAM_MQ_N);
        __BENCHMARK_STOP__(BS_PIOP_MAT_MUL_EXT);
        __BENCHMARK_START__(BS_PIOP_COMPUTE_T1);
        if(is_entry_active_piop_cache(cache, 2*i)){
            get_entry_piop_cache(cache, 2*i, t1[0]);
            get_entry_piop_cache(cache, 2*i+1, t1[1]);
        }
        else{ 
            field_ext_base_mat_mult((field_ext_elt*)A_hat[i], x, (field_ext_elt*)t1[0], (field_ext_elt*)t1[1], MQOM2_PARAM_MQ_N);
            field_ext_vect_add(t1[0], b_hat[2*i], t1[0], MQOM2_PARAM_MQ_N);
            field_ext_vect_add(t1[1], b_hat[2*i+1], t1[1], MQOM2_PARAM_MQ_N);
            set_entry_piop_cache(cache, 2*i, t1[0]);
            set_entry_piop_cache(cache, 2*i+1, t1[1]);
        }
        __BENCHMARK_STOP__(BS_PIOP_COMPUTE_T1);

        /* Compute P_{z,i}(X) = z_{0,i} + z_{1,i} X = P_t(X)^T P_x(X) - y_i X^2 */
        __BENCHMARK_START__(BS_PIOP_COMPUTE_PZI);
        z_0i[0] = field_ext_vect_mult(t0[0], x0_e, MQOM2_PARAM_MQ_N);
        z_0i[1] = field_ext_vect_mult(t0[1], x0_e, MQOM2_PARAM_MQ_N);
    field_ext_elt t0_x[2];
	field_ext_elt t0_x0[2];
	t0_x[0] = field_ext_vect_mult(t1[0], x0_e, MQOM2_PARAM_MQ_N); /* t0^T x */
	t0_x0[0] = field_ext_base_vect_mult(t0[0], x, MQOM2_PARAM_MQ_N);   /* t1^T x0[e] */
    t0_x[1] = field_ext_vect_mult(t1[1], x0_e, MQOM2_PARAM_MQ_N); /* t0^T x */
	t0_x0[1] = field_ext_base_vect_mult(t0[1], x, MQOM2_PARAM_MQ_N);   /* t1^T x0[e] */
        field_ext_vect_add(&t0_x[0], &t0_x0[0], &z_1i[0], 1);
        field_ext_vect_pack(z_0i[0], z0, 2*i);
        field_ext_vect_pack(z_1i[0], z1, 2*i);
        field_ext_vect_add(&t0_x[1], &t0_x0[1], &z_1i[1], 1);
        field_ext_vect_pack(z_0i[1], z0, 2*i+1);
        field_ext_vect_pack(z_1i[1], z1, 2*i+1);
        __BENCHMARK_STOP__(BS_PIOP_COMPUTE_PZI);
    }
    // LAST BLOCK
    #if (MQOM2_PARAM_MQ_M/(MQOM2_PARAM_MU)) & 1
        __BENCHMARK_START__(BS_PIOP_MAT_MUL_EXT);
        field_ext_mat_mult_single((field_ext_elt*)A_hat[i], x0_e, t0[0], MQOM2_PARAM_MQ_N);
        __BENCHMARK_STOP__(BS_PIOP_MAT_MUL_EXT);
        __BENCHMARK_START__(BS_PIOP_COMPUTE_T1);
        if(is_entry_active_piop_cache(cache, 2*i)){
		    get_entry_piop_cache(cache, 2*i, t1[0]);
        }
        else{
            field_ext_base_mat_mult_single((field_ext_elt*)A_hat[i], x, (field_ext_elt*)t1[0], MQOM2_PARAM_MQ_N);
		    field_ext_vect_add(t1[0], b_hat[2*i], t1[0], MQOM2_PARAM_MQ_N); 
            set_entry_piop_cache(cache, 2*i, t1[0]);
        }
        __BENCHMARK_STOP__(BS_PIOP_COMPUTE_T1);

        /* Compute P_{z,i}(X) = z_{0,i} + z_{1,i} X = P_t(X)^T P_x(X) - y_i X^2 */
        __BENCHMARK_START__(BS_PIOP_COMPUTE_PZI);
        z_0i[0] = field_ext_vect_mult(t0[0], x0_e, MQOM2_PARAM_MQ_N);
    field_ext_elt t0_x[2];
	field_ext_elt t0_x0[2];
	t0_x[0] = field_ext_vect_mult(t1[0], x0_e, MQOM2_PARAM_MQ_N); /* t0^T x */
	t0_x0[0] = field_ext_base_vect_mult(t0[0], x, MQOM2_PARAM_MQ_N);   /* t1^T x0[e] */
        field_ext_vect_add(&t0_x[0], &t0_x0[0], &z_1i[0], 1);
        field_ext_vect_pack(z_0i[0], z0, 2*i);
        field_ext_vect_pack(z_1i[0], z1, 2*i);
        __BENCHMARK_STOP__(BS_PIOP_COMPUTE_PZI);
    #endif

    ret = 0;
    return ret;
}
   
int ComputePAlpha_default(const uint8_t com[MQOM2_PARAM_DIGEST_SIZE], const field_ext_elt x0[MQOM2_PARAM_TAU][FIELD_EXT_PACKING(MQOM2_PARAM_MQ_N)], const field_ext_elt u0[MQOM2_PARAM_TAU][FIELD_EXT_PACKING(MQOM2_PARAM_ETA)], const field_ext_elt u1[MQOM2_PARAM_TAU][FIELD_EXT_PACKING(MQOM2_PARAM_ETA)], const field_base_elt x[FIELD_BASE_PACKING(MQOM2_PARAM_MQ_N)], const uint8_t mseed_eq[2 * MQOM2_PARAM_SEED_SIZE], field_ext_elt alpha0[MQOM2_PARAM_TAU][FIELD_EXT_PACKING(MQOM2_PARAM_ETA)], field_ext_elt alpha1[MQOM2_PARAM_TAU][FIELD_EXT_PACKING(MQOM2_PARAM_ETA)])
{
    int ret = -1;
    uint32_t e;
    field_ext_elt *_A_hat = NULL;
    field_ext_elt *_b_hat = NULL;

    /* Initialize the PIOP cache for t1 */
    piop_cache *t1_cache = init_piop_cache(MQOM2_PARAM_MQ_M);

    __BENCHMARK_START__(BS_PIOP_EXPAND_BATCHING_MAT);
#if MQOM2_PARAM_WITH_STATISTICAL_BATCHING == 1
    uint32_t i;
    xof_context xof_ctx;
    field_ext_elt Gamma[MQOM2_PARAM_ETA][FIELD_EXT_PACKING(MQOM2_PARAM_MQ_M/MQOM2_PARAM_MU)];
    uint8_t stream[MQOM2_PARAM_ETA*BYTE_SIZE_FIELD_EXT(MQOM2_PARAM_MQ_M/MQOM2_PARAM_MU)];
    ret = xof_init(&xof_ctx); ERR(ret, err);
    ret = xof_update(&xof_ctx, (const uint8_t*) "\x08", 1); ERR(ret, err);
    ret = xof_update(&xof_ctx, com, MQOM2_PARAM_DIGEST_SIZE); ERR(ret, err);
    ret = xof_squeeze(&xof_ctx, stream, MQOM2_PARAM_ETA*BYTE_SIZE_FIELD_EXT(MQOM2_PARAM_MQ_M/MQOM2_PARAM_MU));
    for(i=0; i<MQOM2_PARAM_ETA; i++){
        field_ext_parse(&stream[i*BYTE_SIZE_FIELD_EXT(MQOM2_PARAM_MQ_M/MQOM2_PARAM_MU)], MQOM2_PARAM_MQ_M/MQOM2_PARAM_MU, Gamma[i]);
    }
    
#else
    (void) com;
#endif
    __BENCHMARK_STOP__(BS_PIOP_EXPAND_BATCHING_MAT);

    /* Expand the public matrices */
    _A_hat = (field_ext_elt*)mqom_malloc(((MQOM2_PARAM_MQ_M+MQOM2_PARAM_MU)/(2*MQOM2_PARAM_MU)) * (MQOM2_PARAM_MQ_N_times8 + 1) * FIELD_EXT_PACKING(MQOM2_PARAM_MQ_N_times8) * sizeof(field_ext_elt));
    if(_A_hat == NULL){
        ret = -1;
        goto err;
    }
    _b_hat = (field_ext_elt*)mqom_malloc((MQOM2_PARAM_MQ_M/MQOM2_PARAM_MU) * FIELD_EXT_PACKING(MQOM2_PARAM_MQ_N) * sizeof(field_ext_elt));
    if(_b_hat == NULL){
        ret = -1;
        goto err;
    }
    MatrixSetMQ A_hat = (MatrixSetMQ)_A_hat;
    VectorSetMQ b_hat = (VectorSetMQ)_b_hat;

    __BENCHMARK_START__(BS_PIOP_EXPAND_MQ);
    ret = ExpandEquations(mseed_eq, A_hat, b_hat); ERR(ret, err);
    __BENCHMARK_STOP__(BS_PIOP_EXPAND_MQ);

    field_ext_elt z0[FIELD_EXT_PACKING(MQOM2_PARAM_MQ_M/MQOM2_PARAM_MU)], z1[FIELD_EXT_PACKING(MQOM2_PARAM_MQ_M/MQOM2_PARAM_MU)];
    for(e = 0; e < MQOM2_PARAM_TAU; e++) {
        ret = ComputePz(x0[e], x, A_hat, b_hat, z0, z1, t1_cache); ERR(ret, err);
        __BENCHMARK_START__(BS_PIOP_BATCH_AND_MASK);
#if MQOM2_PARAM_WITH_STATISTICAL_BATCHING == 1
        for(i=0; i<MQOM2_PARAM_ETA; i++) {
            field_ext_vect_pack(
                field_ext_vect_mult(Gamma[i], z0, MQOM2_PARAM_MQ_M/MQOM2_PARAM_MU),
                alpha0[e], i
            );
        }
        for(i=0; i<MQOM2_PARAM_ETA; i++) {
            field_ext_vect_pack(
                field_ext_vect_mult(Gamma[i], z1, MQOM2_PARAM_MQ_M/MQOM2_PARAM_MU),
                alpha1[e], i
            );
        }
        field_ext_vect_add(alpha0[e], u0[e], alpha0[e], MQOM2_PARAM_ETA);
        field_ext_vect_add(alpha1[e], u1[e], alpha1[e], MQOM2_PARAM_ETA);
#else
        field_ext_vect_add(z0, u0[e], alpha0[e], MQOM2_PARAM_ETA);
        field_ext_vect_add(z1, u1[e], alpha1[e], MQOM2_PARAM_ETA);
#endif
        __BENCHMARK_STOP__(BS_PIOP_BATCH_AND_MASK);
    }

    ret = 0;
err:
    destroy_piop_cache(t1_cache);
    if(_A_hat){
        mqom_free(_A_hat);
    }
    if(_b_hat){
        mqom_free(_b_hat);      
    }
    return ret;
}

/***************************************************************/
/***************************************************************/

static int ComputePzEval(field_ext_elt r, const field_ext_elt v_x[FIELD_EXT_PACKING(MQOM2_PARAM_MQ_N)], field_ext_elt A_hat[MQOM2_PARAM_MQ_M/MQOM2_PARAM_MU][MQOM2_PARAM_MQ_N_times8+1][FIELD_EXT_PACKING(MQOM2_PARAM_MQ_N_times8)], field_ext_elt b_hat[MQOM2_PARAM_MQ_M/MQOM2_PARAM_MU][FIELD_EXT_PACKING(MQOM2_PARAM_MQ_N)], const field_ext_elt y[FIELD_EXT_PACKING(MQOM2_PARAM_MQ_M/MQOM2_PARAM_MU)], field_ext_elt v_z[FIELD_EXT_PACKING(MQOM2_PARAM_MQ_M/MQOM2_PARAM_MU)]) {
    int ret = -1;
    uint32_t i;

    field_ext_elt v_t[2][FIELD_EXT_PACKING(MQOM2_PARAM_MQ_N)];
    field_ext_elt tmp[2][FIELD_EXT_PACKING(MQOM2_PARAM_MQ_N)];
    field_ext_elt v_zi[2];

    field_ext_elt r2 = field_ext_mult(r, r);
    field_ext_elt y_r2[FIELD_EXT_PACKING(MQOM2_PARAM_MQ_M/MQOM2_PARAM_MU)];
    field_ext_constant_vect_mult(r2, y, y_r2, MQOM2_PARAM_MQ_M/MQOM2_PARAM_MU);

    for(i = 0; i < MQOM2_PARAM_MQ_M/(2*MQOM2_PARAM_MU); i++) {
        /* Compute v_t = P_t(r) = A_i P_x(r) + b_i r */
        __BENCHMARK_START__(BS_PIOP_MAT_MUL_EXT_VERIF);
        field_ext_mat_mult((field_ext_elt*)A_hat[i], v_x, tmp[0], tmp[1], MQOM2_PARAM_MQ_N);
        __BENCHMARK_STOP__(BS_PIOP_MAT_MUL_EXT_VERIF);
        field_ext_constant_vect_mult(r, b_hat[2*i], v_t[0], MQOM2_PARAM_MQ_N);
        field_ext_constant_vect_mult(r, b_hat[2*i+1], v_t[1], MQOM2_PARAM_MQ_N);
        field_ext_vect_add(v_t[0], tmp[0], v_t[0], MQOM2_PARAM_MQ_N);
        field_ext_vect_add(v_t[1], tmp[1], v_t[1], MQOM2_PARAM_MQ_N);

        /* Compute v_{z,i} = P_{z,i}(r) = v_t^T v_r - y_i r^2 */
        v_zi[0] = field_ext_vect_mult(v_t[0], v_x, MQOM2_PARAM_MQ_N);
        v_zi[1] = field_ext_vect_mult(v_t[1], v_x, MQOM2_PARAM_MQ_N);
        field_ext_vect_pack(v_zi[0], v_z, 2*i);
        field_ext_vect_pack(v_zi[1], v_z, 2*i+1);
    }
    // LAST BLOCK
    #if (MQOM2_PARAM_MQ_M/(MQOM2_PARAM_MU)) & 1
        __BENCHMARK_START__(BS_PIOP_MAT_MUL_EXT_VERIF);
        field_ext_mat_mult_single((field_ext_elt*)A_hat[i], v_x, tmp[0], MQOM2_PARAM_MQ_N);
        __BENCHMARK_STOP__(BS_PIOP_MAT_MUL_EXT_VERIF);
        field_ext_constant_vect_mult(r, b_hat[2*i], v_t[0], MQOM2_PARAM_MQ_N);
        field_ext_vect_add(v_t[0], tmp[0], v_t[0], MQOM2_PARAM_MQ_N);

        /* Compute v_{z,i} = P_{z,i}(r) = v_t^T v_r - y_i r^2 */
        v_zi[0] = field_ext_vect_mult(v_t[0], v_x, MQOM2_PARAM_MQ_N);
        field_ext_vect_pack(v_zi[0], v_z, 2*i);
    #endif
    field_ext_vect_add(v_z, y_r2, v_z, MQOM2_PARAM_MQ_M/MQOM2_PARAM_MU);

    ret = 0;
    return ret;
}
 
int RecomputePAlpha_default(const uint8_t com[MQOM2_PARAM_DIGEST_SIZE], const field_ext_elt alpha1[MQOM2_PARAM_TAU][FIELD_EXT_PACKING(MQOM2_PARAM_ETA)], const uint16_t i_star[MQOM2_PARAM_TAU], const field_ext_elt x_eval[MQOM2_PARAM_TAU][FIELD_EXT_PACKING(MQOM2_PARAM_MQ_N)], const field_ext_elt u_eval[MQOM2_PARAM_TAU][FIELD_EXT_PACKING(MQOM2_PARAM_ETA)], const uint8_t mseed_eq[2 * MQOM2_PARAM_SEED_SIZE], const field_ext_elt y[FIELD_EXT_PACKING(MQOM2_PARAM_MQ_M/MQOM2_PARAM_MU)], field_ext_elt alpha0[MQOM2_PARAM_TAU][FIELD_EXT_PACKING(MQOM2_PARAM_ETA)])
{
    int ret = -1;
    uint32_t e;
    field_ext_elt *_A_hat = NULL;
    field_ext_elt *_b_hat = NULL;

#if MQOM2_PARAM_WITH_STATISTICAL_BATCHING == 1
    uint32_t i;
    xof_context xof_ctx;
    field_ext_elt Gamma[MQOM2_PARAM_ETA][FIELD_EXT_PACKING(MQOM2_PARAM_MQ_M/MQOM2_PARAM_MU)];
    uint8_t stream[MQOM2_PARAM_ETA*BYTE_SIZE_FIELD_EXT(MQOM2_PARAM_MQ_M/MQOM2_PARAM_MU)];
    ret = xof_init(&xof_ctx); ERR(ret, err);
    ret = xof_update(&xof_ctx, (const uint8_t*) "\x08", 1); ERR(ret, err);
    ret = xof_update(&xof_ctx, com, MQOM2_PARAM_DIGEST_SIZE); ERR(ret, err);
    ret = xof_squeeze(&xof_ctx, stream, MQOM2_PARAM_ETA*BYTE_SIZE_FIELD_EXT(MQOM2_PARAM_MQ_M/MQOM2_PARAM_MU));
    for(i=0; i<MQOM2_PARAM_ETA; i++){
        field_ext_parse(&stream[i*BYTE_SIZE_FIELD_EXT(MQOM2_PARAM_MQ_M/MQOM2_PARAM_MU)], MQOM2_PARAM_MQ_M/MQOM2_PARAM_MU, Gamma[i]);
    }

#else
    (void) com;
#endif

    _A_hat = (field_ext_elt*)mqom_malloc(((MQOM2_PARAM_MQ_M+MQOM2_PARAM_MU)/(2*MQOM2_PARAM_MU)) * (MQOM2_PARAM_MQ_N_times8 + 1) * FIELD_EXT_PACKING(MQOM2_PARAM_MQ_N_times8) * sizeof(field_ext_elt));
    if(_A_hat == NULL){
        ret = -1;
        goto err;
    }
    _b_hat = (field_ext_elt*)mqom_malloc((MQOM2_PARAM_MQ_M/MQOM2_PARAM_MU) * FIELD_EXT_PACKING(MQOM2_PARAM_MQ_N) * sizeof(field_ext_elt));
    if(_b_hat == NULL){
        ret = -1;
        goto err;
    }
    MatrixSetMQ A_hat = (MatrixSetMQ)_A_hat;
    VectorSetMQ b_hat = (VectorSetMQ)_b_hat;

    __BENCHMARK_START__(BS_PIOP_EXPAND_MQ_VERIF);
    ret = ExpandEquations(mseed_eq, A_hat, b_hat); ERR(ret, err);
    __BENCHMARK_STOP__(BS_PIOP_EXPAND_MQ_VERIF);

    field_ext_elt v_z[FIELD_EXT_PACKING(MQOM2_PARAM_MQ_M/MQOM2_PARAM_MU)];
    field_ext_elt v_alpha[FIELD_EXT_PACKING(MQOM2_PARAM_ETA)];
    for(e = 0; e < MQOM2_PARAM_TAU; e++) {
        field_ext_elt r = get_evaluation_point(i_star[e]);
        ret = ComputePzEval(r, x_eval[e], A_hat, b_hat, y, v_z); ERR(ret, err);
#if MQOM2_PARAM_WITH_STATISTICAL_BATCHING == 1
        for(i=0; i<MQOM2_PARAM_ETA; i++) {
            field_ext_vect_pack(
                field_ext_vect_mult(Gamma[i], v_z, MQOM2_PARAM_MQ_M/MQOM2_PARAM_MU),
                v_alpha, i
            );
        }
        field_ext_vect_add(v_alpha, u_eval[e], v_alpha, MQOM2_PARAM_ETA);
        field_ext_constant_vect_mult(r, alpha1[e], alpha0[e], MQOM2_PARAM_ETA);
        field_ext_vect_add(v_alpha, alpha0[e], alpha0[e], MQOM2_PARAM_ETA);
#else
        field_ext_vect_add(v_z, u_eval[e], v_alpha, MQOM2_PARAM_ETA);
        field_ext_constant_vect_mult(r, alpha1[e], alpha0[e], MQOM2_PARAM_ETA);
        field_ext_vect_add(v_alpha, alpha0[e], alpha0[e], MQOM2_PARAM_ETA);
#endif
    }

    ret = 0;
err:
    if(_A_hat){
        mqom_free(_A_hat);
    }
    if(_b_hat){
        mqom_free(_b_hat);      
    }
    return ret;
}
