#include "cuda_helper.h"
#include "cuda_vectors_alexis.h"


__device__ __forceinline__
void G256_Mul2_a1_min3r(uint32_t *regs)
{
    uint32_t tmp = regs[7];
    regs[7] = regs[6];
    regs[6] = regs[5];
    regs[5] = regs[4];
    regs[4] = regs[3] ^ tmp;
    regs[3] = regs[2] ^ tmp;
    regs[2] = regs[1];
    regs[1] = regs[0] ^ tmp;
    regs[0] = tmp;
}

__device__ __forceinline__
void G256_Mul2_a1_min3r(uint4 *r)
{
	uint32_t t = r[1].w;
	r[1].w = r[1].z;
	r[1].z = r[1].y;
	r[1].y = r[1].x;
	r[1].x = r[0].w ^ t;
	r[0].w = r[0].z ^ t;
	r[0].z = r[0].y;
	r[0].y = r[0].x ^ t;
	r[0].x = t;
}

__device__ __forceinline__
void G256_AddRoundConstantQ_quad_a1_min3r(uint32_t &x7, uint32_t &x6, uint32_t &x5, uint32_t &x4, uint32_t &x3, uint32_t &x2, uint32_t &x1, uint32_t &x0, int rnd)
{
    x0 = ~x0;
    x1 = ~x1;
    x2 = ~x2;
    x3 = ~x3;
    x4 = ~x4;
    x5 = ~x5;
    x6 = ~x6;
    x7 = ~x7;

#if 0
    if ((threadIdx.x & 3) != 3)
        return;

    int andmask = 0xFFFF0000;
#else
    /* from sp: faster (branching problem with if ?) */
//    uint32_t andmask = -((threadIdx.x & 3) == 3) & 0xFFFF0000U;
	uint32_t andmask = -((threadIdx.x & 3) -2) & 0xFFFF0000U;
#endif

    x0 ^= ((- (rnd & 0x01)    ) & andmask);
    x1 ^= ((-((rnd & 0x02)>>1)) & andmask);
    x2 ^= ((-((rnd & 0x04)>>2)) & andmask);
    x3 ^= ((-((rnd & 0x08)>>3)) & andmask);

    x4 ^= (0xAAAA0000 & andmask);
    x5 ^= (0xCCCC0000 & andmask);
    x6 ^= (0xF0F00000 & andmask);
    x7 ^= (0xFF000000 & andmask);
}

__device__ __forceinline__
void G256_AddRoundConstantP_quad_a1_min3r(uint32_t &x7, uint32_t &x6, uint32_t &x5, uint32_t &x4, uint32_t &x3, uint32_t &x2, uint32_t &x1, uint32_t &x0, int rnd)
{
	int andmask = 0xFFFF & -((threadIdx.x & 3) == 0);
//	if (threadIdx.x & 3)
//		return;
//	int andmask = 0xFFFF; //& -((threadIdx.x & 3) == 0);

	x0 ^= ((-(rnd & 0x01)) & andmask);
	x1 ^= ((-((rnd & 0x02) >> 1)) & andmask);
	x2 ^= ((-((rnd & 0x04) >> 2)) & andmask);
	x3 ^= ((-((rnd & 0x08) >> 3)) & andmask);

	x4 ^= 0xAAAAU & andmask;
	x5 ^= 0xCCCCU & andmask;
	x6 ^= 0xF0F0U & andmask;
	x7 ^= 0xFF00U & andmask;
	/*
	int andmask = 0xFFFF & ((threadIdx.x & 3) -1);

	x0 ^= ((-(rnd & 0x01)) & andmask);
	x1 ^= ((-((rnd & 0x02) >> 1)) & andmask);
	x2 ^= ((-((rnd & 0x04) >> 2)) & andmask);
	x3 ^= ((-((rnd & 0x08) >> 3)) & andmask);

	x4 ^= 0xAAAAU & andmask;
	x5 ^= 0xCCCCU & andmask;
	x6 ^= 0xF0F0U & andmask;
	x7 ^= 0xFF00U & andmask;
	*/
	/*
    if (threadIdx.x & 3)
        return;

    int andmask = 0xFFFF;

    x0 ^= ((- (rnd & 0x01)    ) & andmask);
    x1 ^= ((-((rnd & 0x02)>>1)) & andmask);
    x2 ^= ((-((rnd & 0x04)>>2)) & andmask);
    x3 ^= ((-((rnd & 0x08)>>3)) & andmask);

    x4 ^= 0xAAAAU;
    x5 ^= 0xCCCCU;
    x6 ^= 0xF0F0U;
    x7 ^= 0xFF00U;
	*/
}

__device__ __forceinline__
void G16mul_quad_a1_min3r(uint32_t &x3, uint32_t &x2, uint32_t &x1, uint32_t &x0,
                                       uint32_t &y3, uint32_t &y2, uint32_t &y1, uint32_t &y0)
{
    uint32_t t0,t1,t2;
    
    t0 = ((x2 ^ x0) ^ (x3 ^ x1)) & ((y2 ^ y0) ^ (y3 ^ y1));
    t1 = ((x2 ^ x0) & (y2 ^ y0)) ^ t0;
    t2 = ((x3 ^ x1) & (y3 ^ y1)) ^ t0 ^ t1;

    t0 = (x2^x3) & (y2^y3);
    x3 = (x3 & y3) ^ t0 ^ t1;
    x2 = (x2 & y2) ^ t0 ^ t2;

    t0 = (x0^x1) & (y0^y1);
    x1 = (x1 & y1) ^ t0 ^ t1;
    x0 = (x0 & y0) ^ t0 ^ t2;
}

__device__ __forceinline__
void G256_inv_quad_a1_min3r(uint32_t &x7, uint32_t &x6, uint32_t &x5, uint32_t &x4, uint32_t &x3, uint32_t &x2, uint32_t &x1, uint32_t &x0)
{
    uint32_t t0,t1,t2,t3,t4,t5,t6,a,b;

    t3 = x7;
    t2 = x6;
    t1 = x5;
    t0 = x4;

	G16mul_quad_a1_min3r(t3, t2, t1, t0, x3, x2, x1, x0);

    a = (x4 ^ x0);
    t0 ^= a;
    t2 ^= (x7 ^ x3) ^ (x5 ^ x1); 
    t1 ^= (x5 ^ x1) ^ a;
    t3 ^= (x6 ^ x2) ^ a;

    b = t0 ^ t1;
    t4 = (t2 ^ t3) & b;
    a = t4 ^ t3 ^ t1;
    t5 = (t3 & t1) ^ a;
    t6 = (t2 & t0) ^ a ^ (t2 ^ t0);

    t4 = (t5 ^ t6) & b;
    t1 = (t6 & t1) ^ t4;
    t0 = (t5 & t0) ^ t4;

    t4 = (t5 ^ t6) & (t2^t3);
    t3 = (t6 & t3) ^ t4;
    t2 = (t5 & t2) ^ t4;

	G16mul_quad_a1_min3r(x3, x2, x1, x0, t1, t0, t3, t2);

	G16mul_quad_a1_min3r(x7, x6, x5, x4, t1, t0, t3, t2);
}

__device__ __forceinline__
void transAtoX_quad_a1_min3r(uint32_t &x0, uint32_t &x1, uint32_t &x2, uint32_t &x3, uint32_t &x4, uint32_t &x5, uint32_t &x6, uint32_t &x7)
{
    uint32_t t0, t1;
    t0 = x0 ^ x1 ^ x2;
    t1 = x5 ^ x6;
    x2 = t0 ^ t1 ^ x7;
    x6 = t0 ^ x3 ^ x6;
    x3 = x0 ^ x1 ^ x3 ^ x4 ^ x7;    
    x4 = x0 ^ x4 ^ t1;
    x2 = t0 ^ t1 ^ x7;
    x1 = x0 ^ x1 ^ t1;
    x7 = x0 ^ t1 ^ x7;
    x5 = x0 ^ t1;
}

__device__ __forceinline__
void transXtoA_quad_a1_min3r(uint32_t &x0, uint32_t &x1, uint32_t &x2, uint32_t &x3, uint32_t &x4, uint32_t &x5, uint32_t &x6, uint32_t &x7)
{
    uint32_t t0,t2,t3,t5;

    x1 ^= x4;
    t0 = x1 ^ x6;
    x1 ^= x5;

    t2 = x0 ^ x2;
    x2 = x3 ^ x5;
    t2 ^= x2 ^ x6;
    x2 ^= x7;
    t3 = x4 ^ x2 ^ x6;

    t5 = x0 ^ x6;
    x4 = x3 ^ x7;
    x0 = x3 ^ x5;

    x6 = t0;    
    x3 = t2;
    x7 = t3;    
    x5 = t5;    
}

__device__ __forceinline__
//void sbox_quad_a1_min3r(uint32_t *r)
void sbox_quad_a1_min3r(uint4 *r)
{
	transAtoX_quad_a1_min3r(r[0].x, r[0].y, r[0].z, r[0].w, r[1].x, r[1].y, r[1].z, r[1].w);

	G256_inv_quad_a1_min3r(r[0].z, r[1].x, r[0].y, r[1].w, r[0].w, r[0].x, r[1].y, r[1].z);

	transXtoA_quad_a1_min3r(r[1].w, r[0].y, r[1].x, r[0].z, r[1].z, r[1].y, r[0].x, r[0].w);

	r[0].x = ~r[0].x;
	r[0].y = ~r[0].y;
	r[1].y = ~r[1].y;
	r[1].z = ~r[1].z;
	/*
	transAtoX_quad_a1_min3r(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);

	G256_inv_quad_a1_min3r(r[2], r[4], r[1], r[7], r[3], r[0], r[5], r[6]);

	transXtoA_quad_a1_min3r(r[7], r[1], r[4], r[2], r[6], r[5], r[0], r[3]);
    
    r[0] = ~r[0];
    r[1] = ~r[1];
    r[5] = ~r[5];
    r[6] = ~r[6];
	*/
}

__device__ __forceinline__
void G256_ShiftBytesP_quad_a1_min3r(uint32_t &x7, uint32_t &x6, uint32_t &x5, uint32_t &x4, uint32_t &x3, uint32_t &x2, uint32_t &x1, uint32_t &x0)
{
    uint32_t t0,t1;

    int tpos = threadIdx.x & 0x03;
    int shift1 = tpos << 1;
    int shift2 = shift1+1 + ((tpos == 3)<<2);

    t0 = __byte_perm(x0, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x0, 0, 0x3232)>>shift2;
    x0 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x1, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x1, 0, 0x3232)>>shift2;
    x1 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x2, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x2, 0, 0x3232)>>shift2;
    x2 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x3, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x3, 0, 0x3232)>>shift2;
    x3 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x4, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x4, 0, 0x3232)>>shift2;
    x4 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x5, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x5, 0, 0x3232)>>shift2;
    x5 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x6, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x6, 0, 0x3232)>>shift2;
    x6 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x7, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x7, 0, 0x3232)>>shift2;
    x7 = __byte_perm(t0, t1, 0x5410);
}

__device__ __forceinline__
void G256_ShiftBytesQ_quad_a1_min3r(uint32_t &x7, uint32_t &x6, uint32_t &x5, uint32_t &x4, uint32_t &x3, uint32_t &x2, uint32_t &x1, uint32_t &x0)
{
    uint32_t t0,t1;

    int tpos = threadIdx.x & 0x03;
    int shift1 = (1-(tpos>>1)) + ((tpos & 0x01)<<2);
    int shift2 = shift1+2 + ((tpos == 1)<<2);

    t0 = __byte_perm(x0, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x0, 0, 0x3232)>>shift2;
    x0 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x1, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x1, 0, 0x3232)>>shift2;
    x1 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x2, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x2, 0, 0x3232)>>shift2;
    x2 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x3, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x3, 0, 0x3232)>>shift2;
    x3 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x4, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x4, 0, 0x3232)>>shift2;
    x4 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x5, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x5, 0, 0x3232)>>shift2;
    x5 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x6, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x6, 0, 0x3232)>>shift2;
    x6 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x7, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x7, 0, 0x3232)>>shift2;
    x7 = __byte_perm(t0, t1, 0x5410);
}

#if __CUDA_ARCH__ < 300
/**
 * __shfl() returns the value of var held by the thread whose ID is given by srcLane.
 * If srcLane is outside the range 0..width-1, the thread’s own value of var is returned.
 */
#undef __shfl
#define __shfl(var, srcLane, width) (uint32_t)(var)
#endif

__device__ __forceinline__
//void G256_MixFunction_quad_a1_min3r(uint32_t *r)
void G256_MixFunction_quad_a1_min3r(uint4 *r)
{
#define SHIFT64_16(hi, lo)    __byte_perm(lo, hi, 0x5432)
#define A(__W, u)             __shfl((int)r __W, ((threadIdx.x+u)&0x03), 4)
#define S(__W, l)            SHIFT64_16( A(__W, (l+1)), A(__W, l) )

#define DOUBLE_ODD(__W, bc)        ( S(__W, (bc)) ^ A(__W, (bc) + 1) )
#define DOUBLE_EVEN(__W, bc)        ( S(__W, (bc)) ^ A(__W, (bc)    ) )

#define SINGLE_ODD(__W, bc)        ( S(__W, (bc)) )
#define SINGLE_EVEN(__W, bc)        ( A(__W, (bc)) )
    uint4 b[2];


	b[0].x = DOUBLE_ODD([0].x, 1) ^ DOUBLE_EVEN([0].x, 3);
	b[0].y = DOUBLE_ODD([0].y, 1) ^ DOUBLE_EVEN([0].y, 3);
	b[0].z = DOUBLE_ODD([0].z, 1) ^ DOUBLE_EVEN([0].z, 3);
	b[0].w = DOUBLE_ODD([0].w, 1) ^ DOUBLE_EVEN([0].w, 3);
	b[1].x = DOUBLE_ODD([1].x, 1) ^ DOUBLE_EVEN([1].x, 3);
	b[1].y = DOUBLE_ODD([1].y, 1) ^ DOUBLE_EVEN([1].y, 3);
	b[1].z = DOUBLE_ODD([1].z, 1) ^ DOUBLE_EVEN([1].z, 3);
	b[1].w = DOUBLE_ODD([1].w, 1) ^ DOUBLE_EVEN([1].w, 3);

	G256_Mul2_a1_min3r(b);

	b[0].x ^= DOUBLE_ODD([0].x, 3) ^ DOUBLE_ODD([0].x, 4) ^ SINGLE_ODD([0].x, 6);
	b[0].y ^= DOUBLE_ODD([0].y, 3) ^ DOUBLE_ODD([0].y, 4) ^ SINGLE_ODD([0].y, 6);
	b[0].z ^= DOUBLE_ODD([0].z, 3) ^ DOUBLE_ODD([0].z, 4) ^ SINGLE_ODD([0].z, 6);
	b[0].w ^= DOUBLE_ODD([0].w, 3) ^ DOUBLE_ODD([0].w, 4) ^ SINGLE_ODD([0].w, 6);
	b[1].x ^= DOUBLE_ODD([1].x, 3) ^ DOUBLE_ODD([1].x, 4) ^ SINGLE_ODD([1].x, 6);
	b[1].y ^= DOUBLE_ODD([1].y, 3) ^ DOUBLE_ODD([1].y, 4) ^ SINGLE_ODD([1].y, 6);
	b[1].z ^= DOUBLE_ODD([1].z, 3) ^ DOUBLE_ODD([1].z, 4) ^ SINGLE_ODD([1].z, 6);
	b[1].w ^= DOUBLE_ODD([1].w, 3) ^ DOUBLE_ODD([1].w, 4) ^ SINGLE_ODD([1].w, 6);

	G256_Mul2_a1_min3r(b);

	b[0].x ^= DOUBLE_EVEN([0].x, 2) ^ DOUBLE_EVEN([0].x, 3) ^ SINGLE_EVEN([0].x, 5);
	b[0].y ^= DOUBLE_EVEN([0].y, 2) ^ DOUBLE_EVEN([0].y, 3) ^ SINGLE_EVEN([0].y, 5);
	b[0].z ^= DOUBLE_EVEN([0].z, 2) ^ DOUBLE_EVEN([0].z, 3) ^ SINGLE_EVEN([0].z, 5);
	b[0].w ^= DOUBLE_EVEN([0].w, 2) ^ DOUBLE_EVEN([0].w, 3) ^ SINGLE_EVEN([0].w, 5);
	b[1].x ^= DOUBLE_EVEN([1].x, 2) ^ DOUBLE_EVEN([1].x, 3) ^ SINGLE_EVEN([1].x, 5);
	b[1].y ^= DOUBLE_EVEN([1].y, 2) ^ DOUBLE_EVEN([1].y, 3) ^ SINGLE_EVEN([1].y, 5);
	b[1].z ^= DOUBLE_EVEN([1].z, 2) ^ DOUBLE_EVEN([1].z, 3) ^ SINGLE_EVEN([1].z, 5);
	b[1].w ^= DOUBLE_EVEN([1].w, 2) ^ DOUBLE_EVEN([1].w, 3) ^ SINGLE_EVEN([1].w, 5);

	*(uint2x4*)&r[0] = *(uint2x4*)&b[0];

	/*
#pragma unroll 8
    for(int i=0;i<8;i++)
        b[i] = DOUBLE_ODD(i, 1) ^ DOUBLE_EVEN(i, 3);

	G256_Mul2_a1_min3r(b);
#pragma unroll 8
    for(int i=0;i<8;i++)
        b[i] = b[i] ^ DOUBLE_ODD(i, 3) ^ DOUBLE_ODD(i, 4) ^ SINGLE_ODD(i, 6);

	G256_Mul2_a1_min3r(b);
#pragma unroll 8
    for(int i=0;i<8;i++)
        r[i] = b[i] ^ DOUBLE_EVEN(i, 2) ^ DOUBLE_EVEN(i, 3) ^ SINGLE_EVEN(i, 5);
	*/
#undef S
#undef A
#undef SHIFT64_16
#undef t
#undef X
}

__device__ __forceinline__
//void groestl512_perm_P_quad_a1_min3r(uint32_t *r)
void groestl512_perm_P_quad_a1_min3r(uint4 *r)
{
	for (int round = 0; round < 14; round++)
	{
		G256_AddRoundConstantP_quad_a1_min3r(r[1].w, r[1].z, r[1].y, r[1].x, r[0].w, r[0].z, r[0].y, r[0].x, round);
		sbox_quad_a1_min3r(r);
		G256_ShiftBytesP_quad_a1_min3r(r[1].w, r[1].z, r[1].y, r[1].x, r[0].w, r[0].z, r[0].y, r[0].x);
		G256_MixFunction_quad_a1_min3r(r);
	}
}
__device__ __forceinline__
void groestl512_perm_P_quad_a1_min3r(uint32_t *r)
{
	for(int round=0;round<14;round++)
    {
		G256_AddRoundConstantP_quad_a1_min3r(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0], round);
		sbox_quad_a1_min3r((uint4*)r);
		G256_ShiftBytesP_quad_a1_min3r(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0]);
		G256_MixFunction_quad_a1_min3r((uint4*)r);
    }
}
__device__ __forceinline__
void groestl512_perm_Q_quad_a1_min3r(uint4 *r)
{
	for (int round = 0; round<14; round++)
	{
		G256_AddRoundConstantQ_quad_a1_min3r(r[1].w, r[1].z, r[1].y, r[1].x, r[0].w, r[0].z, r[0].y, r[0].x, round);
		sbox_quad_a1_min3r(r);
		G256_ShiftBytesQ_quad_a1_min3r(r[1].w, r[1].z, r[1].y, r[1].x, r[0].w, r[0].z, r[0].y, r[0].x);
		G256_MixFunction_quad_a1_min3r(r);
	}
}

__device__ __forceinline__
void groestl512_perm_Q_quad_a1_min3r(uint32_t *r)
{    
    for(int round=0;round<14;round++)
    {
		G256_AddRoundConstantQ_quad_a1_min3r(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0], round);
		sbox_quad_a1_min3r((uint4*)r);
		G256_ShiftBytesQ_quad_a1_min3r(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0]);
		G256_MixFunction_quad_a1_min3r((uint4*)r);
    }
}

__device__ __forceinline__
//void groestl512_progressMessage_quad_a1_min3r(uint32_t *state, uint32_t *message)
void groestl512_progressMessage_quad_a1_min3r(uint4 *state, uint4 *message)
{
	/*
	if (threadIdx.x & 0x3) message[0] = 0;
	if (threadIdx.x & 0x3) message[1] = 0;
	if (threadIdx.x & 0x3) message[2] = 0;
	if (threadIdx.x & 0x3) message[3] = 0;
	if (threadIdx.x & 0x3) message[4] = 0;
	if (threadIdx.x & 0x3) message[5] = 0;
	if (threadIdx.x & 0x3) message[6] = 0;
	if (threadIdx.x & 0x3) message[7] = 0;
	if (!(threadIdx.x & 0x3)) message[0] = 0;
	*/
/*
	message[0] = 0;
	message[1] = 0;
	message[2] = 0;
	message[3] = 0;
	message[4] = 0;
	message[5] = 0;
	message[6] = 0;
	message[7] = 0;
*/
//#pragma unroll 8
//    for(int u=0;u<8;u++) state[u] = message[u];
	*(uint2x4*)&state[0] = *(uint2x4*)&message[0];
	state[0].y ^= 0x8000 & -((threadIdx.x & 0x03) - 2);
//    if ((threadIdx.x & 0x03) == 3) state[0].y ^= 0x00008000;
	groestl512_perm_P_quad_a1_min3r(&state[0]); //! error?
	state[0].y ^= 0x8000 & -((threadIdx.x & 0x03) - 2);
//    if ((threadIdx.x & 0x03) == 3) state[0].y ^= 0x00008000;
	groestl512_perm_Q_quad_a1_min3r(&message[0]); //! error?

	*(uint2x4*)&state[0] ^= *(uint2x4*)&message[0];
	*(uint2x4*)&message[0] = *(uint2x4*)&state[0];

	groestl512_perm_P_quad_a1_min3r(&message[0]);
	*(uint2x4*)&state[0] ^= *(uint2x4*)&message[0];
	/*
#pragma unroll 8
    for(int u=0;u<8;u++) state[u] ^= message[u];
#pragma unroll 8
    for(int u=0;u<8;u++) message[u] = state[u];
	groestl512_perm_P_quad_a1_min3r((uint32_t*)&message[0]);
#pragma unroll 8
    for(int u=0;u<8;u++) state[u] ^= message[u];
	*/
}
