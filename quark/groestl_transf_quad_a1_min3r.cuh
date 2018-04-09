/* File included in quark/groestl (quark/jha,nist5/X11+) and groest/myriad coins for SM 3+ */
#include "miner.h"
#include "cuda_vectors_alexis.h"
#if 1
#define merge8(__z,__x,__y)\
	__z=__byte_perm(__x, __y, 0x5140); \

#define SWAP8(__x,__y)\
	__x=__byte_perm(__x, __y, 0x5410); \
	__y=__byte_perm(__x, __y, 0x7632);

#define SWAP4(__x,__y)\
	t = ((__y)<<4); \
	t = ((__x) ^ t); \
	t = 0xf0f0f0f0UL & t; \
	__x = ((__x) ^ t); \
	t=  t>>4;\
	__y=  (__y) ^ t;

#define SWAP2(__x,__y)\
	t = ((__y)<<2); \
	t = ((__x) ^ t); \
	t = 0xccccccccUL & t; \
	__x = ((__x) ^ t); \
	t=  t>>2;\
	__y=  (__y) ^ t;

#define SWAP1(__x,__y)\
	t = ((__y)+(__y)); \
	t = ((__x) ^ t); \
	t = 0xaaaaaaaaUL & t; \
	__x = ((__x) ^ t); \
	t=  t>>1;\
	__y=  (__y) ^ t;
#else
#define merge8(z,x,y)\
	z=__byte_perm(x, y, 0x5140); \

#define SWAP8(x,y)\
	x=__byte_perm(x, y, 0x5410); \
	y=__byte_perm(x, y, 0x7632);

#define SWAP4(x,y)\
	t = (y<<4); \
	t = (x ^ t); \
	t = 0xf0f0f0f0UL & t; \
	x = (x ^ t); \
	t=  t>>4;\
	y=  y ^ t;

#define SWAP2(x,y)\
	t = (y<<2); \
	t = (x ^ t); \
	t = 0xccccccccUL & t; \
	x = (x ^ t); \
	t=  t>>2;\
	y=  y ^ t;

#define SWAP1(x,y)\
	t = (y+y); \
	t = (x ^ t); \
	t = 0xaaaaaaaaUL & t; \
	x = (x ^ t); \
	t=  t>>1;\
	y=  y ^ t;
#endif
//input[i].__X = __shfl((int)input[i].__X, n ^ /*!fix! 3 & -(n<1 && n>2)*/ (3 * (n >= 1 && n <= 2)), 4); 
//input[i].__X = __byte_perm(input[i].__X, 0, 0x1032); 
//other[i].__X = __byte_perm(other[i].__X, 0, 0x1032); 
//input[i].__X = __shfl((int)input[i].__X, n ^ (3 & -(n < 1 || n > 2)), 4); 
//input[i].__X = __shfl((int)input[i].__X, n ^ (3 * (n >= 1 && n <= 2)), 4); 
/*
#define UNROLL_SETS(__X) {\
		input[i].__X = __shfl((int)input[i].__X, n ^ (3 & -(n < 1 || n > 2)), 4);\
		other[i].__X = __shfl((int)input[i].__X, (threadIdx.x + 1) & 3, 4); \
		input[i].__X = __shfl((int)input[i].__X, threadIdx.x & 2, 4);\
		other[i].__X = __shfl((int)other[i].__X, threadIdx.x & 2, 4);\
		if (threadIdx.x & 1) {\
			input[i].__X = __byte_perm(input[i].__X, 0, 0x1032);\
			other[i].__X = __byte_perm(other[i].__X, 0, 0x1032);\
		}\
};
p = 123, q = 456, r = 1?0;

if(r) p = 456
else p = 123

if(r) p = q;
else p = p;

p = (q & -r) | (p & (r-1))

456 & -1 (456) | -1 & (123)
456 & -0 (0) | 123 & (-1)

p = q & -r | p & (r-1)

input[i].__X = (__byte_perm(input[i].__X, 0, 0x1032) & -(threadIdx.x & 1)) | (input[i].__X & ((threadIdx.x & 1)-1) );\
other[i].__X = (__byte_perm(other[i].__X, 0, 0x1032) & -(threadIdx.x & 1)) | (input[i].__X & ((threadIdx.x & 1)-1) );\
input[i].__X = __shfl((int)input[i].__X, n ^ (3 & -(n < 1 || n > 2)), 4);\

input[i].__X = __shfl((int)input[i].__X, n ^ (3 & -(n >= 1 && n <= 2)), 4);\
*/
//input[i].__X = (__byte_perm(input[i].__X, 0, 0x1032) & (-(threadIdx.x & 1) | (-(threadIdx.x & 1) & input[i].__X));
//other[i].__X = (__byte_perm(other[i].__X, 0, 0x1032) & (-(threadIdx.x & 1) | (-(threadIdx.x & 1) & input[i].__X));
#if 1
#define UNROLL_SETS(__X) {\
		input[i].__X = __shfl((int)input[i].__X, n ^ (3 & -(n >= 1 && n <= 2)), 4);\
		other[i].__X = __shfl((int)input[i].__X, (threadIdx.x + 1) & 3, 4);\
		input[i].__X = __shfl((int)input[i].__X, threadIdx.x & 2, 4);\
		other[i].__X = __shfl((int)other[i].__X, threadIdx.x & 2, 4);\
		{\
		input[i].__X = (__byte_perm(input[i].__X, 0, 0x1032) & -(threadIdx.x & 1)) | ((threadIdx.x & 1)-1 & input[i].__X);\
		other[i].__X = (__byte_perm(other[i].__X, 0, 0x1032) & -(threadIdx.x & 1)) | ((threadIdx.x & 1)-1 & other[i].__X);\
		}\
};
#else
#define UNROLL_SETS(__X) {\
		input[i].__X = __shfl((int)input[i].__X, n ^ (3 * (n >= 1 && n <= 2)), 4);\
		other[i].__X = __shfl((int)input[i].__X, (threadIdx.x + 1) & 3, 4);\
		input[i].__X = __shfl((int)input[i].__X, threadIdx.x & 2, 4);\
		other[i].__X = __shfl((int)other[i].__X, threadIdx.x & 2, 4);\
		if(threadIdx.x & 1) {\
			input[i].__X = __byte_perm(input[i].__X, 0, 0x1032);\
			other[i].__X = __byte_perm(other[i].__X, 0, 0x1032);\
				}\
};
#endif

__device__ __forceinline__
//void to_bitslice_quad_a1_min3r(uint32_t *const __restrict__ input, uint32_t *const __restrict__ output)
void to_bitslice_quad_a1_min3r(uint4 *const __restrict__ input, uint4 *const __restrict__ output)
//void to_bitslice_quad_a1_min3r(uint4 *const __restrict__ qinput, uint4 *const __restrict__ qoutput)
{
#if 1

	uint4 other[2];
	uint4 d[2];
	uint32_t t;
	const unsigned int n = threadIdx.x & 3;
#pragma unroll
	for (int i = 0; i < 2; i++) {
		UNROLL_SETS(x);
		UNROLL_SETS(y);
		UNROLL_SETS(z);
		UNROLL_SETS(w);
	}

	merge8(d[0].x, input[0].x, input[1].x);
	merge8(d[0].y, other[0].x, other[1].x);
	merge8(d[0].z, input[0].y, input[1].y);
	merge8(d[0].w, other[0].y, other[1].y);
	merge8(d[1].x, input[0].z, input[1].z);
	merge8(d[1].y, other[0].z, other[1].z);
	merge8(d[1].z, input[0].w, input[1].w);
	merge8(d[1].w, other[0].w, other[1].w);

	SWAP1(d[0].x, d[0].y);
	SWAP1(d[0].z, d[0].w);
	SWAP1(d[1].x, d[1].y);
	SWAP1(d[1].z, d[1].w);

	SWAP2(d[0].x, d[0].z);
	SWAP2(d[0].y, d[0].w);
	SWAP2(d[1].x, d[1].z);
	SWAP2(d[1].y, d[1].w);

	SWAP4(d[0].x, d[1].x);
	SWAP4(d[0].y, d[1].y);
	SWAP4(d[0].z, d[1].z);
	SWAP4(d[0].w, d[1].w);

	*(uint2x4*)&output[0] = *(uint2x4*)&d[0];
	/*
	uint32_t *input = (uint32_t*)qinput;
	uint32_t *output = (uint32_t*)qoutput;
#pragma unroll
	for (int i = 0; i < 8; i++) {
		input[i] = __shfl((int)input[i], n ^ (3*(n >=1 && n <=2)), 4);
		other[i] = __shfl((int)input[i], (threadIdx.x + 1) & 3, 4);
		input[i] = __shfl((int)input[i], threadIdx.x & 2, 4);
		other[i] = __shfl((int)other[i], threadIdx.x & 2, 4);
		if (threadIdx.x & 1) {
			input[i] = __byte_perm(input[i], 0, 0x1032);
			other[i] = __byte_perm(other[i], 0, 0x1032);
		}
	}
	merge8(d[0], input[0], input[4]);
	merge8(d[1], other[0], other[4]);
	merge8(d[2], input[1], input[5]);
	merge8(d[3], other[1], other[5]);
	merge8(d[4], input[2], input[6]);
	merge8(d[5], other[2], other[6]);
	merge8(d[6], input[3], input[7]);
	merge8(d[7], other[3], other[7]);

	SWAP1(d[0], d[1]);
	SWAP1(d[2], d[3]);
	SWAP1(d[4], d[5]);
	SWAP1(d[6], d[7]);

	SWAP2(d[0], d[2]);
	SWAP2(d[1], d[3]);
	SWAP2(d[4], d[6]);
	SWAP2(d[5], d[7]);

	SWAP4(d[0], d[4]);
	SWAP4(d[1], d[5]);
	SWAP4(d[2], d[6]);
	SWAP4(d[3], d[7]);

	output[0] = d[0];
	output[1] = d[1];
	output[2] = d[2];
	output[3] = d[3];
	output[4] = d[4];
	output[5] = d[5];
	output[6] = d[6];
	output[7] = d[7];
	*/
	//
#else
	uint32_t *input = (uint32_t*)qinput;
	uint32_t *output = (uint32_t*)qoutput;
	uint32_t other[8];
	uint32_t d[8];
	uint32_t t;
	const unsigned int n = threadIdx.x & 3;

#pragma unroll
	for (int i = 0; i < 8; i++) {
		input[i] = __shfl((int)input[i], n ^ (3 * (n >= 1 && n <= 2)), 4);
		other[i] = __shfl((int)input[i], (threadIdx.x + 1) & 3, 4);
		input[i] = __shfl((int)input[i], threadIdx.x & 2, 4);
		other[i] = __shfl((int)other[i], threadIdx.x & 2, 4);
		if (threadIdx.x & 1) {
			input[i] = __byte_perm(input[i], 0, 0x1032);
			other[i] = __byte_perm(other[i], 0, 0x1032);
		}
	}

	merge8(d[0], input[0], input[4]);
	merge8(d[1], other[0], other[4]);
	merge8(d[2], input[1], input[5]);
	merge8(d[3], other[1], other[5]);
	merge8(d[4], input[2], input[6]);
	merge8(d[5], other[2], other[6]);
	merge8(d[6], input[3], input[7]);
	merge8(d[7], other[3], other[7]);

	SWAP1(d[0], d[1]);
	SWAP1(d[2], d[3]);
	SWAP1(d[4], d[5]);
	SWAP1(d[6], d[7]);

	SWAP2(d[0], d[2]);
	SWAP2(d[1], d[3]);
	SWAP2(d[4], d[6]);
	SWAP2(d[5], d[7]);

	SWAP4(d[0], d[4]);
	SWAP4(d[1], d[5]);
	SWAP4(d[2], d[6]);
	SWAP4(d[3], d[7]);

	output[0] = d[0];
	output[1] = d[1];
	output[2] = d[2];
	output[3] = d[3];
	output[4] = d[4];
	output[5] = d[5];
	output[6] = d[6];
	output[7] = d[7];
#endif
}
/*
__device__ __forceinline__
void tmp(uint32_t* output)
{
#pragma unroll 8
	for (int i = 0; i < 16; i += 2) {
		if (threadIdx.x & 1) output[i] = __byte_perm(output[i], 0, 0x1032);
		output[i] = __byte_perm(output[i], __shfl((int)output[i], (threadIdx.x + 1) & 3, 4), 0x7610);
		output[i + 1] = __shfl((int)output[i], (threadIdx.x + 2) & 3, 4);
		if (threadIdx.x & 3) output[i] = output[i + 1] = 0;
	}
}

__device__ __forceinline__
void tmp2(uint32_t* output, uint32_t* d)
{
	output[0] = d[0];
	output[2] = d[1];
	output[4] = d[0] >> 16;
	output[6] = d[1] >> 16;
	output[8] = d[2];
	output[10] = d[3];
	output[12] = d[2] >> 16;
	output[14] = d[3] >> 16;
}
__device__ __forceinline__
void tmp3(uint32_t* d, uint32_t &t)
{
	SWAP1(d[0], d[1]);
	SWAP1(d[2], d[3]);

	SWAP2(d[0], d[2]);
	SWAP2(d[1], d[3]);

	t = __byte_perm(d[0], d[2], 0x5410);
	d[2] = __byte_perm(d[0], d[2], 0x7632);
	d[0] = t;

	t = __byte_perm(d[1], d[3], 0x5410);
	d[3] = __byte_perm(d[1], d[3], 0x7632);
	d[1] = t;

	SWAP4(d[0], d[2]);
	SWAP4(d[1], d[3]);
}

__device__ __forceinline__
void tmp4(uint32_t* input, uint32_t* d)
{
	d[0] = __byte_perm(input[0], input[4], 0x7531);
	d[1] = __byte_perm(input[1], input[5], 0x7531);
	d[2] = __byte_perm(input[2], input[6], 0x7531);
	d[3] = __byte_perm(input[3], input[7], 0x7531);
}

#define SWAP8(x,y)\
	x=__byte_perm(x, y, 0x5410); \
	y=__byte_perm(x, y, 0x7632);

#define SWAP4(x,y)\
	t = (y<<4); \
	t = (x ^ t); \
	t = 0xf0f0f0f0UL & t; \
	x = (x ^ t); \
	t=  t>>4;\
	y=  y ^ t;

#define SWAP2(x,y)\
	t = (y<<2); \
	t = (x ^ t); \
	t = 0xccccccccUL & t; \
	x = (x ^ t); \
	t=  t>>2;\
	y=  y ^ t;

#define SWAP1(x,y)\
	t = (y+y); \
	t = (x ^ t); \
	t = 0xaaaaaaaaUL & t; \
	x = (x ^ t); \
	t=  t>>1;\
	y=  y ^ t;
*/
__device__ __forceinline__
//void from_bitslice_quad_a1_min3r(const uint32_t *const __restrict__ input, uint32_t *const __restrict__ output)
void from_bitslice_quad_a1_min3r(const uint4 *const __restrict__ input, uint4 *const __restrict__ output)
//void from_bitslice_quad_a1_min3r(const uint4 *const __restrict__ sinput, uint4 *const __restrict__ soutput)
{

#if 1
	uint4 d[1];
	uint32_t t;
	/*
	tmp4((uint32_t*)&input[0], (uint32_t*)&d[0]);
	tmp3((uint32_t*)&d[0], t);
	tmp2((uint32_t*)&output[0], (uint32_t*)&d[0]);
	tmp((uint32_t*)&output[0]);
	return;
	*/
	d[0].x = __byte_perm(input[0].x, input[1].x, 0x7531);
	d[0].y = __byte_perm(input[0].y, input[1].y, 0x7531);
	d[0].z = __byte_perm(input[0].z, input[1].z, 0x7531);
	d[0].w = __byte_perm(input[0].w, input[1].w, 0x7531);

	SWAP1(d[0].x, d[0].y);
	SWAP1(d[0].z, d[0].w);

	SWAP2(d[0].x, d[0].z);
	SWAP2(d[0].y, d[0].w);

	t = __byte_perm(d[0].x, d[0].z, 0x5410);
	d[0].z = __byte_perm(d[0].x, d[0].z, 0x7632);
	d[0].x = t;

	t = __byte_perm(d[0].y, d[0].w, 0x5410);
	d[0].w = __byte_perm(d[0].y, d[0].w, 0x7632);
	d[0].y = t;

	SWAP4(d[0].x, d[0].z);
	SWAP4(d[0].y, d[0].w);

	//! looks like if threadIdx.x & 3 == 0 then memclr(output, 16) happens.
//	int32_t flag0 = -((threadIdx.x & 3) == 0);
	int32_t flag1 = (threadIdx.x & 1);


	output[0].x = (__byte_perm(d[0].x, 0, 0x1032) & -flag1) | (d[0].x & (flag1 - 1));//?
	output[0].z = (__byte_perm(d[0].y, 0, 0x1032) & -flag1) | (d[0].y & (flag1 - 1));//?
	output[1].x = (__byte_perm(d[0].x>>16, 0, 0x1032) & -flag1) | ((d[0].x>>16) & (flag1 - 1));//?
	output[1].z = (__byte_perm(d[0].y>>16, 0, 0x1032) & -flag1) | ((d[0].y>>16) & (flag1 - 1));//?
	output[2].x = (__byte_perm(d[0].z, 0, 0x1032) & -flag1) | (d[0].z & (flag1 - 1));//?
	output[2].z = (__byte_perm(d[0].w, 0, 0x1032) & -flag1) | (d[0].w & (flag1 - 1));//?
	output[3].x = (__byte_perm(d[0].z >> 16, 0, 0x1032) & -flag1) | ((d[0].z >> 16) & (flag1 - 1));//?
	output[3].z = (__byte_perm(d[0].w >> 16, 0, 0x1032) & -flag1) | ((d[0].w >> 16) & (flag1 - 1));//?
	//	output[0].x = (d[0].x & flag0);
	//output[0].z = (d[0].y & flag0);
	//output[1].x = ((d[0].x >> 16) & flag0);
	//output[1].z = ((d[0].y >> 16) & flag0);
	//output[2].x = (d[0].z & flag0);
	//output[2].z = (d[0].w & flag0);
	//output[3].x = ((d[0].z >> 16) & flag0);
	//output[3].z = ((d[0].w >> 16) & flag0);

	//! everything looks to be bitshift/select, so if they are 0 before, they are 0 after?
#pragma unroll 4
	for (int i = 0; i < 4; i++) {
//		output[i].x = (__byte_perm(output[i].x, 0, 0x1032) & -flag1) | (output[i].x & (flag1 - 1));//?
		//		if (threadIdx.x & 1) output[i].x = __byte_perm(output[i].x, 0, 0x1032);//?
		output[i].x = __byte_perm(output[i].x, __shfl((int)output[i].x, (threadIdx.x + 1) & 3, 4), 0x7610);
		output[i].y = __shfl((int)output[i].x, (threadIdx.x + 2) & 3, 4);// &flag0;

//		output[i].x = output[i].x & flag0;
//		output[i].y = output[i].y & flag0;
		//		if (threadIdx.x & 3) output[i].x = output[i].y = 0;

//		output[i].z = (__byte_perm(output[i].z, 0, 0x1032) & -flag1) | (output[i].z & (flag1 - 1));//?
		//		if (threadIdx.x & 1) output[i].z = __byte_perm(output[i].z, 0, 0x1032);//?
		output[i].z = __byte_perm(output[i].z, __shfl((int)output[i].z, (threadIdx.x + 1) & 3, 4), 0x7610);
		output[i].w = __shfl((int)output[i].z, (threadIdx.x + 2) & 3, 4);// &flag0;
//		output[i].z = output[i].z & flag0;
//		output[i].w = output[i].w & flag0;
		//		if (threadIdx.x & 3) output[i].z = output[i].w = 0;
	}

	/*
	output[0].x = d[0].x;
	output[0].z = d[0].y;
	output[1].x = d[0].x >> 16;
	output[1].z = d[0].y >> 16;
	output[2].x = d[0].z;
	output[2].z = d[0].w;
	output[3].x = d[0].z >> 16;
	output[3].z = d[0].w >> 16;

#pragma unroll 4
	for (int i = 0; i < 4; i++) {
		output[i].x = (__byte_perm(output[i].x, 0, 0x1032) & -flag1) | (output[i].x & (flag1-1));//?
//		if (threadIdx.x & 1) output[i].x = __byte_perm(output[i].x, 0, 0x1032);//?
		output[i].x = __byte_perm(output[i].x, __shfl((int)output[i].x, (threadIdx.x + 1) & 3, 4), 0x7610);
		output[i].y = __shfl((int)output[i].x, (threadIdx.x + 2) & 3, 4);
		output[i].x = output[i].x & flag0;
		output[i].y = output[i].y & flag0;

//		if (threadIdx.x & 3) output[i].x = output[i].y = 0;
//		if (threadIdx.x & 3) output[i].x = output[i].y = -1;

		output[i].z = (__byte_perm(output[i].z, 0, 0x1032) & -flag1) | (output[i].z & (flag1 - 1));//?
//		if (threadIdx.x & 1) output[i].z = __byte_perm(output[i].z, 0, 0x1032);//?
		output[i].z = __byte_perm(output[i].z, __shfl((int)output[i].z, (threadIdx.x + 1) & 3, 4), 0x7610);
		output[i].w = __shfl((int)output[i].z, (threadIdx.x + 2) & 3, 4);
		output[i].z = output[i].z & flag0;
		output[i].w = output[i].w & flag0;
//		if (threadIdx.x & 3) output[i].z = output[i].w = 0;
//		if (threadIdx.x & 3) output[i].z = output[i].w = -1;
	}
	*/
#else
//	uint32_t* input = (uint32_t*)&sinput[0];
//	uint32_t* output = (uint32_t*)&soutput[0];
	uint32_t t;
	uint32_t d[8];
	d[0] = __byte_perm(input[0], input[4], 0x7531);
	d[1] = __byte_perm(input[1], input[5], 0x7531);
	d[2] = __byte_perm(input[2], input[6], 0x7531);
	d[3] = __byte_perm(input[3], input[7], 0x7531);

	SWAP1(d[0], d[1]);
	SWAP1(d[2], d[3]);

	SWAP2(d[0], d[2]);
	SWAP2(d[1], d[3]);

	t = __byte_perm(d[0], d[2], 0x5410);
	d[2] = __byte_perm(d[0], d[2], 0x7632);
	d[0] = t;

	t = __byte_perm(d[1], d[3], 0x5410);
	d[3] = __byte_perm(d[1], d[3], 0x7632);
	d[1] = t;

	SWAP4(d[0], d[2]);
	SWAP4(d[1], d[3]);

	output[0] = d[0];
	output[2] = d[1];
	output[4] = d[0] >> 16;
	output[6] = d[1] >> 16;
	output[8] = d[2];
	output[10] = d[3];
	output[12] = d[2] >> 16;
	output[14] = d[3] >> 16;

#pragma unroll 8
	for (int i = 0; i < 16; i += 2) {
		if (threadIdx.x & 1) output[i] = __byte_perm(output[i], 0, 0x1032);
		output[i] = __byte_perm(output[i], __shfl((int)output[i], (threadIdx.x + 1) & 3, 4), 0x7610);
		output[i + 1] = __shfl((int)output[i], (threadIdx.x + 2) & 3, 4);
		if (threadIdx.x & 3) output[i] = output[i + 1] = 0;
	}
#endif
}
