// Auf QuarkCoin spezialisierte Version von Groestl inkl. Bitslice

#include <stdio.h>
#include <memory.h>
#include <sys/types.h> // off_t

//#include <cuda_helper.h>
#include "cuda_helper_alexis.h"

#ifdef __INTELLISENSE__
#define __CUDA_ARCH__ 500
#endif

#define TPB 256
#define THF 4U

#if __CUDA_ARCH__ >= 300
#define INTENSIVE_GMF
#include "miner.h"
#include "cuda_vectors_alexis.h"
//#include "../x11/cuda_x11_echo_aes.cuh"
#include "groestl_functions_quad.h"
#include "groestl_functions_quad_a1_min3r.cuh"
#include "groestl_transf_quad.h"
#include "groestl_transf_quad_a1_min3r.cuh"
#endif

#define WANT_GROESTL80
#ifdef WANT_GROESTL80
__constant__ static uint32_t c_Message80[20];
#endif

//#include "cuda_quark_groestl512_sm2.cuh"



__global__ __launch_bounds__(TPB, THF)
//const uint32_t startNounce, 
void quark_groestl512_gpu_hash_64_quad_a1_min3r(int *thr_id, const uint32_t threads, uint4* g_hash)
{
	if ((*(int*)(((uintptr_t)thr_id) & ~15ULL)) & 0x40)
		return;
#if __CUDA_ARCH__ >= 300
	// BEWARE : 4-WAY CODE (one hash need 4 threads)
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x); // >> 2; // done on cpu

	if (thread < threads)
	{
		//uint32_t message[8];
		//uint32_t state[8];
		uint4 state[2], message[2];

		//uint32_t nounce = g_nonceVector ? g_nonceVector[thread] : (startNounce + thread); // assuming compiler doesn't already do this for the gpu...
		//uint32_t nounce = startNounce + (thread >> 2);
		//off_t hashPosition = nounce - startNounce;
		//uint32_t *pHash = &g_hash[hashPosition << 4];
		const uint32_t thr = threadIdx.x & 0x3; // % THF
		//		uint32_t *pHash = &g_hash[thread << 1]; // thread << 4
		uint4 *pHash = (uint4*)&g_hash[thread ^ thr]; // thread << 4


		/*| M0 M1 M2 M3 | M4 M5 M6 M7 | (input)
		--|-------------|-------------|
		T0|  0  4  8 12 | 80          |
		T1|  1  5    13 |             |
		T2|  2  6    14 |             |
		T3|  3  7    15 |          01 |
		--|-------------|-------------| */

		message[0].x = ((uint32_t*)&pHash[0])[thr];
		message[0].y = ((uint32_t*)&pHash[1])[thr];
		message[0].z = ((uint32_t*)&pHash[2])[thr];
		message[0].w = ((uint32_t*)&pHash[3])[thr];
//		__syncthreads();

//#pragma unroll
//		for (int k = 0; k<4; k++) message[k] = pHash[thr + (k * THF)];
#if 0
#pragma unroll
		for (int k = 4; k<8; k++) message[k] = 0;


		if (thr == 0)
		{
			message[4] = 0x80U; // end of data tag
			uint32_t msgBitsliced[8];
			to_bitslice_quad(message, msgBitsliced);

			groestl512_progressMessage_quad(state, msgBitsliced);

			uint32_t hash[16];
			from_bitslice_quad(state, hash);
			uint4 *phash = (uint4*)hash;
			uint4 *outpt = (uint4*)pHash;
			outpt[0] = phash[0];
			outpt[1] = phash[1];
			outpt[2] = phash[2];
			outpt[3] = phash[3];
		}
		else
		{
			if (thr == 3) message[7] = 0x01000000U;

			uint32_t msgBitsliced[8];
			to_bitslice_quad(message, msgBitsliced);

			groestl512_progressMessage_quad(state, msgBitsliced);

			uint32_t hash[16];
			from_bitslice_quad(state, hash);
		}

#else
		message[1].x = (0x80 & (thr-1));
		message[1].y = 0;
		message[1].z = 0;
		message[1].w = 0x01000000 & -(thr == 3);

//		message[1].x = 0;
//		message[1].y = 0;
//		message[1].z = 0;
//		message[1].w = 0;
//		if (thr == 0)
//			message[1].x = 0x80; // if (thr == 0)
//		if (thr == 3)
//			message[1].w = 0x01000000; // if (thr == 3)
//		message[1].x = 0x80 & (thr - 1); // if (thr == 0)
//		message[1].y = 0;
//		message[1].z = 0;

//		message[1].w = 0x01000000 & -((thr + 1) >> 2); // if (thr == 3)

//#pragma unroll
		//		for (int k = 4; k<8; k++) message[k] = 0;
		//		if (thr == 0) message[4] = 0x80U; // end of data tag
//		if (thr == 3) message[7] = 0x01000000U;

//		uint32_t msgBitsliced[8];
		uint4 msgBitsliced[2];
		to_bitslice_quad_a1_min3r(message, msgBitsliced); //! error?!
//		to_bitslice_quad((uint32_t*)message, (uint32_t*)msgBitsliced);
		
//		msgBitsliced[0] |= thr;
		groestl512_progressMessage_quad_a1_min3r(state, msgBitsliced); // works
//		groestl512_progressMessage_quad((uint32_t*)state, (uint32_t*)msgBitsliced);
		//! state is used cross thread?!
//		state[0] |= thr;
		//uint32_t hash[16];
		uint4 hash[4];
		//! optimize vvv
//		*(uint2x4*)&hash[0] = *(uint2x4*)&state[0];
//		*(uint2x4*)&hash[2] = *(uint2x4*)&state[2];

		from_bitslice_quad_a1_min3r(state, hash);//! error :(
//		from_bitslice_quad((uint32_t*)state, (uint32_t*)hash);
		//		if (thr != 0) state[0] = 0;
		/*
		if (0)//(thr == 0)
		{
			uint4 flag0 = { (thr != 0) - 1, (thr != 0) - 1, (thr != 0) - 1, (thr != 0) - 1 };
			uint4 flagn0 = { (thr == 0) - 1, (thr == 0) - 1, (thr == 0) - 1, (thr == 0) - 1 };
			pHash[0] = (hash[0] & flag0) | (pHash[0] & flagn0);
			pHash[1] = (hash[1] & flag0) | (pHash[1] & flagn0);
			pHash[2] = (hash[2] & flag0) | (pHash[2] & flagn0);
			pHash[3] = (hash[3] & flag0) | (pHash[3] & flagn0);
		}
		*/
		/*
		pHash[0] = (hash[0] & flag0) | (pHash[0] & flagn0);
		pHash[1] = (hash[1] & flag0) | (pHash[1] & flagn0);
		pHash[2] = (hash[2] & flag0) | (pHash[2] & flagn0);
		pHash[3] = (hash[3] & flag0) | (pHash[3] & flagn0);
		*/

		if (thr == 0)
		{
			//! hash is unused unless thr == 0 ...
//			uint4* ohash = pHash;
//			uint4* thash = hash;
			*(uint2x4*)&pHash[0] = *(uint2x4*)&hash[0];
			*(uint2x4*)&pHash[2] = *(uint2x4*)&hash[2];
		}

#endif // 0
	}
#endif
}

__global__ __launch_bounds__(TPB, THF)
void quark_groestl512_gpu_hash_64_quad(int *thr_id, const uint32_t threads, const uint32_t startNounce, uint32_t * g_hash, uint32_t * __restrict g_nonceVector)
{
	//! fixme please
#if 0 // __CUDA_ARCH__ >= 300


	// BEWARE : 4-WAY CODE (one hash need 4 threads)
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x) >> 2;

	if (thread < threads)
	{
		uint32_t message[8];
		uint32_t state[8];

		uint32_t nounce = g_nonceVector ? g_nonceVector[thread] : (startNounce + thread);
		off_t hashPosition = nounce - startNounce;
		uint32_t *pHash = &g_hash[hashPosition << 4];

		const uint32_t thr = threadIdx.x & 0x3; // % THF

		/*| M0 M1 M2 M3 | M4 M5 M6 M7 | (input)
		--|-------------|-------------|
		T0|  0  4  8 12 | 80          |
		T1|  1  5    13 |             |
		T2|  2  6    14 |             |
		T3|  3  7    15 |          01 |
		--|-------------|-------------| */

#pragma unroll
		for (int k = 0; k<4; k++) message[k] = pHash[thr + (k * THF)];

#pragma unroll
		for (int k = 4; k<8; k++) message[k] = 0;

		if (thr == 0) message[4] = 0x80U; // end of data tag
		if (thr == 3) message[7] = 0x01000000U; 

		uint32_t msgBitsliced[8];
		to_bitslice_quad(message, msgBitsliced);

		groestl512_progressMessage_quad(state, msgBitsliced);

		uint32_t hash[16];
		from_bitslice_quad(state, hash);

		// uint4 = 4x4 uint32_t = 16 bytes
		if (thr == 0) {
			uint4 *phash = (uint4*)hash;
			uint4 *outpt = (uint4*)pHash;
			outpt[0] = phash[0];
			outpt[1] = phash[1];
			outpt[2] = phash[2];
			outpt[3] = phash[3];
		}
	}
#endif
}

__host__
void quark_groestl512_cpu_init(int thr_id, uint32_t threads)
{
//	int dev_id = device_map[thr_id];
	cuda_get_arch(thr_id);
//	if (device_sm[dev_id] < 300 || cuda_arch[dev_id] < 300)
//		quark_groestl512_sm20_init(thr_id, threads);
}

__host__
void quark_groestl512_cpu_free(int thr_id)
{
//	int dev_id = device_map[thr_id];
//	if (device_sm[dev_id] < 300 || cuda_arch[dev_id] < 300)
//		quark_groestl512_sm20_free(thr_id);
}

__host__
void quark_groestl512_cpu_hash_64(int *thr_id, uint32_t threads, uint32_t *d_hash)
{
	uint32_t threadsperblock = TPB;
	// Compute 3.0 benutzt die registeroptimierte Quad Variante mit Warp Shuffle
	// mit den Quad Funktionen brauchen wir jetzt 4 threads pro Hash, daher Faktor 4 bei der Blockzahl
	const uint32_t factor = THF;

	dim3 grid(factor*((threads + threadsperblock - 1) / threadsperblock));
	dim3 block(threadsperblock);

//	int dev_id = device_map[thr_id];
	//! GTX 1070+ (?) run quark_groestl512_gpu_hash_64_quad

//	if (device_sm[dev_id] >= 300 && cuda_arch[dev_id] >= 300)// && order == -1) //! for x16r, TBD if it will work on other algos.
//	{
	quark_groestl512_gpu_hash_64_quad_a1_min3r << <grid, block >> >(thr_id, threads << 2, (uint4*)d_hash);
//	}
	/*
	else 
		if (device_sm[dev_id] >= 300 && cuda_arch[dev_id] >= 300)
		quark_groestl512_gpu_hash_64_quad<<<grid, block>>>(threads, startNounce, d_hash, d_nonceVector);
	*/
//	else
//		quark_groestl512_sm20_hash_64(thr_id, threads, d_hash, order);
}

// --------------------------------------------------------------------------------------------------------------------------------------------

#ifdef WANT_GROESTL80

__host__
void groestl512_setBlock_80(int thr_id, uint32_t *endiandata)
{
	cudaMemcpyToSymbol(c_Message80, endiandata, sizeof(c_Message80), 0, cudaMemcpyHostToDevice);
}

__global__ __launch_bounds__(TPB, THF)
void groestl512_gpu_hash_80_quad_a1_min3r(const uint32_t threads, const uint32_t startNounce, uint4* g_hash)
{
#if __CUDA_ARCH__ >= 300
	// BEWARE : 4-WAY CODE (one hash need 4 threads)
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x); // >> 2; // done on cpu

	if (thread < threads)
	{
		const uint32_t thr = threadIdx.x & 0x3; // % THF
		uint4 message[2];
		//		uint32_t *pHash = &g_hash[thread << 1]; // thread << 4
		uint4 *pHash = (uint4*)&g_hash[thread ^ thr]; // thread << 4
		/*| M0 M1 M2 M3 M4 | M5 M6 M7 | (input)
		--|----------------|----------|
		T0|  0  4  8 12 16 | 80       |
		T1|  1  5       17 |          |
		T2|  2  6       18 |          |
		T3|  3  7       Nc |       01 |
		--|----------------|----------| TPR */
		message[0].x = c_Message80[thr + (0 * THF)];
		message[0].y = c_Message80[thr + (1 * THF)];
		message[0].z = c_Message80[thr + (2 * THF)];
		message[0].w = c_Message80[thr + (3 * THF)];
		message[1].x = c_Message80[thr + (4 * THF)];

		__syncthreads();


//		message[1].y = 0;

//		message[1].w = 0;

//		message[1].x = cuda_swab32(startNounce + (thread>>2)) & -(thr == 3);
//		message[1].y = 0x80 & -(thr == 0);
		message[1].z = 0;
//		message[1].w = 0x01000000U & -(thr == 3);
//		if (thr == 3) {
//			message[1].x = cuda_swab32(startNounce + (thread >> 2));
//			message[1].w = 0x01000000U;
//		}
//		if (thr == 0)
//			message[1].y = 0x80;
		message[1].y = 0x80 & (thr -1);
		message[1].w = 0x01000000U & -(thr == 3);
		message[1].x = (cuda_swab32(startNounce + (thread >> 2)) & -(thr == 3)) | (message[1].x & -(thr != 3));

		uint4 msgBitsliced[2];
//		to_bitslice_quad((uint32_t*)message, (uint32_t*)msgBitsliced);
		to_bitslice_quad_a1_min3r(message, msgBitsliced);

		uint4 state[2];
//		groestl512_progressMessage_quad((uint32_t*)state, (uint32_t*)msgBitsliced);
		groestl512_progressMessage_quad_a1_min3r(state, msgBitsliced); // works

		uint4 hash[4];
//		from_bitslice_quad((uint32_t*)state, (uint32_t*)hash);
		from_bitslice_quad_a1_min3r(state, hash);
//		from_bitslice_quad_a1_min3r((uint32_t*)state, (uint32_t*)hash);
		 
		if (thr == 0) { /* 4 threads were done */
			*(uint2x4*)&pHash[0] = *(uint2x4*)&hash[0];
			*(uint2x4*)&pHash[2] = *(uint2x4*)&hash[2];
			/*
			pHash[0] = hash[0];
			pHash[1] = hash[1];
			pHash[2] = hash[2];
			pHash[3] = hash[3];
			*/
		}
	}
#endif
}

__global__ __launch_bounds__(TPB, THF)
void groestl512_gpu_hash_80_quad(const uint32_t threads, const uint32_t startNounce, uint32_t * g_outhash)
{
#if __CUDA_ARCH__ >= 300
	// BEWARE : 4-WAY CODE (one hash need 4 threads)
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x) >> 2;
	if (thread < threads)
	{
		const uint32_t thr = threadIdx.x & 0x3; // % THF

		/*| M0 M1 M2 M3 M4 | M5 M6 M7 | (input)
		--|----------------|----------|
		T0|  0  4  8 12 16 | 80       |
		T1|  1  5       17 |          |
		T2|  2  6       18 |          |
		T3|  3  7       Nc |       01 |
		--|----------------|----------| TPR */

		uint32_t message[8];

		#pragma unroll 5
		for(int k=0; k<5; k++) message[k] = c_Message80[thr + (k * THF)];

		#pragma unroll 3
		for(int k=5; k<8; k++) message[k] = 0;

		if (thr == 0) message[5] = 0x80U;
		if (thr == 3) {
			message[4] = cuda_swab32(startNounce + thread);
			message[7] = 0x01000000U;
		}

		uint32_t msgBitsliced[8];
		to_bitslice_quad(message, msgBitsliced);

		uint32_t state[8];
		groestl512_progressMessage_quad(state, msgBitsliced);

		uint32_t hash[16];
		from_bitslice_quad(state, hash);

		if (thr == 0) { /* 4 threads were done */
			const off_t hashPosition = thread;
			//if (!thread) hash[15] = 0xFFFFFFFF;
			uint4 *outpt = (uint4*) &g_outhash[hashPosition << 4];
			uint4 *phash = (uint4*) hash;
			outpt[0] = phash[0];
			outpt[1] = phash[1];
			outpt[2] = phash[2];
			outpt[3] = phash[3];
		}
	}
#endif
}

__host__
void groestl512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNounce, uint32_t *d_hash)
{
//	int dev_id = device_map[thr_id];

//	if (device_sm[dev_id] >= 300 && cuda_arch[dev_id] >= 300) {
		const uint32_t threadsperblock = TPB;
		const uint32_t factor = THF;
		
		dim3 grid(factor*((threads + threadsperblock-1)/threadsperblock));
		dim3 block(threadsperblock);
		//! setup only for x16r(s?)
		groestl512_gpu_hash_80_quad_a1_min3r <<<grid, block>>> (threads << 2, startNounce, (uint4*)d_hash);
//		groestl512_gpu_hash_80_quad<< <grid, block >> > (threads, startNounce, d_hash);
		/*

	} 
	else {

		const uint32_t threadsperblock = 256;
		dim3 grid((threads + threadsperblock-1)/threadsperblock);
		dim3 block(threadsperblock);

		groestl512_gpu_hash_80_sm2 <<<grid, block>>> (threads, startNounce, d_hash);
	}
	*/
}

#endif
