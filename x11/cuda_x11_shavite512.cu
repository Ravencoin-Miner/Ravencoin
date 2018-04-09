#include <memory.h> // memcpy()

#include "cuda_helper_alexis.h"

extern __device__ __device_builtin__ void __threadfence_block(void);
 
//#define TPB 128
#define TPB 384

__constant__ uint32_t c_PaddedMessage80[20]; // padded message (80 bytes + padding)

//#include "cuda_x11_aes.cuh"
#define INTENSIVE_GMF
#include "../x11/cuda_x11_echo_aes.cuh"

__device__ __forceinline__
#if HACK_1
static void round_3_7_11(const uint32_t sharedMemory[4][256], uint32_t* r, uint4 *p, uint4 &x) {
#else
static void round_3_7_11(const uint32_t* __restrict__ sharedMemory, uint32_t* r, uint4 *p, uint4 &x) {
#endif
	KEY_EXPAND_ELT(sharedMemory, &r[0]);
	*(uint4*)&r[0] ^= *(uint4*)&r[28];
	x = p[2] ^ *(uint4*)&r[0];
	KEY_EXPAND_ELT(sharedMemory, &r[4]);
	*(uint4*)&r[4] ^= *(uint4*)&r[0];
	/*
	r[4] ^= r[0];
	r[5] ^= r[1];
	r[6] ^= r[2];
	r[7] ^= r[3];
	*/
	AES_ROUND_NOKEY(sharedMemory, &x);
	x ^= *(uint4*)&r[4];
	/*
	x.x ^= r[4];
	x.y ^= r[5];
	x.z ^= r[6];
	x.w ^= r[7];
	*/
	KEY_EXPAND_ELT(sharedMemory, &r[8]);
	*(uint4*)&r[8] ^= *(uint4*)&r[4];
	/*
	r[8] ^= r[4];
	r[9] ^= r[5];
	r[10] ^= r[6];
	r[11] ^= r[7];
	*/
	AES_ROUND_NOKEY(sharedMemory, &x);
	x ^= *(uint4*)&r[8];
	/*
	x.x ^= r[8];
	x.y ^= r[9];
	x.z ^= r[10];
	x.w ^= r[11];
	*/
	KEY_EXPAND_ELT(sharedMemory, &r[12]);
	*(uint4*)&r[12] ^= *(uint4*)&r[8];
	/*
	r[12] ^= r[8];
	r[13] ^= r[9];
	r[14] ^= r[10];
	r[15] ^= r[11];
	*/
	AES_ROUND_NOKEY(sharedMemory, &x);
	x ^= *(uint4*)&r[12];
	/*
	x.x ^= r[12];
	x.y ^= r[13];
	x.z ^= r[14];
	x.w ^= r[15];
	*/
	AES_ROUND_NOKEY(sharedMemory, &x);
	p[1] ^= x;
	/*
	p[1].x ^= x.x;
	p[1].y ^= x.y;
	p[1].z ^= x.z;
	p[1].w ^= x.w;
	*/
	KEY_EXPAND_ELT(sharedMemory, &r[16]);
	*(uint4*)&r[16] ^= *(uint4*)&r[12];
	x = p[0] ^ *(uint4*)&r[16];
	KEY_EXPAND_ELT(sharedMemory, &r[20]);
	*(uint4*)&r[20] ^= *(uint4*)&r[16];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x ^= *(uint4*)&r[20];
	KEY_EXPAND_ELT(sharedMemory, &r[24]);
	*(uint4*)&r[24] ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x ^= *(uint4*)&r[24];
	KEY_EXPAND_ELT(sharedMemory, &r[28]);
	AES_ROUND_NOKEY(sharedMemory, &x);
	*(uint4*)&r[28] ^= *(uint4*)&r[24];
	x ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY(sharedMemory, &x);
	p[3] ^= x;
}

__device__ __forceinline__
#if HACK_1
static void round_4_8_12(const uint32_t sharedMemory[4][256], uint32_t* r, uint4 *p, uint4 &x){
#else
static void round_4_8_12(const uint32_t* __restrict__ sharedMemory, uint32_t* r, uint4 *p, uint4 &x){
#endif
	*(uint4*)&r[0] ^= *(uint4*)&r[25];
	x = p[1] ^ *(uint4*)&r[0];
	AES_ROUND_NOKEY(sharedMemory, &x);

	r[4] ^= r[29];	r[5] ^= r[30];
	r[6] ^= r[31];	r[7] ^= r[0];

	x ^= *(uint4*)&r[4];
	*(uint4*)&r[8] ^= *(uint4*)&r[1];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x ^= *(uint4*)&r[8];
	*(uint4*)&r[12] ^= *(uint4*)&r[5];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY(sharedMemory, &x);
	p[0] ^= x;
	*(uint4*)&r[16] ^= *(uint4*)&r[9];
	x = p[3] ^ *(uint4*)&r[16];
	AES_ROUND_NOKEY(sharedMemory, &x);
	*(uint4*)&r[20] ^= *(uint4*)&r[13];
	x ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY(sharedMemory, &x);
	*(uint4*)&r[24] ^= *(uint4*)&r[17];
	x ^= *(uint4*)&r[24];
	*(uint4*)&r[28] ^= *(uint4*)&r[21];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY(sharedMemory, &x);
	p[2] ^= x;
}

__device__ __forceinline__
static void c512(const uint32_t* sharedMemory, const uint32_t *state, uint32_t *msg, uint2x4 *Hash, const uint32_t counter)
{
	uint4 p[4];
	uint4 x;
	uint32_t r[32];

	*(uint2x4*)&r[0] = *(uint2x4*)&msg[0];
	*(uint2x4*)&r[8] = *(uint2x4*)&msg[8];
	*(uint4*)&r[16] = *(uint4*)&msg[16];

	*(uint2x4*)&p[0] = *(uint2x4*)&state[0];
	*(uint2x4*)&p[2] = *(uint2x4*)&state[8];

	/* round 0 */
	x = p[1] ^ *(uint4*)&r[0];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x ^= *(uint4*)&r[4];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x ^= *(uint4*)&r[8];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY(sharedMemory, &x);
	p[0] ^= x;
	//
	if (counter == 512)
	{
		*(uint4*)&r[20] = *(uint4*)&msg[20];
		*(uint2x4*)&r[24] = *(uint2x4*)&msg[24];
		x = p[3];
		x.x ^= 0x80;
		AES_ROUND_NOKEY(sharedMemory, &x);
		AES_ROUND_NOKEY(sharedMemory, &x);
		x.w ^= 0x2000000;
		AES_ROUND_NOKEY(sharedMemory, &x);
		x.w ^= 0x2000000;

	}
	else
	{
		x = p[3] ^ *(uint4*)&r[16];

		r[0x14] = 0x80;
		r[0x15] = 0; r[0x16] = 0; r[0x17] = 0; r[0x18] = 0; r[0x19] = 0; r[0x1a] = 0;
		r[0x1b] = 0x2800000;
		r[0x1c] = 0; r[0x1d] = 0; r[0x1e] = 0;
		r[0x1f] = 0x2000000;

		AES_ROUND_NOKEY(sharedMemory, &x);

		x.x ^= 0x80;
		AES_ROUND_NOKEY(sharedMemory, &x);

		x.w ^= 0x02800000;
		AES_ROUND_NOKEY(sharedMemory, &x);

		x.w ^= 0x02000000;
		//
	}
	AES_ROUND_NOKEY(sharedMemory, &x);
	p[2] ^= x;

	// 1
	KEY_EXPAND_ELT(sharedMemory, &r[0]);
	*(uint4*)&r[0] ^= *(uint4*)&r[28];
	r[0] ^= counter; // 0x200/0x280
	r[3] ^= 0xFFFFFFFF;
	x = p[0] ^ *(uint4*)&r[0];
	AES_ROUND_NOKEY(sharedMemory, &x);
	KEY_EXPAND_ELT(sharedMemory, &r[4]);
	*(uint4*)&r[4] ^= *(uint4*)&r[0];
	x ^= *(uint4*)&r[4];
	AES_ROUND_NOKEY(sharedMemory, &x);
	KEY_EXPAND_ELT(sharedMemory, &r[8]);
	*(uint4*)&r[8] ^= *(uint4*)&r[4];
	x ^= *(uint4*)&r[8];
	AES_ROUND_NOKEY(sharedMemory, &x);
	KEY_EXPAND_ELT(sharedMemory, &r[12]);
	*(uint4*)&r[12] ^= *(uint4*)&r[8];
	x ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY(sharedMemory, &x);
	p[3] ^= x;
	KEY_EXPAND_ELT(sharedMemory, &r[16]);
	*(uint4*)&r[16] ^= *(uint4*)&r[12];
	x = p[2] ^ *(uint4*)&r[16];
	AES_ROUND_NOKEY(sharedMemory, &x);
	KEY_EXPAND_ELT(sharedMemory, &r[20]);
	*(uint4*)&r[20] ^= *(uint4*)&r[16];
	x ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY(sharedMemory, &x);
	KEY_EXPAND_ELT(sharedMemory, &r[24]);
	*(uint4*)&r[24] ^= *(uint4*)&r[20];
	x ^= *(uint4*)&r[24];
	AES_ROUND_NOKEY(sharedMemory, &x);
	KEY_EXPAND_ELT(sharedMemory, &r[28]);
	*(uint4*)&r[28] ^= *(uint4*)&r[24];
	x ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY(sharedMemory, &x);
	p[1] ^= x;
	*(uint4*)&r[0] ^= *(uint4*)&r[25];
	x = p[3] ^ *(uint4*)&r[0];
	AES_ROUND_NOKEY(sharedMemory, &x);

	r[4] ^= r[29]; r[5] ^= r[30];
	r[6] ^= r[31]; r[7] ^= r[0];

	x ^= *(uint4*)&r[4];
	AES_ROUND_NOKEY(sharedMemory, &x);
	*(uint4*)&r[8] ^= *(uint4*)&r[1];
	x ^= *(uint4*)&r[8];
	AES_ROUND_NOKEY(sharedMemory, &x);
	*(uint4*)&r[12] ^= *(uint4*)&r[5];
	x ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY(sharedMemory, &x);
	p[2] ^= x;
	*(uint4*)&r[16] ^= *(uint4*)&r[9];
	x = p[1] ^ *(uint4*)&r[16];
	AES_ROUND_NOKEY(sharedMemory, &x);
	*(uint4*)&r[20] ^= *(uint4*)&r[13];
	x ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY(sharedMemory, &x);
	*(uint4*)&r[24] ^= *(uint4*)&r[17];
	x ^= *(uint4*)&r[24];
	AES_ROUND_NOKEY(sharedMemory, &x);
	*(uint4*)&r[28] ^= *(uint4*)&r[21];
	x ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY(sharedMemory, &x);

	p[0] ^= x;

	/* round 3, 7, 11 */
	round_3_7_11(sharedMemory, r, p, x);


	/* round 4, 8, 12 */
	round_4_8_12(sharedMemory, r, p, x);

	// 2
	KEY_EXPAND_ELT(sharedMemory, &r[0]);
	*(uint4*)&r[0] ^= *(uint4*)&r[28];
	x = p[0] ^ *(uint4*)&r[0];
	AES_ROUND_NOKEY(sharedMemory, &x);
	KEY_EXPAND_ELT(sharedMemory, &r[4]);
	*(uint4*)&r[4] ^= *(uint4*)&r[0];
	r[7] ^= (~counter);// 512/640
	x ^= *(uint4*)&r[4];
	AES_ROUND_NOKEY(sharedMemory, &x);
	KEY_EXPAND_ELT(sharedMemory, &r[8]);
	*(uint4*)&r[8] ^= *(uint4*)&r[4];
	x ^= *(uint4*)&r[8];
	AES_ROUND_NOKEY(sharedMemory, &x);
	KEY_EXPAND_ELT(sharedMemory, &r[12]);
	*(uint4*)&r[12] ^= *(uint4*)&r[8];
	x ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY(sharedMemory, &x);
	p[3] ^= x;
	KEY_EXPAND_ELT(sharedMemory, &r[16]);
	*(uint4*)&r[16] ^= *(uint4*)&r[12];
	x = p[2] ^ *(uint4*)&r[16];
	AES_ROUND_NOKEY(sharedMemory, &x);
	KEY_EXPAND_ELT(sharedMemory, &r[20]);
	*(uint4*)&r[20] ^= *(uint4*)&r[16];
	x ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY(sharedMemory, &x);
	KEY_EXPAND_ELT(sharedMemory, &r[24]);
	*(uint4*)&r[24] ^= *(uint4*)&r[20];
	x ^= *(uint4*)&r[24];
	AES_ROUND_NOKEY(sharedMemory, &x);
	KEY_EXPAND_ELT(sharedMemory, &r[28]);
	*(uint4*)&r[28] ^= *(uint4*)&r[24];
	x ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY(sharedMemory, &x);
	p[1] ^= x;

	*(uint4*)&r[0] ^= *(uint4*)&r[25];
	x = p[3] ^ *(uint4*)&r[0];
	AES_ROUND_NOKEY(sharedMemory, &x);
	r[4] ^= r[29];
	r[5] ^= r[30];
	r[6] ^= r[31];
	r[7] ^= r[0];
	x ^= *(uint4*)&r[4];
	AES_ROUND_NOKEY(sharedMemory, &x);
	*(uint4*)&r[8] ^= *(uint4*)&r[1];
	x ^= *(uint4*)&r[8];
	AES_ROUND_NOKEY(sharedMemory, &x);
	*(uint4*)&r[12] ^= *(uint4*)&r[5];
	x ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY(sharedMemory, &x);
	p[2] ^= x;
	*(uint4*)&r[16] ^= *(uint4*)&r[9];
	x = p[1] ^ *(uint4*)&r[16];
	AES_ROUND_NOKEY(sharedMemory, &x);
	*(uint4*)&r[20] ^= *(uint4*)&r[13];
	x ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY(sharedMemory, &x);
	*(uint4*)&r[24] ^= *(uint4*)&r[17];
	x ^= *(uint4*)&r[24];
	AES_ROUND_NOKEY(sharedMemory, &x);
	*(uint4*)&r[28] ^= *(uint4*)&r[21];
	x ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY(sharedMemory, &x);
	p[0] ^= x;

	/* round 3, 7, 11 */
	round_3_7_11(sharedMemory, r, p, x);

	/* round 4, 8, 12 */
	round_4_8_12(sharedMemory, r, p, x);

	// 3
	KEY_EXPAND_ELT(sharedMemory, &r[0]);
	*(uint4*)&r[0] ^= *(uint4*)&r[28];
	x = p[0] ^ *(uint4*)&r[0];
	AES_ROUND_NOKEY(sharedMemory, &x);
	KEY_EXPAND_ELT(sharedMemory, &r[4]);
	*(uint4*)&r[4] ^= *(uint4*)&r[0];
	x ^= *(uint4*)&r[4];
	AES_ROUND_NOKEY(sharedMemory, &x);
	KEY_EXPAND_ELT(sharedMemory, &r[8]);
	*(uint4*)&r[8] ^= *(uint4*)&r[4];
	x ^= *(uint4*)&r[8];
	AES_ROUND_NOKEY(sharedMemory, &x);
	KEY_EXPAND_ELT(sharedMemory, &r[12]);
	*(uint4*)&r[12] ^= *(uint4*)&r[8];
	x ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY(sharedMemory, &x);
	p[3] ^= x;
	KEY_EXPAND_ELT(sharedMemory, &r[16]);
	*(uint4*)&r[16] ^= *(uint4*)&r[12];
	x = p[2] ^ *(uint4*)&r[16];
	AES_ROUND_NOKEY(sharedMemory, &x);
	KEY_EXPAND_ELT(sharedMemory, &r[20]);
	*(uint4*)&r[20] ^= *(uint4*)&r[16];
	x ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY(sharedMemory, &x);
	KEY_EXPAND_ELT(sharedMemory, &r[24]);
	*(uint4*)&r[24] ^= *(uint4*)&r[20];
	x ^= *(uint4*)&r[24];
	AES_ROUND_NOKEY(sharedMemory, &x);
	KEY_EXPAND_ELT(sharedMemory, &r[28]);
	*(uint4*)&r[28] ^= *(uint4*)&r[24];
	r[30] ^= counter; // 512/640
	r[31] ^= 0xFFFFFFFF;
	x ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY(sharedMemory, &x);
	p[1] ^= x;

	*(uint4*)&r[0] ^= *(uint4*)&r[25];
	x = p[3] ^ *(uint4*)&r[0];
	AES_ROUND_NOKEY(sharedMemory, &x);
	r[4] ^= r[29];
	r[5] ^= r[30];
	r[6] ^= r[31];
	r[7] ^= r[0];
	x ^= *(uint4*)&r[4];
	AES_ROUND_NOKEY(sharedMemory, &x);
	*(uint4*)&r[8] ^= *(uint4*)&r[1];
	x ^= *(uint4*)&r[8];
	AES_ROUND_NOKEY(sharedMemory, &x);
	*(uint4*)&r[12] ^= *(uint4*)&r[5];
	x ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY(sharedMemory, &x);
	p[2] ^= x;
	*(uint4*)&r[16] ^= *(uint4*)&r[9];
	x = p[1] ^ *(uint4*)&r[16];
	AES_ROUND_NOKEY(sharedMemory, &x);
	*(uint4*)&r[20] ^= *(uint4*)&r[13];
	x ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY(sharedMemory, &x);
	*(uint4*)&r[24] ^= *(uint4*)&r[17];
	x ^= *(uint4*)&r[24];
	AES_ROUND_NOKEY(sharedMemory, &x);
	*(uint4*)&r[28] ^= *(uint4*)&r[21];
	x ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY(sharedMemory, &x);
	p[0] ^= x;

	/* round 3, 7, 11 */
	round_3_7_11(sharedMemory, r, p, x);

	/* round 4, 8, 12 */
	round_4_8_12(sharedMemory, r, p, x);

	/* round 13 */
	KEY_EXPAND_ELT(sharedMemory, &r[0]);
	*(uint4*)&r[0] ^= *(uint4*)&r[28];
	x = p[0] ^ *(uint4*)&r[0];
	AES_ROUND_NOKEY(sharedMemory, &x);
	KEY_EXPAND_ELT(sharedMemory, &r[4]);
	*(uint4*)&r[4] ^= *(uint4*)&r[0];
	x ^= *(uint4*)&r[4];
	AES_ROUND_NOKEY(sharedMemory, &x);
	KEY_EXPAND_ELT(sharedMemory, &r[8]);
	*(uint4*)&r[8] ^= *(uint4*)&r[4];
	x ^= *(uint4*)&r[8];
	AES_ROUND_NOKEY(sharedMemory, &x);
	KEY_EXPAND_ELT(sharedMemory, &r[12]);
	*(uint4*)&r[12] ^= *(uint4*)&r[8];
	x ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY(sharedMemory, &x);
	p[3] ^= x;
	KEY_EXPAND_ELT(sharedMemory, &r[16]);
	*(uint4*)&r[16] ^= *(uint4*)&r[12];
	x = p[2] ^ *(uint4*)&r[16];
	AES_ROUND_NOKEY(sharedMemory, &x);
	KEY_EXPAND_ELT(sharedMemory, &r[20]);
	*(uint4*)&r[20] ^= *(uint4*)&r[16];
	x ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY(sharedMemory, &x);
	KEY_EXPAND_ELT(sharedMemory, &r[24]);
	*(uint4*)&r[24] ^= *(uint4*)&r[20];
	r[25] ^= counter; // 512/640
	r[27] ^= 0xFFFFFFFF;
	x ^= *(uint4*)&r[24];
	AES_ROUND_NOKEY(sharedMemory, &x);
	KEY_EXPAND_ELT(sharedMemory, &r[28]);
	*(uint4*)&r[28] ^= *(uint4*)&r[24];
	x ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY(sharedMemory, &x);
	p[1] ^= x;

	Hash[0] = *(uint2x4*)&state[0] ^ *(uint2x4*)&p[2];
	Hash[1] = *(uint2x4*)&state[8] ^ *(uint2x4*)&p[0];
}


__device__ __forceinline__
void shavite_gpu_init(uint32_t *sharedMemory)
{
	/* each thread startup will fill a uint32 */
	aes_gpu_init(sharedMemory);
}

// GPU Hash
//__global__ __launch_bounds__(TPB, 7) /* 64 registers with 128,8 - 72 regs with 128,7 */
__global__ __launch_bounds__(TPB, 2) /* 64 registers with 128,8 - 72 regs with 128,7 */
void x11_shavite512_gpu_hash_64(uint32_t threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector)
{
	__shared__ uint32_t sharedMemory[1024];

	shavite_gpu_init(sharedMemory);
	__threadfence_block();

	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

		int hashPosition = nounce - startNounce;
		uint32_t *Hash = (uint32_t*)&g_hash[hashPosition<<3];

		// kopiere init-state
		uint32_t state[16] = {
			SPH_C32(0x72FCCDD8), SPH_C32(0x79CA4727), SPH_C32(0x128A077B), SPH_C32(0x40D55AEC),
			SPH_C32(0xD1901A06), SPH_C32(0x430AE307), SPH_C32(0xB29F5CD1), SPH_C32(0xDF07FBFC),
			SPH_C32(0x8E45D73D), SPH_C32(0x681AB538), SPH_C32(0xBDE86578), SPH_C32(0xDD577E47),
			SPH_C32(0xE275EADE), SPH_C32(0x502D9FCD), SPH_C32(0xB9357178), SPH_C32(0x022A4B9A)
		};

		// nachricht laden
		uint32_t msg[32];

		// fülle die Nachricht mit 64-byte (vorheriger Hash)
		#pragma unroll 16
		for(int i=0;i<16;i++)
			msg[i] = Hash[i];

		// Nachrichtenende
		msg[16] = 0x80;
		#pragma unroll 10
		for(int i=17;i<27;i++)
			msg[i] = 0;

		msg[27] = 0x02000000;
		msg[28] = 0;
		msg[29] = 0;
		msg[30] = 0;
		msg[31] = 0x02000000;

		c512(sharedMemory, state, msg, (uint2x4*)Hash, 512);
		/*
		#pragma unroll 16
		for(int i=0;i<16;i++)
			Hash[i] = state[i];
		*/
	}
}

//__global__ __launch_bounds__(TPB, 7)
__global__ __launch_bounds__(TPB, 2)


#if TPB == 128
__global__ __launch_bounds__(TPB, 7)
#elif TPB == 384
__global__ __launch_bounds__(TPB, 2)
#else
#error "Not set up for this"
#endif
void x11_shavite512_gpu_hash_80(uint32_t threads, uint32_t startNounce, uint64_t *g_hash)
{
	#if TPB == 128
	aes_gpu_init_128(sharedMemory);
	#elif TPB == 384
	//! todo, fix naming and sharedMemory
	__shared__ uint32_t sharedMemory[1024];
	aes_gpu_init(sharedMemory); //! hack 1 must be using 256 threads
	#else
	#error "Not set up for this"
	#endif
//	__threadfence_block();

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	// initial state
	uint32_t state[16] = { // should be constant instead of being used to store data and being overwritten
		SPH_C32(0x72FCCDD8), SPH_C32(0x79CA4727), SPH_C32(0x128A077B), SPH_C32(0x40D55AEC),
		SPH_C32(0xD1901A06), SPH_C32(0x430AE307), SPH_C32(0xB29F5CD1), SPH_C32(0xDF07FBFC),
		SPH_C32(0x8E45D73D), SPH_C32(0x681AB538), SPH_C32(0xBDE86578), SPH_C32(0xDD577E47),
		SPH_C32(0xE275EADE), SPH_C32(0x502D9FCD), SPH_C32(0xB9357178), SPH_C32(0x022A4B9A)
	};

	if (thread < threads)
	{
		const uint32_t nounce = startNounce + thread;
		uint64_t *Hash = &g_hash[thread << 3];

		uint32_t r[20];//32
		*(uint2x4*)&r[0] = *(uint2x4*)&c_PaddedMessage80[0];
		*(uint2x4*)&r[8] = *(uint2x4*)&c_PaddedMessage80[8];
		*(uint4*)&r[16] = *(uint4*)&c_PaddedMessage80[16];

//		__syncthreads();


		r[19] = cuda_swab32(nounce);
		/*
		r[20] = 0x80;
		r[21] = r[22] = r[23] = r[24] = r[25] = r[26] = r[28] = r[29] = r[30] = 0;
		r[27] = 0x2800000;
		r[31] = 0x2000000;
		*/
		c512(sharedMemory, state, r, (uint2x4*)Hash, 640);

	} //thread < threads
}

__host__
void x11_shavite512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	const uint32_t threadsperblock = TPB;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	// note: 128 threads minimum are required to init the shared memory array
	//gpulog(LOG_WARNING, thr_id, "x11 shavite512 is not set up for this algo");
	x11_shavite512_gpu_hash_64<<<grid, block>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);
	//MyStreamSynchronize(NULL, order, thr_id);
}

__host__
void x11_shavite512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_outputHash)
{
	const uint32_t threadsperblock = TPB;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	x11_shavite512_gpu_hash_80<<<grid, block>>>(threads, startNounce, (uint64_t*)d_outputHash);
}

__host__
void x11_shavite512_cpu_init(int thr_id, uint32_t threads)
{
	aes_cpu_init(thr_id);
}

__host__
void x11_shavite512_setBlock_80(void *pdata)
{
	// Message with Padding
	// The nonce is at Byte 76.
/*
	unsigned char PaddedMessage[128];
	memcpy(PaddedMessage, pdata, 80);
	memset(PaddedMessage+80, 0, 48);

	cudaMemcpyToSymbol(c_PaddedMessage80, PaddedMessage, 32*sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
*/
	cudaMemcpyToSymbol(c_PaddedMessage80, pdata, 20 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
}
