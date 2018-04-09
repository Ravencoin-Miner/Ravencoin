/**
 * X16R algorithm (X16 with Randomized chain order)
 *
 * tpruvot 2018 - GPL code
 */

#include <stdio.h>
#include <memory.h>
#include <unistd.h>

extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"

#include "sph/sph_luffa.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"

#include "sph/sph_hamsi.h"
#include "sph/sph_fugue.h"
#include "sph/sph_shabal.h"
#include "sph/sph_whirlpool.h"
#include "sph/sph_sha2.h"
//extern struct work_restart *work_restart;
}

#include "miner.h"
#include "cuda_helper.h"
#include "cuda_x16r.h"

#define GPU_HASH_CHECK_LOG 0
static uint32_t *d_hash[MAX_GPUS];

enum Algo {
	BLAKE = 0,
	BMW,
	GROESTL,
	JH,
	KECCAK,
	SKEIN,
	LUFFA,
	CUBEHASH,
	SHAVITE,
	SIMD,
	ECHO,
	HAMSI,
	FUGUE,
	SHABAL,
	WHIRLPOOL,
	SHA512,
	HASH_FUNC_COUNT
};

static const char* algo_strings[] = {
	"blake",
	"bmw512",
	"groestl",
	"jh512",
	"keccak",
	"skein",
	"luffa",
	"cube",
	"shavite",
	"simd",
	"echo",
	"hamsi",
	"fugue",
	"shabal",
	"whirlpool",
	"sha512",
	NULL
};

static __thread uint32_t s_ntime = UINT32_MAX;
static __thread bool s_implemented = false;
static __thread char hashOrder[HASH_FUNC_COUNT + 1] = { 0 };

static void(*pAlgo64[16])(int, uint32_t, uint32_t*) =
{
	quark_blake512_cpu_hash_64,
	quark_bmw512_cpu_hash_64,
	quark_groestl512_cpu_hash_64,
	quark_jh512_cpu_hash_64,
	quark_keccak512_cpu_hash_64,
	quark_skein512_cpu_hash_64,
	x11_luffa512_cpu_hash_64_alexis,
	x11_cubehash512_cpu_hash_64,
	x11_shavite512_cpu_hash_64_alexis,
	x11_simd512_cpu_hash_64,
	x11_echo512_cpu_hash_64_alexis,
	x13_hamsi512_cpu_hash_64_alexis,
	x13_fugue512_cpu_hash_64_alexis,
	x14_shabal512_cpu_hash_64_alexis,
	x15_whirlpool_cpu_hash_64,
	x17_sha512_cpu_hash_64 
};
static void(*pAlgo80[16])(int, uint32_t, uint32_t, uint32_t*) =
{
	quark_blake512_cpu_hash_80,
	quark_bmw512_cpu_hash_80,
	groestl512_cuda_hash_80,
	jh512_cuda_hash_80,
	keccak512_cuda_hash_80,
	skein512_cpu_hash_80,
	qubit_luffa512_cpu_hash_80_alexis,
	cubehash512_cuda_hash_80,
	x11_shavite512_cpu_hash_80,
	x16_simd512_cuda_hash_80,
	x16_echo512_cuda_hash_80,
	x16_hamsi512_cuda_hash_80,
	x16_fugue512_cuda_hash_80,
	x16_shabal512_cuda_hash_80,
	x16_whirlpool512_hash_80,
	x16_sha512_cuda_hash_80
};

/*
BLAKE = 0,
BMW,1
GROESTL,2
JH,3
KECCAK,4
SKEIN,5
LUFFA,6
CUBEHASH,7
SHAVITE,8
SIMD,9
ECHO,a
HAMSI,b
FUGUE,c
SHABAL,d
WHIRLPOOL,e
SHA512,f
*/

//void quark_blake512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], 0);
#if 0
static void run_x16r_rounds(const uint32_t* prevblock, int thr_id, uint32_t threads, uint32_t startNounce, uint32_t* d_hash)
{
	// big toys for big boys
	pAlgo80[(*(uint64_t*)prevblock >> 60 - (0 * 4)) & 0x0f](thr_id, threads, startNounce, d_hash);
	pAlgo64[(*(uint64_t*)prevblock >> 60 - (1 * 4)) & 0x0f](thr_id, threads, d_hash, 1);
	pAlgo64[(*(uint64_t*)prevblock >> 60 - (2 * 4)) & 0x0f](thr_id, threads, d_hash, 2);
	pAlgo64[(*(uint64_t*)prevblock >> 60 - (3 * 4)) & 0x0f](thr_id, threads, d_hash, 3);
	pAlgo64[(*(uint64_t*)prevblock >> 60 - (4 * 4)) & 0x0f](thr_id, threads, d_hash, 4);
	pAlgo64[(*(uint64_t*)prevblock >> 60 - (5 * 4)) & 0x0f](thr_id, threads, d_hash, 5);
	pAlgo64[(*(uint64_t*)prevblock >> 60 - (6 * 4)) & 0x0f](thr_id, threads, d_hash, 6);
	pAlgo64[(*(uint64_t*)prevblock >> 60 - (7 * 4)) & 0x0f](thr_id, threads, d_hash, 7);
	pAlgo64[(*(uint64_t*)prevblock >> 60 - (8 * 4)) & 0x0f](thr_id, threads, d_hash, 8);
	pAlgo64[(*(uint64_t*)prevblock >> 60 - (9 * 4)) & 0x0f](thr_id, threads, d_hash, 9);
	pAlgo64[(*(uint64_t*)prevblock >> 60 - (10* 4)) & 0x0f](thr_id, threads, d_hash,10);
	pAlgo64[(*(uint64_t*)prevblock >> 60 - (11* 4)) & 0x0f](thr_id, threads, d_hash,11);
	pAlgo64[(*(uint64_t*)prevblock >> 60 - (12* 4)) & 0x0f](thr_id, threads, d_hash,12);
	pAlgo64[(*(uint64_t*)prevblock >> 60 - (13* 4)) & 0x0f](thr_id, threads, d_hash,13);
	pAlgo64[(*(uint64_t*)prevblock >> 60 - (14* 4)) & 0x0f](thr_id, threads, d_hash,14);
	pAlgo64[(*(uint64_t*)prevblock >> 60 - (15* 4)) & 0x0f](thr_id, threads, d_hash,15);
}
#endif
static void getAlgoString(const uint32_t* prevblock, char *output)
{
	for (int i = 0; i < 16; i++)
	{
			*output++ = (*(uint64_t*)prevblock >> 60 - (i * 4)) & 0x0f;
	}
	/*
	char *sptr = output;
	uint8_t* data = (uint8_t*)prevblock;
	//if data == 0x123456789abcdef how does it order?
	for (uint8_t j = 0; j < HASH_FUNC_COUNT; j++) {
		uint8_t b = (15 - j) >> 1; // 16 ascii hex chars, reversed
		uint8_t algoDigit = (j & 1) ? data[b] & 0xF : data[b] >> 4;
		if (algoDigit >= 10)
			sprintf(sptr, "%c", 'A' + (algoDigit - 10));
		else
			sprintf(sptr, "%u", (uint32_t) algoDigit);
		sptr++;
	}
	*sptr = '\0';
	*/
}

// X16R CPU Hash (Validation)
extern "C" void x16r_hash(void *output, const void *input)
{
	//unsigned char _ALIGN(64) hash[128];

	sph_blake512_context ctx_blake;
	sph_bmw512_context ctx_bmw;
	sph_groestl512_context ctx_groestl;
	sph_jh512_context ctx_jh;
	sph_keccak512_context ctx_keccak;
	sph_skein512_context ctx_skein;
	sph_luffa512_context ctx_luffa;
	sph_cubehash512_context ctx_cubehash;
	sph_shavite512_context ctx_shavite;
	sph_simd512_context ctx_simd;
	sph_echo512_context ctx_echo;
	sph_hamsi512_context ctx_hamsi;
	sph_fugue512_context ctx_fugue;
	sph_shabal512_context ctx_shabal;
	sph_whirlpool_context ctx_whirlpool;
	sph_sha512_context ctx_sha512;

	void *in = (void*) input;
	int size = 80;

	uint32_t *in32 = (uint32_t*) input;
//	getAlgoString(&in32[1], hashOrder);
	uint64_t prevblock = *(uint64_t*)&in32[1];

	for (int i = 0; i < 16; i++)
	{
//		const char elem = hashOrder[i];
//		const uint8_t algo = elem >= 'A' ? elem - 'A' + 10 : elem - '0';
//		uint8_t algo = hashOrder[i];
		switch ((prevblock >> 60 - (i << 2)) & 0x0f) {
		case BLAKE:
			sph_blake512_init(&ctx_blake);
			sph_blake512(&ctx_blake, in, size);
			sph_blake512_close(&ctx_blake, output);
			break;
		case BMW:
			sph_bmw512_init(&ctx_bmw);
			sph_bmw512(&ctx_bmw, in, size);
			sph_bmw512_close(&ctx_bmw, output);
			break;
		case GROESTL:
			sph_groestl512_init(&ctx_groestl);
			sph_groestl512(&ctx_groestl, in, size);
			sph_groestl512_close(&ctx_groestl, output);
			break;
		case SKEIN:
			sph_skein512_init(&ctx_skein);
			sph_skein512(&ctx_skein, in, size);
			sph_skein512_close(&ctx_skein, output);
			break;
		case JH:
			sph_jh512_init(&ctx_jh);
			sph_jh512(&ctx_jh, in, size);
			sph_jh512_close(&ctx_jh, output);
			break;
		case KECCAK:
			sph_keccak512_init(&ctx_keccak);
			sph_keccak512(&ctx_keccak, in, size);
			sph_keccak512_close(&ctx_keccak, output);
			break;
		case LUFFA:
			sph_luffa512_init(&ctx_luffa);
			sph_luffa512(&ctx_luffa, in, size);
			sph_luffa512_close(&ctx_luffa, output);
			break;
		case CUBEHASH:
			sph_cubehash512_init(&ctx_cubehash);
			sph_cubehash512(&ctx_cubehash, in, size);
			sph_cubehash512_close(&ctx_cubehash, output);
			break;
		case SHAVITE:
			sph_shavite512_init(&ctx_shavite);
			sph_shavite512(&ctx_shavite, in, size);
			sph_shavite512_close(&ctx_shavite, output);
			break;
		case SIMD:
			sph_simd512_init(&ctx_simd);
			sph_simd512(&ctx_simd, in, size);
			sph_simd512_close(&ctx_simd, output);
			break;
		case ECHO:
			sph_echo512_init(&ctx_echo);
			sph_echo512(&ctx_echo, in, size);
			sph_echo512_close(&ctx_echo, output);
			break;
		case HAMSI:
			sph_hamsi512_init(&ctx_hamsi);
			sph_hamsi512(&ctx_hamsi, in, size);
			sph_hamsi512_close(&ctx_hamsi, output);
			break;
		case FUGUE:
			sph_fugue512_init(&ctx_fugue);
			sph_fugue512(&ctx_fugue, in, size);
			sph_fugue512_close(&ctx_fugue, output);
			break;
		case SHABAL:
			sph_shabal512_init(&ctx_shabal);
			sph_shabal512(&ctx_shabal, in, size);
			sph_shabal512_close(&ctx_shabal, output);
			break;
		case WHIRLPOOL:
			sph_whirlpool_init(&ctx_whirlpool);
			sph_whirlpool(&ctx_whirlpool, in, size);
			sph_whirlpool_close(&ctx_whirlpool, output);
			break;
		case SHA512:
			sph_sha512_init(&ctx_sha512);
			sph_sha512(&ctx_sha512,(const void*) in, size);
			sph_sha512_close(&ctx_sha512, (void*)output);
			break;
		}
		in = (void*) output;
		size = 64;
	}
//	memcpy(output, hash, 32);
}

void whirlpool_midstate(void *state, const void *input)
{
	sph_whirlpool_context ctx;

	sph_whirlpool_init(&ctx);
	sph_whirlpool(&ctx, input, 64);

	memcpy(state, ctx.state, 64);
}

static bool init[MAX_GPUS] = { 0 };

//#define _DEBUG
#define _DEBUG_PREFIX "x16r-"
#include "cuda_debug.cuh"

#if GPU_HASH_CHECK_LOG == 1
static int algo80_tests[HASH_FUNC_COUNT] = { 0 };
static int algo64_tests[HASH_FUNC_COUNT] = { 0 };
#endif
static int algo80_fails[HASH_FUNC_COUNT] = { 0 };
#define NO_ORDER_COUNTER 1

extern "C" int scanhash_x16r(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	const int dev_id = device_map[thr_id];
//	int intensity = (device_sm[dev_id] > 500 && !is_windows()) ? 20 : 19;
//	if (strstr(device_name[dev_id], "GTX 1080")) intensity = 20;
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << 21);

//	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);
	if (init[thr_id]){
		throughput = min(throughput, max_nonce - first_nonce);
		if (throughput == max_nonce - first_nonce)
			applog(LOG_BLUE, "Something did happen, be happy?!");
		if (work_restart[thr_id].restart) return -127;
	}

	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);
		if (throughput2intensity(throughput) > 21) gpulog(LOG_INFO, thr_id, "SIMD throws error on malloc call, TBD if there is a fix");

		quark_groestl512_cpu_init(thr_id, throughput);
//		quark_blake512_cpu_init(thr_id, throughput);
//		quark_bmw512_cpu_init(thr_id, throughput);
//		quark_skein512_cpu_init(thr_id, throughput);
//		quark_jh512_cpu_init(thr_id, throughput);		TEST
		quark_keccak512_cpu_init(thr_id, throughput);
		x11_shavite512_cpu_init(thr_id, throughput);
		if (x11_simd512_cpu_init(thr_id, throughput))
		{
			applog(LOG_WARNING, "SIMD was unable to initialize :( exiting...");
			exit(-1);
		}// 64
//		x16_echo512_cuda_init(thr_id, throughput);
		x13_hamsi512_cpu_init(thr_id, throughput);
		x13_fugue512_cpu_init(thr_id, throughput);
		x16_fugue512_cpu_init(thr_id, throughput);
		// x14_shabal512_cpu_init(thr_id, throughput);
		x15_whirlpool_cpu_init(thr_id, throughput, 0);
		x16_whirlpool512_init(thr_id, throughput);
		x17_sha512_cpu_init(thr_id, throughput);

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t) 64 * throughput), 0);

		cuda_check_cpu_init(thr_id, throughput);

		init[thr_id] = true;
	}

	if (opt_benchmark) {
		/*
		((uint32_t*)ptarget)[7] = 0x003f;
		((uint32_t*)pdata)[1] = 0x88888888;
		((uint32_t*)pdata)[2] = 0x88888888;
		//! Should cause vanila v0.1 code to have shavite CPU invalid hash error at various intervals
		*/

		//testing 0xb, 0xc
		//6FB7C831F4ED0A52
//		((uint32_t*)ptarget)[7] = 0x5ac6acf2;
		((uint32_t*)ptarget)[7] = 0x123f;
		*((uint64_t*)&pdata[1]) = 0x67452301EFCDAB89;//0x31C8B76F520AEDF4;
//		((uint32_t*)pdata)[1] = 0xEFCDAB89;
//		((uint32_t*)pdata)[2] = 0x67452301;
//		((uint32_t*)pdata)[17] = 0x12345678;
//		((uint32_t*)pdata)[1] = 0xd2d8ace2;
//		((uint32_t*)pdata)[2] = 0x56878fc3;
		//c38f8756e2acd8d2
//		((uint32_t*)pdata)[1] = 0xB361F3E4; //E4F361B3
//		((uint32_t*)pdata)[2] = 0x246D7B42; //427B6D24 // 0x4 80 round function
//		((uint32_t*)pdata)[17] = 0xcfac6c5c;//0xcfac6c5a
		/*
		((uint32_t*)pdata)[1] = 0xB361F3E4; //E4F361B3
		((uint32_t*)pdata)[2] = 0x246D7B42; //427B6D24
		((uint32_t*)pdata)[17] = 0xcfac6c5a;
		*/
//		((uint32_t*)pdata)[1] = 0xC0F3C1F9;
//		((uint32_t*)pdata)[2] = 0xF86E8007;
		//07806EF8F9C1F3C0
//		((uint32_t*)ptarget)[7] = 0x12345678;//0x723f;
//		((uint32_t*)pdata)[1] = 0xa8194c32;//0xa8194c3b;
//		((uint32_t*)pdata)[2] = 0x7f5d6e2b;//0x7f5d6e|a|2;

//		((uint8_t*)pdata)[8] = 0x80; // hashOrder[0] = '9'; for simd 80 + blake512 64
//		((uint8_t*)pdata)[8] = 0x90; // hashOrder[0] = '9'; for simd 80 + blake512 64
//		((uint8_t*)pdata)[8] = 0xA0; // hashOrder[0] = 'A'; for echo 80 + blake512 64
//		((uint8_t*)pdata)[8] = 0xB0; // hashOrder[0] = 'B'; for hamsi 80 + blake512 64
//		((uint8_t*)pdata)[8] = 0xC0; // hashOrder[0] = 'C'; for fugue 80 + blake512 64
//		((uint8_t*)pdata)[8] = 0xE0; // hashOrder[0] = 'E'; for whirlpool 80 + blake512 64
	}
	uint32_t _ALIGN(64) endiandata[20];

	for (int k=0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);

	uint32_t ntime = swab32(pdata[17]);
	if (s_ntime != ntime) {
		s_ntime = ntime;
		s_implemented = true;
		//	Display Hash Order as well as name of First Hash function in chain
	/*	if (!thr_id) applog(LOG_INFO, "hash order %s (%08x)", hashOrder, ntime);
		if (hashOrder[0] == '0' && !thr_id) applog(LOG_INFO, "First Algorithm is %s", algo_strings[0]);
		if (hashOrder[0] == '1' && !thr_id) applog(LOG_INFO, "First Algorithm is %s", algo_strings[1]);
		if (hashOrder[0] == '2' && !thr_id) applog(LOG_INFO, "First Algorithm is %s", algo_strings[2]);
		if (hashOrder[0] == '3' && !thr_id) applog(LOG_INFO, "First Algorithm is %s", algo_strings[3]);
		if (hashOrder[0] == '4' && !thr_id) applog(LOG_INFO, "First Algorithm is %s", algo_strings[4]);
		if (hashOrder[0] == '5' && !thr_id) applog(LOG_INFO, "First Algorithm is %s", algo_strings[5]);
		if (hashOrder[0] == '6' && !thr_id) applog(LOG_INFO, "First Algorithm is %s", algo_strings[6]);
		if (hashOrder[0] == '7' && !thr_id) applog(LOG_INFO, "First Algorithm is %s", algo_strings[7]);
		if (hashOrder[0] == '8' && !thr_id) applog(LOG_INFO, "First Algorithm is %s", algo_strings[8]);
		if (hashOrder[0] == '9' && !thr_id) applog(LOG_INFO, "First Algorithm is %s", algo_strings[9]);
		if (hashOrder[0] == 'A' && !thr_id) applog(LOG_INFO, "First Algorithm is %s", algo_strings[10]);
		if (hashOrder[0] == 'B' && !thr_id) applog(LOG_INFO, "First Algorithm is %s", algo_strings[11]);
		if (hashOrder[0] == 'C' && !thr_id) applog(LOG_INFO, "First Algorithm is %s", algo_strings[12]);
		if (hashOrder[0] == 'D' && !thr_id) applog(LOG_INFO, "First Algorithm is %s", algo_strings[13]);
		if (hashOrder[0] == 'E' && !thr_id) applog(LOG_INFO, "First Algorithm is %s", algo_strings[14]);
		if (hashOrder[0] == 'F' && !thr_id) applog(LOG_INFO, "First Algorithm is %s", algo_strings[15]); 	
		if (!thr_id) applog(LOG_INFO, "hash order %X%X (%08x)", endiandata[2], endiandata[1], ntime);  		*/
	}

	if (!s_implemented) {
		sleep(1);
		return -1;
	}

	cuda_check_cpu_setTarget(ptarget);

	uint8_t algo80 = (*(uint64_t*)&endiandata[1] >> 60) & 0x0f;
	switch (algo80) {
		case BLAKE:
			//! low impact, can do a lot to optimize quark_blake512
			quark_blake512_cpu_setBlock_80(thr_id, endiandata);
			break;
		case BMW:
			//! low impact, painfully optimize quark_bmw512
			quark_bmw512_cpu_setBlock_80(endiandata);
			break;
		case GROESTL:
			//! second most used algo historically
			groestl512_setBlock_80(thr_id, endiandata);
			break;
		case JH:
			//! average use, optimization tbd
			jh512_setBlock_80(thr_id, endiandata);
			break;
		case KECCAK:
			//! low impact
			keccak512_setBlock_80(thr_id, endiandata);
			break;
		case SKEIN:
			//! very low impact
			skein512_cpu_setBlock_80((void*)endiandata);
			break;
		case LUFFA:
			//! moderate impact (more than shavite)
			qubit_luffa512_cpu_setBlock_80_alexis((void*)endiandata);
			break;
		case CUBEHASH:
			//! moderate impact (more than shavite)
			cubehash512_setBlock_80(thr_id, endiandata);
			break;
		case SHAVITE:
			//! has been optimized fairly well
			x11_shavite512_setBlock_80((void*)endiandata);
			break;
		case SIMD:
			//! high impact optimization. -i > 21 causes error.
			x16_simd512_setBlock_80((void*)endiandata);
			break;
		case ECHO:
			//! high impact needs more optimizations
			x16_echo512_setBlock_80((void*)endiandata);
			break;
		case HAMSI:
			//! ***highest impact***
			x16_hamsi512_setBlock_80((void*)endiandata);
			break;
		case FUGUE:
			//! very high impact!
			x16_fugue512_setBlock_80((void*)pdata);
			break;
		case SHABAL:
			//! very low impact.
			x16_shabal512_setBlock_80((void*)endiandata);
			break;
		case WHIRLPOOL:
			//! moderate impact (more than shavite by a bit)
			x16_whirlpool512_setBlock_80((void*)endiandata);
			break;
		case SHA512:
			//! second lowest impact.
			x16_sha512_setBlock_80(endiandata);
			break;
	}

	int warn = 0;
	do {
		// Hash with CUDA
/*
		switch (algo80) {
			case BLAKE:
				quark_blake512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); // alexis
				TRACE("blake80:");
				break;
			case BMW:
				quark_bmw512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); // , 0); // alexis x
				TRACE("bmw80  :");
				break;
			case GROESTL:
				groestl512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); // alexis
				TRACE("grstl80:");
				break;
			case JH:
				jh512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); // alexis x
				TRACE("jh51280:");
				break;
			case KECCAK:
				keccak512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); // alexis x
				TRACE("kecck80:");
				break;
			case SKEIN:
				skein512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); // , 1);
				TRACE("skein80:");
				break;
			case LUFFA:
				qubit_luffa512_cpu_hash_80_alexis(thr_id, throughput, pdata[19], d_hash[thr_id]);
				TRACE("luffa80:");
				break;
			case CUBEHASH:
				cubehash512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); // alexis x
				TRACE("cube 80:");
				break;
			case SHAVITE:
				x11_shavite512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); // , 0);
				TRACE("shavite:");
				break;
			case SIMD:
				x16_simd512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); // alexis x
				TRACE("simd512:");
				break;
			case ECHO:
				x16_echo512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]);
				TRACE("echo   :");
				break;
			case HAMSI:
				x16_hamsi512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]);
				TRACE("hamsi  :");
				break;
			case FUGUE:
				x16_fugue512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); // alexis x
				TRACE("fugue  :");
				break;
			case SHABAL:
				x16_shabal512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); // alexis x
				TRACE("shabal :");
				break;
			case WHIRLPOOL:
				x16_whirlpool512_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); // alexis x
				TRACE("whirl  :");
				break;
			case SHA512:
				x16_sha512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); // alexis x
				TRACE("sha512 :");
				break;
		}
*/		
		if (work_restart[thr_id].restart) return -127;
		pAlgo80[(*(uint64_t*)&endiandata[1] >> 60 - (0 * 4)) & 0x0f](thr_id, throughput, pdata[19], d_hash[thr_id]);
		pAlgo64[(*(uint64_t*)&endiandata[1] >> 60 - (1 * 4)) & 0x0f](thr_id, throughput, d_hash[thr_id]);
		pAlgo64[(*(uint64_t*)&endiandata[1] >> 60 - (2 * 4)) & 0x0f](thr_id, throughput, d_hash[thr_id]);
		pAlgo64[(*(uint64_t*)&endiandata[1] >> 60 - (3 * 4)) & 0x0f](thr_id, throughput, d_hash[thr_id]);
		pAlgo64[(*(uint64_t*)&endiandata[1] >> 60 - (4 * 4)) & 0x0f](thr_id, throughput, d_hash[thr_id]);
		pAlgo64[(*(uint64_t*)&endiandata[1] >> 60 - (5 * 4)) & 0x0f](thr_id, throughput, d_hash[thr_id]);
		pAlgo64[(*(uint64_t*)&endiandata[1] >> 60 - (6 * 4)) & 0x0f](thr_id, throughput, d_hash[thr_id]);
		pAlgo64[(*(uint64_t*)&endiandata[1] >> 60 - (7 * 4)) & 0x0f](thr_id, throughput, d_hash[thr_id]);
		pAlgo64[(*(uint64_t*)&endiandata[1] >> 60 - (8 * 4)) & 0x0f](thr_id, throughput, d_hash[thr_id]);
		pAlgo64[(*(uint64_t*)&endiandata[1] >> 60 - (9 * 4)) & 0x0f](thr_id, throughput, d_hash[thr_id]);
		pAlgo64[(*(uint64_t*)&endiandata[1] >> 60 - (10 * 4)) & 0x0f](thr_id, throughput, d_hash[thr_id]);
		pAlgo64[(*(uint64_t*)&endiandata[1] >> 60 - (11 * 4)) & 0x0f](thr_id, throughput, d_hash[thr_id]);
		pAlgo64[(*(uint64_t*)&endiandata[1] >> 60 - (12 * 4)) & 0x0f](thr_id, throughput, d_hash[thr_id]);
		pAlgo64[(*(uint64_t*)&endiandata[1] >> 60 - (13 * 4)) & 0x0f](thr_id, throughput, d_hash[thr_id]);
		pAlgo64[(*(uint64_t*)&endiandata[1] >> 60 - (14 * 4)) & 0x0f](thr_id, throughput, d_hash[thr_id]);
		pAlgo64[(*(uint64_t*)&endiandata[1] >> 60 - (15 * 4)) & 0x0f](thr_id, throughput, d_hash[thr_id]);
		if (work_restart[thr_id].restart) return -127;

//		run_x16r_rounds(&endiandata[1], thr_id, throughput, pdata[19], d_hash[thr_id]);

		*hashes_done = pdata[19] - first_nonce + throughput;

		work->nonces[0] = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
#ifdef _DEBUG
		uint32_t _ALIGN(64) dhash[8];
		be32enc(&endiandata[19], pdata[19]);
		x16r_hash(dhash, endiandata);
		applog_hash(dhash);
		return -1;
#endif
		if (work->nonces[0] != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];
			be32enc(&endiandata[19], work->nonces[0]);
			x16r_hash(vhash, endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work->nonces[1] = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
				work_set_target_ratio(work, vhash);
				if (work->nonces[1] != 0) {
					be32enc(&endiandata[19], work->nonces[1]);
					x16r_hash(vhash, endiandata);
					bn_set_target_ratio(work, vhash, 1);
					work->valid_nonces++;
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				} else {
					pdata[19] = work->nonces[0] + 1; // cursor
				}
#if GPU_HASH_CHECK_LOG == 1
				gpulog(LOG_INFO, thr_id, "hash found with %s 80!", algo_strings[algo80]);

				algo80_tests[algo80] += work->valid_nonces;
				char oks64[128] = { 0 };
				char oks80[128] = { 0 };
				char fails[128] = { 0 };
				for (int a = 0; a < HASH_FUNC_COUNT; a++) {
					const char elem = hashOrder[a];
					const uint8_t algo64 = elem >= 'A' ? elem - 'A' + 10 : elem - '0';
					if (a > 0) algo64_tests[algo64] += work->valid_nonces;
					sprintf(&oks64[strlen(oks64)], "|%X:%2d", a, algo64_tests[a] < 100 ? algo64_tests[a] : 99);
					sprintf(&oks80[strlen(oks80)], "|%X:%2d", a, algo80_tests[a] < 100 ? algo80_tests[a] : 99);
					sprintf(&fails[strlen(fails)], "|%X:%2d", a, algo80_fails[a] < 100 ? algo80_fails[a] : 99);
				}
				applog(LOG_INFO, "K64: %s", oks64);
				applog(LOG_INFO, "K80: %s", oks80);
				applog(LOG_ERR,  "F80: %s", fails);
#endif
				if (work_restart[thr_id].restart) return -127;
				return work->valid_nonces;
			}
			else if (vhash[7] > Htarg) {
				// x11+ coins could do some random error, but not on retry
				gpu_increment_reject(thr_id);
				algo80_fails[algo80]++;
				if (!warn) {
					warn++;
					pdata[19] = work->nonces[0] + 1;
					continue;
				} else {
					if (!opt_quiet)	gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU! %s %X%X",
						work->nonces[0], algo_strings[algo80], endiandata[2], endiandata[1]);
//					work->nonces[0], algo_strings[algo80], hashOrder);
					warn = 0;
//					work->data[19] = max_nonce;
					if (work_restart[thr_id].restart) return -127;
					return -128;
				}
			}
		}

		if ((uint64_t)throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}

		pdata[19] += throughput;

	} while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;
	if (work_restart[thr_id].restart) return -127;
	return 0;
}

// cleanup
extern "C" void free_x16r(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	cudaFree(d_hash[thr_id]);

	quark_blake512_cpu_free(thr_id);
	quark_groestl512_cpu_free(thr_id);
	x11_simd512_cpu_free(thr_id);
	x13_fugue512_cpu_free(thr_id);
	x16_fugue512_cpu_free(thr_id); // to merge with x13_fugue512 ?
	x15_whirlpool_cpu_free(thr_id);

	cuda_check_cpu_free(thr_id);

	cudaDeviceSynchronize();
	init[thr_id] = false;
}
