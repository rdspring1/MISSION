#include "MurmurHash.h"
#include "fast_parser.h"
#include "mp_queue.h"
#include "mem.h"

#include <stdlib.h>
#include <vector>
#include <utility>
#include <iostream>
#include <climits>
#include <random>
#include <chrono>

#include <thread>
#include <mutex>

#include <immintrin.h>
#include <omp.h>

/***** Hyper-Parameters *****/

// Number of Classes
const size_t K = 193;

// Size of Count-Sketch Array
const size_t D = (1 << 24) - 1;

// Number of Arrays in Count-Sketch - Always 1 for Feature Hashing
const size_t N = 1;

// Learning Rate
const float LR = 1e-2;

// Length of String Feature Representation
const size_t LEN = 12;

/***** End of Hyper-Parameters *****/

// AVX Constants
const size_t AVX = 8;
const size_t DIV = K / AVX;
const size_t MOD = K % AVX;
const size_t CNT = (MOD == 0) ? DIV : DIV+1;

typedef std::pair<int, float> fp_t;
typedef std::vector<data_t> x_t;

// Serialize Output
std::mutex mtx;

// Number of threads for parallel data preprocessing
const size_t THREADS = 16;
// Maximum number of features for an example
const size_t MAX_FEATURES = 378;
std::array<std::array<unsigned, MAX_FEATURES>, THREADS> caches;

void producer(fast_parser& p, mp_queue<x_t>& q)
{
		for(std::vector<data_t> x = p.read(' '); p; x = p.read(' '))
		{
				q.enqueue(x);
		}
		//std::cout << "Finished Reading" << std::endl;
}

float process(MEM& sketch, const x_t& x, bool train)
{
		const size_t label = atoi(x[0].data()) - 1;
		assert(label >= 0 && label < K);

		const int tid = omp_get_thread_num();
		std::array<unsigned, MAX_FEATURES>& cache = caches[tid];
		for(size_t idx = 2; idx < x.size(); ++idx)
		{
				const void * key_ptr = (const void *) x[idx].data();
				cache[idx-2] = sketch.hash(key_ptr, LEN);
		}
		assert(x.size() == MAX_FEATURES+2);

		__m256 logits[CNT];
		for(size_t cdx = 0; cdx < CNT; ++cdx)
		{
				logits[cdx] = _mm256_set1_ps(0);
		}

		for(auto& item : cache)
		{
				for(size_t cdx = 0; cdx < CNT; ++cdx)
				{
						__m256 weight = sketch.simd_retrieve(item, cdx);
						logits[cdx] = _mm256_add_ps(logits[cdx], weight);
				}
		}

		float max_value = 0;
		uint32_t argmax = 0;
		maximum(logits, K, max_value, argmax);
		partition(logits, CNT, K, max_value);
		float loss = std::log(get(logits, label) + 1e-10);
		update(logits, label, -1.0);

		if(!train)
		{
				mtx.lock();
				std::cout << label << " " << argmax << std::endl;
				mtx.unlock();
				return loss;
		}

		// Apply Gradient Update
		__m256 LR_AVX = _mm256_set1_ps(-LR);
		for(auto& item : cache)
		{
				for(size_t cdx = 0; cdx < CNT; ++cdx)
				{
						__m256 update = _mm256_mul_ps(LR_AVX, logits[cdx]);
						sketch.simd_update(item, cdx, update);
				}
		}
		return loss;
}

void consumer(MEM& sketch, fast_parser& p, mp_queue<x_t>& q, bool train)
{
		std::vector<x_t> items;
		size_t cnt = 0;
		while(p || q)
		{
				if(!q.full() && p)
				{
						std::this_thread::sleep_for (std::chrono::seconds(1));
						continue;
				}

				// Retrieve items from multiprocess queue
				q.retrieve(items);
				cnt += items.size();

				float loss = 0.0;
				#pragma omp parallel for
				for(size_t cdx = 0; cdx < items.size(); ++cdx)
				{
						loss += process(sketch, items[cdx], train);
				}

				// Debug
				if(train)
				{
						float avg_loss = -loss / items.size();
						std::cout << cnt << "\t" << avg_loss << std::endl;
				}
				items.clear();
		}
		//std::cout << "Finished Consumer" << std::endl;
}

int main(int argc, char* argv[])
{
		MEM sketch(K, D);
		mp_queue<x_t> q(10000);

		for(int iter = 1; iter < argc-1; ++iter)
		{
				std::cout << "Epoch:\t" << iter << std::endl;

				fast_parser train_p(argv[iter]);
				std::thread train_pr([&] { producer(train_p, q); });
				std::thread train_cr([&] { consumer(sketch, train_p, q, true); });
				train_pr.join();
				train_cr.join();

				std::cout << "Validation:\t" << iter << std::endl;
				std::ofstream out("r" + std::to_string(iter) + ".pred");
				std::streambuf* coutbuf = std::cout.rdbuf(); //save old buf
				std::cout.rdbuf(out.rdbuf()); //redirect std::cout to out.txt!

				fast_parser test_p(argv[argc-1]);
				std::thread test_pr([&] { producer(test_p, q); });
				std::thread test_cr([&] { consumer(sketch, test_p, q, false); });
				test_pr.join();
				test_cr.join();

				std::cout.rdbuf(coutbuf); //redirect std::cout to original
		}

		return 0;
}
