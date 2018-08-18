#include "MurmurHash.h"
#include "fast_parser.h"
#include "mp_queue.h"
#include "cms.h"
#include "topk.h"

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

// Size of Top-K Heap
const size_t TOPK = (1 << 20) - 1;

// Number of Classes
const size_t K = 193;

// Size of Count-Sketch Array
const size_t D = (1 << 24) - 1;

// Number of Arrays in Count-Sketch
const size_t N = 3;

// Learning Rate
const float LR = 1e-2;

// Length of String Feature Representation
const size_t LEN = 12;

/***** End of Hyper-Parameters *****/

const size_t AVX = 8;
const size_t DIV = K / AVX;
const size_t MOD = K % AVX;
const size_t CNT = (MOD == 0) ? DIV : DIV+1;

typedef std::pair<int, float> fp_t;
typedef std::vector<data_t> x_t;
typedef std::vector<TopK<data_t, TOPK>> tk_t;

// Serialize Output
std::mutex mtx;

// Number of threads for parallel data preprocessing
const size_t THREADS = 16;
// Maximum number of features for an example
const size_t MAX_FEATURES = 378;
std::array<std::array<hc<N>, MAX_FEATURES>, THREADS> caches;

void producer(fast_parser& p, mp_queue<x_t>& q)
{
		for(std::vector<data_t> x = p.read(' '); p; x = p.read(' '))
		{
				q.enqueue(x);
		}
		//std::cout << "Finished Reading" << std::endl;
}

float process(CMS<N>& sketch, tk_t& topk, const x_t& x, bool train)
{
		const size_t label = atoi(x[0].data()) - 1;
		assert(label >= 0 && label < K);

		const int tid = omp_get_thread_num();
		std::array<hc<N>, MAX_FEATURES>& cache = caches[tid];
		#pragma omp parallel for num_threads(10)
		for(size_t idx = 2; idx < x.size(); ++idx)
		{
				const void * key_ptr = (const void *) x[idx].data();
				sketch.hash(key_ptr, LEN, cache[idx-2]);
		}

		__m256 logits[CNT];
		for(size_t cdx = 0; cdx < CNT; ++cdx)
		{
				logits[cdx] = _mm256_set1_ps(0);
		}

		if(train)
		{
				// During training - Retrieve minimum value for each class using Top-K Heap
				// Threshold any weight less than the minimum value
				// Optimization to use AVX Instruction
				__m256 neg_mask[CNT];
				__m256 pos_mask[CNT];
				for(size_t class_idx = 0; class_idx < K; ++class_idx)
				{
						const size_t cdx = class_idx / AVX;
						const size_t pos = class_idx % AVX;
						float min = topk[class_idx].minimum();
						neg_mask[cdx][pos] = -min;
						pos_mask[cdx][pos] = min;
				}

				for(auto& item : cache)
				{
						for(size_t cdx = 0; cdx < CNT; ++cdx)
						{
								__m256 weight = sketch.cms_retrieve(item, cdx);
								__m256 mask = _mm256_or_ps(_mm256_cmp_ps(weight, pos_mask[cdx], _CMP_GE_OS), _mm256_cmp_ps(weight, neg_mask[cdx], _CMP_LE_OS));
								__m256 iht_weight = _mm256_and_ps(weight, mask);
								logits[cdx] = _mm256_add_ps(logits[cdx], iht_weight);
						}
				}
		}
		else
		{
				// During Testing - Only retrieve values present in the Top-K heap for each class
				#pragma omp parallel for num_threads(10)
				for(size_t class_idx = 0; class_idx < K; ++class_idx)
				{
						for(size_t idx = 2; idx < x.size(); ++idx)
						{
								const data_t& key = x[idx];
								update(logits, class_idx, topk[class_idx][key]);
						}
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
		#pragma omp parallel for num_threads(10)
		for(size_t idx = 2; idx < x.size(); ++idx)
		{
				auto& item = cache[idx-2];
				for(size_t cdx = 0; cdx < CNT; ++cdx)
				{
						__m256 update = _mm256_mul_ps(LR_AVX, logits[cdx]);
						sketch.cms_update(item, cdx, update);
				}
		}

		// Update TopK Heap
		#pragma omp parallel for num_threads(10)
		for(size_t class_idx = 0; class_idx < K; ++class_idx)
		{
				for(size_t idx = 0; idx < MAX_FEATURES; ++idx)
				{
						const data_t& str = x[idx+2];
						float value = sketch.cms_retrieve_single(cache[idx], class_idx);
						topk[class_idx].push(str, my_abs(value));
				}
		}
		return loss;
}

void consumer(CMS<N>& sketch, tk_t& topk, fast_parser& p, mp_queue<x_t>& q, bool train)
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
				for(size_t cdx = 0; cdx < items.size(); ++cdx)
				{
						loss += process(sketch, topk, items[cdx], train);
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
		CMS<N> sketch(K, D);
		mp_queue<x_t> q(10000);
		tk_t topk(K);

		fast_parser train_p(argv[1]);
		std::thread train_pr([&] { producer(train_p, q); });
		std::thread train_cr([&] { consumer(sketch, topk, train_p, q, true); });
		train_pr.join();
		train_cr.join();

		fast_parser test_p(argv[2]);
		std::thread test_pr([&] { producer(test_p, q); });
		std::thread test_cr([&] { consumer(sketch, topk, test_p, q, false); });
		test_pr.join();
		test_cr.join();

		return 0;
}
