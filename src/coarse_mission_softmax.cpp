#include "MurmurHash.h"
#include "fast_parser.h"
#include "mp_queue.h"
#include "cms.h"
#include "topk.h"
#include "util.h"

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
const size_t TOPK = (1 << 22) - 1;

// Number of Classes
const size_t K = 193;

// Size of Count-Sketch Array
const size_t D = (1 << 24) - 1;

// Number of Arrays in Count-Sketch
const size_t N = 3;

// Learning Rate
const float LR = 1e-1;

// Length of String Feature Representation
const size_t LEN = 12;

/***** End of Hyper-Parameters *****/

const size_t AVX = 8;
const size_t DIV = K / AVX;
const size_t MOD = K % AVX;
const size_t CNT = (MOD == 0) ? DIV : DIV+1;

// Number of threads for parallel data preprocessing
const size_t THREADS = 6;

// Maximum number of features for an example
const size_t MAX_FEATURES = 378;

typedef std::pair<int, float> fp_t;
typedef std::vector<data_t> x_t;
typedef std::vector<TopK<data_t, TOPK>> tk_t;

// Serialize Output
std::mutex mtx;
std::array<std::array<hc<N>, MAX_FEATURES>, THREADS> caches;
std::array<std::array<bool, MAX_FEATURES>, THREADS> active_sets;

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
		const int tid = omp_get_thread_num();

		const size_t label = atoi(x[0].data()) - 1;
		assert(label >= 0 && label < K);

		// TopK Heap
		auto& tk = topk[tid];

		// Cache Feature Hashing Indices
		std::array<hc<N>, MAX_FEATURES>& cache = caches[tid];
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
				for(size_t idx = 2; idx < x.size(); ++idx)
				{
						const data_t& key = x[idx];
						if(tk.find(key))
						{
								auto& item = cache[idx-2];
								for(size_t cdx = 0; cdx < CNT; ++cdx)
								{
										__m256 weight = sketch.cms_retrieve(item, cdx);
										logits[cdx] = _mm256_add_ps(logits[cdx], weight);
								}
						}
				}
		}
		else
		{
				// Active Set Boolean Array
				std::array<bool, MAX_FEATURES>& AS = active_sets[tid];
				AS.fill(false);

				// Visit each feature only once
				// Accumulate features across each independent top-k heap
				for(auto& tk : topk)
				{
						for(size_t idx = 2; idx < x.size(); ++idx)
						{
								const data_t& key = x[idx];
								AS[idx-2] = tk.find(key);	
						}
				}

				for(size_t idx = 0; idx < AS.size(); ++idx)
				{
						if(AS[idx])
						{
								auto& item = cache[idx+2];
								for(size_t cdx = 0; cdx < CNT; ++cdx)
								{
										__m256 weight = sketch.cms_retrieve(item, cdx);
										logits[cdx] = _mm256_add_ps(logits[cdx], weight);
								}
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
		for(size_t idx = 2; idx < x.size(); ++idx)
		{
				auto& item = cache[idx-2];
				for(size_t cdx = 0; cdx < CNT; ++cdx)
				{
						__m256 update = _mm256_mul_ps(LR_AVX, logits[cdx]);
						sketch.cms_update(item, cdx, update);
				}
		}

		// Update TopK Heap - L1 Norm for each class feature vector
		for(size_t idx = 2; idx < x.size(); ++idx)
		{
				auto& item = cache[idx-2];

				__m256 l1_norm = _mm256_set1_ps(0);
				for(size_t cdx = 0; cdx < CNT; ++cdx)
				{
						__m256 weight = sketch.cms_retrieve(item, cdx);
						l1_norm = _mm256_add_ps(l1_norm, my_abs(weight));
				}

				float value = 0.0;
				for(size_t pos = 0; pos < AVX; ++pos)
				{
						value += l1_norm[pos];
				}

				const data_t& key = x[idx];
				tk.push(key, value);
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
                #pragma omp parallel for num_threads(THREADS)
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
		tk_t topk(THREADS);

		for(int iter = 1; iter < argc-1; ++iter)
		{
				std::cout << "Epoch:\t" << iter << std::endl;

				fast_parser train_p(argv[iter]);
				std::thread train_pr([&] { producer(train_p, q); });
				std::thread train_cr([&] { consumer(sketch, topk, train_p, q, true); });
				train_pr.join();
				train_cr.join();

				std::cout << "Validation:\t" << iter << std::endl;
				std::ofstream out("r" + std::to_string(iter) + ".pred");
				std::streambuf* coutbuf = std::cout.rdbuf(); //save old buf
				std::cout.rdbuf(out.rdbuf()); //redirect std::cout to out.txt!

				fast_parser test_p(argv[argc-1]);
				std::thread test_pr([&] { producer(test_p, q); });
				std::thread test_cr([&] { consumer(sketch, topk, test_p, q, false); });
				test_pr.join();
				test_cr.join();

				std::cout.rdbuf(coutbuf); //redirect std::cout to original
		}

		return 0;
}
