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
#include <omp.h>

/***** Hyper-Parameters *****/

// Size of Top-K Heap
const size_t TOPK = (1 << 14) - 1;

// Number of Classes - Always 1 for Logistic Regression
const size_t K = 1;

// Size of Count-Sketch Array
const size_t D = (1 << 18) - 1;

// Number of Arrays in Count-Sketch
const size_t N = 3;

// Learning Rate
const float LR = 5e-1;

/***** End of Hyper-Parameters *****/

typedef std::pair<int, float> fp_t;
typedef std::pair<int, std::vector<fp_t> > x_t;
typedef TopK<int, TOPK> tk_t;

// Serialize Output
std::mutex mtx;

// Number of threads for parallel data preprocessing
const size_t THREADS = 2;
// Maximum number of features for an example
const size_t MAX_FEATURES = 5000;

std::array<std::array<hc<N>, MAX_FEATURES>, THREADS> caches;

void split(data_t& item, fp_t& result)
{
		data_t key;
		data_t value;
		memset(key.data(), 0, key.size());
		memset(value.data(), 0, value.size());

		int cdx = 0;
		for(char v = item[cdx]; v != ':'; v = item[cdx])
		{
				key[cdx++] = v;
		}
		result.first = atoi(key.data());

		int initial = ++cdx;
		for(char v = item[cdx]; v != 0; v = item[cdx])
		{
				value[(cdx++ - initial)] = v;
		}
		result.second = atof(value.data());
}

void producer(fast_parser& p, mp_queue<x_t>& q)
{
		for(std::vector<data_t> x = p.read(' '); p; x = p.read(' '))
		{
				const int label = atoi(x[0].data());

				// Parse Features
				std::vector<fp_t> features(x.size()-1);
				for(size_t idx = 1; idx < x.size(); ++idx)
				{
						split(x[idx], features[idx-1]);
				}

				q.enqueue(std::make_pair(label, features));
		}
		//std::cout << "Finished Reading" << std::endl;
}

float process(CMS<N>& sketch, tk_t& topk, const x_t& x, bool train)
{
		float label = (x.first + 1.0f) / 2.0f;
		const std::vector<fp_t>& features = x.second;

		const int tid = omp_get_thread_num();
		std::array<hc<N>, MAX_FEATURES>& cache = caches[tid];
		for(size_t idx = 0; idx < features.size(); ++idx)
		{
				const void * key_ptr = (const void *) &features[idx].first;
				sketch.hash(key_ptr, sizeof(int), cache[idx]);
		}

		float logit = 0;
		for(const fp_t& item : features)
		{
				logit += topk[item.first] * item.second;
		}

		float sigmoid = 1.0 / (1.0 + std::exp(-logit));
		float loss = (label * std::log(sigmoid) + (1.0 - label) * std::log(1 - sigmoid));
		if(!train)
		{
				mtx.lock();
				std::cout << label << " " << sigmoid << std::endl;
				mtx.unlock();
				return loss;
		}

		float gradient = label - sigmoid;
		for(size_t idx = 0; idx < features.size(); ++idx)
		{
				float value = sketch.update(cache[idx], LR * gradient * features[idx].second);
				topk.push(features[idx].first, value);
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
						std::cout << cnt << " " << avg_loss << std::endl;
				}
				items.clear();
		}
		//std::cout << "Finished Consumer" << std::endl;
}

int main(int argc, char* argv[])
{
		CMS<N> sketch(K, D);
		mp_queue<x_t> q(10000);
		tk_t topk;

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
