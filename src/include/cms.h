#ifndef CMS_ML_CMS_H_
#define CMS_ML_CMS_H_

#include "MurmurHash.h"
#include "util.h"

#include <random>
#include <climits>
#include <cstring>
#include <array>
#include <iostream>
#include <fstream>
#include <stdlib.h>

#include <immintrin.h>

// Struct for caching hash indices and signs for a feature
template<size_t N>
struct hc
{
		std::array<int, N> hash;
		std::array<float, N> sign;
};

template<size_t N>
class CMS
{
		private:
				const size_t AVX = 8;
				const size_t K;
				const size_t D;
				const size_t DIV;
				const size_t MOD;
				const size_t CNT;
				const size_t NK;
				const size_t SIZE;

				float * data;
				uint32_t * seeds;
				__m256i mask;

		public:
				/*
				   Initialize Memory and Random Seeds for Count-Sketch

				   @param _K - Number of classes represented by Count-Sketch
				   @param _D - Number of weights allocated for each class
				 */
				CMS(size_t _K, size_t _D) :
						K(_K),
						D(_D),
						DIV(K/AVX),
						MOD(K%AVX),
						CNT((MOD == 0) ? DIV : DIV+1),
						NK(CNT*AVX),
						SIZE(NK*N*D)
						{
								data = (float*) aligned_alloc(32, sizeof(float)*SIZE);
								seeds = new uint32_t[N];

								// Dynamic Mask
								mask = _mm256_set1_epi32(0);
								for(size_t idx = 0; idx < MOD; ++idx)
								{
										mask[idx] = -1;
								}

								// Clear Sketch
								memset(data, 0, SIZE);

								// Initialize seeds for universal hashing
								std::default_random_engine generator;
								std::uniform_int_distribution<uint32_t> seed_gen(0, UINT_MAX);
								for(size_t idx = 0; idx < N; ++idx)
								{
										seeds[idx] = seed_gen(generator);
								}
						}

				~CMS()
				{
						delete [] data;
						delete [] seeds;
				}

				/*
				   Erase all values in the Count-Sketch
				 */
				void clear()
				{
						memset(data, 0, sizeof(float) * SIZE);
				}

				/*
				   Initialize Count-Sketch from a file
				   @param filename - Count-Sketch Weight File
				   @return true if successfully loaded weights from file
				 */
				bool initialize(const char* filename)
				{
						std::string line;
						std::ifstream myfile;
						myfile.open (filename);
						if(!myfile.is_open())
						{
								return false;
						}
						assert(myfile.is_open());

						// Read Random Seeds
						getline (myfile, line);
						const size_t file_seeds = std::atol(line.c_str());
						assert(file_seeds == N);

						for(size_t idx = 0; idx < N; ++idx)
						{
								getline (myfile, line);
								seeds[idx] = std::atol(line.c_str());
						}

						// Read Sketch Values
						getline (myfile, line);
						const size_t file_sketch = std::atol(line.c_str());
						assert(file_sketch == SIZE);

						for(size_t idx = 0; idx < SIZE; ++idx)
						{
								getline (myfile, line);
								data[idx] = std::atof(line.c_str());
						}
						myfile.close();
						return true;
				}

				/*
				   Save Count-Sketch weights to a file
				   @param filename - Count-Sketch Weight File
				 */
				void save(const char* filename) const
				{
						std::ofstream myfile;
						myfile.open (filename);
						assert(myfile.is_open());

						// Write Random Seeds
						myfile << N << std::endl;
						for(size_t idx = 0; idx < N; ++idx)
						{
								myfile << seeds[idx] << std::endl;
						}

						// Write Sketch Values
						myfile << SIZE << std::endl;
						for(size_t idx = 0; idx < SIZE; ++idx)
						{
								myfile << data[idx] << std::endl;
						}
						myfile.close();
				}

				/*
				   Update feature in the Count-Sketch
				   @param cache - Cached indices and signs for the feature
				   @param value - Update value for the feature
				   @return the new value for the feature
				 */
				float update(const hc<N>& cache, const float value)
				{
						std::vector<float> values(N, 0);
						for(size_t idx = 0; idx < N; ++idx)
						{
								int index = cache.hash[idx];
								int sign = cache.sign[idx];
								data[index] += sign * value;
								values[idx] = sign * data[index];
						}
						return median(values);
				}

				/*
				   Get feature in the Count-Sketch
				   @param cache - Cached indices and signs for the feature
				   @return the value for the feature
				 */
				float retrieve(const hc<N>& cache) const
				{
						// For each hash function
						std::vector<float> values(N, 0);
						for(size_t idx = 0; idx < N; ++idx)
						{
								values[idx] = cache.sign[idx] * data[cache.hash[idx]];
						}
						return median(values);
				}

				/*
				   Update feature in the Count-Sketch
				   @param key - pointer to feature representation
				   @param len - length of the feature representation
				   @param value - Update value for the feature
				   @return the new value for the feature
				 */
				float update(const void* key, const int len, float value)
				{
						std::vector<float> values(N, 0);
						for(size_t idx = 0; idx < N; ++idx)
						{
								const uint32_t hash = MurmurHash3_x86_32 (key, len, seeds[idx]) % D;
								const uint32_t index = idx * D + hash;

								const bool sign_bit = MurmurHash3_x86_32 (key, len, seeds[idx]+3) & 0x1;
								const float sign = (sign_bit) ? 1.0 : -1.0;
								data[index] += sign * value;
								values[idx] = sign * data[index];
						}
						return median(values);
				}

				/*
				   Get feature in the Count-Sketch
				   @param key - pointer to feature representation
				   @param len - length of the feature representation
				   @return the value for the feature
				 */
				float retrieve(const void* key, const int len) const
				{
						std::vector<float> values(N, 0);
						for(size_t idx = 0; idx < N; ++idx)
						{
								const uint32_t hash = MurmurHash3_x86_32 (key, len, seeds[idx]) % D;
								const uint32_t index = idx * D + hash;

								const bool sign_bit = MurmurHash3_x86_32 (key, len, seeds[idx]+3) & 0x1;
								const float sign = (sign_bit) ? 1.0 : -1.0;
								values[idx] = sign * data[index];
						}
						return median(values);
				}

				/*
				   Update the same feature for multiple classes using AVX instructions - For Softmax Regression
				   @param cache - Cached indices and signs for the feature
				   @param cdx - Class offset
				   @param value - Update value for the features
				 */
				void cms_update(const hc<N>& cache, const size_t cdx, __m256 value)
				{
						for(size_t idx = 0; idx < N; ++idx)
						{
								const size_t index = cache.hash[idx] * NK + cdx * AVX;
								__m256 sign = _mm256_set1_ps(cache.sign[idx]);

								if(cdx < DIV)
								{
										__m256 current = _mm256_load_ps( &data[index] );
										__m256 result = _mm256_add_ps(current, _mm256_mul_ps(sign, value));
										_mm256_store_ps(&data[index], result);
								}
								else
								{
										__m256 current = _mm256_maskload_ps( &data[index], mask );
										__m256 result = _mm256_add_ps(current, _mm256_mul_ps(sign, value));
										_mm256_maskstore_ps(&data[index], mask, result);
								}
						}
				}

				/*
				   Get the same feature for multiple classes using AVX instructions - For Softmax Regression
				   @param cache - Cached indices and signs for the feature
				   @param cdx - Class offset
				 */
				__m256 cms_retrieve(const hc<N>& cache, const size_t cdx) const
				{
						__m256 values[N];
						for(size_t idx = 0; idx < N; ++idx)
						{
								const size_t index = cache.hash[idx] * NK + cdx * AVX;
								__m256 sign = _mm256_set1_ps(cache.sign[idx]);

								if(cdx < DIV)
								{
										__m256 w = _mm256_load_ps( &data[index] );
										values[idx] = _mm256_mul_ps(sign, w);
								}
								else
								{
										__m256 w = _mm256_maskload_ps( &data[index], mask );
										values[idx] = _mm256_mul_ps(sign, w);
								}
						}
						return median(values[0], values[1], values[2]);
				}

				/*
				   Get the feature for a specific class - For Softmax Regression
				   @param cache - Cached indices and signs for the features
				   @param class_idx - Class Index
				 */
				float cms_retrieve_single(const hc<N>& cache, const size_t class_idx) const
				{
						std::vector<float> values(N, 0);
						for(size_t idx = 0; idx < N; ++idx)
						{
								const size_t index = cache.hash[idx] * NK + class_idx;
								values[idx] = cache.sign[idx] * data[index];
						}
						return median(values);
				}

				/*
				   Precompute the Hash Index and Sign
				   @param key - pointer to feature representation
				   @param len - length of the feature representation
				   @param result - struct to store the cached values
				 */
				void hash(const void* key, const int len, hc<N>& result)
				{
						for(size_t idx = 0; idx < N; ++idx)
						{
								const uint32_t hash = MurmurHash3_x86_32 (key, len, seeds[idx]) % D;
								const bool sign_bit = MurmurHash3_x86_32 (key, len, seeds[idx]+3) & 0x1;
								result.hash[idx] = idx * D + hash;
								result.sign[idx] = (sign_bit) ? 1.0 : -1.0;
						}
				}
};

#endif // CMS_ML_CMS_H_
