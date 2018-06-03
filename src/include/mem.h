#ifndef CMS_ML_MEM_H_
#define CMS_ML_MEM_H_

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

class MEM
{
		private:
				std::default_random_engine generator;
				const unsigned AVX = 8;

				const unsigned K;
				const unsigned D;
				const unsigned DIV;
				const unsigned MOD;
				const unsigned CNT;
				const unsigned NK;
				const unsigned SIZE;
				std::uniform_int_distribution<unsigned> seed_gen;
				unsigned seed;

				float * data;
				__m256i mask;

		public:
				/*
				   Initialize Memory and Random Seeds

				   @param _K - Number of classes
				   @param _D - Number of weights allocated for each class
				 */
				MEM(unsigned _K, unsigned _D) :
						K(_K),
						D(_D),
						DIV(K/AVX),
						MOD(K%AVX),
						CNT((MOD == 0) ? DIV : DIV+1),
						NK(CNT*AVX),
						SIZE(NK*D),
						seed_gen(0, UINT_MAX),
						seed(seed_gen(generator))
						{
								data = (float*) aligned_alloc(32, sizeof(float)*SIZE);

								// Dynamic Mask
								mask = _mm256_set1_epi32(0);
								for(unsigned idx = 0; idx < MOD; ++idx)
								{
										mask[idx] = -1;
								}

								// Clear Sketch
								memset(data, 0, SIZE);
						}

				~MEM()
				{
						delete [] data;
				}

				/*
				   Erase all values
				 */
				void clear()
				{
						memset(data, 0, sizeof(float)*SIZE);
				}

				/*
				   Initialize memory from a file
				   @param filename - Weight File
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
						seed = std::atol(line.c_str());

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
				   Save weights to a file
				   @param filename - Weight File
				 */
				void save(const char* filename) const
				{
						std::ofstream myfile;
						myfile.open (filename);
						assert(myfile.is_open());

						myfile << seed << std::endl;
						myfile << SIZE << std::endl;
						for(size_t idx = 0; idx < SIZE; ++idx)
						{
								myfile << data[idx] << std::endl;
						}
						myfile.close();
				}

				/*
				   Update feature
				   @param hash - Hash index for the feature
				   @param value - Update value for the feature
				 */
				void update(const unsigned hash, const float value)
				{
						data[hash] += value;
				}

				/*
				   Get feature
				   @param hash - Hash index for the feature
				 */
				float retrieve(const unsigned hash) const
				{
						return data[hash];
				}

				/*
				   Update feature
				   @param key - pointer to feature representation
				   @param len - length of the feature representation
				   @param value - Update value for the feature
				 */
				void update(const void* key, const int len, float value)
				{
						const unsigned hash = MurmurHash3_x86_32 (key, len, seed) % D;
						data[hash] = value;
				}

				/*
				   Get feature
				   @param key - pointer to feature representation
				   @param len - length of the feature representation
				 */
				float retrieve(const void* key, const int len) const
				{
						const unsigned hash = MurmurHash3_x86_32 (key, len, seed) % D;
						return data[hash];
				}

				/*
				   Update the same feature for multiple classes using AVX instructions - For Softmax Regression
				   @param hash - Hash index for the feature
				   @param cdx - Class offset
				   @param value - Update value for the features
				 */
				void simd_update(const unsigned hash, const unsigned cdx, __m256 value)
				{
						const unsigned AVX = 8;
						const unsigned idx = hash * NK + cdx * AVX;
						if(cdx < DIV)
						{
								__m256 current = _mm256_load_ps( &data[idx] );
								__m256 result = _mm256_add_ps(current, value);
								_mm256_store_ps(&data[idx], result);
						}
						else
						{
								__m256 current = _mm256_maskload_ps( &data[idx], mask );
								__m256 result = _mm256_add_ps(current, value);
								_mm256_maskstore_ps(&data[idx], mask, result);
						}
				}

				/*
				   Get the same feature for multiple classes using AVX instructions - For Softmax Regression
				   @param cache - Cached indices and signs for the feature
				   @param cdx - Class offset
				 */
				__m256 simd_retrieve(const unsigned hash, const unsigned cdx) const
				{
						const unsigned AVX = 8;
						const unsigned idx = hash * NK + cdx * AVX;

						if(cdx < DIV)
						{
								return _mm256_load_ps( &data[idx] );
						}
						else
						{
								return _mm256_maskload_ps( &data[idx], mask );
						}
				}

				/*
				   Precompute the Hash Index for a feature
				   @param key - pointer to feature representation
				   @param len - length of the feature representation
				   @param result - struct to store the cached values
				 */
				unsigned hash(const void* key, const int len)
				{
						return MurmurHash3_x86_32 (key, len, seed) % D;
				}
};

#endif // CMS_ML_MEM_H_
