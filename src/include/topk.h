#ifndef CMS_ML_TOPK_H_
#define CMS_ML_TOPK_H_

#include "util.h"
#include <utility>
#include <array>
#include <unordered_map>

#include <algorithm>
#include <assert.h>
#include <cfloat>
#include <cmath>

#include <string>
#include <cstring>
#include <fstream>

const float EPS = 1.05;

/*
   Fast Parser - Read data efficiently using Memory-Mapped I/O
 */
namespace std
{
		template<typename T, size_t N>
				struct hash<array<T, N> >
				{
						typedef array<T, N> argument_type;
						typedef size_t result_type;

						result_type operator()(const argument_type& item) const
						{
								return MurmurHash3_x86_32 (item.data(), N, 8192);
						}
				};
}

template <typename key_t, int N>
class TopK
{
		private:
				typedef std::pair<float, int> ftr;

				// data - memory_index => <weight, ptr>
				// keys - ptr => feature
				// dict - feature => memory_index
				std::vector<ftr> data;
				std::vector<key_t> keys;
				std::unordered_map<key_t, int> dict;
				std::unordered_map<key_t, float> fdict;
				size_t count;

		public:
				TopK() : data(N), keys(N), count(0) {}

				/*
				   @param key - feature representation
				   @return the corresponding value for the feature if present in Top-K Heap
				 */
				const float operator[] (const key_t& key) const
				{
						return (fdict.find(key) == fdict.end()) ? 0.0 : fdict.at(key);
				}

				/*
				   @param key - feature representation
				   @return true if the feature is present in Top-K Heap
				 */
				const bool find (const key_t& key) const
				{
						return (fdict.find(key) == fdict.end()) ? false : true;
				}

				bool full() const
				{
						return (count >= N);
				}

				/*
				   @return the minimum value in the heap if the Top-K Heap is full
				 */
				float minimum() const
				{
						return (count < N) ? 0.0 : data[0].first;
				}

				/*
				   Insert new key into Top-K heap if greater than the minimum value
				   @param key - feature representation
				   @param value - corresponding value for the feature
				 */
				void push(const key_t& key, const float value)
				{
						float abs_value = my_abs(value);
						if(dict.find(key) != dict.end())
						{
								fdict[key] = value;

								int pos = dict[key];
								float& current = data[pos].first;
								bool top = (abs_value >= current * EPS);
								bool bottom = (abs_value <= current / EPS);
								if(count < N)
								{
										current = abs_value;
								}
								else if(top || bottom)
								{
										current = abs_value;
										heapify(pos+1, true);
								}
						}
						else if(count < N)
						{
								data[count].first = abs_value;
								data[count].second = count;
								keys[count] = key;
								dict[key] = count;
								fdict[key] = value;
								++count;

								// Build Heap
								if(count == N)
								{
										for(int idx = N/2; idx > 0; --idx)
										{
												heapify(idx);
										}
								}
						}
						else if(abs_value > (this->minimum() * EPS))
						{
								insert(key, value);
						}
						assert(dict.size() <= N);
				}

				/*
				   Maintain the Heap Invariant
				   @param idx - current updated index
				   @param update - recursively call function to maintain heap
				 */
				void heapify(int idx, bool update = false)
				{
						ftr& current = data[idx-1];

						if(update && idx > 1)
						{
								// Swap Parent
								int p_idx = idx/2;
								ftr& parent = data[p_idx-1];

								if(current.first < parent.first)
								{
										dict[keys[current.second]] = p_idx-1;
										dict[keys[parent.second]] = idx-1;
										std::swap(current, parent);
										heapify(p_idx, true);
								}
						}

						int lc_idx = 2*idx;
						int rc_idx = 2*idx+1;
						if(lc_idx <= N && rc_idx <= N)
						{
								// Swap Smallest Child
								ftr& lc = data[lc_idx-1];
								ftr& rc = data[rc_idx-1];
								bool left_smallest = lc.first <= rc.first;
								int sc_idx = (left_smallest) ? lc_idx : rc_idx;
								ftr& sc = (left_smallest) ? lc : rc;

								if(sc.first < current.first)
								{
										dict[keys[current.second]] = sc_idx-1;
										dict[keys[sc.second]] = idx-1;
										std::swap(sc, current);
										heapify(sc_idx, update);
								}
						}
				}

				/*
				   Delete old minimum value and replace with new value
				   @param key - feature representation
				   @param value - corresponding value for the feature
				 */
				void insert(const key_t& key, float value)
				{
						// Delete Minimum
						int min_pos = data[0].second;
						key_t& min_key = keys[min_pos];
						dict.erase(min_key);
						fdict.erase(min_key);

						min_key = key;
						dict[key] = 0;
						fdict[key] = value;
						data[0].first = my_abs(value);
						heapify(1);
				}

				/*
				   Check the Heap Invariant
				 */
				void check() const
				{
						for(int idx = N/2; idx > 0; --idx)
						{
								int p_idx = std::max(idx/2, 1);
								int lc_idx = std::min(2*idx, N);
								int rc_idx = std::min(2*idx+1, N);

								const ftr& lc = data[lc_idx-1];
								const ftr& rc = data[rc_idx-1];
								const ftr& current = data[idx-1];
								const ftr& parent = data[p_idx-1];

								assert(current.first <= lc.first);
								assert(current.first <= rc.first);
								assert(parent.first <= current.first);
						}
				}

				/*
				   Load Top-K Heap from file
				   @param myfile - Top-K Heap File
				 */
				void load(std::ifstream& myfile)
				{
						// Read Random Seeds
						std::string line;
						getline (myfile, line);

						const size_t FN = std::atol(line.c_str());
						for(size_t idx = 0; idx < FN; ++idx)
						{
								getline (myfile, line);

								// Key
								data_t key;
								std::strcpy (key.data(), line.c_str());

								// Float Weight
								getline (myfile, line);
								float value = std::atof(line.c_str());

								data[count].first = my_abs(value);
								data[count].second = count;
								keys[count] = key;
								dict[key] = count;
								fdict[key] = value;
								++count;
						}
				}

				/*
				   Save Top-K Heap to file
				   @param myfile - Top-K Heap File
				 */
				void save(std::ofstream& myfile) const
				{
						assert(myfile.is_open());

						myfile << dict.size() << std::endl;
						for(const auto& item : data)
						{
								const key_t& key = keys[item.second];
								const float value = fdict.at(key);
								myfile << key << std::endl;
								myfile << value << std::endl;
						}
				}

				/*
				   @return current size of the Top-K Heap
				 */
				size_t size() const
				{
						return count;
				}
};

#endif // CMS_ML_TOPK_H
