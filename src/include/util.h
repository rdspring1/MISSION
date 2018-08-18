#ifndef CMS_ML_UTIL_H_
#define CMS_ML_UTIL_H_

#include <stdint.h> 
#include <cstddef>
#include <vector>
#include <cmath>

#include <immintrin.h>

// AVX Functions
float get (__m256* data, size_t idx);
void replace (__m256* data, size_t idx, float value);
void update (__m256* data, size_t idx, float value);
void maximum(__m256* data, size_t len, float& value, uint32_t& argmax);
void partition(__m256* data, const size_t CNT, const size_t len, const float max_value);
__m256 median(__m256 a, __m256 b, __m256 c);
__m256 my_abs(__m256 x);

// Standard Functions
void maximum(float* data, size_t len, float& value, uint32_t& argmax);
float sum(float* data, size_t len);
float median(float a, float b, float c);
float median(std::vector<float> values);
float my_abs(float x);

#endif /* CMS_ML_UTIL_H_ */
