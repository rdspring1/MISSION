#include "util.h"

#include <stdio.h>
#include <cfloat>
#include <algorithm>
#include <assert.h>
#include <iostream>

float get (__m256* data, size_t idx)
{
		const size_t AVX = 8;
		size_t row = idx / AVX;
		size_t col = idx % AVX;
		return data[row][col];
}

void replace (__m256* data, size_t idx, float value)
{
		const size_t AVX = 8;
		size_t row = idx / AVX;
		size_t col = idx % AVX;
		data[row][col] = value;
}

void update (__m256* data, size_t idx, float value)
{
		const size_t AVX = 8;
		size_t row = idx / AVX;
		size_t col = idx % AVX;
		data[row][col] += value;
}

void maximum(__m256* data, size_t len, float& value, uint32_t& argmax)
{
		value = FLT_MIN;
		argmax = 0;
		for(size_t idx = 0; idx < len; ++idx)
		{
				float nv = get(data, idx);
				if(nv > value)
				{
						value = nv;
						argmax = idx;
				}
		}
}

void partition(__m256* data, const size_t CNT, const size_t len, const float max_value)
{
		// Subtract Maximum Value
		__m256 mv = _mm256_set1_ps(max_value);
		for(size_t cdx = 0; cdx < CNT; ++cdx)
		{
				data[cdx] = _mm256_sub_ps(data[cdx], mv);
		}

		// Exponentiate + Sum
		float sum = 0.0;
		for(size_t idx = 0; idx < len; ++idx)
		{
				float v = std::exp(get(data, idx));
				sum += v;
				replace(data, idx, v);
		}

		// Divide by Partition Function
		__m256 sv = _mm256_set1_ps(sum);
		for(size_t cdx = 0; cdx < CNT; ++cdx)
		{
				data[cdx] = _mm256_div_ps(data[cdx], sv);
		}
}

__m256 median(__m256 a, __m256 b, __m256 c)
{
		__m256 ab_min = _mm256_min_ps(a, b);
		__m256 ab_max = _mm256_max_ps(a, b);
		__m256 ab_max_c_min = _mm256_min_ps(ab_max, c);
		return _mm256_max_ps(ab_min, ab_max_c_min);
}

__m256 my_abs(__m256 x)
{
		const __m256 MASK = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
		return _mm256_andnot_ps(MASK, x);
}

void maximum(float* data, size_t len, float& value, uint32_t& argmax)
{
		value = FLT_MIN;
		argmax = 0;
		for(size_t idx = 0; idx < len; ++idx)
		{
				if(data[idx] > value)
				{
						value = data[idx];
						argmax = idx;
				}
		}
}

float sum(float* data, size_t len)
{
		float value = 0;
		for(size_t idx = 0; idx < len; ++idx)
		{
				value += data[idx];
		}
		return value;
}

float median(float a, float b, float c)
{
		return std::max(std::min(a,b), std::min(std::max(a,b),c));
}

float median(std::vector<float> values)
{
		size_t size = values.size();
		std::sort(values.begin(), values.end());

		if (size % 2 == 0)
		{
				return (values[size / 2 - 1] + values[size / 2]) / 2;
		}
		else
		{
				return values[size / 2];
		}
}

float my_abs(float x)
{
		return (x >= 0) ? x : -x;
}
