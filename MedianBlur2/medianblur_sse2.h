#ifndef __MEDIANBLUR_SSE2_H__
#define __MEDIANBLUR_SSE2_H__

#include "avisynth.h"
#include <emmintrin.h>
#include <cassert>
#include "common.h"

template<typename T>
static MB_FORCEINLINE __m128i simd_adds(const __m128i& a, const __m128i& b) {
  assert(false);
}

template<>
MB_FORCEINLINE __m128i simd_adds<uint8_t>(const __m128i& a, const __m128i& b) {
  return _mm_adds_epu8(a, b);
}

template<>
MB_FORCEINLINE __m128i simd_adds<uint16_t>(const __m128i& a, const __m128i& b) {
  return _mm_adds_epu16(a, b);
}

template<>
MB_FORCEINLINE __m128i simd_adds<int32_t>(const __m128i& a, const __m128i& b) {
  return _mm_add_epi32(a, b);
}

template<typename T>
MB_FORCEINLINE __m128i simd_subs(const __m128i& a, const __m128i& b) {
  assert(false);
}

template<>
MB_FORCEINLINE __m128i simd_subs<uint8_t>(const __m128i& a, const __m128i& b) {
  return _mm_subs_epu8(a, b);
}

template<>
MB_FORCEINLINE __m128i simd_subs<uint16_t>(const __m128i& a, const __m128i& b) {
  return _mm_subs_epu16(a, b);
}

template<>
MB_FORCEINLINE __m128i simd_subs<int32_t>(const __m128i& a, const __m128i& b) {
  return _mm_sub_epi32(a, b);
}

#define MEDIANPROCESSOR_SSE2
#include "medianblur.h"
#undef MEDIANPROCESSOR_SSE2


#endif
