#ifdef INTEL_INTRINSICS

#include "avisynth.h"
#include "medianblur_avx2.h"

#if defined (__GNUC__) && ! defined (__INTEL_COMPILER)
#include <x86intrin.h>
// x86intrin.h includes header files for whatever instruction
// sets are specified on the compiler command line, such as: xopintrin.h, fma4intrin.h
#else
#include <immintrin.h> // MS version of immintrin.h covers AVX, AVX2 and FMA3
#endif // __GNUC__

#if !defined(__FMA__)
// Assume that all processors that have AVX2 also have FMA3
#if defined (__GNUC__) && ! defined (__INTEL_COMPILER) && ! defined (__clang__)
// Prevent error message in g++ when using FMA intrinsics with avx2:
#pragma message "It is recommended to specify also option -mfma when using -mavx2 or higher"
#else
#define __FMA__  1
#endif
#endif
// FMA3 instruction set
#if defined (__FMA__) && (defined(__GNUC__) || defined(__clang__))  && ! defined (__INTEL_COMPILER)
#include <fmaintrin.h>
#endif // __FMA__ 



template<typename T>
static MB_FORCEINLINE __m256i simd_adds(const __m256i& a, const __m256i& b) {
  assert(false);
  return _mm256_setzero_si256();
}

template<>
MB_FORCEINLINE __m256i simd_adds<uint8_t>(const __m256i& a, const __m256i& b) {
  return _mm256_adds_epu8(a, b);
}

template<>
MB_FORCEINLINE __m256i simd_adds<uint16_t>(const __m256i& a, const __m256i& b) {
  return _mm256_adds_epu16(a, b);
}

template<>
MB_FORCEINLINE __m256i simd_adds<int32_t>(const __m256i& a, const __m256i& b) {
  return _mm256_add_epi32(a, b);
}

template<typename T>
MB_FORCEINLINE __m256i simd_subs(const __m256i& a, const __m256i& b) {
  assert(false);
  return _mm256_setzero_si256();
}

template<>
MB_FORCEINLINE __m256i simd_subs<uint8_t>(const __m256i& a, const __m256i& b) {
  return _mm256_subs_epu8(a, b);
}

template<>
MB_FORCEINLINE __m256i simd_subs<uint16_t>(const __m256i& a, const __m256i& b) {
  return _mm256_subs_epu16(a, b);
}

template<>
MB_FORCEINLINE __m256i simd_subs<int32_t>(const __m256i& a, const __m256i& b) {
  return _mm256_sub_epi32(a, b);
}

template<typename pixel_t>
static MB_FORCEINLINE __m256i simd_median(const __m256i& a, const __m256i& b, const __m256i& c) {
  assert(false);
  return _mm256_setzero_si256();
}

template<>
MB_FORCEINLINE __m256i simd_median<uint8_t>(const __m256i& a, const __m256i& b, const __m256i& c) {
  auto t1 = _mm256_min_epu8(a, b);
  auto t2 = _mm256_max_epu8(a, b);
  auto t3 = _mm256_min_epu8(t2, c);
  auto median = _mm256_max_epu8(t1, t3);
  return median;
}

template<>
MB_FORCEINLINE __m256i simd_median<uint16_t>(const __m256i& a, const __m256i& b, const __m256i& c) {
  auto t1 = _mm256_min_epu16(a, b);
  auto t2 = _mm256_max_epu16(a, b);
  auto t3 = _mm256_min_epu16(t2, c);
  auto median = _mm256_max_epu16(t1, t3);
  return median;
}

template<>
MB_FORCEINLINE __m256i simd_median<float>(const __m256i& aa, const __m256i& bb, const __m256i& cc) {
  auto a = _mm256_castsi256_ps(aa);
  auto b = _mm256_castsi256_ps(bb);
  auto c = _mm256_castsi256_ps(cc);
  auto t1 = _mm256_min_ps(a, b);
  auto t2 = _mm256_max_ps(a, b);
  auto t3 = _mm256_min_ps(t2, c);
  auto median = _mm256_max_ps(t1, t3);
  return _mm256_castps_si256(median);
}

template<typename pixel_t>
MB_FORCEINLINE __m256i simd_median5(const __m256i& a, const __m256i& b, const __m256i& c, const __m256i& d, const __m256i& e) {
  assert(false);
  return _mm256_setzero_si256();
}

template<>
MB_FORCEINLINE __m256i simd_median5<uint8_t>(const __m256i& a, const __m256i& b, const __m256i& c, const __m256i& d, const __m256i& e) {
  auto f = _mm256_max_epu8(_mm256_min_epu8(a, b), _mm256_min_epu8(c, d)); // discards lowest from first 4
  auto g = _mm256_min_epu8(_mm256_max_epu8(a, b), _mm256_max_epu8(c, d)); // discards biggest from first 4
  return simd_median<uint8_t>(e, f, g);
}

template<>
MB_FORCEINLINE __m256i simd_median5<uint16_t>(const __m256i& a, const __m256i& b, const __m256i& c, const __m256i& d, const __m256i& e) {
  auto f = _mm256_max_epu16(_mm256_min_epu16(a, b), _mm256_min_epu16(c, d)); // discards lowest from first 4
  auto g = _mm256_min_epu16(_mm256_max_epu16(a, b), _mm256_max_epu16(c, d)); // discards biggest from first 4
  return simd_median<uint16_t>(e, f, g);
}

template<>
MB_FORCEINLINE __m256i simd_median5<float>(const __m256i& aa, const __m256i& bb, const __m256i& cc, const __m256i& dd, const __m256i& ee) {
  auto a = _mm256_castsi256_ps(aa);
  auto b = _mm256_castsi256_ps(bb);
  auto c = _mm256_castsi256_ps(cc);
  auto d = _mm256_castsi256_ps(dd);
  auto e = _mm256_castsi256_ps(ee);
  auto f = _mm256_max_ps(_mm256_min_ps(a, b), _mm256_min_ps(c, d)); // discards lowest from first 4
  auto g = _mm256_min_ps(_mm256_max_ps(a, b), _mm256_max_ps(c, d)); // discards biggest from first 4
  return simd_median<float>(_mm256_castps_si256(e), _mm256_castps_si256(f), _mm256_castps_si256(g));
}

#define MEDIANPROCESSOR_AVX2
#include "medianblur.hpp"
#undef MEDIANPROCESSOR_AVX2

// instantiate
template void MedianProcessor_avx2<uint8_t, 8, InstructionSet::AVX2>::calculate_median<uint8_t, 8, false>(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height, int radius, void* buffer);
template void MedianProcessor_avx2<uint8_t, 10, InstructionSet::AVX2>::calculate_median<uint16_t, 10, false>(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height, int radius, void* buffer);
template void MedianProcessor_avx2<uint8_t, 12, InstructionSet::AVX2>::calculate_median<uint16_t, 12, false>(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height, int radius, void* buffer);
template void MedianProcessor_avx2<uint8_t, 14, InstructionSet::AVX2>::calculate_median<uint16_t, 14, false>(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height, int radius, void* buffer);
template void MedianProcessor_avx2<uint8_t, 16, InstructionSet::AVX2>::calculate_median<uint16_t, 16, false>(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height, int radius, void* buffer);
template void MedianProcessor_avx2<uint8_t, 16, InstructionSet::AVX2>::calculate_median<float, 16, false>(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height, int radius, void* buffer);
template void MedianProcessor_avx2<uint8_t, 16, InstructionSet::AVX2>::calculate_median<float, 16, true>(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height, int radius, void* buffer);

template void MedianProcessor_avx2<uint16_t, 8, InstructionSet::AVX2>::calculate_median<uint8_t, 8, false>(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height, int radius, void* buffer);
template void MedianProcessor_avx2<uint16_t, 10, InstructionSet::AVX2>::calculate_median<uint16_t, 10, false>(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height, int radius, void* buffer);
template void MedianProcessor_avx2<uint16_t, 12, InstructionSet::AVX2>::calculate_median<uint16_t, 12, false>(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height, int radius, void* buffer);
template void MedianProcessor_avx2<uint16_t, 14, InstructionSet::AVX2>::calculate_median<uint16_t, 14, false>(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height, int radius, void* buffer);
template void MedianProcessor_avx2<uint16_t, 16, InstructionSet::AVX2>::calculate_median<uint16_t, 16, false>(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height, int radius, void* buffer);
template void MedianProcessor_avx2<uint16_t, 16, InstructionSet::AVX2>::calculate_median<float, 16, false>(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height, int radius, void* buffer);
template void MedianProcessor_avx2<uint16_t, 16, InstructionSet::AVX2>::calculate_median<float, 16, true>(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height, int radius, void* buffer);

template void MedianProcessor_avx2<int32_t, 8, InstructionSet::AVX2>::calculate_temporal_median<uint8_t, 8, false>(uint8_t* dstp, int dst_pitch, const uint8_t** src_ptrs, const int* src_pitches, int frames_count, int width, int height, int radius, void* buffer);
template void MedianProcessor_avx2<int32_t, 10, InstructionSet::AVX2>::calculate_temporal_median<uint16_t, 10, false>(uint8_t* dstp, int dst_pitch, const uint8_t** src_ptrs, const int* src_pitches, int frames_count, int width, int height, int radius, void* buffer);
template void MedianProcessor_avx2<int32_t, 12, InstructionSet::AVX2>::calculate_temporal_median<uint16_t, 12, false>(uint8_t* dstp, int dst_pitch, const uint8_t** src_ptrs, const int* src_pitches, int frames_count, int width, int height, int radius, void* buffer);
template void MedianProcessor_avx2<int32_t, 14, InstructionSet::AVX2>::calculate_temporal_median<uint16_t, 14, false>(uint8_t* dstp, int dst_pitch, const uint8_t** src_ptrs, const int* src_pitches, int frames_count, int width, int height, int radius, void* buffer);
template void MedianProcessor_avx2<int32_t, 16, InstructionSet::AVX2>::calculate_temporal_median<uint16_t, 16, false>(uint8_t* dstp, int dst_pitch, const uint8_t** src_ptrs, const int* src_pitches, int frames_count, int width, int height, int radius, void* buffer);
template void MedianProcessor_avx2<int32_t, 16, InstructionSet::AVX2>::calculate_temporal_median<float, 16, false>(uint8_t* dstp, int dst_pitch, const uint8_t** src_ptrs, const int* src_pitches, int frames_count, int width, int height, int radius, void* buffer);
template void MedianProcessor_avx2<int32_t, 16, InstructionSet::AVX2>::calculate_temporal_median<float, 16, true>(uint8_t* dstp, int dst_pitch, const uint8_t** src_ptrs, const int* src_pitches, int frames_count, int width, int height, int radius, void* buffer);

template void calculate_temporal_median_sr0_tr1_avx2<uint8_t>(uint8_t* dstp, int dst_pitch, const uint8_t** src_ptrs, const int* src_pitches, int frames_count, int width, int height, int radius, void* buffer);
template void calculate_temporal_median_sr0_tr1_avx2<uint16_t>(uint8_t* dstp, int dst_pitch, const uint8_t** src_ptrs, const int* src_pitches, int frames_count, int width, int height, int radius, void* buffer);
template void calculate_temporal_median_sr0_tr1_avx2<float>(uint8_t* dstp, int dst_pitch, const uint8_t** src_ptrs, const int* src_pitches, int frames_count, int width, int height, int radius, void* buffer);

template void calculate_temporal_median_sr0_tr2_avx2<uint8_t>(uint8_t* dstp, int dst_pitch, const uint8_t** src_ptrs, const int* src_pitches, int frames_count, int width, int height, int radius, void* buffer);
template void calculate_temporal_median_sr0_tr2_avx2<uint16_t>(uint8_t* dstp, int dst_pitch, const uint8_t** src_ptrs, const int* src_pitches, int frames_count, int width, int height, int radius, void* buffer);
template void calculate_temporal_median_sr0_tr2_avx2<float>(uint8_t* dstp, int dst_pitch, const uint8_t** src_ptrs, const int* src_pitches, int frames_count, int width, int height, int radius, void* buffer);

#endif // INTEL_INTRINSICS

