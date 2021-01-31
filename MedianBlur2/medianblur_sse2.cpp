#ifdef ENABLE_INTEL_SIMD
#include "avisynth.h"
#include "medianblur_sse2.h"

template<typename T>
static MB_FORCEINLINE __m128i simd_adds(const __m128i& a, const __m128i& b) {
  assert(false);
  return _mm_setzero_si128();
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
static MB_FORCEINLINE __m128i simd_subs(const __m128i& a, const __m128i& b) {
  assert(false);
  return _mm_setzero_si128();
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

template<typename pixel_t>
static MB_FORCEINLINE __m128i simd_median_sse4(const __m128i& a, const __m128i& b, const __m128i& c) {
  assert(false);
  return _mm_setzero_si128();
}

template<>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
MB_FORCEINLINE __m128i simd_median_sse4<uint8_t>(const __m128i& a, const __m128i& b, const __m128i& c) {
  auto t1 = _mm_min_epu8(a, b);
  auto t2 = _mm_max_epu8(a, b);
  auto t3 = _mm_min_epu8(t2, c);
  auto median = _mm_max_epu8(t1, t3);
  return median;
}

template<>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
MB_FORCEINLINE __m128i simd_median_sse4<uint16_t>(const __m128i& a, const __m128i& b, const __m128i& c) {
  auto t1 = _mm_min_epu16(a, b);
  auto t2 = _mm_max_epu16(a, b);
  auto t3 = _mm_min_epu16(t2, c);
  auto median = _mm_max_epu16(t1, t3);
  return median;
}

template<>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
MB_FORCEINLINE __m128i simd_median_sse4<float>(const __m128i& aa, const __m128i& bb, const __m128i& cc) {
  auto a = _mm_castsi128_ps(aa);
  auto b = _mm_castsi128_ps(bb);
  auto c = _mm_castsi128_ps(cc);
  auto t1 = _mm_min_ps(a, b);
  auto t2 = _mm_max_ps(a, b);
  auto t3 = _mm_min_ps(t2, c);
  auto median = _mm_max_ps(t1, t3);
  return _mm_castps_si128(median);
}

template<typename pixel_t>
static MB_FORCEINLINE __m128i simd_median(const __m128i& a, const __m128i& b, const __m128i& c) {
  assert(false);
  return _mm_setzero_si128();
}

template<>
MB_FORCEINLINE __m128i simd_median<uint8_t>(const __m128i& a, const __m128i& b, const __m128i& c) {
  auto t1 = _mm_min_epu8(a, b);
  auto t2 = _mm_max_epu8(a, b);
  auto t3 = _mm_min_epu8(t2, c);
  auto median = _mm_max_epu8(t1, t3);
  return median;
}

template<>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
MB_FORCEINLINE __m128i simd_median<uint16_t>(const __m128i& a, const __m128i& b, const __m128i& c) {
  auto t1 = _mm_min_epu16(a, b);
  auto t2 = _mm_max_epu16(a, b);
  auto t3 = _mm_min_epu16(t2, c);
  auto median = _mm_max_epu16(t1, t3);
  return median;
}

template<>
MB_FORCEINLINE __m128i simd_median<float>(const __m128i& aa, const __m128i& bb, const __m128i& cc) {
  auto a = _mm_castsi128_ps(aa);
  auto b = _mm_castsi128_ps(bb);
  auto c = _mm_castsi128_ps(cc);
  auto t1 = _mm_min_ps(a, b);
  auto t2 = _mm_max_ps(a, b);
  auto t3 = _mm_min_ps(t2, c);
  auto median = _mm_max_ps(t1, t3);
  return _mm_castps_si128(median);
}

template<typename pixel_t>
MB_FORCEINLINE __m128i simd_median5(const __m128i& a, const __m128i& b, const __m128i& c, const __m128i& d, const __m128i& e) {
  assert(false);
  return _mm_setzero_si128();
}

template<>
MB_FORCEINLINE __m128i simd_median5<uint8_t>(const __m128i& a, const __m128i& b, const __m128i& c, const __m128i& d, const __m128i& e) {
  auto f = _mm_max_epu8(_mm_min_epu8(a, b), _mm_min_epu8(c, d)); // discards lowest from first 4
  auto g = _mm_min_epu8(_mm_max_epu8(a, b), _mm_max_epu8(c, d)); // discards biggest from first 4
  return simd_median<uint8_t>(e, f, g);
}

template<>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
MB_FORCEINLINE __m128i simd_median5<uint16_t>(const __m128i& a, const __m128i& b, const __m128i& c, const __m128i& d, const __m128i& e) {
  auto f = _mm_max_epu16(_mm_min_epu16(a, b), _mm_min_epu16(c, d)); // discards lowest from first 4
  auto g = _mm_min_epu16(_mm_max_epu16(a, b), _mm_max_epu16(c, d)); // discards biggest from first 4
  return simd_median<uint16_t>(e, f, g);
}

template<>
MB_FORCEINLINE __m128i simd_median5<float>(const __m128i& aa, const __m128i& bb, const __m128i& cc, const __m128i& dd, const __m128i& ee) {
  auto a = _mm_castsi128_ps(aa);
  auto b = _mm_castsi128_ps(bb);
  auto c = _mm_castsi128_ps(cc);
  auto d = _mm_castsi128_ps(dd);
  auto e = _mm_castsi128_ps(ee);
  auto f = _mm_max_ps(_mm_min_ps(a, b), _mm_min_ps(c, d)); // discards lowest from first 4
  auto g = _mm_min_ps(_mm_max_ps(a, b), _mm_max_ps(c, d)); // discards biggest from first 4
  return simd_median<float>(_mm_castps_si128(e), _mm_castps_si128(f), _mm_castps_si128(g));
}

#define MEDIANPROCESSOR_SSE2
#include "medianblur.hpp"
#undef MEDIANPROCESSOR_SSE2

// instantiate
template void MedianProcessor_sse2<uint8_t, 8, InstructionSet::SSE2>::calculate_median<uint8_t, 8, false>(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height, int radius, void* buffer);
template void MedianProcessor_sse2<uint8_t, 10, InstructionSet::SSE2>::calculate_median<uint16_t, 10, false>(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height, int radius, void* buffer);
template void MedianProcessor_sse2<uint8_t, 12, InstructionSet::SSE2>::calculate_median<uint16_t, 12, false>(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height, int radius, void* buffer);
template void MedianProcessor_sse2<uint8_t, 14, InstructionSet::SSE2>::calculate_median<uint16_t, 14, false>(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height, int radius, void* buffer);
template void MedianProcessor_sse2<uint8_t, 16, InstructionSet::SSE2>::calculate_median<uint16_t, 16, false>(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height, int radius, void* buffer);
template void MedianProcessor_sse2<uint8_t, 16, InstructionSet::SSE2>::calculate_median<float, 16, false>(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height, int radius, void* buffer);
template void MedianProcessor_sse2<uint8_t, 16, InstructionSet::SSE2>::calculate_median<float, 16, true>(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height, int radius, void* buffer);

template void MedianProcessor_sse2<uint16_t, 8, InstructionSet::SSE2>::calculate_median<uint8_t, 8, false>(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height, int radius, void* buffer);
template void MedianProcessor_sse2<uint16_t, 10, InstructionSet::SSE2>::calculate_median<uint16_t, 10, false>(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height, int radius, void* buffer);
template void MedianProcessor_sse2<uint16_t, 12, InstructionSet::SSE2>::calculate_median<uint16_t, 12, false>(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height, int radius, void* buffer);
template void MedianProcessor_sse2<uint16_t, 14, InstructionSet::SSE2>::calculate_median<uint16_t, 14, false>(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height, int radius, void* buffer);
template void MedianProcessor_sse2<uint16_t, 16, InstructionSet::SSE2>::calculate_median<uint16_t, 16, false>(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height, int radius, void* buffer);
template void MedianProcessor_sse2<uint16_t, 16, InstructionSet::SSE2>::calculate_median<float, 16, false>(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height, int radius, void* buffer);
template void MedianProcessor_sse2<uint16_t, 16, InstructionSet::SSE2>::calculate_median<float, 16, true>(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height, int radius, void* buffer);

template void MedianProcessor_sse2<int32_t, 8, InstructionSet::SSE2>::calculate_temporal_median<uint8_t, 8, false>(uint8_t* dstp, int dst_pitch, const uint8_t** src_ptrs, const int* src_pitches, int frames_count, int width, int height, int radius, void* buffer);
template void MedianProcessor_sse2<int32_t, 10, InstructionSet::SSE2>::calculate_temporal_median<uint16_t, 10, false>(uint8_t* dstp, int dst_pitch, const uint8_t** src_ptrs, const int* src_pitches, int frames_count, int width, int height, int radius, void* buffer);
template void MedianProcessor_sse2<int32_t, 12, InstructionSet::SSE2>::calculate_temporal_median<uint16_t, 12, false>(uint8_t* dstp, int dst_pitch, const uint8_t** src_ptrs, const int* src_pitches, int frames_count, int width, int height, int radius, void* buffer);
template void MedianProcessor_sse2<int32_t, 14, InstructionSet::SSE2>::calculate_temporal_median<uint16_t, 14, false>(uint8_t* dstp, int dst_pitch, const uint8_t** src_ptrs, const int* src_pitches, int frames_count, int width, int height, int radius, void* buffer);
template void MedianProcessor_sse2<int32_t, 16, InstructionSet::SSE2>::calculate_temporal_median<uint16_t, 16, false>(uint8_t* dstp, int dst_pitch, const uint8_t** src_ptrs, const int* src_pitches, int frames_count, int width, int height, int radius, void* buffer);
template void MedianProcessor_sse2<int32_t, 16, InstructionSet::SSE2>::calculate_temporal_median<float, 16, false>(uint8_t* dstp, int dst_pitch, const uint8_t** src_ptrs, const int* src_pitches, int frames_count, int width, int height, int radius, void* buffer);
template void MedianProcessor_sse2<int32_t, 16, InstructionSet::SSE2>::calculate_temporal_median<float, 16, true>(uint8_t * dstp, int dst_pitch, const uint8_t * *src_ptrs, const int* src_pitches, int frames_count, int width, int height, int radius, void* buffer);

template
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void calculate_temporal_median_sr0_tr1_sse4<uint8_t>(uint8_t* dstp, int dst_pitch, const uint8_t** src_ptrs, const int* src_pitches, int frames_count, int width, int height, int radius, void* buffer);

template
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void calculate_temporal_median_sr0_tr1_sse4<uint16_t>(uint8_t* dstp, int dst_pitch, const uint8_t** src_ptrs, const int* src_pitches, int frames_count, int width, int height, int radius, void* buffer);

template
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void calculate_temporal_median_sr0_tr1_sse4<float>(uint8_t* dstp, int dst_pitch, const uint8_t** src_ptrs, const int* src_pitches, int frames_count, int width, int height, int radius, void* buffer);

template
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void calculate_temporal_median_sr0_tr2_sse4<uint8_t>(uint8_t* dstp, int dst_pitch, const uint8_t** src_ptrs, const int* src_pitches, int frames_count, int width, int height, int radius, void* buffer);

template
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void calculate_temporal_median_sr0_tr2_sse4<uint16_t>(uint8_t* dstp, int dst_pitch, const uint8_t** src_ptrs, const int* src_pitches, int frames_count, int width, int height, int radius, void* buffer);

template
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void calculate_temporal_median_sr0_tr2_sse4<float>(uint8_t* dstp, int dst_pitch, const uint8_t** src_ptrs, const int* src_pitches, int frames_count, int width, int height, int radius, void* buffer);

#endif // ENABLE_INTEL_SIMD
