#include "avisynth.h"
#include "medianblur_sse2.h"

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
static MB_FORCEINLINE __m128i simd_subs(const __m128i& a, const __m128i& b) {
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


