#include "avisynth.h"
#include "medianblur_avx2.h"

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

// n/a
