#include "avisynth.h"
#include <stdint.h>
#include <cassert>
#include <algorithm>

static MB_FORCEINLINE int calculate_window_side_length(int radius, int x, int width) {
  int length = radius + 1;
  if (x <= radius) {
    length += x;
  }
  else if (x >= width - radius) {
    length += width - x - 1;
  }
  else {
    length += radius;
  }
  return length;
}

/*
add and sum work on 1/2/4 byte sized counters,
size is template parameter uint8_t, uint16_t and int32_t.
depends on how many pixels (radius+temporary frames) should be summed up
for radius < 8 non-temporal: byte is enough

rad=1 rad=2     rad=7 rad=8
3x3   5x5       15x15 17x17
9:    25:   ... 225   289! byte is not enough
***   *****
*X*   *****
***   **X**
      *****
      *****

      T: uint8_t, uint16_t or int32_t: whichever fits the max element count of the Histogram

T is the counter type
uint8_t for Median with radius<8: Processing width <= 7+1+7; max pixel count for a histogram entry <= 15x15, counters (bins) fit in a byte (uint8_t)
uint16_t for Median with radius>=8: Processing width >= 8+1+8; max pixel count for a histogram entry >= 17x17, counters (bins) fit in a uint16_t
uint32_t for TemporalMedian, independently from radius

--> AVX2 benefits when course_histogram_size*sizeof(T)>=32 (all cases except (8bit@radius<=7))

*/

// template is specialized for avx2

#ifdef MEDIANPROCESSOR_AVX2
#define MEDIANPROC MedianProcessor_avx2
#define MEDIANPROC_SR0_T1 calculate_temporal_median_sr0_tr1_avx2
#define MEDIANPROC_SR0_T2 calculate_temporal_median_sr0_tr2_avx2
#endif
#ifdef MEDIANPROCESSOR_SSE2
#define MEDIANPROC MedianProcessor_sse2
#define MEDIANPROC_SR0_T1 calculate_temporal_median_sr0_tr1_sse4
#define MEDIANPROC_SR0_T2 calculate_temporal_median_sr0_tr2_sse4
#endif
#ifdef MEDIANPROCESSOR_C
#define MEDIANPROC MedianProcessor_c
#define MEDIANPROC_SR0_T1 calculate_temporal_median_sr0_tr1_c
#define MEDIANPROC_SR0_T2 calculate_temporal_median_sr0_tr2_c
#endif

#ifdef MEDIANPROCESSOR_C
template< typename T, int histogram_resolution_bits, InstructionSet instruction_set >
MB_FORCEINLINE void MEDIANPROC<T, histogram_resolution_bits, instruction_set >::add_16_bins_coarse(T* a, const T* b) {
  for (int i = 0; i < coarse_histogram_size; ++i) {
    a[i] += b[i];
  }
}

template< typename T, int histogram_resolution_bits, InstructionSet instruction_set >
MB_FORCEINLINE void MEDIANPROC<T, histogram_resolution_bits, instruction_set >::add_16_bins(T* a, const T* b) {
  for (int i = 0; i < histogram_refining_part_size; ++i) {
    a[i] += b[i];
    }
  }

template< typename T, int histogram_resolution_bits, InstructionSet instruction_set >
MB_FORCEINLINE void MEDIANPROC<T, histogram_resolution_bits, instruction_set >::sub_16_bins_coarse(T* a, const T* b) {
  for (int i = 0; i < coarse_histogram_size; ++i) {
    a[i] -= b[i];
  }
}

template< typename T, int histogram_resolution_bits, InstructionSet instruction_set >
MB_FORCEINLINE void MEDIANPROC<T, histogram_resolution_bits, instruction_set >::sub_16_bins(T* a, const T* b) {
  for (int i = 0; i < histogram_refining_part_size; ++i) {
    a[i] -= b[i];
  }
}

template< typename T, int histogram_resolution_bits, InstructionSet instruction_set >
MB_FORCEINLINE void MEDIANPROC<T, histogram_resolution_bits, instruction_set >::zero_single_bin(T* a) {
  for (int i = 0; i < histogram_refining_part_size; ++i) {
    a[i] = 0;
  }
}
#endif

#ifdef MEDIANPROCESSOR_SSE2
template< typename T, int histogram_resolution_bits, InstructionSet instruction_set >
MB_FORCEINLINE void MEDIANPROC<T, histogram_resolution_bits, instruction_set >::add_16_bins_coarse(T* a, const T* b) {
  for (int i = 0; i < sizeof(T) * coarse_histogram_size / 16; ++i) {
    __m128i aval = _mm_load_si128(reinterpret_cast<const __m128i*>(a) + i);
    __m128i bval = _mm_load_si128(reinterpret_cast<const __m128i*>(b) + i);
    __m128i sum = simd_adds<T>(aval, bval);
    _mm_store_si128(reinterpret_cast<__m128i*>(a) + i, sum);
    }
  }

template< typename T, int histogram_resolution_bits, InstructionSet instruction_set >
MB_FORCEINLINE void MEDIANPROC<T, histogram_resolution_bits, instruction_set >::add_16_bins(T* a, const T* b) {
  for (int i = 0; i < sizeof(T) * histogram_refining_part_size / 16; ++i) {
    __m128i aval = _mm_load_si128(reinterpret_cast<const __m128i*>(a) + i);
    __m128i bval = _mm_load_si128(reinterpret_cast<const __m128i*>(b) + i);
    __m128i sum = simd_adds<T>(aval, bval);
    _mm_store_si128(reinterpret_cast<__m128i*>(a) + i, sum);
  }
}

template< typename T, int histogram_resolution_bits, InstructionSet instruction_set >
MB_FORCEINLINE void MEDIANPROC<T, histogram_resolution_bits, instruction_set >::sub_16_bins_coarse(T* a, const T* b) {
  for (int i = 0; i < sizeof(T) * coarse_histogram_size / 16; ++i) {
    __m128i aval = _mm_load_si128(reinterpret_cast<const __m128i*>(a) + i);
    __m128i bval = _mm_load_si128(reinterpret_cast<const __m128i*>(b) + i);
    __m128i sum = simd_subs<T>(aval, bval);
    _mm_store_si128(reinterpret_cast<__m128i*>(a) + i, sum);
  }
}

template< typename T, int histogram_resolution_bits, InstructionSet instruction_set >
MB_FORCEINLINE void MEDIANPROC<T, histogram_resolution_bits, instruction_set >::sub_16_bins(T* a, const T* b) {
  for (int i = 0; i < sizeof(T) * histogram_refining_part_size / 16; ++i) {
    __m128i aval = _mm_load_si128(reinterpret_cast<const __m128i*>(a) + i);
    __m128i bval = _mm_load_si128(reinterpret_cast<const __m128i*>(b) + i);
    __m128i sum = simd_subs<T>(aval, bval);
    _mm_store_si128(reinterpret_cast<__m128i*>(a) + i, sum);
  }
}

template< typename T, int histogram_resolution_bits, InstructionSet instruction_set >
MB_FORCEINLINE void MEDIANPROC<T, histogram_resolution_bits, instruction_set >::zero_single_bin(T* a) {
  __m128i zero = _mm_setzero_si128();
  for (int i = 0; i < sizeof(T) * histogram_refining_part_size / 16; ++i) {
    _mm_store_si128(reinterpret_cast<__m128i*>(a) + i, zero);
  }
}
#endif

#ifdef MEDIANPROCESSOR_AVX2
template< typename T, int histogram_resolution_bits, InstructionSet instruction_set >
#ifdef MEDIANPROCESSOR_AVX2
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("fma,avx2")))
#endif
#endif
MB_FORCEINLINE void MEDIANPROC<T, histogram_resolution_bits, instruction_set >::add_16_bins_coarse(T* a, const T* b) {
  for (int i = 0; i < sizeof(T) * coarse_histogram_size / 32; ++i) {
    auto aval = _mm256_load_si256(reinterpret_cast<const __m256i*>(a) + i);
    auto bval = _mm256_load_si256(reinterpret_cast<const __m256i*>(b) + i);
    auto sum = simd_adds<T>(aval, bval);
    _mm256_store_si256(reinterpret_cast<__m256i*>(a) + i, sum);
  }
}

template< typename T, int histogram_resolution_bits, InstructionSet instruction_set >
#ifdef MEDIANPROCESSOR_AVX2
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("fma,avx2")))
#endif
#endif
MB_FORCEINLINE void MEDIANPROC<T, histogram_resolution_bits, instruction_set >::add_16_bins(T* a, const T* b) {
  for (int i = 0; i < sizeof(T) * histogram_refining_part_size / 32; ++i) {
    auto aval = _mm256_load_si256(reinterpret_cast<const __m256i*>(a) + i);
    auto bval = _mm256_load_si256(reinterpret_cast<const __m256i*>(b) + i);
    auto sum = simd_adds<T>(aval, bval);
    _mm256_store_si256(reinterpret_cast<__m256i*>(a) + i, sum);
  }
}

template< typename T, int histogram_resolution_bits, InstructionSet instruction_set >
#ifdef MEDIANPROCESSOR_AVX2
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("fma,avx2")))
#endif
#endif
MB_FORCEINLINE void MEDIANPROC<T, histogram_resolution_bits, instruction_set >::sub_16_bins_coarse(T* a, const T* b) {
  for (int i = 0; i < sizeof(T) * coarse_histogram_size / 32; ++i) {
    auto aval = _mm256_load_si256(reinterpret_cast<const __m256i*>(a) + i);
    auto bval = _mm256_load_si256(reinterpret_cast<const __m256i*>(b) + i);
    auto sum = simd_subs<T>(aval, bval);
    _mm256_store_si256(reinterpret_cast<__m256i*>(a) + i, sum);
  }
}

template< typename T, int histogram_resolution_bits, InstructionSet instruction_set >
#ifdef MEDIANPROCESSOR_AVX2
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("fma,avx2")))
#endif
#endif
MB_FORCEINLINE void MEDIANPROC<T, histogram_resolution_bits, instruction_set >::sub_16_bins(T* a, const T* b) {
  for (int i = 0; i < sizeof(T) * histogram_refining_part_size / 32; ++i) {
    auto aval = _mm256_load_si256(reinterpret_cast<const __m256i*>(a) + i);
    auto bval = _mm256_load_si256(reinterpret_cast<const __m256i*>(b) + i);
    auto sum = simd_subs<T>(aval, bval);
    _mm256_store_si256(reinterpret_cast<__m256i*>(a) + i, sum);
  }
}

template< typename T, int histogram_resolution_bits, InstructionSet instruction_set >
#ifdef MEDIANPROCESSOR_AVX2
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("fma,avx2")))
#endif
#endif
MB_FORCEINLINE void MEDIANPROC<T, histogram_resolution_bits, instruction_set >::zero_single_bin(T* a) {
  auto zero = _mm256_setzero_si256();
  for (int i = 0; i < sizeof(T) * histogram_refining_part_size / 32; ++i) {
    _mm256_store_si256(reinterpret_cast<__m256i*>(a) + i, zero);
  }
}
#endif

template< typename T, int histogram_resolution_bits, InstructionSet instruction_set >
template<typename pixel_t, int bits_per_pixel, bool chroma>
#ifdef MEDIANPROCESSOR_AVX2
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("fma,avx2")))
#endif
#endif
#ifdef MEDIANPROCESSOR_SSE2
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse2")))
#endif
#endif
MB_FORCEINLINE void MEDIANPROC<T, histogram_resolution_bits, instruction_set>::process_line(uint8_t* dstp, Histogram* histograms, ColumnPair* fine_columns, Histogram* current_hist, int radius, int temporal_radius, int width, int current_length_y) {
  memset(current_hist, 0, HISTOGRAM_SIZE);  // also nullifies fine 
  memset(fine_columns, -1, sizeof(ColumnPair) * coarse_histogram_size);

  //init histogram of the leftmost column
  for (int x = 0; x < radius; ++x) {
    add_16_bins_coarse(current_hist->coarse, histograms[x].coarse);
  }

  for (int x = 0; x < width; ++x) {
    //add one column to the right
    if (x < (width - radius)) {
      add_16_bins_coarse(current_hist->coarse, histograms[x + radius].coarse);
    }

    int current_length_x = calculate_window_side_length(radius, x, width);
    int start_x = std::max(0, x - radius);
    int end_x = start_x + current_length_x;
    int half_elements = (current_length_y * current_length_x * temporal_radius + 1) / 2;

    //finding median on coarse level
    int count = 0;
    int coarse_idx = -1;
    while (count < half_elements) {
      count += current_hist->coarse[++coarse_idx];
    }
    count -= current_hist->coarse[coarse_idx];

#ifdef _DEBUG
    assert(coarse_idx < coarse_histogram_size);
#endif

    int fine_offset = coarse_idx * histogram_refining_part_size; // << 4

    //partially updating fine histogram
    if (fine_columns[coarse_idx].end < start_x) {
      //any data we gathered is useless, drop it and build the whole block from scratch
      zero_single_bin(current_hist->fine + fine_offset);

      for (int i = start_x; i < end_x; ++i) {
        add_16_bins(current_hist->fine + fine_offset, histograms[i].fine + fine_offset);
      }
    }
    else {
      int i = fine_columns[coarse_idx].start;
      while (i < start_x) {
        sub_16_bins(current_hist->fine + fine_offset, histograms[i++].fine + fine_offset);
      }
      i = fine_columns[coarse_idx].end;
      while (++i < end_x) {
        add_16_bins(current_hist->fine + fine_offset, histograms[i].fine + fine_offset);
      }
    }
    fine_columns[coarse_idx].start = start_x;
    fine_columns[coarse_idx].end = end_x - 1;

    //finding median on fine level
    int fine_idx = fine_offset - 1;
    while (count < half_elements) {
      count += current_hist->fine[++fine_idx];
    }

    if constexpr (bits_per_pixel <= 16)
    {
#ifdef _DEBUG
      assert(fine_idx < (1 << bits_per_pixel));
#endif
      reinterpret_cast<pixel_t*>(dstp)[x] = (pixel_t)fine_idx;
    }
    else {
      // Weird. Histogram and median on float :) Anyway we are doing in 16 bit granurality
      float fine_idx_f;
      if constexpr (chroma)
        fine_idx_f = (fine_idx - 32768) / 65534.0f; // yes, 65534, even
      else
        fine_idx_f = fine_idx / 65535.0f;
      reinterpret_cast<pixel_t*>(dstp)[x] = fine_idx_f;
    }

    //subtract leftmost histogram
    if (x >= radius) {
      sub_16_bins_coarse(current_hist->coarse, histograms[x - radius].coarse);
    }
  }
}

template< typename T, int histogram_resolution_bits, InstructionSet instruction_set >
template<typename pixel_t, int bits_per_pixel, bool chroma>
void MEDIANPROC<T, histogram_resolution_bits, instruction_set>::calculate_median(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height, int radius, void* buffer) {
  Histogram* histograms = reinterpret_cast<Histogram*>(buffer);

  // these are be too big for the stack
  // __declspec(align(16)) Histogram current_hist;
  // ColumnPair fine_columns[coarse_histogram_size]; //indexes of last and first column in every fine histogram segment

  static constexpr int ALIGN = 32;
  auto mem1 = std::make_unique<uint8_t[]>(sizeof(Histogram) + ALIGN - 1);
  auto current_hist = (Histogram*)(((uintptr_t)mem1.get() + ALIGN - 1) & ~(uintptr_t)(ALIGN - 1));

  auto mem2 = std::make_unique<uint8_t[]>(sizeof(ColumnPair) * coarse_histogram_size + ALIGN - 1);
  auto fine_columns = (ColumnPair*)(((uintptr_t)mem2.get() + ALIGN - 1) & ~(uintptr_t)(ALIGN - 1));

  memset(histograms, 0, width * HISTOGRAM_SIZE);

  constexpr int max_pixel_value = sizeof(pixel_t) == 4 ? 65535 : (1 << bits_per_pixel) - 1;

  //init histograms
  for (int y = 0; y < radius + 1; ++y) {
    for (int x = 0; x < width; ++x) {
      int new_element;
      if constexpr (bits_per_pixel == 8) {
        new_element = srcp[y * src_pitch + x];
      }
      else if constexpr (bits_per_pixel <= 16) {
        new_element = *(reinterpret_cast<const uint16_t*>(srcp + y * src_pitch) + x);
        new_element = std::min(new_element, max_pixel_value);
      }
      else {
        // 32 bit float: 0..1 luma, -0.5..-0.5 chroma range to 0..65535
        float new_element_f = *(reinterpret_cast<const float*>(srcp + y * src_pitch) + x);
        if constexpr (chroma)
          new_element = (int)((new_element_f + 0.5f) * 65534.0f + 0.5f); // mul with even, keep center
        else
          new_element = (int)(new_element_f * 65535.0f + 0.5f);
        new_element = std::max(0, std::min(new_element, 65535));
      }
      histograms[x].coarse[new_element >> shift_to_coarse]++;
      histograms[x].fine[new_element]++;
    }
  }

  for (int y = 0; y < height; ++y) {
    int current_length_y = calculate_window_side_length(radius, y, height);

    process_line<pixel_t, bits_per_pixel, chroma>(dstp, histograms, fine_columns, current_hist, radius, 1, width, current_length_y);

    //updating column histograms
    if (y >= radius) {
      for (int x = 0; x < width; ++x) {
        int old_element;
        if constexpr (bits_per_pixel == 8) {
          old_element = srcp[(y - radius) * src_pitch + x];
        }
        else if constexpr (bits_per_pixel <= 16) {
          old_element = *(reinterpret_cast<const uint16_t*>(srcp + (y - radius) * src_pitch) + x);
          old_element = std::min(old_element, max_pixel_value);
        }
        else {
          // 32 bit float: 0..1 luma, -0.5..-0.5 chroma range to 0..65535
          float old_element_f = *(reinterpret_cast<const float*>(srcp + (y - radius) * src_pitch) + x);
          if constexpr (chroma)
            old_element = (int)((old_element_f + 0.5f) * 65534.0f + 0.5f); // mul with even, keep center
          else
            old_element = (int)(old_element_f * 65535.0f + 0.5f);
          old_element = std::max(0, std::min(old_element, 65535));
        }
        histograms[x].coarse[old_element >> shift_to_coarse]--;
        histograms[x].fine[old_element]--;
      }
    }
    if (y < height - radius - 1) {
      for (int x = 0; x < width; ++x) {
        int new_element;
        if constexpr (bits_per_pixel == 8) {
          new_element = srcp[(y + radius + 1) * src_pitch + x];
        }
        else if constexpr (bits_per_pixel <= 16) {
          new_element = *(reinterpret_cast<const uint16_t*>(srcp + (y + radius + 1) * src_pitch) + x);
          new_element = std::min(new_element, max_pixel_value);
        }
        else {
          // 32 bit float: 0..1 luma, -0.5..-0.5 chroma range to 0..65535
          float new_element_f = *(reinterpret_cast<const float*>(srcp + (y + radius + 1) * src_pitch) + x);
          if constexpr (chroma)
            new_element = (int)((new_element_f + 0.5f) * 65534.0f + 0.5f); // mul with even, keep center
          else
            new_element = (int)(new_element_f * 65535.0f + 0.5f);
          new_element = std::max(0, std::min(new_element, 65535));
        }

        histograms[x].coarse[new_element >> shift_to_coarse]++;
        histograms[x].fine[new_element]++;
      }
    }
    dstp += dst_pitch;
  }
#ifdef MEDIANPROCESSOR_AVX2
  _mm256_zeroupper();
#endif
}

template< typename T, int histogram_resolution_bits, InstructionSet instruction_set >
template<typename pixel_t, int bits_per_pixel, bool chroma>
void MEDIANPROC<T, histogram_resolution_bits, instruction_set>::calculate_temporal_median(uint8_t* dstp, int dst_pitch, const uint8_t** src_ptrs, const int* src_pitches, int frames_count, int width, int height, int radius, void* buffer) {
  Histogram* histograms = reinterpret_cast<Histogram*>(buffer);

  // these can be too big for the stack
  // __declspec(align(16)) Histogram current_hist;
  // ColumnPair fine_columns[coarse_histogram_size]; //indexes of last and first column in every fine histogram segment

  static constexpr int ALIGN = 32;
  auto mem1 = std::make_unique<uint8_t[]>(sizeof(Histogram) + ALIGN - 1);
  auto current_hist = (Histogram*)(((uintptr_t)mem1.get() + ALIGN - 1) & ~(uintptr_t)(ALIGN - 1));

  auto mem2 = std::make_unique<uint8_t[]>(sizeof(ColumnPair) * coarse_histogram_size + ALIGN - 1);
  auto fine_columns = (ColumnPair*)(((uintptr_t)mem2.get() + ALIGN - 1) & ~(uintptr_t)(ALIGN - 1));

  memset(histograms, 0, width * HISTOGRAM_SIZE);

  constexpr int max_pixel_value = sizeof(pixel_t) == 4 ? 65535 : (1 << bits_per_pixel) - 1;

  //init histograms
  for (int i = 0; i < frames_count; ++i) {
    auto srcp = src_ptrs[i];
    auto pitch = src_pitches[i];
    for (int y = 0; y < radius + 1; ++y) {
      for (int x = 0; x < width; ++x) {
        int new_element;
        if constexpr (bits_per_pixel == 8) {
          new_element = srcp[x];
        }
        else if constexpr (bits_per_pixel <= 16) {
          new_element = *(reinterpret_cast<const uint16_t*>(srcp) + x);
          new_element = std::min(new_element, max_pixel_value); // guard histogram buffer
        }
        else {
          // 32 bit float: 0..1 luma, -0.5..-0.5 chroma range to 0..65535
          float new_element_f = *(reinterpret_cast<const float*>(srcp) + x);
          if constexpr (chroma)
            new_element = (int)((new_element_f + 0.5f) * 65534.0f + 0.5f); // mul with even, keep center
          else
            new_element = (int)(new_element_f * 65535.0f + 0.5f);
          new_element = std::max(0, std::min(new_element, 65535));
        }
        histograms[x].coarse[new_element >> shift_to_coarse]++;
        histograms[x].fine[new_element]++;
      }
      srcp += pitch;
    }
  }

  for (int y = 0; y < height; ++y) {
    int current_length_y = calculate_window_side_length(radius, y, height);

    process_line<pixel_t, bits_per_pixel, chroma>(dstp, histograms, fine_columns, current_hist, radius, frames_count, width, current_length_y);

    //updating column histograms
    if (y >= radius) {
      for (int x = 0; x < width; ++x) {
        for (int i = 0; i < frames_count; ++i) {
          int old_element;
          if constexpr (bits_per_pixel == 8) {
            old_element = src_ptrs[i][(y - radius) * src_pitches[i] + x];
          }
          else if constexpr (bits_per_pixel <= 16) {
            old_element = *(reinterpret_cast<const uint16_t*>(src_ptrs[i] + (y - radius) * src_pitches[i]) + x);
            old_element = std::min(old_element, max_pixel_value);
          }
          else {
            // 32 bit float: 0..1 luma, -0.5..-0.5 chroma range to 0..65535
            float old_element_f = *(reinterpret_cast<const float*>(src_ptrs[i] + (y - radius) * src_pitches[i]) + x);
            if constexpr (chroma)
              old_element = (int)((old_element_f + 0.5f) * 65534.0f + 0.5f); // mul with even, keep center
            else
              old_element = (int)(old_element_f * 65535.0f + 0.5f);
            old_element = std::max(0, std::min(old_element, 65535));
          }
          histograms[x].coarse[old_element >> shift_to_coarse]--;
          histograms[x].fine[old_element]--;
        }
      }
    }
    if (y < height - radius - 1) {
      for (int x = 0; x < width; ++x) {
        for (int i = 0; i < frames_count; ++i) {
          int new_element;
          if constexpr (bits_per_pixel == 8) {
            new_element = src_ptrs[i][(y + radius + 1) * src_pitches[i] + x];
          }
          else if constexpr (bits_per_pixel <= 16) {
            new_element = *(reinterpret_cast<const uint16_t*>(src_ptrs[i] + (y + radius + 1) * src_pitches[i]) + x);
            new_element = std::min(new_element, max_pixel_value);
          }
          else {
            // 32 bit float: 0..1 luma, -0.5..-0.5 chroma range to 0..65535
            float new_element_f = *(reinterpret_cast<const float*>(src_ptrs[i] + (y + radius + 1) * src_pitches[i]) + x);
            if constexpr (chroma)
              new_element = (int)((new_element_f + 0.5f) * 65534.0f + 0.5f); // mul with even, keep center
            else
              new_element = (int)(new_element_f * 65535.0f + 0.5f);
            new_element = std::max(0, std::min(new_element, 65535));
          }
          histograms[x].coarse[new_element >> shift_to_coarse]++;
          histograms[x].fine[new_element]++;
        }
      }
    }

    dstp += dst_pitch;
  }
#ifdef MEDIANPROCESSOR_AVX2
  _mm256_zeroupper();
#endif
}

// special case for spatial radius 0 and temporal radius 1 (frame_count==3)
template<typename pixel_t>
#ifdef MEDIANPROCESSOR_SSE2
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
#endif
void MEDIANPROC_SR0_T1(uint8_t* dstp, int dst_pitch, const uint8_t** src_ptrs, const int* src_pitches, int frames_count, int width, int height, int radius, void* buffer) {
  auto srcp0 = src_ptrs[0];
  auto srcp1 = src_ptrs[1];
  auto srcp2 = src_ptrs[2];
  auto pitch0 = src_pitches[0];
  auto pitch1 = src_pitches[1];
  auto pitch2 = src_pitches[2];
  for (int y = 0; y < height; ++y) {
#ifdef MEDIANPROCESSOR_C
    for (int x = 0; x < width; ++x) {
      auto A = reinterpret_cast<const pixel_t*>(srcp0)[x];
      auto B = reinterpret_cast<const pixel_t*>(srcp1)[x];
      auto C = reinterpret_cast<const pixel_t*>(srcp2)[x];
      auto t1 = std::min(A, B);
      auto t2 = std::max(A, B);
      auto t3 = std::min(t2, C);
      auto median = std::max(t1, t3);
      reinterpret_cast<pixel_t*>(dstp)[x] = median;
    }
#endif
#ifdef MEDIANPROCESSOR_SSE2
    for (int x = 0; x < width * sizeof(pixel_t); x += 16) {
      auto A = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp0 + x));
      auto B = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp1 + x));
      auto C = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp2 + x));
      auto median = simd_median<pixel_t>(A, B, C);
      _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), median);
    }
#endif
#ifdef MEDIANPROCESSOR_AVX2
    for (int x = 0; x < width * sizeof(pixel_t); x += 32) {
      auto A = _mm256_load_si256(reinterpret_cast<const __m256i*>(srcp0 + x));
      auto B = _mm256_load_si256(reinterpret_cast<const __m256i*>(srcp1 + x));
      auto C = _mm256_load_si256(reinterpret_cast<const __m256i*>(srcp2 + x));
      //auto A = _mm256_loadu2_m128i(reinterpret_cast<const __m128i*>(srcp0 + x + 16), reinterpret_cast<const __m128i*>(srcp0 + x + 0));
      //auto B = _mm256_loadu2_m128i(reinterpret_cast<const __m128i*>(srcp1 + x + 16), reinterpret_cast<const __m128i*>(srcp1 + x + 0));
      //auto C = _mm256_loadu2_m128i(reinterpret_cast<const __m128i*>(srcp2 + x + 16), reinterpret_cast<const __m128i*>(srcp2 + x + 0));
      auto median = simd_median<pixel_t>(A, B, C);
      _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), _mm256_castsi256_si128(median));
      _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x + 16), _mm256_extracti128_si256(median,1));
    }
#endif
    srcp0 += pitch0;
    srcp1 += pitch1;
    srcp2 += pitch2;
    dstp += dst_pitch;
  }

#ifdef MEDIANPROCESSOR_AVX2
  _mm256_zeroupper();
#endif
}

// special case for spatial radius 0 and temporal radius 2 (frame_count==5)
template<typename pixel_t>
#ifdef MEDIANPROCESSOR_SSE2
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
#endif
void MEDIANPROC_SR0_T2(uint8_t* dstp, int dst_pitch, const uint8_t** src_ptrs, const int* src_pitches, int frames_count, int width, int height, int radius, void* buffer) {
  auto srcp0 = src_ptrs[0];
  auto srcp1 = src_ptrs[1];
  auto srcp2 = src_ptrs[2];
  auto srcp3 = src_ptrs[3];
  auto srcp4 = src_ptrs[4];
  auto pitch0 = src_pitches[0];
  auto pitch1 = src_pitches[1];
  auto pitch2 = src_pitches[2];
  auto pitch3 = src_pitches[3];
  auto pitch4 = src_pitches[4];
  for (int y = 0; y < height; ++y) {
#ifdef MEDIANPROCESSOR_C
    for (int x = 0; x < width; ++x) {
      auto A = reinterpret_cast<const pixel_t*>(srcp0)[x];
      auto B = reinterpret_cast<const pixel_t*>(srcp1)[x];
      auto C = reinterpret_cast<const pixel_t*>(srcp2)[x];
      auto D = reinterpret_cast<const pixel_t*>(srcp3)[x];
      auto E = reinterpret_cast<const pixel_t*>(srcp4)[x];

      auto f = std::max(std::min(A, B), std::min(C, D)); // discards lowest from first 4
      auto g = std::min(std::max(A, B), std::max(C, D)); // discards biggest from first 4

      auto t1 = std::min(f, g);
      auto t2 = std::max(f, g);
      auto t3 = std::min(t2, E);
      auto median = std::max(t1, t3);
      reinterpret_cast<pixel_t*>(dstp)[x] = median;
    }
#endif
#ifdef MEDIANPROCESSOR_SSE2
    for (int x = 0; x < width * sizeof(pixel_t); x += 16) {
      auto A = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp0 + x));
      auto B = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp1 + x));
      auto C = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp2 + x));
      auto D = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp3 + x));
      auto E = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp4 + x));
      auto median = simd_median5<pixel_t>(A, B, C, D, E);
      _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), median);
    }
#endif
#ifdef MEDIANPROCESSOR_AVX2
    for (int x = 0; x < width * sizeof(pixel_t); x += 32) {
      auto A = _mm256_load_si256(reinterpret_cast<const __m256i*>(srcp0 + x));
      auto B = _mm256_load_si256(reinterpret_cast<const __m256i*>(srcp1 + x));
      auto C = _mm256_load_si256(reinterpret_cast<const __m256i*>(srcp2 + x));
      auto D = _mm256_load_si256(reinterpret_cast<const __m256i*>(srcp3 + x));
      auto E = _mm256_load_si256(reinterpret_cast<const __m256i*>(srcp4 + x));
      //auto A = _mm256_loadu2_m128i(reinterpret_cast<const __m128i*>(srcp0 + x + 16), reinterpret_cast<const __m128i*>(srcp0 + x + 0));
      //auto B = _mm256_loadu2_m128i(reinterpret_cast<const __m128i*>(srcp1 + x + 16), reinterpret_cast<const __m128i*>(srcp1 + x + 0));
      //auto C = _mm256_loadu2_m128i(reinterpret_cast<const __m128i*>(srcp2 + x + 16), reinterpret_cast<const __m128i*>(srcp2 + x + 0));
      auto median = simd_median5<pixel_t>(A, B, C, D, E);
      _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), _mm256_castsi256_si128(median));
      _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x + 16), _mm256_extracti128_si256(median, 1));
    }
#endif
    srcp0 += pitch0;
    srcp1 += pitch1;
    srcp2 += pitch2;
    srcp3 += pitch3;
    srcp4 += pitch4;
    dstp += dst_pitch;
  }

#ifdef MEDIANPROCESSOR_AVX2
  _mm256_zeroupper();
#endif
}


template<class V>
inline V median(const V& a, const V& b, const V& c)
{
  return max(min(a, b), min(c, max(a, b)));
}

template<class V>
inline V median(const V& a, const V& b, const V& c, const V& d, const V& e)
{
  V f = max(min(a, b), min(c, d)); // discards lowest from first 4
  V g = min(max(a, b), max(c, d)); // discards biggest from first 4
  return median(e, f, g);
}
