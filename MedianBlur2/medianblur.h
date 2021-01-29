#include <stdint.h>
#include <memory>

/*
medianblur.h

included either from C, SSE or AVX2
as
MEDIANPROCESSOR_C
MEDIANPROCESSOR_SSE2
MEDIANPROCESSOR_AVX2
defined
*/


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

#ifdef MEDIANPROCESSOR_SSE2
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse2")))
#endif
#endif
#ifdef MEDIANPROCESSOR_AVX2
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2, fma3")))
#endif
#endif
template<typename T, int histogram_resolution_bits, InstructionSet instruction_set>
#ifdef MEDIANPROCESSOR_AVX2
class MedianProcessor_avx2
#endif
#ifdef MEDIANPROCESSOR_SSE2
  class MedianProcessor_sse2
#endif
#ifdef MEDIANPROCESSOR_C
  class MedianProcessor_c
#endif
{
  static constexpr int histogram_size = 1 << histogram_resolution_bits;
  static constexpr int coarse_histogram_resolution_bits = histogram_resolution_bits / 2; // tested, optimum is at half
  static constexpr int shift_to_coarse = histogram_resolution_bits - coarse_histogram_resolution_bits;
  static constexpr int coarse_histogram_size = 1 << coarse_histogram_resolution_bits;
  static constexpr int histogram_refining_part_size = 1 << shift_to_coarse;

  // histogram_resolution_bits|histogram_size|coarse_histogram_resolution_bits|shift_to_coarse|coarse_histogram_size|histogram_refining_part_size
  //          8                      256                        4                       4              16                     16
  //         10                     1024                        5                       5              32                     32
  // etc..
  //         16                    65536                        8                       8             256                    256

  typedef struct {
    T coarse[coarse_histogram_size];
    T fine[histogram_size];
  } Histogram;

  typedef struct {
    int start;
    int end;
  } ColumnPair;

  static MB_FORCEINLINE void add_16_bins_coarse(T* a, const T* b);
  static MB_FORCEINLINE void add_16_bins(T* a, const T* b);
  static MB_FORCEINLINE void sub_16_bins_coarse(T* a, const T* b);
  static MB_FORCEINLINE void sub_16_bins(T* a, const T* b);
  static MB_FORCEINLINE void zero_single_bin(T* a);

  template<typename pixel_t, int bits_per_pixel, bool chroma>
  static MB_FORCEINLINE void process_line(uint8_t* dstp, Histogram* histograms, ColumnPair* fine_columns, Histogram* current_hist, int radius, int temporal_radius, int width, int current_length_y);

public:
  static constexpr int HISTOGRAM_SIZE = sizeof(Histogram);

  template<typename pixel_t, int bits_per_pixel, bool chroma>
  static void calculate_median(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height, int radius, void* buffer);

  template<typename pixel_t, int bits_per_pixel, bool chroma>
  static void calculate_temporal_median(uint8_t* dstp, int dst_pitch, const uint8_t** src_ptrs, const int* src_pitches, int frames_count, int width, int height, int radius, void* buffer);
};
