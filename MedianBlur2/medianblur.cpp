#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include "avisynth.h"
#include <math.h>
#include <assert.h>
#include <stdint.h>
#include <algorithm>
#include <emmintrin.h>
#include "special.h"
#include <memory>
#include <functional>

enum class InstructionSet
{
    SSE2,
    PLAIN_C
};

static MB_FORCEINLINE int calculate_window_side_length(int radius, int x, int width) {
    int length = radius + 1;
    if (x <= radius) {
        length += x;
    } else if (x >= width-radius) {
        length += width-x-1;
    } else {
        length += radius;
    }
    return length;
}

template<typename T>
static MB_FORCEINLINE __m128i simd_adds(const __m128i &a, const __m128i &b) {
    assert(false);
}

template<>
MB_FORCEINLINE __m128i simd_adds<uint8_t>(const __m128i &a, const __m128i &b) {
    return _mm_adds_epu8(a, b);
}

template<>
MB_FORCEINLINE __m128i simd_adds<uint16_t>(const __m128i &a, const __m128i &b) {
    return _mm_adds_epu16(a, b);
}

template<>
MB_FORCEINLINE __m128i simd_adds<int32_t>(const __m128i &a, const __m128i &b) {
    return _mm_add_epi32(a, b);
}

template<typename T>
MB_FORCEINLINE __m128i simd_subs(const __m128i &a, const __m128i &b) {
    assert(false);
}

template<>
MB_FORCEINLINE __m128i simd_subs<uint8_t>(const __m128i &a, const __m128i &b) {
    return _mm_subs_epu8(a, b);
}

template<>
MB_FORCEINLINE __m128i simd_subs<uint16_t>(const __m128i &a, const __m128i &b) {
    return _mm_subs_epu16(a, b);
}

template<>
MB_FORCEINLINE __m128i simd_subs<int32_t>(const __m128i &a, const __m128i &b) {
    return _mm_sub_epi32(a, b);
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

*/

template<typename T, int histogram_resolution_bits, InstructionSet instruction_set>
class MedianProcessor
{
  static constexpr int histogram_size = 1 << histogram_resolution_bits;
  static constexpr int coarse_histogram_resolution_bits = histogram_resolution_bits / 2; // tested, optimum is at half
  static constexpr int shift_to_coarse = histogram_resolution_bits - coarse_histogram_resolution_bits;
  static constexpr int coarse_histogram_size = 1 << coarse_histogram_resolution_bits;
  static constexpr int histogram_refining_part_size = 1 << shift_to_coarse;

    typedef struct {
        T coarse[coarse_histogram_size];
        T fine[histogram_size];
    } Histogram;

    typedef struct {
        int start;
        int end;
    } ColumnPair;

    static MB_FORCEINLINE void add_16_bins_coarse_c(T* a, const T* b) {
      for (int i = 0; i < coarse_histogram_size; ++i) {
        a[i] += b[i];
      }
    }

    static MB_FORCEINLINE void add_16_bins_c(T* a, const T *b) {
        for (int i = 0; i < histogram_refining_part_size; ++i) {
            a[i] += b[i];
        }
    }

    static MB_FORCEINLINE void sub_16_bins_coarse_c(T* a, const T* b) {
      for (int i = 0; i < coarse_histogram_size; ++i) {
        a[i] -= b[i];
      }
    }

    static MB_FORCEINLINE void sub_16_bins_c(T* a, const T *b) {
        for (int i = 0; i < histogram_refining_part_size; ++i) {
            a[i] -= b[i];
        }
    }

    static MB_FORCEINLINE void zero_single_bin_c(T* a) {
        for (int i = 0; i < histogram_refining_part_size; ++i) {
            a[i] = 0;
        }
    }

    static MB_FORCEINLINE void add_16_bins_coarse_sse2(T* a, const T* b) {
      for (int i = 0; i < sizeof(T) * coarse_histogram_size/ 16; ++i) {
        __m128i aval = _mm_load_si128(reinterpret_cast<const __m128i*>(a) + i);
        __m128i bval = _mm_load_si128(reinterpret_cast<const __m128i*>(b) + i);
        __m128i sum = simd_adds<T>(aval, bval);
        _mm_store_si128(reinterpret_cast<__m128i*>(a) + i, sum);
      }
    }

    static MB_FORCEINLINE void add_16_bins_sse2(T* a, const T *b) {
        for (int i = 0; i < sizeof(T) * histogram_refining_part_size / 16; ++i) {
            __m128i aval = _mm_load_si128(reinterpret_cast<const __m128i*>(a)+i);
            __m128i bval = _mm_load_si128(reinterpret_cast<const __m128i*>(b)+i);
            __m128i sum = simd_adds<T>(aval, bval);
            _mm_store_si128(reinterpret_cast<__m128i*>(a)+i, sum);
        }
    }

    static MB_FORCEINLINE void sub_16_bins_coarse_sse2(T* a, const T *b) {
        for (int i = 0; i < sizeof(T) * coarse_histogram_size / 16; ++i) {
            __m128i aval = _mm_load_si128(reinterpret_cast<const __m128i*>(a)+i);
            __m128i bval = _mm_load_si128(reinterpret_cast<const __m128i*>(b)+i);
            __m128i sum = simd_subs<T>(aval, bval);
            _mm_store_si128(reinterpret_cast<__m128i*>(a)+i, sum);
        }
    }

    static MB_FORCEINLINE void sub_16_bins_sse2(T* a, const T* b) {
      for (int i = 0; i < sizeof(T) * histogram_refining_part_size / 16; ++i) {
        __m128i aval = _mm_load_si128(reinterpret_cast<const __m128i*>(a) + i);
        __m128i bval = _mm_load_si128(reinterpret_cast<const __m128i*>(b) + i);
        __m128i sum = simd_subs<T>(aval, bval);
        _mm_store_si128(reinterpret_cast<__m128i*>(a) + i, sum);
      }
    }

    static MB_FORCEINLINE void zero_single_bin_sse2(T* a) {
        __m128i zero = _mm_setzero_si128();
        for (int i = 0; i < sizeof(T) * histogram_refining_part_size / 16; ++i) {
            _mm_store_si128(reinterpret_cast<__m128i*>(a)+i, zero);
        }
    }

    static MB_FORCEINLINE void add_16_bins_coarse(T* a, const T* b) {
      if constexpr (instruction_set == InstructionSet::SSE2) {
        add_16_bins_coarse_sse2(a, b);
      }
      else {
        add_16_bins_coarse_c(a, b);
      }
    }

    static MB_FORCEINLINE void add_16_bins(T* a, const T *b) {
        if constexpr (instruction_set == InstructionSet::SSE2) {
            add_16_bins_sse2(a, b);
        } else {
            add_16_bins_c(a, b);
        }
    }

    static MB_FORCEINLINE void sub_16_bins_coarse(T* a, const T *b) {
        if constexpr(instruction_set == InstructionSet::SSE2) {
            sub_16_bins_coarse_sse2(a, b);
        } else {
            sub_16_bins_coarse_c(a, b);
        }
    }

    static MB_FORCEINLINE void sub_16_bins(T* a, const T* b) {
      if constexpr (instruction_set == InstructionSet::SSE2) {
        sub_16_bins_sse2(a, b);
      }
      else {
        sub_16_bins_c(a, b);
      }
    }

    static MB_FORCEINLINE void zero_single_bin(T* a) {
        if constexpr (instruction_set == InstructionSet::SSE2) {
            zero_single_bin_sse2(a);
        } else {
            zero_single_bin_c(a);
        }
    }

    template<typename pixel_t, int bits_per_pixel, bool chroma>
    static MB_FORCEINLINE void process_line(uint8_t *dstp, Histogram* histograms, ColumnPair *fine_columns, Histogram *current_hist, int radius, int temporal_radius, int width, int current_length_y) {
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

public:
    static const int HISTOGRAM_SIZE = sizeof(Histogram);

    template<typename pixel_t, int bits_per_pixel, bool chroma>
    static void calculate_median(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height, int radius, void *buffer) {
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
        for (int y = 0; y < radius+1; ++y) {
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
            if (y < height-radius-1) {
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
    }

    template<typename pixel_t, int bits_per_pixel, bool chroma = false>
    static void calculate_temporal_median(uint8_t* dstp, int dst_pitch, const uint8_t** src_ptrs, const int* src_pitches, int frames_count, int width, int height, int radius, void *buffer) {
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
        for (int y = 0; y < radius+1; ++y) {
            for (int x = 0; x < width; ++x) {
                for (int i = 0; i < frames_count; ++i) {
                  int new_element;
                  if constexpr (bits_per_pixel == 8) {
                    new_element = src_ptrs[i][y * src_pitches[i] + x];
                  }
                  else if constexpr(bits_per_pixel <= 16) {
                    new_element = *(reinterpret_cast<const uint16_t *>(src_ptrs[i] + y * src_pitches[i]) + x);
                    new_element = std::min(new_element, max_pixel_value);
                  }
                  else {
                    // 32 bit float: 0..1 luma, -0.5..-0.5 chroma range to 0..65535
                    float new_element_f = *(reinterpret_cast<const float *>(src_ptrs[i] + y * src_pitches[i]) + x);
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
            if (y < height-radius-1) {
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
    }
};


class MedianBlur : public GenericVideoFilter {
public:
    MedianBlur(PClip child, int radius_y, int radius_u, int radius_v, IScriptEnvironment* env);
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env);

    ~MedianBlur() {
        _aligned_free(buffer_);
    }

private:
    int radius_y_;
    int radius_u_;
    int radius_v_;
    void *buffer_;
    decltype(&MedianProcessor<uint8_t, 8, InstructionSet::SSE2>::calculate_median<uint8_t, 8, false>) processors_[4];

    static const int MAX_RADIUS = 127;
};

MedianBlur::MedianBlur(PClip child, int radius_y, int radius_u, int radius_v, IScriptEnvironment* env)
: GenericVideoFilter(child), radius_y_(radius_y), radius_u_(radius_u), radius_v_(radius_v), buffer_(nullptr) {
    if (!vi.IsPlanar()) {
        env->ThrowError("MedianBlur: only planar formats allowed");
    }

    if (radius_y > MAX_RADIUS || radius_u > MAX_RADIUS || radius_v > MAX_RADIUS) {
        env->ThrowError("MedianBlur: radius is too large. Must be between 0 and %i", MAX_RADIUS);
    }

    const int bits_per_pixel = vi.BitsPerComponent();

    const int planesYUV[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    const int planesRGB[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
    const int* planes = vi.IsYUV() || vi.IsYUVA() ? planesYUV : planesRGB;

    // alpha is copied
    int radii[] = { radius_y, radius_u, radius_v, 0 };

    bool sse2 = !!(env->GetCPUFlags() & CPUF_SSE2);

    int hist_size = 0;
    for (int i = 0; i < vi.NumComponents(); ++i) {
      if (radii[i] <= 0) {
        continue;
      }
      int plane = planes[i];
      bool chroma = plane == PLANAR_U || plane == PLANAR_V;
      int width = vi.width >> vi.GetPlaneWidthSubsampling(plane);
      int height = vi.height >> vi.GetPlaneHeightSubsampling(plane);
      int core_size = radii[i] * 2 + 1;
      if (width < core_size || height < core_size) {
        env->ThrowError("MedianBlur: image is too small for this radius!");
      }

      //special cases make sense only when SSE2 is available, otherwise generic routine will be faster
      if (bits_per_pixel == 8 && radii[i] == 1 && sse2 && width > 16) {
        processors_[i] = &calculate_median_r1;
      }
      else if (bits_per_pixel == 8 && radii[i] == 2 && sse2 && width > 16) {
        processors_[i] = &calculate_median_r2;
      }
      else if (radii[i] < 8) {
        // radius<8, processing width <= 7+1+7
        // max pixel count for a histogram entry <= 15x15, counters (bins) fit in a byte, using uint8_t for counter type
        if (sse2) {
          switch (bits_per_pixel) {
          case 8:
            processors_[i] = &MedianProcessor<uint8_t, 8, InstructionSet::SSE2>::calculate_median<uint8_t, 8, false>;
            break;
          case 10:
            processors_[i] = &MedianProcessor<uint8_t, 10, InstructionSet::SSE2>::calculate_median<uint16_t, 10, false>;
            break;
          case 12:
            processors_[i] = &MedianProcessor<uint8_t, 12, InstructionSet::SSE2>::calculate_median<uint16_t, 12, false>;
            break;
          case 14:
            processors_[i] = &MedianProcessor<uint8_t, 14, InstructionSet::SSE2>::calculate_median<uint16_t, 14, false>;
            break;
          case 16:
            processors_[i] = &MedianProcessor<uint8_t, 16, InstructionSet::SSE2>::calculate_median<uint16_t, 16, false>;
            break;
          default:
            // float histogram is simulated as 16 bits
            // 32 bit float processors need special chroma handling
            processors_[i] =
              chroma ?
              &MedianProcessor<uint8_t, 16, InstructionSet::SSE2>::calculate_median<float, 32, true> :
              &MedianProcessor<uint8_t, 16, InstructionSet::SSE2>::calculate_median<float, 32, false>;
            break;
          }
        }
        else {
          switch (bits_per_pixel) {
          case 8:
            processors_[i] = &MedianProcessor<uint8_t, 8, InstructionSet::PLAIN_C>::calculate_median<uint8_t, 8, false>;
            break;
          case 10:
            processors_[i] = &MedianProcessor<uint8_t, 10, InstructionSet::PLAIN_C>::calculate_median<uint16_t, 10, false>;
            break;
          case 12:
            processors_[i] = &MedianProcessor<uint8_t, 12, InstructionSet::PLAIN_C>::calculate_median<uint16_t, 12, false>;
            break;
          case 14:
            processors_[i] = &MedianProcessor<uint8_t, 14, InstructionSet::PLAIN_C>::calculate_median<uint16_t, 14, false>;
            break;
          case 16:
            processors_[i] = &MedianProcessor<uint8_t, 16, InstructionSet::PLAIN_C>::calculate_median<uint16_t, 16, false>;
            break;
          default:
            // float histogram is simulated as 16 bits
            // 32 bit float processors need special chroma handling
            processors_[i] =
              chroma ?
              &MedianProcessor<uint8_t, 16, InstructionSet::PLAIN_C>::calculate_median<float, 32, true> :
              &MedianProcessor<uint8_t, 16, InstructionSet::PLAIN_C>::calculate_median<float, 32, false>;
            break;
          }
        }

        int buffersize;
        switch (bits_per_pixel) {
        case 8: buffersize = MedianProcessor<uint8_t, 8, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
        case 10: buffersize = MedianProcessor<uint8_t, 10, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
        case 12: buffersize = MedianProcessor<uint8_t, 12, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
        case 14: buffersize = MedianProcessor<uint8_t, 14, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
        case 16: buffersize = MedianProcessor<uint8_t, 16, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
        default: // 32 bit is simulated on 16 bit histogram
          buffersize = MedianProcessor<uint8_t, 16, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
        }

        hist_size = std::max(hist_size, buffersize);
      }
      else {
        // radius>=8, processing width >= 8+1+8
        // max pixel count for a histogram entry >= 17x17, counters does not fit in a byte, using uint16_t for counter type
        if (bits_per_pixel == 8 && sse2) {
          switch (bits_per_pixel) {
          case 8:processors_[i] = &MedianProcessor<uint16_t, 8, InstructionSet::SSE2>::calculate_median<uint8_t, 8, false>;
            break;
          case 10:processors_[i] = &MedianProcessor<uint16_t, 10, InstructionSet::SSE2>::calculate_median<uint16_t, 10, false>;
            break;
          case 12:processors_[i] = &MedianProcessor<uint16_t, 12, InstructionSet::SSE2>::calculate_median<uint16_t, 12, false>;
            break;
          case 14:processors_[i] = &MedianProcessor<uint16_t, 14, InstructionSet::SSE2>::calculate_median<uint16_t, 14, false>;
            break;
          case 16:processors_[i] = &MedianProcessor<uint16_t, 16, InstructionSet::SSE2>::calculate_median<uint16_t, 16, false>;
            break;
          default: // float
            // float histogram is simulated as 16 bits
            // 32 bit float processors need special chroma handling
            processors_[i] =
              chroma ?
              processors_[i] = &MedianProcessor<uint16_t, 16, InstructionSet::SSE2>::calculate_median<float, 32, true> :
              processors_[i] = &MedianProcessor<uint16_t, 16, InstructionSet::SSE2>::calculate_median<float, 32, false>;
            break;
          }
        }
        else {
          switch (bits_per_pixel) {
          case 8:processors_[i] = &MedianProcessor<uint16_t, 8, InstructionSet::PLAIN_C>::calculate_median<uint8_t, 8, false>;
            break;
          case 10:processors_[i] = &MedianProcessor<uint16_t, 10, InstructionSet::PLAIN_C>::calculate_median<uint16_t, 10, false>;
            break;
          case 12:processors_[i] = &MedianProcessor<uint16_t, 12, InstructionSet::PLAIN_C>::calculate_median<uint16_t, 12, false>;
            break;
          case 14:processors_[i] = &MedianProcessor<uint16_t, 14, InstructionSet::PLAIN_C>::calculate_median<uint16_t, 14, false>;
            break;
          case 16:processors_[i] = &MedianProcessor<uint16_t, 16, InstructionSet::PLAIN_C>::calculate_median<uint16_t, 16, false>;
            break;
          default: // float
            // float histogram is simulated as 16 bits
            // 32 bit float processors need special chroma handling
            processors_[i] =
              chroma ?
              processors_[i] = &MedianProcessor<uint16_t, 16, InstructionSet::PLAIN_C>::calculate_median<float, 32, true> :
              processors_[i] = &MedianProcessor<uint16_t, 16, InstructionSet::PLAIN_C>::calculate_median<float, 32, false>;
            break;
          }
        }

        int buffersize;
        switch (bits_per_pixel) {
        case 8: buffersize = MedianProcessor<uint16_t, 8, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
        case 10: buffersize = MedianProcessor<uint16_t, 10, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
        case 12: buffersize = MedianProcessor<uint16_t, 12, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
        case 14: buffersize = MedianProcessor<uint16_t, 14, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
        case 16: buffersize = MedianProcessor<uint16_t, 16, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
        default: // 32 bit is simulated on 16 bit histogram
          buffersize = MedianProcessor<uint16_t, 16, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
        }

        hist_size = std::max(hist_size, buffersize);

      }
    }

    if (hist_size) {
        //allocate buffer only for generic approach
        buffer_ = _aligned_malloc(vi.width  * hist_size, 32);
        if (!buffer_) {
            env->ThrowError("MedianBlurTemp: Couldn't callocate buffer.");
        }
    }
}

PVideoFrame MedianBlur::GetFrame(int n, IScriptEnvironment *env) {
    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame dst = env->NewVideoFrame(vi);

    const int planesYUV[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    const int planesRGB[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
    const int* planes = vi.IsYUV() || vi.IsYUVA() ? planesYUV : planesRGB;

    // alpha is copied
    int radii[] = { radius_y_, radius_u_, radius_v_, 0 };
    for (int i = 0; i < vi.NumComponents(); i++) {
        int plane = planes[i];
        int radius = radii[i];
        int width = src->GetRowSize(plane);
        int height = src->GetHeight(plane);
        int realwidth = width / vi.ComponentSize();

        if (radius > 0) {
            processors_[i](dst->GetWritePtr(plane), src->GetReadPtr(plane), dst->GetPitch(plane),
                src->GetPitch(plane), realwidth, height, radius, buffer_);
        } else if (radius == 0) {
            env->BitBlt(dst->GetWritePtr(plane), dst->GetPitch(plane),
                src->GetReadPtr(plane), src->GetPitch(plane), width, height);
        } else if (radius > -256) {
          // fixme for hbd: special "fill" like in masktools
            memset(dst->GetWritePtr(plane), -radius, dst->GetPitch(plane)*height);
        } else {
            memset(dst->GetWritePtr(plane), 0, dst->GetPitch(plane)*height);
        }
    }

    return dst;
}



class MedianBlurTemp : public GenericVideoFilter {
public:
    MedianBlurTemp(PClip child, int radius_y, int radius_u, int radius_v, int radius_temp, IScriptEnvironment* env);
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env);

    ~MedianBlurTemp() {
        _aligned_free(buffer_);
    }

private:
    int radius_y_;
    int radius_u_;
    int radius_v_;
    int radius_temp_;
    void *buffer_;
    decltype(&MedianProcessor<int32_t, 8, InstructionSet::SSE2>::calculate_temporal_median<uint8_t, 8>) processor_;
    decltype(&MedianProcessor<int32_t, 8, InstructionSet::SSE2>::calculate_temporal_median<uint8_t, 8>) processor_chroma_;

    static const int MAX_RADIUS = 1024;
};

MedianBlurTemp::MedianBlurTemp(PClip child, int radius_y, int radius_u, int radius_v, int radius_temp, IScriptEnvironment* env)
: GenericVideoFilter(child), radius_y_(radius_y), radius_u_(radius_u), radius_v_(radius_v), radius_temp_(radius_temp), buffer_(nullptr) {
    if (!vi.IsPlanar()) {
        env->ThrowError("MedianBlurTemp: only planar formats allowed");
    }

    if (radius_y > MAX_RADIUS || radius_u > MAX_RADIUS || radius_v > MAX_RADIUS) {
        env->ThrowError("MedianBlurTemp: radius is too large. Must be between 0 and %i", MAX_RADIUS);
    }
    if (radius_temp <= 0) {
        env->ThrowError("MedianBlurTemp: Invalid temporal radius. Should be greater than zero.");
    }

    const int bits_per_pixel = vi.BitsPerComponent();

    const int planesYUV[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    const int planesRGB[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
    const int* planes = vi.IsYUV() || vi.IsYUVA() ? planesYUV : planesRGB;

    // alpha is copied
    int radii[] = { radius_y, radius_u, radius_v, 0 };

    for (int i = 0; i < vi.NumComponents(); ++i) {
        if (radii[i] < 0) {
            continue;
        }
        int plane = planes[i];
        int width = vi.width >> vi.GetPlaneWidthSubsampling(plane);
        int height = vi.height >> vi.GetPlaneHeightSubsampling(plane);
        int core_size = radii[i]*2 + 1;
        if (width < core_size || height < core_size) {
            env->ThrowError("MedianBlurTemp: image is too small for this radius!");
        }
    }

    // all counter accumulators are 32 bit ints, no need to differentiate by planes
    processor_chroma_ = nullptr;

    if (bits_per_pixel == 8 && !!(env->GetCPUFlags() & CPUF_SSE2)) {
        processor_ = &MedianProcessor<int32_t, 8, InstructionSet::SSE2>::calculate_temporal_median<uint8_t, 8>;
    } else {
      switch (bits_per_pixel) {
      case 8: processor_ = &MedianProcessor<int32_t, 8, InstructionSet::PLAIN_C>::calculate_temporal_median<uint8_t, 8>; break;
      case 10: processor_ = &MedianProcessor<int32_t, 10, InstructionSet::PLAIN_C>::calculate_temporal_median<uint16_t, 10>; break;
      case 12: processor_ = &MedianProcessor<int32_t, 12, InstructionSet::PLAIN_C>::calculate_temporal_median<uint16_t, 12>; break;
      case 14: processor_ = &MedianProcessor<int32_t, 14, InstructionSet::PLAIN_C>::calculate_temporal_median<uint16_t, 14>; break;
      case 16: processor_ = &MedianProcessor<int32_t, 16, InstructionSet::PLAIN_C>::calculate_temporal_median<uint16_t, 16>; break;
      default: // float histogram is simulated on 16 bits
        processor_ = &MedianProcessor<int32_t, 16, InstructionSet::PLAIN_C>::calculate_temporal_median<float, 32, false>;
        processor_chroma_ = &MedianProcessor<int32_t, 16, InstructionSet::PLAIN_C>::calculate_temporal_median<float, 32, true>;
        break;
      }
    }

    int buffersize;
    switch (bits_per_pixel) {
    case 8: buffersize = MedianProcessor<int32_t, 8, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
    case 10: buffersize = MedianProcessor<int32_t, 10, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
    case 12: buffersize = MedianProcessor<int32_t, 12, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
    case 14: buffersize = MedianProcessor<int32_t, 14, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
    case 16: buffersize = MedianProcessor<int32_t, 16, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
    default: // 32 bit is simulated on 16 bit histogram
      buffersize = MedianProcessor<int32_t, 16, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
    }
    
    buffer_ = _aligned_malloc(vi.width  * buffersize, 32);
    if (!buffer_) {
        env->ThrowError("MedianBlurTemp: Couldn't callocate buffer.");
    }
}

PVideoFrame MedianBlurTemp::GetFrame(int n, IScriptEnvironment *env) {
    int frame_buffer_size = sizeof(PVideoFrame)* (radius_temp_*2+1);
    PVideoFrame* src_frames = reinterpret_cast<PVideoFrame*>(alloca(frame_buffer_size));
    if (src_frames == nullptr) {
        env->ThrowError("MedianBlurTemp: Couldn't allocate memory on stack. This is a bug, please report");
    }
    memset(src_frames, 0, frame_buffer_size);

    PVideoFrame src = child->GetFrame(n, env);
    int frame_count = 0;
    for (int i = -radius_temp_; i <= radius_temp_; ++i) {
        int frame_number = n + i;
        //don't get the n'th frame twice
        src_frames[frame_count++] = (i == 0 ? src : child->GetFrame(frame_number, env));
    }
    const uint8_t **src_ptrs = reinterpret_cast<const uint8_t **>(alloca(sizeof(uint8_t*)* frame_count));
    int *src_pitches = reinterpret_cast<int*>(alloca(sizeof(int)* frame_count));
    if (src_ptrs == nullptr || src_pitches == nullptr) {
        env->ThrowError("MedianBlurTemp: Couldn't allocate memory on stack. This is a bug, please report");
    }

    PVideoFrame dst = env->NewVideoFrame(vi);

    const int planesYUV[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    const int planesRGB[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
    const int* planes = vi.IsYUV() || vi.IsYUVA() ? planesYUV : planesRGB;

    // alpha is copied
    int radii[] = { radius_y_, radius_u_, radius_v_, -1 };
    for (int i = 0; i < vi.NumComponents(); i++) {
        int plane = planes[i];
        int radius = radii[i];
        bool chroma = plane == PLANAR_U || plane == PLANAR_V;
        int width = dst->GetRowSize(plane);
        int height = dst->GetHeight(plane);
        int realwidth = width / vi.ComponentSize();

        for (int i = 0; i < frame_count; ++i) {
            src_ptrs[i] = src_frames[i]->GetReadPtr(plane);
            src_pitches[i] = src_frames[i]->GetPitch(plane);
        }

        if (radius >= 0) {
          if (chroma && processor_chroma_ != nullptr)
            processor_chroma_(dst->GetWritePtr(plane), dst->GetPitch(plane),
              src_ptrs, src_pitches, frame_count, realwidth, height, radius, buffer_);
          else
            processor_(dst->GetWritePtr(plane), dst->GetPitch(plane),
              src_ptrs, src_pitches, frame_count, realwidth, height, radius, buffer_);
        } else if (radius == -1) {
            env->BitBlt(dst->GetWritePtr(plane), dst->GetPitch(plane),
                src->GetReadPtr(plane), src->GetPitch(plane), width, height);
        } else if (radius > -256) {
            memset(dst->GetWritePtr(plane), -radius, dst->GetPitch(plane)*height);
        } else {
            memset(dst->GetWritePtr(plane), 0, dst->GetPitch(plane)*height);
        }
    }

    for (int i = 0; i < frame_count; ++i) {
        src_frames[i].~PVideoFrame();
    }

    return dst;
}

AVSValue __cdecl create_median_blur(AVSValue args, void*, IScriptEnvironment* env) {
    enum { CLIP, RADIUS, RADIUS_U, RADIUS_V };
    const bool isRGB = args[0].AsClip()->GetVideoInfo().IsRGB();
    const int radius_y = args[RADIUS].AsInt(2);
    const int radius_u = isRGB ? radius_y : args[RADIUS_U].AsInt(2);
    const int radius_v = isRGB ? radius_y : args[RADIUS_V].AsInt(2);
    return new MedianBlur(args[CLIP].AsClip(), radius_y, radius_u, radius_v, env);
}

AVSValue __cdecl create_temporal_median_blur(AVSValue args, void*, IScriptEnvironment* env) {
    enum { CLIP, RADIUS, RADIUS_U, RADIUS_V, TEMPORAL_RADIUS };
    const bool isRGB = args[0].AsClip()->GetVideoInfo().IsRGB();
    const int radius_y = args[RADIUS].AsInt(2);
    const int radius_u = isRGB ? radius_y : args[RADIUS_U].AsInt(2);
    const int radius_v = isRGB ? radius_y : args[RADIUS_V].AsInt(2);
    return new MedianBlurTemp(args[CLIP].AsClip(), radius_y, radius_u, radius_v, args[TEMPORAL_RADIUS].AsInt(1), env);
}

const AVS_Linkage *AVS_linkage = nullptr;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors) {
    AVS_linkage = vectors;
    env->AddFunction("MedianBlur", "c[radiusy]i[radiusu]i[radiusv]i", create_median_blur, 0);
    env->AddFunction("MedianBlurTemporal", "c[radiusy]i[radiusu]i[radiusv]i[temporalradius]i", create_temporal_median_blur, 0);
    return "Kawaikunai";
}
