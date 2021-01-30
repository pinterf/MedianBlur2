#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include "avisynth.h"
#include "common.h"

#define MEDIANPROCESSOR_C
#include "medianblur.h"
#include "medianblur.hpp"
#undef MEDIANPROCESSOR_C

#include "medianblur_avx2.h"
#include "medianblur_sse2.h"
#include <math.h>
#include <assert.h>
#include <stdint.h>
#include <algorithm>
#include <emmintrin.h>
#include "special.h"
#include <memory>
#include <functional>


class MedianBlur : public GenericVideoFilter {
public:
    MedianBlur(PClip child, int radius_y, int radius_u, int radius_v, int opt, IScriptEnvironment* env);
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

    // Auto register AVS+ mode
    int __stdcall SetCacheHints(int cachehints, int frame_range) override {
      return cachehints == CACHE_GET_MTMODE ? MT_MULTI_INSTANCE : 0;
    }

    ~MedianBlur() {
        _aligned_free(buffer_);
    }

private:
    int radius_y_;
    int radius_u_;
    int radius_v_;
    int opt_;
    void *buffer_;
    decltype(&MedianProcessor_c<uint8_t, 8, InstructionSet::PLAIN_C>::calculate_median<uint8_t, 8, false>) processors_[4];
 
    bool has_at_least_v8; // v8 interface frameprop copy support

    static const int MAX_RADIUS = 127;
};

MedianBlur::MedianBlur(PClip child, int radius_y, int radius_u, int radius_v, int opt, IScriptEnvironment* env)
: GenericVideoFilter(child), radius_y_(radius_y), radius_u_(radius_u), radius_v_(radius_v), opt_(opt), buffer_(nullptr) {
    if (!vi.IsPlanar()) {
        env->ThrowError("MedianBlur: only planar formats allowed");
    }

    if (radius_y > MAX_RADIUS || radius_u > MAX_RADIUS || radius_v > MAX_RADIUS) {
        env->ThrowError("MedianBlur: radius is too large. Must be between 0 and %i", MAX_RADIUS);
    }

    has_at_least_v8 = true;
    try { env->CheckVersion(8); }
    catch (const AvisynthError&) { has_at_least_v8 = false; }

    const int bits_per_pixel = vi.BitsPerComponent();

    const int planesYUV[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    const int planesRGB[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
    const int* planes = vi.IsYUV() || vi.IsYUVA() ? planesYUV : planesRGB;

    // alpha is copied
    int radii[] = { radius_y, radius_u, radius_v, 0 };

    bool sse2 = !!(env->GetCPUFlags() & CPUF_SSE2);
    bool sse41 = !!(env->GetCPUFlags() & CPUF_SSE4_1);
    bool avx2 = !!(env->GetCPUFlags() & CPUF_AVX2);
    if (opt_ >= 0) {
      if (opt_ <= 0)
        sse2 = false;
      if (opt_ <= 1)
        sse41 = false;
      if (opt_ <= 2)
        avx2 = false;
    }

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
        if (avx2 && bits_per_pixel > 8) {
          // no avx for small radii and 8 bit. uint8_t counters will fit in sse2
          switch (bits_per_pixel) {
          case 8: processors_[i] = &MedianProcessor_avx2<uint8_t, 8, InstructionSet::AVX2>::calculate_median<uint8_t, 8, false>; break;
          case 10: processors_[i] = &MedianProcessor_avx2<uint8_t, 10, InstructionSet::AVX2>::calculate_median<uint16_t, 10, false>; break;
          case 12: processors_[i] = &MedianProcessor_avx2<uint8_t, 12, InstructionSet::AVX2>::calculate_median<uint16_t, 12, false>; break;
          case 14: processors_[i] = &MedianProcessor_avx2<uint8_t, 14, InstructionSet::AVX2>::calculate_median<uint16_t, 14, false>; break;
          case 16: processors_[i] = &MedianProcessor_avx2<uint8_t, 16, InstructionSet::AVX2>::calculate_median<uint16_t, 16, false>; break;
          default:
            // float histogram is simulated as 16 bits
            // 32 bit float processors need special chroma handling
            processors_[i] =
              chroma ?
              &MedianProcessor_avx2<uint8_t, 16, InstructionSet::AVX2>::calculate_median<float, 16, true> :
              &MedianProcessor_avx2<uint8_t, 16, InstructionSet::AVX2>::calculate_median<float, 16, false>;
            break;
          }
        }
        if (sse2) {
          switch (bits_per_pixel) {
          case 8: processors_[i] = &MedianProcessor_sse2<uint8_t, 8, InstructionSet::SSE2>::calculate_median<uint8_t, 8, false>; break;
          case 10: processors_[i] = &MedianProcessor_sse2<uint8_t, 10, InstructionSet::SSE2>::calculate_median<uint16_t, 10, false>; break;
          case 12: processors_[i] = &MedianProcessor_sse2<uint8_t, 12, InstructionSet::SSE2>::calculate_median<uint16_t, 12, false>; break;
          case 14: processors_[i] = &MedianProcessor_sse2<uint8_t, 14, InstructionSet::SSE2>::calculate_median<uint16_t, 14, false>; break;
          case 16: processors_[i] = &MedianProcessor_sse2<uint8_t, 16, InstructionSet::SSE2>::calculate_median<uint16_t, 16, false>; break;
          default:
            // float histogram is simulated as 16 bits
            // 32 bit float processors need special chroma handling
            processors_[i] =
              chroma ?
              &MedianProcessor_sse2<uint8_t, 16, InstructionSet::SSE2>::calculate_median<float, 16, true> :
              &MedianProcessor_sse2<uint8_t, 16, InstructionSet::SSE2>::calculate_median<float, 16, false>;
            break;
          }
        }
        else {
          switch (bits_per_pixel) {
          case 8: processors_[i] = &MedianProcessor_c<uint8_t, 8, InstructionSet::PLAIN_C>::calculate_median<uint8_t, 8, false>; break;
          case 10: processors_[i] = &MedianProcessor_c<uint8_t, 10, InstructionSet::PLAIN_C>::calculate_median<uint16_t, 10, false>; break;
          case 12: processors_[i] = &MedianProcessor_c<uint8_t, 12, InstructionSet::PLAIN_C>::calculate_median<uint16_t, 12, false>; break;
          case 14: processors_[i] = &MedianProcessor_c<uint8_t, 14, InstructionSet::PLAIN_C>::calculate_median<uint16_t, 14, false>; break;
          case 16: processors_[i] = &MedianProcessor_c<uint8_t, 16, InstructionSet::PLAIN_C>::calculate_median<uint16_t, 16, false>; break;
          default:
            // float histogram is simulated as 16 bits
            // 32 bit float processors need special chroma handling
            processors_[i] =
              chroma ?
              &MedianProcessor_c<uint8_t, 16, InstructionSet::PLAIN_C>::calculate_median<float, 16, true> :
              &MedianProcessor_c<uint8_t, 16, InstructionSet::PLAIN_C>::calculate_median<float, 16, false>;
            break;
          }
        }

        int buffersize;
        switch (bits_per_pixel) {
        case 8: buffersize = MedianProcessor_c<uint8_t, 8, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
        case 10: buffersize = MedianProcessor_c<uint8_t, 10, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
        case 12: buffersize = MedianProcessor_c<uint8_t, 12, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
        case 14: buffersize = MedianProcessor_c<uint8_t, 14, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
        case 16: buffersize = MedianProcessor_c<uint8_t, 16, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
        default: // 32 bit is simulated on 16 bit histogram
          buffersize = MedianProcessor_c<uint8_t, 16, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
        }

        hist_size = std::max(hist_size, buffersize);
      }
      else {
        // radius>=8, processing width >= 8+1+8
        // max pixel count for a histogram entry >= 17x17, counters does not fit in a byte, using uint16_t for counter type
        if (avx2) {
          switch (bits_per_pixel) {
          case 8: processors_[i] = &MedianProcessor_avx2<uint16_t, 8, InstructionSet::AVX2>::calculate_median<uint8_t, 8, false>; break;
          case 10: processors_[i] = &MedianProcessor_avx2<uint16_t, 10, InstructionSet::AVX2>::calculate_median<uint16_t, 10, false>; break;
          case 12: processors_[i] = &MedianProcessor_avx2<uint16_t, 12, InstructionSet::AVX2>::calculate_median<uint16_t, 12, false>; break;
          case 14: processors_[i] = &MedianProcessor_avx2<uint16_t, 14, InstructionSet::AVX2>::calculate_median<uint16_t, 14, false>; break;
          case 16: processors_[i] = &MedianProcessor_avx2<uint16_t, 16, InstructionSet::AVX2>::calculate_median<uint16_t, 16, false>; break;
          default: // float
            // float histogram is simulated as 16 bits
            // 32 bit float processors need special chroma handling
            processors_[i] =
              chroma ?
              processors_[i] = &MedianProcessor_avx2<uint16_t, 16, InstructionSet::AVX2>::calculate_median<float, 16, true> :
              processors_[i] = &MedianProcessor_avx2<uint16_t, 16, InstructionSet::AVX2>::calculate_median<float, 16, false>;
            break;
          }
        }
        if (sse2) {
          switch (bits_per_pixel) {
          case 8: processors_[i] = &MedianProcessor_sse2<uint16_t, 8, InstructionSet::SSE2>::calculate_median<uint8_t, 8, false>; break;
          case 10: processors_[i] = &MedianProcessor_sse2<uint16_t, 10, InstructionSet::SSE2>::calculate_median<uint16_t, 10, false>; break;
          case 12: processors_[i] = &MedianProcessor_sse2<uint16_t, 12, InstructionSet::SSE2>::calculate_median<uint16_t, 12, false>; break;
          case 14: processors_[i] = &MedianProcessor_sse2<uint16_t, 14, InstructionSet::SSE2>::calculate_median<uint16_t, 14, false>; break;
          case 16: processors_[i] = &MedianProcessor_sse2<uint16_t, 16, InstructionSet::SSE2>::calculate_median<uint16_t, 16, false>; break;
          default: // float
            // float histogram is simulated as 16 bits
            // 32 bit float processors need special chroma handling
            processors_[i] =
              chroma ?
              processors_[i] = &MedianProcessor_sse2<uint16_t, 16, InstructionSet::SSE2>::calculate_median<float, 16, true> :
              processors_[i] = &MedianProcessor_sse2<uint16_t, 16, InstructionSet::SSE2>::calculate_median<float, 16, false>;
            break;
          }
        }
        else {
          switch (bits_per_pixel) {
          case 8: processors_[i] = &MedianProcessor_c<uint16_t, 8, InstructionSet::PLAIN_C>::calculate_median<uint8_t, 8, false>; break;
          case 10: processors_[i] = &MedianProcessor_c<uint16_t, 10, InstructionSet::PLAIN_C>::calculate_median<uint16_t, 10, false>; break;
          case 12: processors_[i] = &MedianProcessor_c<uint16_t, 12, InstructionSet::PLAIN_C>::calculate_median<uint16_t, 12, false>; break;
          case 14: processors_[i] = &MedianProcessor_c<uint16_t, 14, InstructionSet::PLAIN_C>::calculate_median<uint16_t, 14, false>; break;
          case 16: processors_[i] = &MedianProcessor_c<uint16_t, 16, InstructionSet::PLAIN_C>::calculate_median<uint16_t, 16, false>; break;
          default: // float
            // float histogram is simulated as 16 bits
            // 32 bit float processors need special chroma handling
            processors_[i] =
              chroma ?
              processors_[i] = &MedianProcessor_c<uint16_t, 16, InstructionSet::PLAIN_C>::calculate_median<float, 16, true> :
              processors_[i] = &MedianProcessor_c<uint16_t, 16, InstructionSet::PLAIN_C>::calculate_median<float, 16, false>;
            break;
          }
        }

        int buffersize;
        switch (bits_per_pixel) {
        case 8: buffersize = MedianProcessor_c<uint16_t, 8, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
        case 10: buffersize = MedianProcessor_c<uint16_t, 10, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
        case 12: buffersize = MedianProcessor_c<uint16_t, 12, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
        case 14: buffersize = MedianProcessor_c<uint16_t, 14, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
        case 16: buffersize = MedianProcessor_c<uint16_t, 16, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
        default: // 32 bit is simulated on 16 bit histogram
          buffersize = MedianProcessor_c<uint16_t, 16, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
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
    PVideoFrame dst = has_at_least_v8 ? env->NewVideoFrameP(vi, &src) : env->NewVideoFrame(vi); // frame property support

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
    MedianBlurTemp(PClip child, int radius_y, int radius_u, int radius_v, int radius_temp, int opt, IScriptEnvironment* env);
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env) override;

    // Auto register AVS+ mode
    int __stdcall SetCacheHints(int cachehints, int frame_range) override {
      return cachehints == CACHE_GET_MTMODE ? MT_MULTI_INSTANCE : 0;
    }

    ~MedianBlurTemp() {
        _aligned_free(buffer_);
    }

private:
    int radius_y_;
    int radius_u_;
    int radius_v_;
    int radius_temp_;
    int opt_;
    void *buffer_;
    decltype(&MedianProcessor_c<int32_t, 8, InstructionSet::PLAIN_C>::calculate_temporal_median<uint8_t, 8, false>) processors_[4];
    decltype(&MedianProcessor_c<int32_t, 8, InstructionSet::PLAIN_C>::calculate_temporal_median<uint8_t, 8, false>) processors_chroma_[4];

    bool has_at_least_v8; // v8 interface frameprop copy support

    static constexpr int MAX_RADIUS = 1024;
};

MedianBlurTemp::MedianBlurTemp(PClip child, int radius_y, int radius_u, int radius_v, int radius_temp, int opt, IScriptEnvironment* env)
: GenericVideoFilter(child), radius_y_(radius_y), radius_u_(radius_u), radius_v_(radius_v), radius_temp_(radius_temp), opt_(opt), buffer_(nullptr) {

    if (!vi.IsPlanar()) {
        env->ThrowError("MedianBlurTemp: only planar formats allowed");
    }

    if (radius_y > MAX_RADIUS || radius_u > MAX_RADIUS || radius_v > MAX_RADIUS) {
        env->ThrowError("MedianBlurTemp: radius is too large. Must be between 0 and %i", MAX_RADIUS);
    }
    if (radius_temp <= 0) {
        env->ThrowError("MedianBlurTemp: Invalid temporal radius. Should be greater than zero.");
    }

    has_at_least_v8 = true;
    try { env->CheckVersion(8); }
    catch (const AvisynthError&) { has_at_least_v8 = false; }

    const int bits_per_pixel = vi.BitsPerComponent();

    const int planesYUV[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    const int planesRGB[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
    const int* planes = vi.IsYUV() || vi.IsYUVA() ? planesYUV : planesRGB;

    buffer_ = nullptr;

    // alpha is copied
    int radii[] = { radius_y, radius_u, radius_v, 0 };

    bool sse2 = !!(env->GetCPUFlags() & CPUF_SSE2);
    bool sse41 = !!(env->GetCPUFlags() & CPUF_SSE4_1);
    bool avx2 = !!(env->GetCPUFlags() & CPUF_AVX2);
    if (opt_ >= 0) {
      if (opt_ <= 0)
        sse2 = false;
      if (opt_ <= 1)
        sse41 = false;
      if (opt_ <= 2)
        avx2 = false;
    }

    for (int i = 0; i < vi.NumComponents(); ++i) {
      if (radii[i] < 0) {
        continue;
      }
      int plane = planes[i];
      int width = vi.width >> vi.GetPlaneWidthSubsampling(plane);
      int height = vi.height >> vi.GetPlaneHeightSubsampling(plane);
      int core_size = radii[i] * 2 + 1;
      if (width < core_size || height < core_size) {
        env->ThrowError("MedianBlurTemp: image is too small for this radius!");
      }

      // all counter accumulators are 32 bit ints, no need to differentiate by planes
      processors_chroma_[i] = nullptr;

      if (radii[i] == 0 && radius_temp == 1) {
        // special case: spatial radius 0, temporal radius = 1
        if (avx2) {
          switch (bits_per_pixel) {
          case 8: processors_[i] = &calculate_temporal_median_sr0_tr1_avx2<uint8_t>; break;
          case 10:
          case 12:
          case 14:
          case 16: processors_[i] = &calculate_temporal_median_sr0_tr1_avx2<uint16_t>; break;
          default: // float histogram is simulated on 16 bits
            processors_[i] = &calculate_temporal_median_sr0_tr1_avx2<float>; break;
            break;
          }
        }
        else if (sse41) {
          switch (bits_per_pixel) {
          case 8: processors_[i] = &calculate_temporal_median_sr0_tr1_sse4<uint8_t>; break;
          case 10:
          case 12:
          case 14:
          case 16: processors_[i] = &calculate_temporal_median_sr0_tr1_sse4<uint16_t>; break;
          default: // float histogram is simulated on 16 bits
            processors_[i] = &calculate_temporal_median_sr0_tr1_sse4<float>; break;
            break;
          }
        }
        else {
          switch (bits_per_pixel) {
          case 8: processors_[i] = &calculate_temporal_median_sr0_tr1_c<uint8_t>; break;
          case 10:
          case 12:
          case 14:
          case 16: processors_[i] = &calculate_temporal_median_sr0_tr1_c<uint16_t>; break;
          default: // float histogram is simulated on 16 bits
            processors_[i] = &calculate_temporal_median_sr0_tr1_c<float>; break;
            break;
          }
        }

      }
      else if (radii[i] == 0 && radius_temp == 2) {
        // special case: spatial radius 0, temporal radius = 2
        if (avx2) {
          switch (bits_per_pixel) {
          case 8: processors_[i] = &calculate_temporal_median_sr0_tr2_avx2<uint8_t>; break;
          case 10:
          case 12:
          case 14:
          case 16: processors_[i] = &calculate_temporal_median_sr0_tr2_avx2<uint16_t>; break;
          default: // float histogram is simulated on 16 bits
            processors_[i] = &calculate_temporal_median_sr0_tr2_avx2<float>; break;
            break;
          }
        }
        else if (sse41) {
          switch (bits_per_pixel) {
          case 8: processors_[i] = &calculate_temporal_median_sr0_tr2_sse4<uint8_t>; break;
          case 10:
          case 12:
          case 14:
          case 16: processors_[i] = &calculate_temporal_median_sr0_tr2_sse4<uint16_t>; break;
          default: // float histogram is simulated on 16 bits
            processors_[i] = &calculate_temporal_median_sr0_tr2_sse4<float>; break;
            break;
          }
        }
        else {
          switch (bits_per_pixel) {
          case 8: processors_[i] = &calculate_temporal_median_sr0_tr2_c<uint8_t>; break;
          case 10:
          case 12:
          case 14:
          case 16: processors_[i] = &calculate_temporal_median_sr0_tr2_c<uint16_t>; break;
          default: // float histogram is simulated on 16 bits
            processors_[i] = &calculate_temporal_median_sr0_tr2_c<float>; break;
            break;
          }
        }

      }
      else {
        if (avx2) {
          switch (bits_per_pixel) {
          case 8: processors_[i] = &MedianProcessor_avx2<int32_t, 8, InstructionSet::AVX2>::calculate_temporal_median<uint8_t, 8, false>; break;
          case 10: processors_[i] = &MedianProcessor_avx2<int32_t, 10, InstructionSet::AVX2>::calculate_temporal_median<uint16_t, 10, false>; break;
          case 12: processors_[i] = &MedianProcessor_avx2<int32_t, 12, InstructionSet::AVX2>::calculate_temporal_median<uint16_t, 12, false>; break;
          case 14: processors_[i] = &MedianProcessor_avx2<int32_t, 14, InstructionSet::AVX2>::calculate_temporal_median<uint16_t, 14, false>; break;
          case 16: processors_[i] = &MedianProcessor_avx2<int32_t, 16, InstructionSet::AVX2>::calculate_temporal_median<uint16_t, 16, false>; break;
          default: // float histogram is simulated on 16 bits
            processors_[i] = &MedianProcessor_avx2<int32_t, 16, InstructionSet::AVX2>::calculate_temporal_median<float, 16, false>;
            processors_chroma_[i] = &MedianProcessor_avx2<int32_t, 16, InstructionSet::AVX2>::calculate_temporal_median<float, 16, true>;
            break;
          }
        }
        else if (sse2) {
          switch (bits_per_pixel) {
          case 8: processors_[i] = &MedianProcessor_sse2<int32_t, 8, InstructionSet::SSE2>::calculate_temporal_median<uint8_t, 8, false>; break;
          case 10: processors_[i] = &MedianProcessor_sse2<int32_t, 10, InstructionSet::SSE2>::calculate_temporal_median<uint16_t, 10, false>; break;
          case 12: processors_[i] = &MedianProcessor_sse2<int32_t, 12, InstructionSet::SSE2>::calculate_temporal_median<uint16_t, 12, false>; break;
          case 14: processors_[i] = &MedianProcessor_sse2<int32_t, 14, InstructionSet::SSE2>::calculate_temporal_median<uint16_t, 14, false>; break;
          case 16: processors_[i] = &MedianProcessor_sse2<int32_t, 16, InstructionSet::SSE2>::calculate_temporal_median<uint16_t, 16, false>; break;
          default: // float histogram is simulated on 16 bits
            processors_[i] = &MedianProcessor_sse2<int32_t, 16, InstructionSet::SSE2>::calculate_temporal_median<float, 16, false>;
            processors_chroma_[i] = &MedianProcessor_sse2<int32_t, 16, InstructionSet::SSE2>::calculate_temporal_median<float, 16, true>;
            break;
          }
        }
        else {
          switch (bits_per_pixel) {
          case 8: processors_[i] = &MedianProcessor_c<int32_t, 8, InstructionSet::PLAIN_C>::calculate_temporal_median<uint8_t, 8, false>; break;
          case 10: processors_[i] = &MedianProcessor_c<int32_t, 10, InstructionSet::PLAIN_C>::calculate_temporal_median<uint16_t, 10, false>; break;
          case 12: processors_[i] = &MedianProcessor_c<int32_t, 12, InstructionSet::PLAIN_C>::calculate_temporal_median<uint16_t, 12, false>; break;
          case 14: processors_[i] = &MedianProcessor_c<int32_t, 14, InstructionSet::PLAIN_C>::calculate_temporal_median<uint16_t, 14, false>; break;
          case 16: processors_[i] = &MedianProcessor_c<int32_t, 16, InstructionSet::PLAIN_C>::calculate_temporal_median<uint16_t, 16, false>; break;
          default: // float histogram is simulated on 16 bits
            processors_[i] = &MedianProcessor_c<int32_t, 16, InstructionSet::PLAIN_C>::calculate_temporal_median<float, 16, false>;
            processors_chroma_[i] = &MedianProcessor_c<int32_t, 16, InstructionSet::PLAIN_C>::calculate_temporal_median<float, 16, true>;
            break;
          }
        }

        int buffersize;
        switch (bits_per_pixel) {
        case 8: buffersize = MedianProcessor_c<int32_t, 8, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
        case 10: buffersize = MedianProcessor_c<int32_t, 10, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
        case 12: buffersize = MedianProcessor_c<int32_t, 12, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
        case 14: buffersize = MedianProcessor_c<int32_t, 14, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
        case 16: buffersize = MedianProcessor_c<int32_t, 16, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
        default: // 32 bit is simulated on 16 bit histogram
          buffersize = MedianProcessor_c<int32_t, 16, InstructionSet::PLAIN_C>::HISTOGRAM_SIZE; break;
        }

        if (buffer_ == nullptr) {
          buffer_ = _aligned_malloc(vi.width * buffersize, 32);
          if (!buffer_) {
            env->ThrowError("MedianBlurTemp: Couldn't callocate buffer.");
          }
        }
      }
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

    PVideoFrame dst = has_at_least_v8 ? env->NewVideoFrameP(vi, &src) : env->NewVideoFrame(vi); // frame property support

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
          if (chroma && processors_chroma_[i] != nullptr)
            processors_chroma_[i](dst->GetWritePtr(plane), dst->GetPitch(plane),
              src_ptrs, src_pitches, frame_count, realwidth, height, radius, buffer_);
          else
            processors_[i](dst->GetWritePtr(plane), dst->GetPitch(plane),
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
    enum { CLIP, RADIUS, RADIUS_U, RADIUS_V, OPT };
    const bool isRGB = args[0].AsClip()->GetVideoInfo().IsRGB();
    const int radius_y = args[RADIUS].AsInt(2);
    const int radius_u = isRGB ? radius_y : args[RADIUS_U].AsInt(2);
    const int radius_v = isRGB ? radius_y : args[RADIUS_V].AsInt(2);
    const int opt = args[OPT].AsInt(-1); // -1: auto, 0: C, other: SIMD
    return new MedianBlur(args[CLIP].AsClip(), radius_y, radius_u, radius_v, opt, env);
}

AVSValue __cdecl create_temporal_median_blur(AVSValue args, void*, IScriptEnvironment* env) {
    enum { CLIP, RADIUS, RADIUS_U, RADIUS_V, TEMPORAL_RADIUS, OPT };
    const bool isRGB = args[0].AsClip()->GetVideoInfo().IsRGB();
    const int radius_y = args[RADIUS].AsInt(2);
    const int radius_u = isRGB ? radius_y : args[RADIUS_U].AsInt(2);
    const int radius_v = isRGB ? radius_y : args[RADIUS_V].AsInt(2);
    const int opt = args[OPT].AsInt(-1); // -1: auto, 0: C, other: SIMD
    return new MedianBlurTemp(args[CLIP].AsClip(), radius_y, radius_u, radius_v, args[TEMPORAL_RADIUS].AsInt(1), opt, env);
}

const AVS_Linkage *AVS_linkage = nullptr;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors) {
    AVS_linkage = vectors;
    env->AddFunction("MedianBlur", "c[radiusy]i[radiusu]i[radiusv]i[opt]i", create_median_blur, 0);
    env->AddFunction("MedianBlurTemporal", "c[radiusy]i[radiusu]i[radiusv]i[temporalradius]i[opt]i", create_temporal_median_blur, 0);
    return "Kawaikunai";
}
