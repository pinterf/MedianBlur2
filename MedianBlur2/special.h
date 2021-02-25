#include "common.h"
#include <stdint.h>

#ifdef INTEL_INTRINSICS

#include "emmintrin.h"

/*
 * Specialized routines for faster processing of small radii
 */

//this is far from optimal but simple and sufficiently fast for border cases
class CMedianizer
{
    int elements_count;
    uint8_t coarse[16];
    uint8_t fine[256];

public:
    inline void reset() {
        elements_count = 0;
        memset(fine, 0, sizeof(fine));
        memset(coarse, 0, sizeof(coarse));
    }

    inline void add(int val) {
        coarse[val>>4]++;
        fine[val]++;
        elements_count++;
    }

    inline uint8_t finalize() const
    {
        int count = 0;
        int coarse_idx = -1;
        int half_elements = (elements_count + 1) >> 1;
        while (count < half_elements) {
            count += coarse[++coarse_idx];
        }
        count -= coarse[coarse_idx];
        int fine_offset = coarse_idx*16;

        int fine_idx = fine_offset - 1;
        while (count < half_elements) {
            count += fine[++fine_idx];
        }
        return fine_idx;
    }
};

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse2")))
#endif
static MB_FORCEINLINE void sort(__m128i &a1, __m128i &a2) {
    const __m128i tmp = _mm_min_epu8(a1, a2);
    a2 = _mm_max_epu8(a1, a2);
    a1 = tmp;
}

// special case: radius=1, 8 bits
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse2")))
#endif
static void calculate_median_r1(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height, int /*radius*/, void* /*buffer*/) {
    CMedianizer medianizer;

    //first line
    medianizer.reset();
    medianizer.add(srcp[0]); medianizer.add(srcp[1]); medianizer.add(srcp[src_pitch]); medianizer.add(srcp[src_pitch+1]);
    dstp[0] = medianizer.finalize();

    for (int x = 1; x < width-1; ++x) {
        medianizer.reset();
        medianizer.add(srcp[x-1]);
        medianizer.add(srcp[x]);
        medianizer.add(srcp[x+1]);
        medianizer.add(srcp[x+src_pitch-1]);
        medianizer.add(srcp[x+src_pitch]);
        medianizer.add(srcp[x+src_pitch+1]);
        dstp[x] = medianizer.finalize();
    }

    medianizer.reset();
    medianizer.add(srcp[width-2]); medianizer.add(srcp[width-1]); medianizer.add(srcp[src_pitch+width-2]); medianizer.add(srcp[src_pitch+width-1]);
    dstp[width-1] = medianizer.finalize();

    //main part
    srcp += src_pitch;
    dstp += dst_pitch;
    for (int y = 1; y < height-1; ++y) {
        medianizer.reset();
        medianizer.add(srcp[-src_pitch]); medianizer.add(srcp[-src_pitch+1]);
        medianizer.add(srcp[0]); medianizer.add(srcp[1]);
        medianizer.add(srcp[src_pitch]); medianizer.add(srcp[src_pitch+1]);
        dstp[0] = medianizer.finalize();

        for (int x = 1; x < width-1; x += 16) {
            __m128i a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp+x-src_pitch-1));
            __m128i a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp+x-src_pitch));
            __m128i a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp+x-src_pitch+1));
            __m128i a4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp+x-1));
            __m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp+x));
            __m128i a5 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp+x+1));
            __m128i a6 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp+x+src_pitch-1));
            __m128i a7 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp+x+src_pitch));
            __m128i a8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp+x+src_pitch+1));
            sort(a1, a2);
            sort(a3, a4);
            sort(a5, a6);
            sort(a7, a8);

            sort(a1, a3);
            sort(a2, a4);
            sort(a5, a7);
            sort(a6, a8);

            sort(a2, a3);
            sort(a6, a7);

            a5 = _mm_max_epu8(a1, a5);
            a6 = _mm_max_epu8(a2, a6);
            a3 = _mm_min_epu8(a3, a7);
            a4 = _mm_min_epu8(a4, a8);

            a5 = _mm_max_epu8(a3, a5);
            a4 = _mm_min_epu8(a4, a6);

            sort(a4, a5);
            __m128i result = _mm_max_epu8(_mm_min_epu8(c, a5), a4);;
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dstp+x), result);
        }

        medianizer.reset();
        medianizer.add(srcp[width-src_pitch-2]); medianizer.add(srcp[width-src_pitch-1]);
        medianizer.add(srcp[width-2]); medianizer.add(srcp[width-1]);
        medianizer.add(srcp[width+src_pitch-2]); medianizer.add(srcp[width+src_pitch-1]);
        dstp[width-1] = medianizer.finalize();

        srcp += src_pitch;
        dstp += dst_pitch;
    }

    //last line
    medianizer.reset();
    medianizer.add(srcp[0]); medianizer.add(srcp[1]); medianizer.add(srcp[-src_pitch]); medianizer.add(srcp[-src_pitch+1]);
    dstp[0] = medianizer.finalize();

    for (int x = 1; x < width-1; ++x) {
        medianizer.reset();
        medianizer.add(srcp[x-src_pitch-1]);
        medianizer.add(srcp[x-src_pitch]);
        medianizer.add(srcp[x-src_pitch+1]);
        medianizer.add(srcp[x-1]);
        medianizer.add(srcp[x]);
        medianizer.add(srcp[x+1]);
        dstp[x] = medianizer.finalize();
    }

    medianizer.reset();
    medianizer.add(srcp[width-2]); medianizer.add(srcp[width-1]); medianizer.add(srcp[-src_pitch+width-2]); medianizer.add(srcp[-src_pitch+width-1]);
    dstp[width-1] = medianizer.finalize();
}


template<bool left_border, bool right_border, bool top_border, bool bottom_border>
static void calculate_partial_median(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height, int radius) {
    CMedianizer medianizer;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            medianizer.reset();
            for (int local_y = top_border ? 0 : y-radius; local_y <= (bottom_border ? height-1 : y+radius); ++local_y) {
                for (int local_x = left_border ? 0 : x-radius; local_x <= (right_border ? width-1 : x+radius); ++local_x) {
                    medianizer.add(srcp[src_pitch*local_y + local_x]);
                }
            }
            dstp[x] = medianizer.finalize();
        }
        dstp += dst_pitch;
    }
}

// special case: radius=2, 8 bits
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse2")))
#endif
static void calculate_median_r2(uint8_t* dstp, const uint8_t* srcp, int dst_pitch, int src_pitch, int width, int height, int /*radius*/, void* /*buffer*/) {
    //top left corder
    calculate_partial_median<true, false, true, false>(dstp, srcp, dst_pitch, src_pitch, 2, 2, 2);
    //first two lines
    calculate_partial_median<false, false, true, false>(dstp+2, srcp+2, dst_pitch, src_pitch, width-4, 2, 2);
    //top right corder
    calculate_partial_median<false, true, true, false>(dstp+width-2, srcp+width-2, dst_pitch, src_pitch, 2, 2, 2);  

    //main part. Don't laugh.
    __m128i a[25];
    int offsets[] = {
        -src_pitch*2-2, -src_pitch*2-1, -src_pitch*2, -src_pitch*2+1, -src_pitch*2+2,
        -src_pitch*1-2, -src_pitch*1-1, -src_pitch*1, -src_pitch*1+1, -src_pitch*1+2,
        -src_pitch*0-2, -src_pitch*0-1, -src_pitch*0, -src_pitch*0+1, -src_pitch*0+2,
        +src_pitch*1-2, +src_pitch*1-1, +src_pitch*1, +src_pitch*1+1, +src_pitch*1+2,
        +src_pitch*2-2, +src_pitch*2-1, +src_pitch*2, +src_pitch*2+1, +src_pitch*2+2
    };
    
    const uint8_t* srcp_saved = srcp;
    uint8_t* dstp_saved = dstp;

    srcp += src_pitch*2;
    dstp += dst_pitch*2;
    for (int y = 2; y < height-2; ++y) {
        for (int x = 2; x < width-2; x += 16) {
            for (int i = 0; i < 25; ++i) {
                a[i] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp+x+offsets[i]));
            }
            //sorting network taken from Simd Library by Yermalayeu Ihar
            //probably doing some unnecessary comparisons but works
            sort(a[0], a[1]);   sort(a[3], a[4]);   sort(a[2], a[4]);
            sort(a[2], a[3]);   sort(a[6], a[7]);   sort(a[5], a[7]);
            sort(a[5], a[6]);   sort(a[9], a[10]);  sort(a[8], a[10]);
            sort(a[8], a[9]);   sort(a[12], a[13]); sort(a[11], a[13]);
            sort(a[11], a[12]); sort(a[15], a[16]); sort(a[14], a[16]);
            sort(a[14], a[15]); sort(a[18], a[19]); sort(a[17], a[19]);
            sort(a[17], a[18]); sort(a[21], a[22]); sort(a[20], a[22]);
            sort(a[20], a[21]); sort(a[23], a[24]); sort(a[2], a[5]);
            sort(a[3], a[6]);   sort(a[0], a[6]);   sort(a[0], a[3]);
            sort(a[4], a[7]);   sort(a[1], a[7]);   sort(a[1], a[4]);
            sort(a[11], a[14]); sort(a[8], a[14]);  sort(a[8], a[11]);
            sort(a[12], a[15]); sort(a[9], a[15]);  sort(a[9], a[12]);
            sort(a[13], a[16]); sort(a[10], a[16]); sort(a[10], a[13]);
            sort(a[20], a[23]); sort(a[17], a[23]); sort(a[17], a[20]);
            sort(a[21], a[24]); sort(a[18], a[24]); sort(a[18], a[21]);
            sort(a[19], a[22]); sort(a[8], a[17]);  sort(a[9], a[18]);
            sort(a[0], a[18]);  sort(a[0], a[9]);   sort(a[10], a[19]);
            sort(a[1], a[19]);  sort(a[1], a[10]);  sort(a[11], a[20]);
            sort(a[2], a[20]);  sort(a[2], a[11]);  sort(a[12], a[21]);
            sort(a[3], a[21]);  sort(a[3], a[12]);  sort(a[13], a[22]);
            sort(a[4], a[22]);  sort(a[4], a[13]);  sort(a[14], a[23]);
            sort(a[5], a[23]);  sort(a[5], a[14]);  sort(a[15], a[24]);
            sort(a[6], a[24]);  sort(a[6], a[15]);  sort(a[7], a[16]);
            sort(a[7], a[19]);  sort(a[13], a[21]); sort(a[15], a[23]);
            sort(a[7], a[13]);  sort(a[7], a[15]);  sort(a[1], a[9]);
            sort(a[3], a[11]);  sort(a[5], a[17]);  sort(a[11], a[17]);
            sort(a[9], a[17]);  sort(a[4], a[10]);  sort(a[6], a[12]);
            sort(a[7], a[14]);  sort(a[4], a[6]);   sort(a[4], a[7]);
            sort(a[12], a[14]); sort(a[10], a[14]); sort(a[6], a[7]);
            sort(a[10], a[12]); sort(a[6], a[10]);  sort(a[6], a[17]);
            sort(a[12], a[17]); sort(a[7], a[17]);  sort(a[7], a[10]);
            sort(a[12], a[18]); sort(a[7], a[12]);  sort(a[10], a[18]);
            sort(a[12], a[20]); sort(a[10], a[20]); sort(a[10], a[12]);

            _mm_storeu_si128(reinterpret_cast<__m128i*>(dstp+x), a[12]);
        }
        srcp += src_pitch;
        dstp += dst_pitch;
    }

    dstp = dstp_saved;
    srcp = srcp_saved;
    //two left columns
    calculate_partial_median<true, false, false, false>(dstp+dst_pitch*2, srcp+src_pitch*2, dst_pitch, src_pitch, 2, height-4, 2);
    //two right columns
    calculate_partial_median<false, true, false, false>(dstp+dst_pitch*2+width-2, srcp+src_pitch*2+width-2, dst_pitch, src_pitch, 2, height-4, 2);
    dstp += dst_pitch * (height-2);
    srcp += src_pitch * (height-2);
    //bottom left corner
    calculate_partial_median<true, false, false, true>(dstp, srcp, dst_pitch, src_pitch, 2, 2, 2);
    //two bottom rows
    calculate_partial_median<false, false, false, true>(dstp+2, srcp+2, dst_pitch, src_pitch, width-4, 2, 2);
    //bottom right corner
    calculate_partial_median<false, true, false, true>(dstp+width-2, srcp+width-2, dst_pitch, src_pitch, 2, 2, 2);
}

#endif // INTEL_INTRINSICS