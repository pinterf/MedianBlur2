#ifndef __MEDIANBLUR_SSE2_H__
#define __MEDIANBLUR_SSE2_H__

#include "avisynth.h"
#include <emmintrin.h>
#include <smmintrin.h> // SSE 4.1
#include <cassert>
#include "common.h"


#define MEDIANPROCESSOR_SSE2
#include "medianblur.h"
#undef MEDIANPROCESSOR_SSE2

#endif
