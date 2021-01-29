#ifndef __COMMON_H__
#define __COMMON_H__

#include <stdint.h>

#if defined(__clang__)
// Check clang first. clang-cl also defines __MSC_VER
// We set MSVC because they are mostly compatible
#   define CLANG
#if defined(_MSC_VER)
#   define MSVC
#   define MB_FORCEINLINE __attribute__((always_inline))
#else
#   define MB_FORCEINLINE __attribute__((always_inline)) inline
#endif
#elif   defined(_MSC_VER)
#   define MSVC
#   define MSVC_PURE
#   define MB_FORCEINLINE __forceinline
#elif defined(__GNUC__)
#   define GCC
#   define MB_FORCEINLINE __attribute__((always_inline)) inline
#else
#   error Unsupported compiler.
#   define MB_FORCEINLINE inline
#   undef __forceinline
#   define __forceinline inline
#endif 

enum class InstructionSet
{
  AVX2,
  SSE2,
  PLAIN_C
};

#endif
