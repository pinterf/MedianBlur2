## MedianBlur2 ##

Implementation of [constant time median filter](http://nomis80.org/ctmf.html) for AviSynth. 

### Usage
This plugin provides two functions: `MedianBlur(int radiusy, int radiusu, int radiusv)` is a spatial-only version and `MedianBlurTemporal(int radiusy, int radiusu, int radiusv, int temporalRadius)` is a spatio-temporal version.

Maximum radius on every plane is limited to 127.

If alpha plane is available it will be simply copied.
Planar RGB is using value for radiusy for all its channels. radius u and radius v are ignored.

### Performance
Unlike the old MedianBlur, this implementation has constant runtime complexity, meaning that theoretical performance is same for any radius. A few additional optimizations are included:

1. Radii 1 and 2 are special-cased (a lot faster but less generic algorithm) when SSE2 is available. (8 bit videos)
2. For 2 < radius < 8, generic approach with 8-bit bin size is used.
3. For large radii, 16-bit bins are used, as described in the paper.
4. TemporalMedianBlur is using 32 bit bins
5. TemporalMedianBlur quick special case for radius=0 and temporal radius 1 or 2

In other words, you can expect huge performance drop when going from 1 to 2, not so huge but still large from 2 to 3 and a noticeable slowdown from 7 to 8. Between them the fps should be constant and it actually might get a bit faster with larger radii. 

Performance with radius > 2 also depends on the actual frame content. Processing colorbars() is a lot faster than addgrainc(10000).

High bit depth considerations:

At larger bit depths performance drops because internal histogram tables contains statistical data for 10, 12, 14 and 16 bits precision. Also: higher bit depths need more memory.
Unfortunately 32 bit float is faked: float data have to be quantized before processing and is using 16 bit histogram so resulting pixel values are quantized to 16 bits as well.
Radii 1 and 2 are special case only for 8 bits as of v1.0.

### Change log ###

- (no version change) (20210131)
  - Source to compile with GCC and on Linux (tested on Ubuntu 19.10 WSL GCC 9.21, Windows GCC 10.2)
  - Add CMake build system (GCC/Linux)
  - Build instructions in README.md
  - Support non-x86 systems (C-only) by CMake: use ENABLE_INTEL_SIMD off

- 1.1 (20210130) - pinterf
  - Speed: SSE2 and AVX2 for 10+ bits (generic case, MedianBlur)
  - Speed: SSE2 and AVX2 for TemporalMedianBlur
  - Speed: Much-much quicker: TemporalMedianBlur special case: temporal radius=1 or 2, spatial radius=0) (C, SSE4.1, AVX2)
  - Pass frame properties when Avisynth interface>=8
  - Debug helper parameter 'opt': integer default -1
    <0: autodetect CPU
    0: C only (disable SSE2 and AVX2)
    1: SSE2 (disable SSE4.1 and AVX2)
    2: SSE4 (disable AVX2)
    3: AVX2
  - (planned: linux build support, pure C support, CMake build environment)

- 1.0 (20200402) - pinterf
  add high bitdepth support
  add planar RGB support
  add version resource to DLL
  move to Visual Studio 2019, ClangCL, v141_xp and v142 configurations
  project hosted at https://github.com/pinterf/MedianBlur2
  
- 0.94 (2014) - tp7
  Initial version
  https://github.com/tp7/MedianBlur2

### License ###
This project is licensed under the [MIT license][mit_license]. Binaries are [GPL v2][gpl_v2] because if I understand licensing stuff right (please tell me if I don't) they must be.

[mit_license]: http://opensource.org/licenses/MIT
[gpl_v2]: http://www.gnu.org/licenses/gpl-2.0.html

Build instructions
==================
VS2019: 
  use IDE

Windows GCC (mingw installed by msys2):
  from the 'build' folder under project root:

  del ..\CMakeCache.txt
  cmake .. -G "MinGW Makefiles" -DENABLE_INTEL_SIMD:bool=on
  @rem test: cmake .. -G "MinGW Makefiles" -DENABLE_INTEL_SIMD:bool=off
  cmake --build . --config Release  

Linux
* Clone repo
    
        git clone https://github.com/pinterf/MedianBlur2
        cd MedianBlur2
        cmake -B build -S .
        cmake --build build

  Possible option test for C only on x86 arhitectures:
        cmake -B build -S . -DENABLE_INTEL_SIMD:bool=off
        cmake --build build

        Note: ENABLE_INTEL_SIMD is automatically off for non x86 arhitectures and ON for x86

* Find binaries at
    
        build/MedianBlur2/libmedianblur2.so

* Install binaries

        cd build
        sudo make install

Links
=====
Project: https://github.com/pinterf/MedianBlur2
Additional info: http://avisynth.nl/index.php/MedianBlur2
