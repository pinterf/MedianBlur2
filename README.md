## MedianBlur2 ##

Implementation of [constant time median filter](http://nomis80.org/ctmf.html) for AviSynth. 

### Usage
This plugin provides two functions: `MedianBlur(int radiusy, int radiusu, int radiusv)` is a spatial-only version and `MedianBlurTemporal(int radiusy, int radiusu, int radiusv, int temporalRadius)` is a spatio-temporal version.

Maximum radius on every plane is limited to 127.

If alpha plane is available it will be simply copied.
Planar RGB is using value for radiusy for all its channels. radius u and radius v are ignored.

### Performance
Unlike the old MedianBlur, this implementation has constant runtime complexity, meaning that theoretical performance is same for any radius. A few additional optimizations are included:

1. Radii 1 and 2 are special-cased (a lot faster but less generic algorithm) when SSE2 is available.
2. For 2 < radius < 8, generic approach with 8-bit bin size is used.
3. For large radii, 16-bit bins are used, as described in the paper. 

In other words, you can expect huge performance drop when going from 1 to 2, not so huge but still large from 2 to 3 and a noticeable slowdown from 7 to 8. Between them the fps should be constant and it actually might get a bit faster with larger radii. 

Performance with radius > 2 also depends on the actual frame content. Processing colorbars() is a lot faster than addgrainc(10000).

High bit depth considerations:

At larger bit depths performance drops because internal histogram tables contains statistical data for 10, 12, 14 and 16 bits precision. Also: higher bit depths need more memory.
Unfortunately 32 bit float is faked: float data have to be quantized before processing and is using 16 bit histogram so resulting pixel values are quantized to 16 bits as well.
Radii 1 and 2 are special case only for 8 bits as of v1.0.

### Change log ###

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
