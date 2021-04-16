# BTF Extractor
[![PyPI version](https://img.shields.io/pypi/v/btf-extractor?style=flat-square)](https://pypi.org/project/btf-extractor/#history)
[![GitHub version](https://img.shields.io/github/v/tag/2-propanol/BTF_extractor?style=flat-square)](https://github.com/2-propanol/BTF_extractor/releases)
[![Python Versions](https://img.shields.io/pypi/pyversions/btf-extractor?style=flat-square)](https://pypi.org/project/btf-extractor/)

Extract UBO BTF archive format([UBO2003](https://cg.cs.uni-bonn.de/en/projects/btfdbb/download/ubo2003/), [ATRIUM](https://cg.cs.uni-bonn.de/en/projects/btfdbb/download/atrium/), [UBO2014](https://cg.cs.uni-bonn.de/en/projects/btfdbb/download/ubo2014/)).

This repository uses [zeroeffects/btf](https://github.com/zeroeffects/btf)'s [btf.hh](https://github.com/zeroeffects/btf/blob/master/btf.hh) (MIT License).

Extract to ndarray compatible with openCV(BGR, channels-last).

## Install
```bash
pip install btf-extractor
```

This package uses the [Cython](https://cython.readthedocs.io/en/latest/src/quickstart/install.html).
To install this package, a C++ build environment is required.

### Build is tested on
- Windows 10 20H2 + MSVC v14.28 ([Build Tools for Visual Studio 2019](https://visualstudio.microsoft.com/downloads/))
- MacOS 11(Big Sur) + clang 12.0.0 (Command line tools for Xcode (`xcode-select --install`))
- Ubuntu 20.04 + GCC 9.3.0 ([build-essential](https://packages.ubuntu.com/focal/build-essential))

## Example
```python
>>> from btf_extractor import Ubo2003, AtriumHdr, Ubo2014

>>> btf = Ubo2003("UBO_CORDUROY256.zip")
>>> angles_list = list(btf.angles_set)
>>> print(angles_list[0])
(0, 0, 0, 0)
>>> image = btf.angles_to_image(*angles_list[0])
>>> print(image.shape)
(256, 256, 3)
>>> print(image.dtype)
uint8

>>> btf = AtriumHdr("CEILING_HDR.zip")
>>> angles_list = list(btf.angles_set)
>>> print(angles_list[0])
(0, 0, 0, 0)
>>> image = btf.angles_to_image(*angles_list[0])
>>> print(image.shape)
(256, 256, 3)
>>> print(image.dtype)
float32

>>> btf = Ubo2014("carpet01_resampled_W400xH400_L151xV151.btf")
>>> print(btf.img_shape)
(400, 400, 3)
>>> angles_list = list(btf.angles_set)
>>> print(angles_list[0])
(60.0, 270.0, 60.0, 135.0)
>>> image = btf.angles_to_image(*angles_list[0])
>>> print(image.shape)
(400, 400, 3)
>>> print(image.dtype)
float32
```

## Supported Datasets
### UBO2003
6561 images, 256x256 resolution, 81 view and 81 light directions. 

![ubo2003](https://user-images.githubusercontent.com/42978570/114306638-59518580-9b17-11eb-9961-baa775ab235f.jpg)
> Mirko Sattler, Ralf Sarlette and Reinhard Klein "[Efficient and Realistic Visualization of Cloth](http://cg.cs.uni-bonn.de/de/publikationen/paper-details/sattler-2003-efficient/)", EGSR 2003.

### ATRIUM (non-HDR and HDR)
6561 images, 800x800 resolution, 81 view and 81 light directions.

![atrium](https://user-images.githubusercontent.com/42978570/114306641-5c4c7600-9b17-11eb-8251-9a4a92a16b55.jpg)

### UBO2014
22,801 images, 512x512(400x400) resolution, 151 view and 151 light directions.

![ubo2014](https://user-images.githubusercontent.com/42978570/114306647-5f476680-9b17-11eb-9fb6-5332e104f341.jpg)
> [Michael Weinmann](https://cg.cs.uni-bonn.de/en/people/dr-michael-weinmann/), [Juergen Gall](http://www.iai.uni-bonn.de/~gall/) and [Reinhard Klein](https://cg.cs.uni-bonn.de/en/people/prof-dr-reinhard-klein/). "[Material Classification based on Training Data Synthesized Using a BTF Database](https://cg.cs.uni-bonn.de/de/publikationen/paper-details/weinmann-2014-materialclassification/)", accepted at ECCV 2014.
