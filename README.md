# BTF Extractor
Extract UBO BTF archive format([UBO2003](https://cg.cs.uni-bonn.de/en/projects/btfdbb/download/ubo2003/), [UBO2014](https://cg.cs.uni-bonn.de/en/projects/btfdbb/download/ubo2014/)).

This repository uses [zeroeffects/btf](https://github.com/zeroeffects/btf)'s [btf.hh](https://github.com/zeroeffects/btf/blob/master/btf.hh).

Extract to ndarray compatible with openCV(BGR, channels-last).


## Build tested on
- Windows 10 20H2 + MSVC v14.20
- MacOS 11(Big Sur) + Homebrew GCC 10.2.0
- Ubuntu 20.04 + GCC 9.3.0

## Install
```bash
pip install btf-extractor
```

## Example
```python
>>> from btf_extractor import Ubo2003, Ubo2014

>>> btf = Ubo2003("UBO_CORDUROY256.zip")
>>> angles_list = list(btf.angles_set)
>>> image = btf.angles_to_image(*angles_list[0])
>>> print(image.shape)
(256, 256, 3)
>>> print(angles_list[0])
(0, 0, 0, 0)

>>> btf = Ubo2014("carpet01_resampled_W400xH400_L151xV151.btf")
>>> print(btf.img_shape)
(400, 400, 3)
>>> angles_list = list(btf.angles_set)
>>> image = btf.angles_to_image(*angles_list[0])
>>> print(image.shape)
(400, 400, 3)
>>> print(angles_list[0])
(60.0, 270.0, 60.0, 135.0)
```
