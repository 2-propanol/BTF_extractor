# cython: language_level=3
from itertools import product
from typing import Any, Tuple

cimport cython
import numpy as np
cimport numpy as np
from cython.parallel import prange

from btf_extractor.c_ext.btf cimport BTF, Spectrum, Vector3
from btf_extractor.c_ext.btf cimport BTFFetchSpectrum, DestroyBTF, LoadBTF

DTYPE_F32 = np.float32
ctypedef np.float32_t DTYPE_F32_t


cdef class Ubo2014:
    """BTFDBBのbtfファイルから角度や画像を取り出す

    角度は全て度数法(degree)を用いている。
    画像の実体はopencvと互換性のあるndarray形式(BGR, channels-last)で出力する。

    `col_decode_range`、`row_decode_range`は負の値による指定も可能。
    `img_shape`の範囲外を指定しようとすると`ValueError`を返す。
    `*_decode_range`の書き換えで`img_shape`は変更されない。

    Attributes:
        btf_filepath (str):
            コンストラクタに指定したbtfファイルパス。読み取り専用。
        img_shape (tuple[float,float,float]):
            btfファイルに含まれている画像のshape。読み取り専用。
        angles_set (set[tuple[float,float,float,float]]):
            btfファイルに含まれる画像の角度条件の集合。読み取り専用。
        col_decode_range (tuple[int,int]):
            `angles_to_image()`でデコードする列の範囲。読み書き可能。
        row_decode_range (tuple[int,int]):
            `angles_to_image()`でデコードする行の範囲。読み書き可能。

    Example:
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
        >>> btf.row_decode_range = (136, 264)
        >>> btf.col_decode_range = (136, -136)
        >>> print(btf.angles_to_image(*angles_list[0]).shape)
        (128, 128, 3)
        >>> print(btf.img_shape)
        (400, 400, 3)
        >>> btf.row_decode_range = (136, 401)
        ValueError: out of range
        >>> btf.row_decode_range = ("136", "264")
        TypeError: indices must be integers
        >>> btf.row_decode_range = (136, 264, 1)
        ValueError: too many values to unpack (expected 2)
    """
    cdef readonly str btf_filepath
    cdef BTF* __raw_btf
    cdef readonly tuple img_shape
    cdef readonly frozenset angles_set
    cdef readonly dict angles_vs_index_dict
    cdef Py_ssize_t row_decode_begin
    cdef Py_ssize_t row_decode_length
    cdef Py_ssize_t col_decode_begin
    cdef Py_ssize_t col_decode_length

    def __cinit__(self, btf_filepath: str) -> None:
        """使用するbtfファイルを指定する"""
        if not type(btf_filepath) is str:
            raise TypeError("`btf_filepath` is not str")
        self.btf_filepath = btf_filepath

        # C++のstringの変換にはポインタの参照が必要。
        # なので、`LoadBTF(btf_filepath.encode(), NULL)`はコンパイルエラーになる。
        cdef bytes py_byte_string = btf_filepath.encode()
        self.__raw_btf = LoadBTF(py_byte_string, NULL)

        # self.__raw_btf.ChannelCountを3と仮定する。
        self.img_shape = (self.__raw_btf.Width, self.__raw_btf.Height, 3)
        self.row_decode_begin = 0
        self.row_decode_length = self.__raw_btf.Width
        self.col_decode_begin = 0
        self.col_decode_length = self.__raw_btf.Height
        cdef Py_ssize_t num_views = self.__raw_btf.ViewCount
        cdef Py_ssize_t num_lights = self.__raw_btf.LightCount

        # *_in_cartesian: 直交座標系で格納された角度情報
        cdef np.ndarray[DTYPE_F32_t, ndim=2] _lights_in_cartesian = np.empty(
            (num_lights, 3), dtype=DTYPE_F32
        )
        cdef Vector3 light
        for light_idx in range(num_lights):
            light = self.__raw_btf.Lights[light_idx]
            _lights_in_cartesian[light_idx,0] = light.x
            _lights_in_cartesian[light_idx,1] = light.y
            _lights_in_cartesian[light_idx,2] = light.z

        cdef np.ndarray[DTYPE_F32_t, ndim=2] _views_in_cartesian = np.empty(
            (num_views, 3), dtype=DTYPE_F32
        )
        cdef Vector3 view
        for view_idx in range(num_views):
            view = self.__raw_btf.Views[view_idx]
            _views_in_cartesian[view_idx,0] = view.x
            _views_in_cartesian[view_idx,1] = view.y
            _views_in_cartesian[view_idx,2] = view.z

        # *_in_spherical: 球面座標系で格納された角度情報
        cdef np.ndarray[DTYPE_F32_t, ndim=2] lights_in_spherical = (
            self._cartesian_to_spherical(_lights_in_cartesian)
        )

        cdef np.ndarray[DTYPE_F32_t, ndim=2] views_in_spherical = (
            self._cartesian_to_spherical(_views_in_cartesian)
        )

        # cdef list angles_list
        cdef list angles_list = [
            (float(l[0]), float(l[1]), float(v[0]), float(v[1]))
            for l, v in product(lights_in_spherical, views_in_spherical)
        ]

        # このBTFファイルに含まれる(tl, pl, tv, pv)の組からindexを引く辞書
        self.angles_vs_index_dict = {
            angles: (lidx, vidx)
            for angles, (lidx, vidx)
            in zip(angles_list, product(range(num_lights), range(num_views)))
        }

        # このBTFファイルに含まれる(tl, pl, tv, pv)の組が格納されたset
        self.angles_set = frozenset(angles_list)

    def __dealloc__(self) -> None:
        DestroyBTF(self.__raw_btf)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef np.ndarray[DTYPE_F32_t, ndim=2] _cartesian_to_spherical(
        self, np.ndarray[DTYPE_F32_t, ndim=2] xyz
    ):
        """直交座標系から球面座標系に変換する"""
        cdef np.ndarray[DTYPE_F32_t, ndim=1] theta
        cdef np.ndarray[DTYPE_F32_t, ndim=1] phi
        theta = np.rad2deg(np.arccos(xyz[:,2]))
        # -180から+180を、0から360に収める
        phi = np.rad2deg(np.arctan2(xyz[:,1], xyz[:,0])) % 360
        # UBO2014の角度情報の精度は0.5度単位なので小数点以下1桁に丸める
        return np.array((theta, phi)).T.round(1)

    @property
    def row_decode_range(self) -> Tuple[int, int]:
        return (self.row_decode_begin, self.row_decode_end)

    @row_decode_range.setter
    def row_decode_range(self, row_index: Tuple[int, int]) -> None:
        if not (type(row_index[0]) is int and type(row_index[1]) is int):
            raise TypeError("indices must be integers")

        cdef int left, right
        left, right = row_index
        if left < 0:
            left = self.__raw_btf.Width + left
        if right < 0:
            right = self.__raw_btf.Width + right

        if right < self.__raw_btf.Width and left < right:
            self.row_decode_begin = left
            self.row_decode_length = right - left
        else:
            raise ValueError("out of range")

    @property
    def col_decode_range(self) -> Tuple[int, int]:
        return (self.col_decode_begin, self.col_decode_end)

    @col_decode_range.setter
    def col_decode_range(self, col_index: Tuple[int, int]) -> None:
        if not (type(col_index[0]) is int and type(col_index[1]) is int):
            raise TypeError("indices must be integers")

        cdef int top, bottom
        top, bottom = col_index
        if top < 0:
            top = self.__raw_btf.Height + top
        if bottom < 0:
            bottom = self.__raw_btf.Height + bottom

        if bottom < self.__raw_btf.Height and top < bottom:
            self.col_decode_begin = top
            self.col_decode_length = bottom - top
        else:
            raise ValueError("out of range")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef np.ndarray[DTYPE_F32_t, ndim=3] _index_to_image_skip_validation(
        self, int light_idx, int view_idx
    ):
        cdef Py_ssize_t col_begin = self.col_decode_begin
        cdef Py_ssize_t row_begin = self.row_decode_begin
        cdef Py_ssize_t height = self.col_decode_length
        cdef Py_ssize_t width = self.row_decode_length
        cdef Py_ssize_t lidx = light_idx
        cdef Py_ssize_t vidx = view_idx
        cdef Spectrum spec
        cdef np.ndarray[DTYPE_F32_t, ndim=3] img3d = np.empty(
            (height, width, 3), dtype=DTYPE_F32
        )
        cdef float[:, :, ::1] img3d_view = img3d
        cdef int x, y
        for y in prange(height, nogil=True, schedule="dynamic"):
            for x in range(width):
                spec = BTFFetchSpectrum(
                    self.__raw_btf, lidx, vidx, x+row_begin, y+col_begin
                )
                img3d_view[y,x,0] = spec.z
                img3d_view[y,x,1] = spec.y
                img3d_view[y,x,2] = spec.x
        return img3d

    def angles_to_image(
        self, tl: float, pl: float, tv: float, pv: float
    ) -> np.ndarray:
        """`tl`, `pl`, `tv`, `pv`の角度条件の画像をndarray形式で返す"""
        cdef tuple key = (tl, pl, tv, pv)
        cdef tuple indices = self.angles_vs_index_dict.get(key)
        if not indices:
            raise ValueError(
                f"Condition {key} does not exist in '{self.btf_filepath}'."
            )
        return self._index_to_image_skip_validation(indices[0], indices[1])

    @cython.nonecheck(False)
    def _angleindex_to_image_unsafe(
        self, light_idx: int, view_idx: int
    ) -> np.ndarray:
        """`light_idx`, `view_idx`で指定される角度条件の画像をndarray形式で返す

        値(引数)の妥当性をチェックしない。不正な値を入力するとクラッシュする。
        """
        return self._index_to_image_skip_validation(light_idx, view_idx)
