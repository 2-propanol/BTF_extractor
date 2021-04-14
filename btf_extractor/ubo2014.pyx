# cython: language_level=3
from itertools import product
from typing import Any, Tuple

cimport cython
import numpy as np
cimport numpy as np

from btf cimport BTF, Spectrum, Vector3
from btf cimport BTFFetchSpectrum, DestroyBTF, LoadBTF

DTYPE_F32 = np.float32
ctypedef np.float32_t DTYPE_F32_t


cdef class Ubo2014:
    """BTFDBBのbtfファイルから角度や画像を取り出す

    角度は全て度数法(degree)を用いている。
    画像の実体はopencvと互換性のあるndarray形式(BGR, channels-last)で出力する。

    Attributes:
        btf_filepath (str): コンストラクタに指定したbtfファイルパス。
        img_shape (tuple[float,float,float]): btfファイルに含まれている画像のshape。
        angles_set (set[tuple[float,float,float,float]]):
            btfファイルに含まれる画像の角度条件の集合。

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
    """
    cdef public str btf_filepath
    cdef BTF* __raw_btf
    cdef public tuple img_shape
    cdef public frozenset angles_set
    cdef dict __angles_vs_index_dict

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
        cdef Py_ssize_t num_views = self.__raw_btf.ViewCount
        cdef Py_ssize_t num_lights = self.__raw_btf.LightCount

        # *_in_cartesian: 直交座標系で格納された角度情報
        cdef np.ndarray[DTYPE_F32_t, ndim=2] _views_in_cartesian = np.empty(
            (num_views, 3), dtype=DTYPE_F32
        )
        cdef Vector3 view
        for view_idx in range(num_views):
            view = self.__raw_btf.Views[view_idx]
            _views_in_cartesian[view_idx,0] = view.x
            _views_in_cartesian[view_idx,1] = view.y
            _views_in_cartesian[view_idx,2] = view.z

        cdef np.ndarray[DTYPE_F32_t, ndim=2] _lights_in_cartesian = np.empty(
            (num_lights, 3), dtype=DTYPE_F32
        )
        cdef Vector3 light
        for light_idx in range(num_lights):
            light = self.__raw_btf.Lights[light_idx]
            _lights_in_cartesian[light_idx,0] = light.x
            _lights_in_cartesian[light_idx,1] = light.y
            _lights_in_cartesian[light_idx,2] = light.z

        # *_in_spherical: 球面座標系で格納された角度情報
        cdef np.ndarray[DTYPE_F32_t, ndim=2] lights_in_spherical = np.empty(
            (num_lights, 3), dtype=DTYPE_F32
        )
        cdef np.ndarray[DTYPE_F32_t, ndim=2] views_in_spherical = np.empty(
            (num_views, 3), dtype=DTYPE_F32
        )
        lights_in_spherical = self._cartesian_to_spherical(
            _lights_in_cartesian
        )
        views_in_spherical = self._cartesian_to_spherical(
            _views_in_cartesian
        )

        # cdef np.ndarray[DTYPE_F32_t, ndim=2] _views_in_spherical = np.empty((self.__raw_btf.ViewCount, 2), dtype=DTYPE_F32)
        # cdef Vector3 view
        # for view_idx in range(self.__raw_btf.ViewCount):
        #     view = self.__raw_btf.Views[view_idx]
        #     _views_in_spherical[view_idx,0] = acos(view.z)  #tv
        #     _views_in_spherical[view_idx,1] = atan2(view.y, view.x) % 360  #pv
        # self.num_views = len(_views_in_spherical)

        # cdef np.ndarray[DTYPE_F32_t, ndim=2] _lights_in_spherical = np.empty((self.__raw_btf.LightCount, 2), dtype=DTYPE_F32)
        # cdef Vector3 light
        # for view_idx in range(self.__raw_btf.LightCount):
        #     view = self.__raw_btf.Views[view_idx]
        #     _lights_in_spherical[view_idx,0] = acos(light.z)  #tl
        #     _lights_in_spherical[view_idx,1] = atan2(light.y, light.x) % 360  #pl
        # self.num_lights = len(_lights_in_spherical)

        # UBO2014の角度情報の精度は0.5度単位なので小数点以下1桁に丸める
        # self.lights_in_spherical = self.lights_in_spherical.round(1)
        # self.views_in_spherical = self.views_in_spherical.round(1)

        cdef list angles_list
        angles_list = [
            (float(l[0]), float(l[1]), float(v[0]), float(v[1]))
            for l, v in product(lights_in_spherical, views_in_spherical)
        ]

        # このBTFファイルに含まれる(tl, pl, tv, pv)の組からindexを引く辞書
        self.__angles_vs_index_dict = {
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
        phi = np.rad2deg(np.arctan2(xyz[:,1], xyz[:,0])) % 360  # -180から+180を、0から360に収める
        return np.array((theta, phi)).T.round(1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef np.ndarray[DTYPE_F32_t, ndim=3] _index_to_image_skip_validation(
        self, int light_idx, int view_idx
    ):
        cdef Py_ssize_t height = self.__raw_btf.Height
        cdef Py_ssize_t width = self.__raw_btf.Width
        cdef Py_ssize_t lidx = light_idx
        cdef Py_ssize_t vidx = view_idx
        cdef Spectrum spec
        cdef np.ndarray[DTYPE_F32_t, ndim=3] img3d = np.empty(
            (height, width, 3), dtype=DTYPE_F32
        )
        cdef float[:, :, ::1] img3d_view = img3d
        for y in range(height):
            for x in range(width):
                spec = BTFFetchSpectrum(self.__raw_btf, lidx, vidx, x, y)
                img3d_view[y,x,0] = spec.x
                img3d_view[y,x,1] = spec.y
                img3d_view[y,x,2] = spec.z
        return img3d

    def angles_to_image(self, tl: float, pl: float, tv: float, pv: float):
        """`tl`, `pl`, `tv`, `pv`の角度条件の画像をndarray形式で返す"""
        cdef tuple key = (tl, pl, tv, pv)
        cdef tuple indices = self.__angles_vs_index_dict.get(key)
        if not indices:
            raise ValueError(
                f"Condition {key} does not exist in '{self.btf_filepath}'."
            )
        return self._index_to_image_skip_validation(indices[0], indices[1])

    # def angles_xy_to_pixel(
    #     self, light_idx: float, view_idx: float, x: int, y: int
    # ) -> NDArray[(2), np.float16]:
    #     """`tl`, `pl`, `tv`, `pv`の角度条件の`x`, `y`の座標の画素値をRGBのfloatで返す"""
    #     if light_idx >= self.num_lights and view_idx >= self.num_views:
    #         raise IndexError(f"angle index out of range")
    #     if x >= self.img_shape[0] and y >= self.img_shape[1]:
    #         raise IndexError(f"xy index out of range")

    #     pixel = np.array(FetchBTF_pixel(self.__raw_btf, light_idx, view_idx, x, y))
    #     return pixel.astype(np.float16)
