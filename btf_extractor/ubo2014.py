"""BTFDBBのbtfファイルをbtfファイルのまま使用するためのライブラリ

BTFDBB UBO2014(*)形式のbtfファイルを参照し、
・btfファイルに含まれる角度情報の取得
・「撮影条件の角度(tl, pl, tv, pv)」から
　「画像の実体(ndarray形式(BGR, channels-last))」を取得
する関数を提供する

(*) https://cg.cs.uni-bonn.de/en/projects/btfdbb/download/ubo2014/
"""
from itertools import product
from typing import Any, Tuple

import numpy as np
from nptyping import NDArray

from ubo2014_cpp import FetchBTF, FetchBTF_pixel, LoadBTF, SniffBTF

BGRImage = NDArray[(Any, Any, 3), np.uint8]


class Ubo2014:
    """BTFDBBのbtfファイルから角度や画像を取り出す

    角度は全て度数法(degree)を用いている。
    画像の実体はopencvと互換性のあるndarray形式(BGR, channels-last)で出力する。

    Attributes:
        btf_filepath (str): コンストラクタに指定したbtfファイルパス。
        img_shape (tuple[int,int,int]): btfファイルに含まれている画像のshape。
        angles_set (set[tuple[float,float,float,float]]):
            btfファイルに含まれる画像の角度条件の集合。

    Example:
        >>> btf = Ubo2014("carpet01_resampled_W400xH400_L151xV151.btf")
        >>> print(btf.img_shape)
        (400, 400, 3)
        >>> angles_list = list(btf.angles_set)
        >>> image = btf.angles_to_image(*angles_list[0])
        >>> print(image.shape)
        (400, 400, 3)
        >>> print(angles_list[0])
        (60.0, 270.0, 60.0, 135.0)
    """

    def __init__(self, btf_filepath: str) -> None:
        """使用するbtfファイルを指定する"""
        if not type(btf_filepath) is str:
            raise (TypeError, "`btf_filepath` is not str")
        self.btf_filepath = btf_filepath
        self.__raw_btf = LoadBTF(btf_filepath)

        # *_in_cartesian: 直交座標系で格納された角度情報
        self._views_in_cartesian, self._lights_in_cartesian, self.img_shape = SniffBTF(
            self.__raw_btf
        )
        self._lights_in_cartesian: NDArray[(Any, 2), np.float64] = np.array(
            self._lights_in_cartesian
        )
        self._views_in_cartesian: NDArray[(Any, 2), np.float64] = np.array(
            self._views_in_cartesian
        )

        self.num_lights = len(self._lights_in_cartesian)
        self.num_views = len(self._views_in_cartesian)

        # *_in_spherical: 球面座標系で格納された角度情報
        tl, pl = self._cartesian_to_spherical(
            self._lights_in_cartesian[:, 0],
            self._lights_in_cartesian[:, 1],
            self._lights_in_cartesian[:, 2],
        )
        tv, pv = self._cartesian_to_spherical(
            self._views_in_cartesian[:, 0],
            self._views_in_cartesian[:, 1],
            self._views_in_cartesian[:, 2],
        )
        self.lights_in_spherical: NDArray[(Any, 2), np.float16] = np.array((tl, pl)).T
        self.views_in_spherical: NDArray[(Any, 2), np.float16] = np.array((tv, pv)).T

        # UBO2014の角度情報の精度は0.5度単位なので小数点以下1桁に丸める
        self.lights_in_spherical = self.lights_in_spherical.round(1)
        self.views_in_spherical = self.views_in_spherical.round(1)

        # このBTFファイルに含まれる(tl, pl, tv, pv)の組が格納されたset
        self.angles_set = frozenset(
            {
                tuple(l) + tuple(v)
                for l, v in product(self.lights_in_spherical, self.views_in_spherical)
            }
        )

        # 不測の書き換えを防止する
        self._views_in_cartesian.flags.writeable = False
        self._lights_in_cartesian.flags.writeable = False
        self.lights_in_spherical.flags.writeable = False
        self.views_in_spherical.flags.writeable = False

    @staticmethod
    def _cartesian_to_spherical(x: float, y: float, z: float) -> Tuple[float, float]:
        """直交座標系から球面座標系に変換する"""
        theta = np.rad2deg(np.arccos(z))
        phi = np.rad2deg(np.arctan2(y, x)) % 360  # -180から+180を、0から360に収める
        return theta, phi

    def _index_to_image(self, light_idx: int, view_idx: int) -> BGRImage:
        """`index_to_image()` インデックスチェック無し"""
        return np.array(FetchBTF(self.__raw_btf, light_idx, view_idx), dtype=np.float16).reshape(self.img_shape)[..., ::-1]

    def index_to_image(self, light_idx: int, view_idx: int) -> BGRImage:
        """`self._*s_in_spherical`のインデックスで画像を指定し、ndarray形式で返す"""
        if light_idx >= self.num_lights and view_idx >= self.num_views:
            raise IndexError(f"angle index out of range")

        return self._index_to_image(light_idx, view_idx)

    def light_angles_to_index(self, theta_phi: NDArray[(2), np.float16]):
        """`self.lights_in_spherical`から`theta_phi`が格納されているindexを探索する"""
        return np.where(np.all(self.lights_in_spherical == theta_phi, axis=1))[0]

    def view_angles_to_index(self, theta_phi: NDArray[(2), np.float16]):
        """`self.views_in_spherical`から`theta_phi`が格納されているindexを探索する"""
        return np.where(np.all(self.views_in_spherical == theta_phi, axis=1))[0]

    def angles_to_image(self, tl: float, pl: float, tv: float, pv: float) -> BGRImage:
        """`tl`, `pl`, `tv`, `pv`の角度条件の画像をndarray形式で返す"""
        light_idx = self.light_angles_to_index(np.array((tl, pl)))
        view_idx = self.view_angles_to_index(np.array((tv, pv)))

        if light_idx.size == 0 or view_idx.size == 0:
            raise ValueError(f"tl:{tl}, pl:{pl}, tv:{tv}, pv:{pv} not found")

        return self._index_to_image(light_idx[0], view_idx[0])

    def angles_xy_to_pixel(
        self, light_idx: float, view_idx: float, x: int, y: int
    ) -> NDArray[(2), np.float16]:
        """`tl`, `pl`, `tv`, `pv`の角度条件の`x`, `y`の座標の画素値をRGBのfloatで返す"""
        if light_idx >= self.num_lights and view_idx >= self.num_views:
            raise IndexError(f"angle index out of range")
        if x >= self.img_shape[0] and y >= self.img_shape[1]:
            raise IndexError(f"xy index out of range")

        pixel = np.array(FetchBTF_pixel(self.__raw_btf, light_idx, view_idx, x, y))
        return pixel.astype(np.float16)
