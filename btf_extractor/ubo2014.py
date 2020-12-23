"""BTFDBBのbtfファイルをbtfファイルのまま使用するためのライブラリ

BTFDBB UBO2014(*)形式のbtfファイルを参照し、
・btfファイルに含まれる角度情報の取得
・「撮影条件の角度(tl, pl, tv, pv)」から
　「画像の実体(ndarray形式(BGR, channels-last))」を取得
する関数を提供する

(*) https://cg.cs.uni-bonn.de/en/projects/btfdbb/download/ubo2014/
"""
from itertools import product
from typing import Tuple

import numpy as np

from ubo2014_cpp import FetchBTF, FetchBTF_pixel, LoadBTF, SniffBTF


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
        self._view_vecs, self._light_vecs, self.img_shape = SniffBTF(self.__raw_btf)
        self._light_vecs = np.array(self._light_vecs)
        self._view_vecs = np.array(self._view_vecs)
        self.light_set, self.view_set = self.__get_angles_set()
        self.light_set = self.light_set.round(1)
        self.view_set = self.view_set.round(1)
        self.angles_set = frozenset(
            {tuple(l) + tuple(v) for l, v in product(self.light_set, self.view_set)}
        )

        self._view_vecs.flags.writeable = False
        self._light_vecs.flags.writeable = False
        self.light_set.flags.writeable = False
        self.view_set.flags.writeable = False

    @staticmethod
    def _cartesian_to_sphere(x: float, y: float, z: float) -> Tuple[float, float]:
        """直交座標系から球面座標系に変換する"""
        theta = np.rad2deg(np.arccos(z))
        phi = np.rad2deg(np.arctan2(y, x)) % 360  # -180から+180を、0から360に収める
        return theta, phi

    def __get_angles_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """btfファイルから角度情報を取得し、`float`のndarrayで返す"""
        tl, pl = self._cartesian_to_sphere(
            self._light_vecs[:, 0], self._light_vecs[:, 1], self._light_vecs[:, 2]
        )
        tv, pv = self._cartesian_to_sphere(
            self._view_vecs[:, 0], self._view_vecs[:, 1], self._view_vecs[:, 2]
        )
        return np.array((tl, pl)).T, np.array((tv, pv)).T

    def index_to_image(self, light_idx: int, view_idx: int) -> np.ndarray:
        """`self.light_set`, `self.view_set`のインデックスで画像を指定し、ndarray形式で返す"""
        if light_idx+1 > len(self.light_set) and view_idx > len(self.view_set):
            raise IndexError(f"angle index out of range")
        img = np.array(FetchBTF(self.__raw_btf, light_idx, view_idx))
        return img[:, :, ::-1]

    def angles_to_image(self, tl: float, pl: float, tv: float, pv: float) -> np.ndarray:
        """`tl`, `pl`, `tv`, `pv`の角度条件の画像をndarray形式で返す"""
        light_idx = -1
        view_idx = -1

        for idx in range(len(self.light_set)):
            if np.allclose(self.light_set[idx], np.array((tl, pl))):
                light_idx = idx
        if light_idx == -1:
            raise ValueError(f"tl:{tl}, pl:{pl} not found")

        for idx in range(len(self.view_set)):
            if np.allclose(self.light_set[idx], np.array((tv, pv))):
                view_idx = idx
        if view_idx == -1:
            raise ValueError(f"tv:{tv}, pv:{pv} not found")

        return self.index_to_image(light_idx, view_idx)

    def angles_xy_to_pixel(
        self, light_idx: float, view_idx: float, x: int, y: int
    ) -> np.ndarray:
        """`tl`, `pl`, `tv`, `pv`の角度条件で`x`, `y`の座標の画素値をRGBのfloatで返す"""
        if light_idx+1 > len(self.light_set) and view_idx > len(self.view_set):
            raise IndexError(f"angle index out of range")
        if x+1 > self.img_shape[0] and y > self.img_shape[1]:
            raise IndexError(f"angle index out of range")
        pixel = np.array(FetchBTF_pixel(self.__raw_btf, light_idx, view_idx, x, y))
        return pixel
