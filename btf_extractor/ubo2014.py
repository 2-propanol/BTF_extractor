from typing import Set, Tuple, Sequence

import numpy as np

from ubo2014_cpp import LoadBTF, SniffBTF, FetchBTF


class Ubo2014:
    """BTFDBBのzipファイルから角度や画像を取り出す"""

    def __init__(self, btf_filepath: str) -> None:
        """使用するzipファイルを指定する"""
        if not type(btf_filepath) is str:
            raise (TypeError, "`btf_filepath` is not str")
        self.btf_filepath = btf_filepath
        self.__raw_btf = LoadBTF(btf_filepath)
        self._view_vecs, self._light_vecs = SniffBTF(self.__raw_btf)
        self._light_vecs = np.array(self._light_vecs)
        self._view_vecs = np.array(self._view_vecs)
        self.light_set, self.view_set = self.__get_angles_set()
        self.light_set = self.light_set.round(1)
        self.view_set = self.view_set.round(1)

        self._view_vecs.flags.writeable = False
        self._light_vecs.flags.writeable = False
        self.light_set.flags.writeable = False
        self.view_set.flags.writeable = False

    @staticmethod
    def _cartesian_to_sphere(x: float, y: float, z: float) -> Tuple[float, float]:
        """直交座標系から球面座標系に変換"""
        theta = np.rad2deg(np.arccos(z))
        phi = np.rad2deg(np.arctan2(y, x)) % 360  # -180から+180を、0から360に収める
        return theta, phi

    def __get_angles_set(self) -> np.ndarray:
        """btfファイルから角度情報を取得し、floatのndarrayで返す"""
        tl, pl = self._cartesian_to_sphere(
            self._light_vecs[:, 0], self._light_vecs[:, 1], self._light_vecs[:, 2]
        )
        tv, pv = self._cartesian_to_sphere(
            self._view_vecs[:, 0], self._view_vecs[:, 1], self._view_vecs[:, 2]
        )
        return np.array((tl, pl)).T, np.array((tv, pv)).T

    def index_to_image(self, light_idx: int, view_idx: int) -> np.ndarray:
        """`tl`, `pl`, `tv`, `pv`の角度を持つ画像の実体をndarray形式(BGR)で返す"""
        img = np.array(FetchBTF(self.__raw_btf, light_idx, view_idx))
        return img[:, :, ::-1]

    def angles_to_image(self, tl: float, pl: float, tv: float, pv: float) -> np.ndarray:
        """`tl`, `pl`, `tv`, `pv`の角度を持つ画像の実体をndarray形式で返す"""
        light_idx = 0
        view_idx = 0
        for light_idx in range(len(self.light_set)):
            if np.allclose(self.light_set[light_idx], np.array((tl, pl))):
                print("hit", light_idx)
                break
        for view_idx in range(len(self.view_set)):
            if np.allclose(self.light_set[view_idx], np.array((tv, pv))):
                print("hit", view_idx)
                break
        return self.index_to_image(light_idx, view_idx)
