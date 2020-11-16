from typing import Set, Tuple

import numpy as np

from ubo2014_cpp import LoadBTF

AnglesTuple = Tuple[int, int, int, int]


class Ubo2014:
    """BTFDBBのzipファイルから角度や画像を取り出す"""

    def __init__(self, btf_filepath: str) -> None:
        """使用するzipファイルを指定する"""
        self.btf_filepath = btf_filepath

    def get_angles_set(self) -> Set[AnglesTuple]:
        """zip内の"jpg"ファイル名から角度情報を取得し、intのタプルの集合で返す"""
        raise NotImplementedError("get_angles_set")
        # return {self.filename_to_angles(path) for path in self.get_filepath_set()}

    def angles_to_image(self, tl: int, pl: int, tv: int, pv: int) -> np.ndarray:
        """`tl`, `pl`, `tv`, `pv`の角度を持つ画像の実体をndarray形式で返す"""
        raise NotImplementedError("angles_to_image")
        # filename = f"tl{tl:03} pl{pl:03} tv{tv:03} pv{pv:03}.jpg"
        # return self.filename_to_image(filename)

    def raw_index_to_image(self, l: int, v: int) -> np.ndarray:
        img = np.array(LoadBTF(self.btf_filename, l, v))
        return img[:,:,::-1]
