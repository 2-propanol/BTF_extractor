"""BTFDBBのzipファイルをzipファイルのまま使用するためのライブラリ

BTFDBB UBO2003(*)形式, ATRIUM(**)形式のzipファイルを参照し、
・zipファイルに含まれる角度情報の取得
・「撮影条件の角度(tl, pl, tv, pv)」から
　「画像の実体(ndarray形式(BGR, channels-last))」を取得
する関数を提供する

(*) http://cg.cs.uni-bonn.de/en/projects/btfdbb/download/ubo2003/
(**) http://cg.cs.uni-bonn.de/en/projects/btfdbb/download/atrium/
"""
from sys import stderr
from typing import Any, Set, Tuple
from zipfile import ZipFile

import numpy as np
from nptyping import NDArray
from PIL import Image

AnglesTuple = Tuple[int, int, int, int]
BGRImage = NDArray[(Any, Any, 3), np.uint8]

class Ubo2003:
    """BTFDBBのzipファイルから角度や画像を取り出す

    角度は全て度数法(degree)を用いている。
    zipファイルに含まれる角度情報の順番は保証せず、並べ替えもしない。
    `angles_set`には`list`ではなく、順序の無い`set`を用いている。

    画像の実体はopencvと互換性のあるndarray形式(BGR, channels-last)で出力する。

    zipファイル要件:
        f"tl{tl:03} pl{pl:03} tv{tv:03} pv{pv:03}.jpg"を格納しているzipファイル

    Attributes:
        zip_filepath (str): コンストラクタに指定したzipファイルパス。
        angles_set (set[tuple[int,int,int,int]]): zipファイルに含まれる画像の角度条件の集合。

    Example:
        >>> btf = Ubo2003("UBO_CORDUROY256.zip")
        >>> angles_list = list(btf.angles_set)
        >>> image = btf.angles_to_image(*angles_list[0])
        >>> print(image.shape)
        (256, 256, 3)
        >>> print(angles_list[0])
        (0, 0, 0, 0)
    """

    def __init__(self, zip_filepath: str) -> None:
        """使用するzipファイルを指定する"""
        self.zip_filepath = zip_filepath
        self.__z = ZipFile(zip_filepath)
        self.__filepath_set = self.__get_filepath_set()
        self.angles_set = frozenset(self.__get_angles_set())

    def __get_filepath_set(self) -> Set[str]:
        """zip内の"jpg"ファイルのファイルパスの集合を取得する"""
        return {path for path in self.__z.namelist() if path.endswith(".jpg")}

    def __get_angles_set(self) -> Set[AnglesTuple]:
        """zip内の"jpg"ファイル名から角度情報を取得し、`int`のタプルの集合で返す"""
        return {self._filename_to_angles(path) for path in self.__get_filepath_set()}

    @staticmethod
    def _filename_to_angles(filename: str) -> AnglesTuple:
        """ファイル名(orパス)から角度(`int`)のタプル(`tl`, `pl`, `tv`, `pv`)を取得する"""
        # ファイルパスの長さの影響を受けないように後ろから数えている
        tl = int(filename[-25:-22])
        pl = int(filename[-19:-16])
        tv = int(filename[-13:-10])
        pv = int(filename[-7:-4])
        return (tl, pl, tv, pv)

    def _filename_to_image(self, filename: str) -> BGRImage:
        """`filename`が含まれるファイルを探し、その画像をndarray形式で返す

        `filename`が含まれるファイルが存在しない場合は`ValueError`
        `filename`が含まれるファイルが複数ある場合は`print`で警告を表示
        """
        # filepath_listからfilenameが含まれるものを探し、filepathに入れる
        filepaths = [t for t in self.__filepath_set if filename in t]

        found_files = len(filepaths)
        if found_files == 0:
            raise ValueError(f"'{filename}' does not exist in '{self.zip_filepath}'.")
        # 同じ角度情報を持つファイルが複数存在する場合に警告
        elif found_files > 1:
            print(
                "WARN:",
                f"'{self.zip_filepath}' has {found_files} '{filename}'.",
                file=stderr,
            )

        img = Image.open(self.__z.open(filepaths[0]))
        return np.array(img)[:, :, ::-1]

    def angles_to_image(self, tl: int, pl: int, tv: int, pv: int) -> BGRImage:
        """`tl`, `pl`, `tv`, `pv`の角度条件の画像をndarray形式で返す"""
        filename = f"tl{tl:03} pl{pl:03} tv{tv:03} pv{pv:03}.jpg"
        return self._filename_to_image(filename)
