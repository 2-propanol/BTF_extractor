"""BTFDBBのzipファイルをzipファイルのまま使用するためのライブラリ

BTFDBB UBO2003(*)形式, ATRIUM(**)形式のzipファイルを参照し、
・zipファイルに含まれる角度情報の取得
・「撮影条件の角度(tl, pl, tv, pv)」から
　「画像の実体(ndarray形式(BGR, channels-last))」を取得
する関数を提供する

(*) http://cg.cs.uni-bonn.de/en/projects/btfdbb/download/ubo2003/
(**) http://cg.cs.uni-bonn.de/en/projects/btfdbb/download/atrium/
"""
from collections import Counter
from sys import stderr
from typing import Any, Tuple
from zipfile import ZipFile

import numpy as np
from nptyping import NDArray
from simplejpeg import decode_jpeg

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
        """使用するzipファイルを指定する
        
        指定したzipファイルに角度条件の重複がある場合、
        何が重複しているか表示し、`RuntimeError`を投げる。
        """
        self.zip_filepath = zip_filepath
        self.__z = ZipFile(zip_filepath)

        # ファイルパスは重複しないので`filepath_set`はsetで良い
        filepath_set = {path for path in self.__z.namelist() if path.endswith(".jpg")}
        self.__angles_vs_filepath_dict = {
            self._filename_to_angles(path): path for path in filepath_set
        }
        self.angles_set = frozenset(self.__angles_vs_filepath_dict.keys())

        # 角度条件の重複がある場合、何が重複しているか調べる
        if len(filepath_set) != len(self.angles_set):
            angles_list = [self._filename_to_angles(path) for path in filepath_set]
            angle_collection = Counter(angles_list)
            for angles, counter in angle_collection.items():
                if counter > 1:
                    print(
                        f"[BTF-Extractor] '{self.zip_filepath}' has"
                        + f"{counter} files with condition {angles}.",
                        file=stderr,
                    )
            raise RuntimeError(f"'{self.zip_filepath}' has duplicated conditions.")

    @staticmethod
    def _filename_to_angles(filename: str) -> AnglesTuple:
        """ファイル名(orパス)から角度(`int`)のタプル(`tl`, `pl`, `tv`, `pv`)を取得する"""
        # ファイルパスの長さの影響を受けないように後ろから数えている
        tl = int(filename[-25:-22])
        pl = int(filename[-19:-16])
        tv = int(filename[-13:-10])
        pv = int(filename[-7:-4])
        return (tl, pl, tv, pv)

    def angles_to_image(self, tl: int, pl: int, tv: int, pv: int) -> BGRImage:
        """`tl`, `pl`, `tv`, `pv`の角度条件の画像をndarray形式で返す

        `filename`が含まれるファイルが存在しない場合は`ValueError`を投げる。
        """
        key = (tl, pl, tv, pv)
        filepath = self.__angles_vs_filepath_dict.get(key)
        if not filepath:
            raise ValueError(
                f"Condition {key} does not exist in '{self.zip_filepath}'."
            )

        with self.__z.open(filepath) as f:
            return decode_jpeg(f.read(), colorspace="BGR")
