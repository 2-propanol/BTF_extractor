import cv2
import numpy as np

from ubo2014_cpp import LoadBTF

btf_filename = "wallpaper11_resampled_W400xH400_L151xV151.btf"

for l, v in ((0, 0), (30, 30), (60, 60), (90, 90), (120, 120), (150, 150)):
    img = np.array(LoadBTF(btf_filename, l, v))
    cv2.imwrite(f"out_{l}_{v}.jpg", np.clip(img * 255, 0, 255))
