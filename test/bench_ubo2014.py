from btf_extractor import Ubo2014
from tqdm import tqdm, trange

FILEPATH = "carpet01_resampled_W400xH400_L151xV151.btf"


print("[UBO2014] instantiation")
for _ in trange(100):
    btf = Ubo2014(FILEPATH)

print()
print("[UBO2014] extract image")
btf = Ubo2014(FILEPATH)
l = list(btf.angles_set)[:100]

for a in tqdm(l):
    _ = btf.angles_to_image(*a)
