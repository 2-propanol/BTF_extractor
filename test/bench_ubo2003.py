from btf_extractor import Ubo2003
from tqdm import tqdm, trange

FILEPATH = "UBO_CORDUROY64.zip"


print("[UBO2003] instantiation")
for _ in trange(100):
    btf = Ubo2003(FILEPATH)

print()
print("[UBO2003] extract image")
btf = Ubo2003(FILEPATH)
l = list(btf.angles_set)[:100]

for a in tqdm(l):
    _ = btf.angles_to_image(*a)
