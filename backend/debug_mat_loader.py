import scipy.io
import os
import numpy as np

data_dir = os.path.join(os.path.dirname(__file__), '../data')
mat = scipy.io.loadmat(os.path.join(data_dir, 'all_splines_0.mat'))

root = mat['all_splines']  # (1, 100) object array
print(f"root shape: {root.shape}")

# Pick first tree
tree = root[0, 0]  # (1, 81) object array
print(f"tree shape: {tree.shape}")

# Pick first spline in tree
sp = tree.flatten()[0]
print(f"\nsp type: {type(sp)}, dtype: {sp.dtype}, shape: {sp.shape}")

# Access fields
for field in sp.dtype.names:
    val = sp[field]
    print(f"\n--- {field} ---")
    print(f"  type: {type(val)}, dtype: {val.dtype}, shape: {val.shape}")
    
    # Unwrap nested object arrays
    v = val
    depth = 0
    while v.dtype == object and v.size == 1:
        v = v.flat[0]
        depth += 1
        print(f"  unwrap {depth}: type={type(v)}, ", end="")
        if hasattr(v, 'dtype'):
            print(f"dtype={v.dtype}, shape={v.shape}")
        else:
            print(f"value={v}")
    
    if hasattr(v, 'dtype') and v.dtype != object:
        print(f"  FINAL: dtype={v.dtype}, shape={v.shape}")
        if v.size < 20:
            print(f"  values: {v}")
