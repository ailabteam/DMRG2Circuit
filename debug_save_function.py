# file: debug_save_function.py
import tenpy
import numpy as np

print("Đang tạo đối tượng MPS để kiểm tra...")
# --- Phần code tối giản để tạo đối tượng psi ---
L = 10
model_params = dict(L=L, J=1.0, g=0.5, bc_MPS='finite', conserve=None)
M = tenpy.models.tf_ising.TFIChain(model_params)
psi = tenpy.networks.mps.MPS.from_product_state(M.lat.mps_sites(), ["up"] * L)

print("\n" + "="*80)
print("TÀI LIỆU HƯỚNG DẪN CHO HÀM `psi.save_hdf5`:")
print("="*80)

# In ra tài liệu hướng dẫn của hàm
help(psi.save_hdf5)

print("="*80)
