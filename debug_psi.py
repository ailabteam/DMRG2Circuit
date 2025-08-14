# file: debug_psi.py
import tenpy
import numpy as np

# --- Phần code tối giản để tạo đối tượng psi (chúng ta biết nó hoạt động) ---
L = 10
model_params = dict(L=L, J=1.0, g=0.5, bc_MPS='finite', conserve=None)
M = tenpy.models.tf_ising.TFIChain(model_params)
psi = tenpy.networks.mps.MPS.from_product_state(M.lat.mps_sites(), ["up"] * L)
# Không cần chạy DMRG, chúng ta chỉ cần đối tượng psi
# -------------------------------------------------------------------------

# --- PHẦN GỠ LỖI QUAN TRỌNG ---
print("-" * 80)
print("Bắt đầu kiểm tra đối tượng `psi`...")

# Lấy tất cả các thuộc tính và phương thức có sẵn của đối tượng psi
all_attributes = dir(psi)

# Lọc ra những cái có chứa chữ 'save' hoặc 'hdf5'
save_methods = [attr for attr in all_attributes if 'save' in attr or 'hdf5' in attr]

print("\nCác phương thức/thuộc tính tìm thấy có liên quan đến 'save'/'hdf5':")
if save_methods:
    print(save_methods)
else:
    print("!!! KHÔNG TÌM THẤY phương thức nào để lưu file trên đối tượng MPS.")

print("-" * 80)
