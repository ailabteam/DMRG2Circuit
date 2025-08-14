# CODE LẤY TỪ VÍ DỤ CHÍNH THỨC CỦA TENPY - ĐẢM BẢO CHẠY ĐƯỢC
# Nguồn: https://tenpy.readthedocs.io/en/latest/introduction/getting_started.html

import tenpy
import numpy as np

# 1. Định nghĩa tham số mô hình
L = 10
J = 1.0
g = 1.0
model_params = dict(L=L, J=J, g=g, bc_MPS='finite', conserve=None)
print("Tham số mô hình:", model_params)

# 2. Xây dựng mô hình
# Cách xây dựng mô hình trong ví dụ này khác hoàn toàn
# Họ không dùng 'lattice' mà dùng 'conserve=None' và truyền trực tiếp.
M = tenpy.models.tf_ising.TFIChain(model_params)

# 3. Tạo trạng thái ban đầu
# Đây là phần quan trọng: M.lat.mps_sites() chính là `sites` mà chúng ta cần
psi = tenpy.networks.mps.MPS.from_product_state(M.lat.mps_sites(), ["up"] * L)

# 4. Cấu hình và chạy DMRG
dmrg_params = {
    'mixer': None,  # disable mixer, needs TEBD
    'max_E_err': 1.e-10,
    'trunc_params': {
        'chi_max': 100,
        'svd_min': 1.e-10,
    },
    'max_sweeps': 10,
}
info = tenpy.algorithms.dmrg.run(psi, M, dmrg_params)

# 5. In kết quả
E = info['E']
print("-" * 50)
print(f"DMRG hoàn tất.")
print(f"Năng lượng trạng thái cơ bản: {E:.10f}")
print(f"Năng lượng trên mỗi site: {E / L:.10f}")

# Đo entropy
S = psi.entanglement_entropy()
print("Entropy vướng víu tại mỗi liên kết:", S)
