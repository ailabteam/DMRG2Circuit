import tenpy
import numpy as np
import tenpy.tools.hdf5_io as hdf5_io
import matplotlib.pyplot as plt

# ==============================================================================
# HÀM CỐT LÕI TỪ BÀI HỌC 2 (chúng ta biết nó hoạt động)
# Tôi đã thêm một tùy chọn 'verbose' để tắt bớt việc in ra màn hình
# ==============================================================================
def find_ising_ground_state(L=10, J=1.0, g=0.5, chi_max=100, verbose=True):
    """
    Tìm trạng thái cơ bản của mô hình Ising.
    Trả về (năng lượng, entropy vướng víu tại trung tâm).
    """
    if verbose:
        print("-" * 80)
        print(f"Bắt đầu DMRG cho L={L}, J={J}, g={g}")
    
    model_params = dict(L=L, J=J, g=g, bc_MPS='finite', conserve=None)
    M = tenpy.models.tf_ising.TFIChain(model_params)
    psi = tenpy.networks.mps.MPS.from_product_state(M.lat.mps_sites(), ["up"] * L)

    dmrg_params = {
        'mixer': None,
        'max_E_err': 1.e-10,
        'trunc_params': {'chi_max': chi_max, 'svd_min': 1.e-10},
        'max_sweeps': 10,
    }
    info = tenpy.algorithms.dmrg.run(psi, M, dmrg_params)
    
    E = info['E']
    all_entropies = psi.entanglement_entropy()
    S_center = all_entropies[L // 2 - 1]
    
    if verbose:
        print(f"Năng lượng trạng thái cơ bản: {E:.10f}")
        print(f"Entropy vướng víu tại trung tâm: {S_center:.5f}")
    
    return E, S_center


# ==============================================================================
# HÀM MỚI CHO BÀI HỌC 3: CHẠY THÍ NGHIỆM
# ==============================================================================
def run_phase_transition_experiment():
    """
    Chạy DMRG cho nhiều giá trị của 'g' và vẽ đồ thị kết quả.
    """
    # Các tham số của thí nghiệm
    L = 20  # Tăng kích thước hệ thống lên một chút để thấy rõ hơn
    J = 1.0
    chi_max = 100
    
    # Tạo một dải các giá trị 'g' để quét qua điểm chuyển pha (g=1.0)
    g_values = np.linspace(0.1, 2.0, 20)
    
    # Các danh sách để lưu trữ kết quả
    energies = []
    entropies = []
    
    print("Bắt đầu thí nghiệm quét chuyển pha lượng tử...")
    print(f"Hệ thống: L={L}, J={J}, chi_max={chi_max}")
    
    # Vòng lặp chính: chạy qua từng giá trị của g
    for g in g_values:
        print(f"  Đang tính toán cho g = {g:.3f}...")
        E, S = find_ising_ground_state(L=L, J=J, g=g, chi_max=chi_max, verbose=False)
        energies.append(E / L) # Lưu năng lượng trên mỗi site
        entropies.append(S)

    print("Thí nghiệm hoàn tất. Đang vẽ đồ thị...")

    # Dùng matplotlib để vẽ đồ thị
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Đồ thị năng lượng
    ax1 = axes[0]
    ax1.plot(g_values, energies, 'o-')
    ax1.set_ylabel("Năng lượng trên mỗi site (E/L)")
    ax1.set_title(f"Chuyển pha lượng tử trong mô hình Ising (L={L})")
    ax1.grid(True)
    
    # Đồ thị entropy
    ax2 = axes[1]
    ax2.plot(g_values, entropies, 'o-', color='r')
    ax2.set_xlabel("Từ trường ngang (g/J)")
    ax2.set_ylabel("Entropy vướng víu tại trung tâm")
    # Vẽ một đường thẳng đứng tại điểm chuyển pha lý thuyết
    ax2.axvline(1.0, linestyle='--', color='k', label='Điểm chuyển pha (g=1.0)')
    ax2.legend()
    ax2.grid(True)

    # Lưu hình ảnh ra file
    output_filename = f"ising_phase_transition_L{L}.png"
    plt.savefig(output_filename)
    print(f"Đã lưu đồ thị vào file: '{output_filename}'")
    
    # Hiển thị đồ thị (nếu bạn có giao diện đồ họa)
    # plt.show() # Tạm thời comment dòng này lại vì bạn đang dùng server shell

# ==============================================================================
# CHẠY CHƯƠG TRÌNH CHÍNH
# ==============================================================================
if __name__ == '__main__':
    run_phase_transition_experiment()
