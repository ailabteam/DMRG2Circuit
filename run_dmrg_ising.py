import tenpy
import numpy as np

# THAY ĐỔI QUAN TRỌNG NHẤT: IMPORT MODULE CHUYÊN DỤNG
import tenpy.tools.hdf5_io as hdf5_io

def find_ising_ground_state(L=10, J=1.0, g=0.5):
    """
    Hàm này tìm trạng thái cơ bản của mô hình Ising,
    đo lường các đại lượng và lưu kết quả MPS vào file HDF5.
    """
    print("-" * 80)
    print(f"Bắt đầu DMRG cho L={L}, J={J}, g={g}")
    
    model_params = dict(L=L, J=J, g=g, bc_MPS='finite', conserve=None)
    M = tenpy.models.tf_ising.TFIChain(model_params)
    psi = tenpy.networks.mps.MPS.from_product_state(M.lat.mps_sites(), ["up"] * L)

    dmrg_params = {
        'mixer': None,
        'max_E_err': 1.e-10,
        'trunc_params': {'chi_max': 100, 'svd_min': 1.e-10},
        'max_sweeps': 10,
    }
    info = tenpy.algorithms.dmrg.run(psi, M, dmrg_params)
    
    E = info['E']
    print(f"Năng lượng trạng thái cơ bản: {E:.10f}")

    center_site = L // 2
    Sz_center = psi.expectation_value("Sz", [center_site])[0]
    print(f"Từ hóa Sz tại site {center_site}: {Sz_center:.5f}")
    ZZ_corr = psi.correlation_function("Sz", "Sz", [0], [L - 1])[0, 0]
    print(f"Tương quan Sz-Sz giữa site 0 và {L - 1}: {ZZ_corr:.5f}")
    all_entropies = psi.entanglement_entropy()
    S_center = all_entropies[center_site - 1] 
    print(f"Entropy vướng víu tại trung tâm: {S_center:.5f}")
    
    # --- CÁCH LƯU FILE ĐÚNG 100% ---
    filename = f"gs_ising_L{L}_J{J}_g{g}.h5"
    
    # Tạo một dictionary chứa tất cả dữ liệu chúng ta muốn lưu
    data_to_save = {
        "psi": psi,
        "model": M,
        "params": model_params
    }

    # Gọi hàm save từ module hdf5_io
    hdf5_io.save(data_to_save, filename)

    print(f"Đã lưu thành công dữ liệu vào file: '{filename}'")
    
    return E, psi

if __name__ == '__main__':
    ground_energy, ground_state_psi = find_ising_ground_state(L=10, g=0.5)
    
    print("-" * 80)
    print("Thực thi hoàn tất.")
    print("Đối tượng MPS trả về:", ground_state_psi)
    print("Bond dimensions của MPS:", ground_state_psi.chi)
