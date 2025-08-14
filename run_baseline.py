import tenpy
import tenpy.tools.hdf5_io as hdf5_io
import numpy as np
from qiskit import QuantumCircuit
import time
import os
import json

# ==============================================================================
# PHẦN 1: LOAD DỮ LIỆU MỤC TIÊU (TENPY)
# ==============================================================================
def get_target_statevector(filename):
    """
    Đọc file MPS và chuyển nó thành một statevector đầy đủ.
    """
    print(f"  Đang đọc và chuyển đổi MPS từ file: '{filename}'")
    psi_mps = hdf5_io.load(filename)["psi"]
    psi_mps.canonical_form()
    full_tensor_array = psi_mps.get_theta(0, psi_mps.L)
    target_state = full_tensor_array.to_ndarray().flatten()
    print(f"  Đã chuyển đổi MPS thành statevector. Shape: {target_state.shape}")
    return target_state

# ==============================================================================
# PHẦN 2: HÀM CHẠY THÍ NGHIỆM BASELINE
# ==============================================================================
def run_qiskit_baseline(L, J, g):
    """
    Chạy thí nghiệm baseline hoàn chỉnh cho một bộ tham số và lưu kết quả.
    """
    exp_name = f"L{L}_g{g}_baseline"
    print("\n" + "#"*80 + f"\n# Baseline Experiment: {exp_name} #" + "\n" + "#"*80)

    results_dir = f"results/{exp_name}"; os.makedirs(results_dir, exist_ok=True)
    
    # --- Chuẩn bị dữ liệu ---
    mps_filename = f"data/gs_ising_L{L}_J{J}_g{g}.h5"
    if not os.path.exists(mps_filename):
        print(f"Lỗi: File dữ liệu '{mps_filename}' không tồn tại. Hãy chạy main_experiment.py trước để tạo nó.")
        return

    target_sv = get_target_statevector(mps_filename)
    
    print("\nBắt đầu xây dựng và phân rã mạch baseline (quá trình này có thể rất lâu)...")
    start_time = time.time()
    
    try:
        # --- Quy trình xây dựng mạch ---
        num_qubits = L
        statevector_np = target_sv / np.linalg.norm(target_sv)
        
        dim = 2**num_qubits
        unitary_matrix = np.zeros((dim, dim), dtype=complex)
        unitary_matrix[:, 0] = statevector_np
        q, _ = np.linalg.qr(unitary_matrix)
        
        qc = QuantumCircuit(num_qubits)
        qc.unitary(q, range(num_qubits), label=f'StatePrep_L{L}')
        
        # Phân rã mạch
        decomposed_qc = qc.decompose()
        
    except Exception as e:
        print(f"LỖI trong quá trình xây dựng baseline: {e}"); return
        
    end_time = time.time()
    run_time = end_time - start_time
    print(f"Xây dựng và phân rã hoàn tất sau {run_time:.2f} giây.")

    # --- Phân tích và Lưu kết quả ---
    print("\nĐang phân tích và lưu kết quả...")
    
    metrics = {
        "method": "Baseline (Qiskit Isometry)",
        "L": L, "g": g,
        "final_global_fidelity": 1.0,
        "depth": decomposed_qc.depth(),
        "num_cnots": decomposed_qc.count_ops().get('cx', 0),
        "total_gates": sum(decomposed_qc.count_ops().values()),
        "ops_count": {k: v for k, v in decomposed_qc.count_ops().items()},
        "run_time_seconds": run_time
    }
    
    with open(f"{results_dir}/metrics.json", 'w') as f: json.dump(metrics, f, indent=4)
    print("Các chỉ số đã được lưu vào 'metrics.json':")
    print(json.dumps(metrics, indent=4))

    # Lưu bản vẽ mạch
    try:
        decomposed_qc.draw('mpl', style='iqx').savefig(f"{results_dir}/circuit_diagram.png", dpi=150); plt.close()
        print("Đã lưu bản vẽ mạch vào 'circuit_diagram.png'")
    except Exception as e:
        print(f"Không thể lưu bản vẽ mạch dạng ảnh: {e}")
        # Lưu dạng text thay thế
        with open(f"{results_dir}/circuit_diagram.txt", 'w') as f:
            f.write(decomposed_qc.draw('text', max_length=120))
        print("Đã lưu bản vẽ mạch dạng text.")

# ==============================================================================
# PHẦN 3: ĐIỀU KHIỂN CÁC THÍ NGHIỆM BASELINE
# ==============================================================================
def main():
    """
    Định nghĩa và chạy tất cả các thí nghiệm baseline cần thiết.
    """
    # === CHIẾN DỊCH 1: Baseline cho các chế độ vật lý (L=10) ===
    print("\n\n--- STARTING BASELINE CAMPAIGN 1: PHYSICAL REGIMES (L=10) ---")
    L_fixed = 10
    g_points = [0.5, 1.0, 1.5]
    for g in g_points:
        run_qiskit_baseline(L=L_fixed, J=1.0, g=g)

    # === CHIẾN DỊCH 2: Baseline cho kiểm tra khả năng mở rộng (g=0.8) ===
    print("\n\n--- STARTING BASELINE CAMPAIGN 2: SCALABILITY TEST (g=0.8) ---")
    g_fixed = 0.8
    L_points = [6, 8, 10]
    for L in L_points:
        run_qiskit_baseline(L=L, J=1.0, g=g_fixed)

if __name__ == '__main__':
    # Lưu ý: Cần cài đặt `pylatexenc` để vẽ mạch bằng matplotlib
    try:
        import pylatexenc
    except ImportError:
        print("Cảnh báo: Thư viện 'pylatexenc' chưa được cài đặt.")
        print("Bản vẽ mạch có thể không đẹp. Hãy chạy: pip install pylatexenc")

    main()
