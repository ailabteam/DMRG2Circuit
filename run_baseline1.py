import tenpy
import tenpy.tools.hdf5_io as hdf5_io
import numpy as np
from qiskit import QuantumCircuit
import time
import os
import json
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue

# ==============================================================================
# PHẦN 1: TẠO/LOAD DỮ LIỆU MỤC TIÊU (TENPY)
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
# PHẦN 2: HÀM THỰC HIỆN TÁC VỤ NẶNG (ĐỂ CHẠY TRONG PROCESS RIÊNG)
# ==============================================================================
def decompose_task(statevector_np, result_queue):
    """
    Hàm này thực hiện việc xây dựng và phân rã mạch.
    Nó sẽ đặt kết quả vào một Queue để tiến trình chính có thể nhận được.
    """
    try:
        num_qubits = int(np.log2(len(statevector_np)))
        
        dim = 2**num_qubits
        unitary_matrix = np.zeros((dim, dim), dtype=complex)
        unitary_matrix[:, 0] = statevector_np
        q, _ = np.linalg.qr(unitary_matrix)
        
        qc = QuantumCircuit(num_qubits)
        qc.unitary(q, range(num_qubits))
        
        # Đây là tác vụ nặng nhất
        decomposed_qc = qc.decompose()
        
        # Lấy các chỉ số
        depth = decomposed_qc.depth()
        ops_count = decomposed_qc.count_ops()
        num_cnots = ops_count.get('cx', 0)
        total_gates = sum(ops_count.values())
        
        result_queue.put({
            "status": "success",
            "depth": depth,
            "num_cnots": num_cnots,
            "total_gates": total_gates,
            "ops_count": ops_count,
            "circuit": decomposed_qc
        })
    except Exception as e:
        result_queue.put({"status": "error", "message": str(e)})

# ==============================================================================
# PHẦN 3: HÀM CHẠY THÍ NGHIỆM BASELINE VỚI TIMEOUT
# ==============================================================================
def run_qiskit_baseline(L, J, g, timeout_seconds):
    exp_name = f"L{L}_g{g}_baseline"
    print("\n" + "#"*80 + f"\n# Baseline Experiment: {exp_name} (Timeout: {timeout_seconds}s) #" + "\n" + "#"*80)

    results_dir = f"results/{exp_name}"; os.makedirs(results_dir, exist_ok=True)
    
    mps_filename = f"data/gs_ising_L{L}_J{J}_g{g}.h5"
    if not os.path.exists(mps_filename):
        print(f"Lỗi: File dữ liệu '{mps_filename}' không tồn tại. Hãy chạy main_experiment.py trước."); return

    target_sv = get_target_statevector(mps_filename)
    
    print(f"\nBắt đầu xây dựng và phân rã mạch baseline...")
    start_time = time.time()
    
    result_queue = Queue()
    p = Process(target=decompose_task, args=(target_sv, result_queue))
    p.start()
    
    # Chờ tiến trình hoàn thành hoặc hết thời gian
    p.join(timeout_seconds)
    
    run_time = time.time() - start_time
    
    metrics = {"method": "Baseline (Qiskit Isometry)", "L": L, "g": g}
    
    if p.is_alive():
        print(f"\nQUÁ TRÌNH VƯỢT QUÁ TIMEOUT ({timeout_seconds} giây)! Đang dừng lại.")
        p.terminate() # Buộc dừng tiến trình
        p.join()
        metrics.update({
            "final_global_fidelity": 1.0, "depth": f"> {timeout_seconds}s",
            "num_cnots": f"> {timeout_seconds}s", "total_gates": f"> {timeout_seconds}s",
            "run_time_seconds": f"> {timeout_seconds}"
        })
    else:
        result = result_queue.get()
        if result["status"] == "success":
            print(f"Xây dựng và phân rã hoàn tất sau {run_time:.2f} giây.")
            metrics.update({
                "final_global_fidelity": 1.0, "depth": result["depth"],
                "num_cnots": result["num_cnots"], "total_gates": result["total_gates"],
                "ops_count": {k: v for k, v in result["ops_count"].items()},
                "run_time_seconds": run_time
            })
            # Lưu bản vẽ mạch
            try:
                result["circuit"].draw('mpl', style='iqx').savefig(f"{results_dir}/circuit_diagram.png", dpi=150); plt.close()
            except Exception as e:
                with open(f"{results_dir}/circuit_diagram.txt", 'w') as f: f.write(result["circuit"].draw('text', max_length=120))
        else:
            print(f"LỖI trong quá trình phân rã: {result['message']}")
            return

    with open(f"{results_dir}/metrics.json", 'w') as f: json.dump(metrics, f, indent=4)
    print("\nFinal Metrics:"); print(json.dumps(metrics, indent=4))

# ==============================================================================
# PHẦN 4: ĐIỀU KHIỂN CÁC THÍ NGHIỆM BASELINE
# ==============================================================================
def main():
    """
    Định nghĩa và chạy tất cả các thí nghiệm baseline cần thiết.
    """
    # === CHIẾN DỊCH SCALABILITY (g=0.8) ===
    print("\n\n--- STARTING BASELINE CAMPAIGN: SCALABILITY TEST (g=0.8) ---")
    g_fixed = 0.8
    # Chạy cho các hệ nhỏ, chúng ta kỳ vọng sẽ chạy xong
    run_qiskit_baseline(L=6, J=1.0, g=g_fixed, timeout_seconds=3600) # Timeout 1 giờ
    run_qiskit_baseline(L=8, J=1.0, g=g_fixed, timeout_seconds=3600) # Timeout 1 giờ
    
    # Chạy lại cho L=10 với timeout ngắn hơn để xác nhận tính không khả thi
    run_qiskit_baseline(L=10, J=1.0, g=g_fixed, timeout_seconds=600) # Timeout 10 phút

    print("\n\n--- BASELINE EXPERIMENTS COMPLETED ---")


if __name__ == '__main__':
    main()
