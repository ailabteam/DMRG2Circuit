import tenpy
import tenpy.tools.hdf5_io as hdf5_io
import numpy as np
from qiskit import QuantumCircuit

def get_target_statevector(filename):
    """
    Đọc file MPS và chuyển nó thành một statevector đầy đủ.
    """
    print(f"Đang đọc và chuyển đổi MPS từ file: '{filename}'")
    psi_mps = hdf5_io.load(filename)["psi"]
    psi_mps.canonical_form()
    full_tensor_array = psi_mps.get_theta(0, psi_mps.L)
    target_state = full_tensor_array.to_ndarray().flatten()
    print(f"Đã chuyển đổi MPS thành statevector. Shape: {target_state.shape}")
    return target_state

def run_qiskit_baseline(statevector_np):
    """
    Xây dựng một mạch lượng tử từ một statevector bằng phương pháp Isometry.
    """
    print("\nBắt đầu xây dựng mạch baseline bằng Qiskit Isometry...")
    
    num_qubits = int(np.log2(len(statevector_np)))
    
    try:
        # --- QUY TRÌNH MỚI ---
        # 1. Chuẩn hóa statevector
        statevector_np = statevector_np / np.linalg.norm(statevector_np)
        
        # 2. Xây dựng ma trận Unitary từ statevector
        # Đây là một thủ thuật toán học tiêu chuẩn
        dim = 2**num_qubits
        unitary_matrix = np.zeros((dim, dim), dtype=complex)
        unitary_matrix[:, 0] = statevector_np
        
        # Dùng phân rã QR để hoàn thiện ma trận thành unitary
        q, r = np.linalg.qr(unitary_matrix)
        
        # 3. Tạo một mạch rỗng và khởi tạo nó bằng ma trận unitary
        baseline_circuit = QuantumCircuit(num_qubits)
        baseline_circuit.unitary(q, range(num_qubits), label='State Prep')
        
        # 4. Phân rã mạch thành các cổng cơ bản
        baseline_circuit = baseline_circuit.decompose()
        
        print("Xây dựng và phân rã mạch thành công.")
    except Exception as e:
        print(f"\n[LỖI] Qiskit không thể xây dựng mạch từ statevector này.")
        print(f"Lỗi chi tiết: {e}")
        return

    # Phân tích mạch
    print("\n--- Phân tích Mạch Baseline ---")
    
    depth = baseline_circuit.depth()
    print(f"Độ sâu (Depth): {depth}")

    ops_count = baseline_circuit.count_ops()
    print("Thống kê số lượng cổng:")
    for op, count in ops_count.items():
        print(f"  - {op}: {count}")
        
    num_cnots = ops_count.get('cx', 0)
    print(f"\nTổng số cổng CNOT: {num_cnots}")
    
    return depth, num_cnots

if __name__ == '__main__':
    mps_filename = "gs_ising_L10_J1.0_g0.5.h5"
    
    target_sv = get_target_statevector(mps_filename)
    
    run_qiskit_baseline(target_sv)
