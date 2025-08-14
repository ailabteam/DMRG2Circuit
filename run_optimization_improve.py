import tenpy
import tenpy.tools.hdf5_io as hdf5_io
import pennylane as qml
from pennylane import numpy as np

# ==============================================================================
# BƯỚC 1: LOAD DỮ LIỆU - ĐÃ HOẠT ĐỘNG
# ==============================================================================
def get_target_statevector(filename):
    """Đọc file MPS và chuyển nó thành một statevector đầy đủ."""
    print(f"Đang đọc và chuyển đổi MPS từ file: '{filename}'")
    try:
        psi_mps = hdf5_io.load(filename)["psi"]
    except FileNotFoundError:
        print(f"\n!!! Lỗi: Không tìm thấy file '{filename}'.")
        print("!!! Hãy chắc chắn rằng bạn đã chạy file 'run_dmrg_ising.py' trước đó để tạo file này.")
        exit()
        
    print("  Đang thực hiện chuyển đổi MPS sang statevector...")
    psi_mps.canonical_form()
    full_tensor_array = psi_mps.get_theta(0, psi_mps.L)
    target_state = full_tensor_array.to_ndarray().flatten()
    print(f"Đã chuyển đổi MPS thành statevector. Shape: {target_state.shape}")
    
    # Chuyển đổi target_state sang numpy của PennyLane và nói rõ không cần gradient
    return np.array(target_state, requires_grad=False)

# ==============================================================================
# BƯỚC 2: ĐỊNH NGHĨA MẠCH LƯỢNG TỬ THAM SỐ (ANSATZ) - CẢI TIẾN
# ==============================================================================
# Các tham số chung của mạch
num_qubits = 10
# --- THAY ĐỔI 1: Tăng số lớp để mạch mạnh hơn ---
num_layers = 4  # Tăng từ 2 lên 4

dev = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(dev)
def create_ansatz(params):
    """Mạch ansatz của chúng ta, trả về trạng thái cuối cùng."""
    for i in range(num_qubits):
        qml.Hadamard(wires=i)
        
    for l in range(num_layers):
        for i in range(num_qubits):
            qml.RX(params[l, i, 0], wires=i)
            qml.RY(params[l, i, 1], wires=i)
            qml.RZ(params[l, i, 2], wires=i)
        # Cải tiến lớp vướng víu: thêm một vòng CNOT ngược lại để tăng khả năng vướng víu
        for i in range(num_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        # Thêm một CNOT ở đầu và cuối để tạo vướng víu vòng
        if num_qubits > 1:
            qml.CNOT(wires=[num_qubits - 1, 0])

    return qml.state()

# ==============================================================================
# BƯỚC 3: ĐỊNH NGHĨA HÀM MẤT MÁT - ĐÃ HOẠT ĐỘNG
# ==============================================================================
def cost_function(params, target_state):
    """
    Tính toán "khoảng cách" giữa trạng thái mạch và trạng thái mục tiêu.
    """
    circuit_state = create_ansatz(params)
    
    # Tự tính fidelity bằng các phép toán cơ bản
    overlap = qml.math.sum(qml.math.conj(target_state) * circuit_state)
    fidelity_sq = qml.math.abs(overlap)**2
    
    return 1.0 - fidelity_sq

# ==============================================================================
# BƯỚC 4: THỰC HIỆN VÒNG LẶP TỐI ƯU HÓA - CẢI TIẾN
# ==============================================================================
if __name__ == '__main__':
    mps_filename = "gs_ising_L10_J1.0_g0.5.h5"
    target_state = get_target_statevector(mps_filename)
    
    param_shape = (num_layers, num_qubits, 3)
    # Khởi tạo tham số từ một phân phối chuẩn để có sự đa dạng hơn
    params = np.random.normal(0, np.pi, size=param_shape, requires_grad=True)
    
    # --- THAY ĐỔI 2: Giảm tốc độ học một chút để tối ưu hóa ổn định hơn ---
    optimizer = qml.AdamOptimizer(stepsize=0.05)
    
    print("\n" + "="*80)
    print(f"Bắt đầu quá trình tối ưu hóa (Nâng cao: {num_layers} lớp, 200 bước)...")
    
    # --- THAY ĐỔI 3: Tăng số bước tối ưu hóa ---
    num_steps = 200

    for step in range(num_steps):
        params, cost = optimizer.step_and_cost(lambda p: cost_function(p, target_state), params)
        
        # In ra kết quả sau mỗi 10 bước
        if (step + 1) % 10 == 0:
            print(f"Bước {step+1:3d}:  Cost (1 - Fidelity) = {cost:.6f}")
            
    print("Quá trình tối ưu hóa hoàn tất.")
    final_cost = cost_function(params, target_state)
    print(f"\nCost cuối cùng: {final_cost:.6f}")
    print(f"Fidelity cuối cùng: {1.0 - final_cost:.6f}")
