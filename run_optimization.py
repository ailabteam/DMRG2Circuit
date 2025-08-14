import tenpy
import tenpy.tools.hdf5_io as hdf5_io
import pennylane as qml
from pennylane import numpy as np

# ==============================================================================
# BƯỚC 1: LOAD DỮ LIỆU - ĐÃ HOẠT ĐỘNG
# ==============================================================================
def get_target_statevector(filename):
    print(f"Đang đọc và chuyển đổi MPS từ file: '{filename}'")
    psi_mps = hdf5_io.load(filename)["psi"]
    print("  Đang thử phương pháp chuyển đổi cuối cùng: gọi .to_ndarray() trên Array trả về từ get_theta()...")
    psi_mps.canonical_form()
    full_tensor_array = psi_mps.get_theta(0, psi_mps.L)
    target_state = full_tensor_array.to_ndarray().flatten()
    print(f"Đã chuyển đổi MPS thành statevector. Shape: {target_state.shape}")
    # Chuyển đổi target_state sang numpy của PennyLane để đảm bảo tương thích
    return np.array(target_state, requires_grad=False)

# ==============================================================================
# BƯỚC 2: ĐỊNH NGHĨA ANSATZ - KHÔNG ĐỔI
# ==============================================================================
num_qubits = 10
num_layers = 2
dev = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(dev)
def create_ansatz(params):
    for i in range(num_qubits):
        qml.Hadamard(wires=i)
    for l in range(num_layers):
        for i in range(num_qubits):
            qml.RX(params[l, i, 0], wires=i)
            qml.RY(params[l, i, 1], wires=i)
            qml.RZ(params[l, i, 2], wires=i)
        for i in range(num_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
    return qml.state()

# ==============================================================================
# BƯỚC 3: ĐỊNH NGHĨA HÀM MẤT MÁT - PHIÊN BẢN TỐI GIẢN
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
# BƯỚC 4: THỰC HIỆN VÒNG LẶP TỐI ƯU HÓA
# ==============================================================================
if __name__ == '__main__':
    mps_filename = "gs_ising_L10_J1.0_g0.5.h5"
    target_state = get_target_statevector(mps_filename)
    
    param_shape = (num_layers, num_qubits, 3)
    params = np.random.uniform(0, 2 * np.pi, size=param_shape, requires_grad=True)
    optimizer = qml.AdamOptimizer(stepsize=0.1)

    print("\n" + "="*80)
    print("Bắt đầu quá trình tối ưu hóa...")
    num_steps = 50
    
    for step in range(num_steps):
        # Truyền target_state vào lambda function
        params, cost = optimizer.step_and_cost(lambda p: cost_function(p, target_state), params)
        if (step + 1) % 5 == 0:
            print(f"Bước {step+1:3d}:  Cost (1 - Fidelity) = {cost:.6f}")
            
    print("Quá trình tối ưu hóa hoàn tất.")
    final_cost = cost_function(params, target_state)
    print(f"\nCost cuối cùng: {final_cost:.6f}")
    print(f"Fidelity cuối cùng: {1.0 - final_cost:.6f}")
