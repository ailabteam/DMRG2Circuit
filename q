import tenpy
import tenpy.tools.hdf5_io as hdf5_io
import pennylane as qml
from pennylane import numpy as np # Đây là numpy của PennyLane
import numpy as regular_np # Import numpy gốc để dùng cho reshape

# ==============================================================================
# BƯỚC 1: LOAD DỮ LIỆU MỤC TIÊU TỪ TENPY (PHIÊN BẢN ĐÃ SỬA LẦN 2)
# ==============================================================================
def get_target_statevector(filename):
    """Đọc file MPS và chuyển nó thành một statevector đầy đủ."""
    print(f"Đang đọc và chuyển đổi MPS từ file: '{filename}'")
    try:
        data = hdf5_io.load(filename)
    except FileNotFoundError:
        print(f"\n!!! Lỗi: Không tìm thấy file '{filename}'. Chạy lại file run_dmrg_ising.py.")
        exit()
        
    psi_mps = data["psi"]
    
    # Lấy tensor lớn đại diện cho toàn bộ chuỗi
    full_tensor = psi_mps.get_theta(0, psi_mps.L)

    # --- SỬA Ở ĐÂY ---
    # Thay vì gọi full_tensor.reshape(...),
    # chúng ta dùng hàm np.reshape(...) và truyền full_tensor vào.
    # Dùng numpy gốc (regular_np) để đảm bảo tương thích.
    target_state = regular_np.reshape(full_tensor, (2**psi_mps.L))
    
    print(f"Đã chuyển đổi MPS thành statevector. Shape: {target_state.shape}")
    return target_state

# ==============================================================================
# CÁC PHẦN CÒN LẠI GIỮ NGUYÊN
# ==============================================================================
# BƯỚC 2: ĐỊNH NGHĨA MẠCH LƯỢNG TỬ THAM SỐ (ANSATZ)
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

# BƯỚC 3: ĐỊNH NGHĨA HÀM MẤT MÁT (LOSS FUNCTION)
def cost_function(params, target_state):
    circuit_state = create_ansatz(params)
    fidelity = qml.math.fidelity(circuit_state, target_state)
    return 1.0 - fidelity

# BƯỚC 4: THỰC HIỆN VÒNG LẶP TỐI ƯU HÓA
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
        params, cost = optimizer.step_and_cost(lambda p: cost_function(p, target_state), params)
        if (step + 1) % 5 == 0:
            print(f"Bước {step+1:3d}:  Cost (1 - Fidelity) = {cost:.6f}")
            
    print("Quá trình tối ưu hóa hoàn tất.")
    final_cost = cost_function(params, target_state)
    print(f"\nCost cuối cùng: {final_cost:.6f}")
    print(f"Fidelity cuối cùng: {1.0 - final_cost:.6f}")
