import tenpy
import tenpy.tools.hdf5_io as hdf5_io
import pennylane as qml
from pennylane import numpy as np

# ==============================================================================
# BƯỚC 1: LOAD DỮ LIỆU - LẤY RA CÁC RDM MỤC TIÊU
# ==============================================================================
pauli_I = np.array([[1, 0], [0, 1]])
pauli_X = np.array([[0, 1], [1, 0]])
pauli_Y = np.array([[0, -1j], [1j, 0]])
pauli_Z = np.array([[1, 0], [0, -1]])

def get_target_rdms(filename):
    """Đọc file MPS và xây dựng các RDM 1-site."""
    print(f"Đang đọc MPS và tính toán RDM mục tiêu từ file: '{filename}'")
    psi_mps = hdf5_io.load(filename)["psi"]
    L = psi_mps.L
    psi_mps.canonical_form()
    target_rdms = {}
    for i in range(L):
        exp_X = 2 * psi_mps.expectation_value("Sx", [i])[0]
        exp_Y = 2 * psi_mps.expectation_value("Sy", [i])[0]
        exp_Z = 2 * psi_mps.expectation_value("Sz", [i])[0]
        rdm_i = 0.5 * (pauli_I + exp_X * pauli_X + exp_Y * pauli_Y + exp_Z * pauli_Z)
        target_rdms[(i,)] = rdm_i
    print("Đã tính toán xong các RDM 1-site mục tiêu.")
    return target_rdms

# ==============================================================================
# BƯỚC 2: ĐỊNH NGHĨA ANSATZ ĐỂ TRẢ VỀ CÁC RDM
# ==============================================================================
num_qubits = 10
num_layers = 4
dev = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(dev)
def create_ansatz_and_get_rdms(params):
    for i in range(num_qubits):
        qml.Hadamard(wires=i)
    for l in range(num_layers):
        for i in range(num_qubits):
            qml.RX(params[l, i, 0], wires=i)
            qml.RY(params[l, i, 1], wires=i)
            qml.RZ(params[l, i, 2], wires=i)
        for i in range(num_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        if num_qubits > 1:
            qml.CNOT(wires=[num_qubits - 1, 0])
    return [qml.density_matrix(wires=i) for i in range(num_qubits)]

# ==============================================================================
# BƯỚC 3: ĐỊNH NGHĨA HÀM MẤT MÁT CỤC BỘ (LOCAL LOSS)
# ==============================================================================
def local_cost_function(params, target_rdms):
    """
    Tính toán sự khác biệt giữa các RDM của mạch và RDM mục tiêu.
    """
    circuit_rdms_list = create_ansatz_and_get_rdms(params)
    total_loss = 0.0
    for i in range(num_qubits):
        rdm_circ = circuit_rdms_list[i]
        rdm_target = target_rdms[(i,)]
        diff = rdm_circ - rdm_target
        
        # Thay thế .conj().T bằng các hàm của qml.math
        diff_dagger = qml.math.T(qml.math.conj(diff))
        loss_i = qml.math.real(qml.math.trace(qml.math.dot(diff_dagger, diff)))
        total_loss += loss_i
        
    return total_loss

# ==============================================================================
# BƯỚC 4: THỰC HIỆN VÒNG LẶP TỐI ƯU HÓA
# ==============================================================================
if __name__ == '__main__':
    mps_filename = "gs_ising_L10_J1.0_g0.5.h5"
    target_rdms_dict = get_target_rdms(mps_filename)
    
    param_shape = (num_layers, num_qubits, 3)
    params = np.random.normal(0, np.pi, size=param_shape, requires_grad=True)
    optimizer = qml.AdamOptimizer(stepsize=0.05)
    
    print("\n" + "="*80)
    print("Bắt đầu quá trình tối ưu hóa với HÀM MẤT MÁT CỤC BỘ...")
    num_steps = 200

    for step in range(num_steps):
        params, cost = optimizer.step_and_cost(lambda p: local_cost_function(p, target_rdms_dict), params)
        if (step + 1) % 10 == 0:
            print(f"Bước {step+1:3d}:  Local Cost = {cost:.6f}")
            
    print("Quá trình tối ưu hóa hoàn tất.")
    final_cost = local_cost_function(params, target_rdms_dict)
    print(f"\nCost cục bộ cuối cùng: {final_cost:.6f}")
