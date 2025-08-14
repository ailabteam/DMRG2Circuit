import tenpy
import tenpy.tools.hdf5_io as hdf5_io
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================================================================
# PHẦN 1: TẠO/LOAD DỮ LIỆU MỤC TIÊU (TENPY)
# ==============================================================================
# (Các hàm này đã được kiểm chứng và hoạt động tốt)
pauli_I = np.array([[1, 0], [0, 1]])
pauli_X = np.array([[0, 1], [1, 0]])
pauli_Y = np.array([[0, -1j], [1j, 0]])
pauli_Z = np.array([[1, 0], [0, -1]])

def get_or_create_target_rdms(L, J, g):
    """
    Kiểm tra xem file MPS đã tồn tại chưa. Nếu chưa, tạo nó.
    Sau đó, đọc file và trả về các RDM 1-site.
    """
    filename = f"data/gs_ising_L{L}_J{J}_g{g}.h5"
    
    # Tạo thư mục 'data' nếu nó chưa tồn tại
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    if not os.path.exists(filename):
        print(f"File '{filename}' không tồn tại. Đang tạo mới bằng DMRG...")
        model_params = dict(L=L, J=J, g=g, bc_MPS='finite', conserve=None)
        M = tenpy.models.tf_ising.TFIChain(model_params)
        psi = tenpy.networks.mps.MPS.from_product_state(M.lat.mps_sites(), ["up"] * L)
        dmrg_params = {'mixer': None, 'max_E_err': 1.e-10, 'trunc_params': {'chi_max': 100}}
        info = tenpy.algorithms.dmrg.run(psi, M, dmrg_params)
        data_to_save = {"psi": psi}
        hdf5_io.save(data_to_save, filename)
        print("Đã tạo và lưu file MPS mới.")

    print(f"Đang đọc MPS và tính toán RDM mục tiêu từ file: '{filename}'")
    psi_mps = hdf5_io.load(filename)["psi"]
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
# PHẦN 2: MẠCH LƯỢNG TỬ VÀ TỐI ƯU HÓA (PENNYLANE)
# ==============================================================================
def create_ansatz_and_get_rdms(params, num_qubits, num_layers):
    # Phần xây dựng mạch không đổi
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

def local_cost_function(params, target_rdms, num_qubits, num_layers, qnode):
    circuit_rdms_list = qnode(params)
    total_loss = 0.0
    for i in range(num_qubits):
        rdm_circ = circuit_rdms_list[i]
        rdm_target = target_rdms[(i,)]
        diff = rdm_circ - rdm_target
        diff_dagger = qml.math.T(qml.math.conj(diff))
        loss_i = qml.math.real(qml.math.trace(qml.math.dot(diff_dagger, diff)))
        total_loss += loss_i
    return total_loss

# ==============================================================================
# PHẦN 3: HÀM THÍ NGHIỆM CHÍNH
# ==============================================================================
def run_experiment(L, J, g, num_layers, num_steps, step_size):
    """
    Chạy một thí nghiệm hoàn chỉnh cho một bộ tham số và lưu tất cả kết quả.
    """
    print("\n" + "#"*80)
    print(f"# Bắt đầu Thí nghiệm: L={L}, J={J}, g={g}, layers={num_layers}, steps={num_steps} #")
    print("#"*80)

    # --- Setup ---
    # Tạo một thư mục riêng cho kết quả của lần chạy này
    results_dir = f"results/L{L}_g{g}_layers{num_layers}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Lấy dữ liệu mục tiêu
    target_rdms_dict = get_or_create_target_rdms(L, J, g)

    # Cấu hình PennyLane
    dev = qml.device("default.qubit", wires=L)
    qnode = qml.QNode(lambda p: create_ansatz_and_get_rdms(p, L, num_layers), dev)

    # Khởi tạo tham số và optimizer
    param_shape = (num_layers, L, 3)
    params = np.random.normal(0, np.pi, size=param_shape, requires_grad=True)
    optimizer = qml.AdamOptimizer(stepsize=step_size)
    cost_history = []

    # --- Tối ưu hóa ---
    print("\nBắt đầu quá trình tối ưu hóa...")
    for step in range(num_steps):
        params, cost = optimizer.step_and_cost(
            lambda p: local_cost_function(p, target_rdms_dict, L, num_layers, qnode), params
        )
        cost_history.append(cost)
        if (step + 1) % 20 == 0:
            print(f"Bước {step+1:4d}:  Local Cost = {cost:.8f}")

    print("Quá trình tối ưu hóa hoàn tất.")
    print(f"Cost cuối cùng: {cost_history[-1]:.8f}")

    # --- Lưu kết quả ---
    print("\nĐang lưu kết quả...")
    # 1. Lưu lịch sử cost
    np.save(f"{results_dir}/cost_history.npy", cost_history)
    # 2. Lưu các tham số tối ưu
    np.save(f"{results_dir}/optimal_params.npy", params)

    # --- Vẽ và lưu figures ---
    # 1. Figure: Đồ thị hội tụ
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_steps + 1), cost_history)
    plt.xlabel("Bước tối ưu hóa")
    plt.ylabel("Local Cost (log scale)")
    plt.yscale('log')
    plt.title(f"Lịch sử hội tụ cho L={L}, g={g}, {num_layers} lớp")
    plt.grid(True, which="both", ls="--")
    plt.savefig(f"{results_dir}/convergence_plot.png")
    plt.close() # Đóng figure để không hiển thị

    # 2. Figure: Bản vẽ mạch
    # Tạo một drawer riêng để vẽ
    drawer = qml.draw(qnode, max_length=120)
    circuit_drawing_txt = drawer(params)
    with open(f"{results_dir}/circuit_drawing.txt", 'w') as f:
        f.write(circuit_drawing_txt)
    
    print(f"Đã lưu tất cả kết quả và figures vào thư mục: '{results_dir}'")
    print("#"*80)


# ==============================================================================
# PHẦN 4: ĐIỀU KHIỂN CÁC THÍ NGHIỆM
# ==============================================================================
if __name__ == '__main__':
    # --- Định nghĩa các thí nghiệm chúng ta muốn chạy ---
    
    # Thí nghiệm 1: Chạy một trường hợp cơ bản
    run_experiment(L=10, J=1.0, g=0.5, num_layers=4, num_steps=200, step_size=0.05)
    
    # Thí nghiệm 2: Chạy tại điểm chuyển pha (khó hơn)
    run_experiment(L=10, J=1.0, g=1.0, num_layers=4, num_steps=300, step_size=0.05)

    # Thí nghiệm 3: Thử với hệ thống nhỏ hơn nhưng nhiều lớp hơn
    # run_experiment(L=6, J=1.0, g=0.8, num_layers=6, num_steps=200, step_size=0.05)
    
    # Bạn có thể thêm nhiều lần gọi run_experiment ở đây để quét qua nhiều giá trị
    # ví dụ:
    # for g_val in [0.5, 0.8, 1.0, 1.2, 1.5]:
    #     run_experiment(L=10, J=1.0, g=g_val, num_layers=4, num_steps=200, step_size=0.05)

    print("\nTất cả các thí nghiệm đã hoàn tất.")
