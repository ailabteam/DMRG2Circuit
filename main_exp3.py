import tenpy
import tenpy.tools.hdf5_io as hdf5_io
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import os
import json
from qiskit import QuantumCircuit
import time

# ==============================================================================
# PHẦN 1: TẠO/LOAD DỮ LIỆU MỤC TIÊU (TENPY)
# ==============================================================================
pauli_I = np.array([[1, 0], [0, 1]], dtype=complex)
pauli_X = np.array([[0, 1], [1, 0]], dtype=complex)
pauli_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
pauli_Z = np.array([[1, 0], [0, -1]], dtype=complex)

def get_or_create_target_data(L, J, g):
    filename = f"data/gs_ising_L{L}_J{J}_g{g}.h5"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not os.path.exists(filename):
        print(f"File '{filename}' không tồn tại. Đang tạo mới bằng DMRG...")
        model_params = dict(L=L, J=J, g=g, bc_MPS='finite', conserve=None)
        M = tenpy.models.tf_ising.TFIChain(model_params)
        psi = tenpy.networks.mps.MPS.from_product_state(M.lat.mps_sites(), ["up"] * L)
        dmrg_params = {'mixer': None, 'max_E_err': 1.e-10, 'trunc_params': {'chi_max': 100}}
        info = tenpy.algorithms.dmrg.run(psi, M, dmrg_params)
        hdf5_io.save({"psi": psi}, filename)
        print(f"Đã tạo và lưu file MPS mới cho L={L}, g={g}.")
    psi_mps = hdf5_io.load(filename)["psi"]
    psi_mps.canonical_form()
    target_rdms = {}
    for i in range(L):
        exp_X = 2*psi_mps.expectation_value("Sx",[i])[0]; exp_Y = 2*psi_mps.expectation_value("Sy",[i])[0]; exp_Z = 2*psi_mps.expectation_value("Sz",[i])[0]
        rdm_i = 0.5 * (pauli_I + exp_X*pauli_X + exp_Y*pauli_Y + exp_Z*pauli_Z)
        target_rdms[(i,)] = np.array(rdm_i, requires_grad=False)
    full_tensor_array = psi_mps.get_theta(0, psi_mps.L)
    target_sv = full_tensor_array.to_ndarray().flatten()
    target_sv = np.array(target_sv, requires_grad=False)
    return target_rdms, target_sv

# ==============================================================================
# PHẦN 2: CÁC HÀM TIỆN ÍCH
# ==============================================================================
def create_ansatz(params, num_qubits, num_layers):
    for i in range(num_qubits): qml.Hadamard(wires=i)
    for l in range(num_layers):
        for i in range(num_qubits):
            qml.RX(params[l, i, 0], wires=i); qml.RY(params[l, i, 1], wires=i); qml.RZ(params[l, i, 2], wires=i)
        for i in range(num_qubits - 1): qml.CNOT(wires=[i, i + 1])
        if num_qubits > 1: qml.CNOT(wires=[num_qubits - 1, 0])

def hybrid_cost_function(params, target_rdms, target_sv, alpha, rdm_qnode, state_qnode):
    circuit_rdms_list = rdm_qnode(params)
    local_loss = 0.0
    for i in range(len(circuit_rdms_list)):
        diff = circuit_rdms_list[i] - target_rdms[(i,)]
        local_loss += qml.math.real(qml.math.trace(qml.math.dot(qml.math.T(qml.math.conj(diff)), diff)))
    circuit_state = state_qnode(params)
    overlap = qml.math.sum(qml.math.conj(target_sv) * circuit_state)
    fidelity_sq = qml.math.abs(overlap)**2
    global_loss = 1.0 - fidelity_sq
    return local_loss + alpha * global_loss

def build_qiskit_circuit(params, num_qubits, num_layers):
    qc = QuantumCircuit(num_qubits); qc.h(range(num_qubits)); qc.barrier()
    for l in range(num_layers):
        for i in range(num_qubits):
            rx, ry, rz = float(params[l,i,0]), float(params[l,i,1]), float(params[l,i,2])
            qc.rx(rx, i); qc.ry(ry, i); qc.rz(rz, i)
        qc.barrier()
        for i in range(num_qubits - 1): qc.cx(i, i + 1)
        if num_qubits > 1: qc.cx(num_qubits - 1, 0)
        qc.barrier()
    return qc

# ==============================================================================
# PHẦN 3: CÁC HÀM VẼ ĐỒ THỊ
# ==============================================================================
def plot_convergence(history, L, g, num_layers, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history) + 1), history)
    plt.xlabel("Optimization Step"); plt.ylabel("Hybrid Cost (log scale)"); plt.yscale('log')
    plt.title(f"Convergence History (L={L}, g={g}, layers={num_layers})")
    plt.grid(True, which="both", ls="--")
    plt.savefig(filename); plt.close()

def plot_probabilities(target_sv, final_sv, L, g, fidelity, filename):
    probs_target = np.abs(target_sv)**2; probs_circuit = np.abs(final_sv)**2
    plt.figure(figsize=(12, 6));
    basis_states = range(2**L)
    width = 0.8
    plt.bar(basis_states, probs_target, width=width, alpha=0.7, label=f'Target (MPS)')
    plt.bar(basis_states, probs_circuit, width=width*0.5, alpha=0.9, label=f'Optimized Circuit (Fidelity={fidelity:.4f})')
    plt.xlabel("Basis State"); plt.ylabel("Probability"); plt.title(f"Probability Distribution (L={L}, g={g})")
    plt.legend();
    if L > 6: plt.xlim(-0.5, 2**6 - 0.5)
    plt.savefig(filename); plt.close()

# ==============================================================================
# PHẦN 4: HÀM CHẠY THÍ NGHIỆM CHÍNH
# ==============================================================================
def run_variational_experiment(L, J, g, num_layers, num_steps, step_size, alpha, num_runs=1):
    exp_name = f"L{L}_g{g}_layers{num_layers}_alpha{alpha}"
    print("\n" + "#"*80 + f"\n# Variational Experiment: {exp_name} | Runs: {num_runs} #" + "\n" + "#"*80)

    results_dir = f"results/{exp_name}"; os.makedirs(results_dir, exist_ok=True)
    
    print(f"Đang chuẩn bị dữ liệu mục tiêu cho L={L}, g={g}...")
    target_rdms, target_sv = get_or_create_target_data(L, J, g)

    best_run_metrics = None
    best_run_fidelity = -1.0
    all_fidelities = []

    for run in range(num_runs):
        print(f"\n--- Starting Run {run+1}/{num_runs} ---")
        start_time = time.time()
        
        dev = qml.device("default.qubit", wires=L)
        @qml.qnode(dev)
        def rdm_circuit(params):
            create_ansatz(params, L, num_layers); return [qml.density_matrix(wires=i) for i in range(L)]
        @qml.qnode(dev)
        def state_circuit(params):
            create_ansatz(params, L, num_layers); return qml.state()

        param_shape = (num_layers, L, 3)
        params = np.random.normal(0, np.pi, size=param_shape, requires_grad=True)
        optimizer = qml.AdamOptimizer(stepsize=step_size)
        cost_history = []

        for step in range(num_steps):
            params, cost = optimizer.step_and_cost(lambda p: hybrid_cost_function(p, target_rdms, target_sv, alpha, rdm_circuit, state_circuit), params)
            cost_history.append(cost)
        
        final_cost = cost_history[-1]
        final_state = state_circuit(params)
        final_fidelity = np.abs(np.sum(np.conj(target_sv) * final_state))**2
        all_fidelities.append(final_fidelity)
        
        print(f"Run {run+1} completed. Final Fidelity: {final_fidelity:.6f}")

        if final_fidelity > best_run_fidelity:
            best_run_fidelity = final_fidelity
            qiskit_qc = build_qiskit_circuit(params, L, num_layers)
            
            end_time = time.time()
            best_run_metrics = {
                "method": "Variational (Ours)", "L": L, "g": g, "num_layers": num_layers, "alpha": alpha, "num_steps": num_steps,
                "best_run_fidelity": float(final_fidelity), "final_hybrid_cost": float(final_cost),
                "depth": qiskit_qc.depth(), "num_cnots": qiskit_qc.count_ops().get('cx', 0),
                "run_time_seconds": end_time - start_time
            }
            # Lưu lại các kết quả tốt nhất
            plot_convergence(cost_history, L, g, num_layers, f"{results_dir}/best_run_convergence.png")
            plot_probabilities(target_sv, final_state, L, g, final_fidelity, f"{results_dir}/best_run_probabilities.png")
            qiskit_qc.draw('mpl', style='iqx').savefig(f"{results_dir}/best_run_circuit.png", dpi=150); plt.close()
            np.save(f"{results_dir}/best_run_params.npy", params)

    # Thống kê tổng hợp
    summary_metrics = best_run_metrics.copy()
    summary_metrics["fidelities_all_runs"] = all_fidelities
    summary_metrics["fidelity_mean"] = np.mean(all_fidelities)
    summary_metrics["fidelity_std"] = np.std(all_fidelities)
    
    with open(f"{results_dir}/summary_metrics.json", 'w') as f: json.dump(summary_metrics, f, indent=4)
    print("\nFinal Summary Metrics for this experiment:")
    print(json.dumps(summary_metrics, indent=4))
    print(f"Đã lưu tất cả kết quả vào thư mục: '{results_dir}'")

# ==============================================================================
# PHẦN 5: ĐIỀU KHIỂN CHIẾN DỊCH
# ==============================================================================
def main():
    """
    Định nghĩa và chạy tất cả các thí nghiệm cho bài báo.
    """
    # === CHIẾN DỊCH 1: SO SÁNH HIỆU QUẢ TRÊN CÁC CHẾ ĐỘ VẬT LÝ (L=10) ===
    print("\n\n--- STARTING CAMPAIGN 1: PHYSICAL REGIMES (L=10) ---")
    L_fixed = 10
    g_points = [0.5, 1.0, 1.5]
    for g in g_points:
        run_variational_experiment(L=L_fixed, J=1.0, g=g, num_layers=8, num_steps=1000, step_size=0.05, alpha=0.1, num_runs=3)
        # run_baseline_experiment(L=L_fixed, J=1.0, g=g) # Tạm thời comment

    # === CHIẾN DỊCH 2: KIỂM TRA KHẢ NĂNG MỞ RỘNG (g=0.8) ===
    print("\n\n--- STARTING CAMPAIGN 2: SCALABILITY TEST (g=0.8) ---")
    g_fixed = 0.8
    L_points = [6, 8, 10]
    for L in L_points:
        # Tự động điều chỉnh số lớp và bước cho phù hợp
        num_layers = 2 * L - 12 if L > 6 else 4  # Heuristic: 4, 8, 8 lớp
        num_steps = 200 * (L - 4)              # Heuristic: 400, 800, 1200 bước
        run_variational_experiment(L=L, J=1.0, g=g_fixed, num_layers=num_layers, num_steps=num_steps, step_size=0.05, alpha=0.1, num_runs=3)
        # run_baseline_experiment(L=L, J=1.0, g=g_fixed) # Tạm thời comment

    print("\n\n--- ALL EXPERIMENTS COMPLETED ---")


if __name__ == '__main__':
    # Kiểm tra các thư viện tùy chọn
    try:
        import pylatexenc
    except ImportError:
        print("Warning: 'pylatexenc' not found. Circuit diagrams may not render optimally. Install with: pip install pylatexenc")
    
    main()
