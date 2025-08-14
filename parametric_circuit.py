import pennylane as qml
from pennylane import numpy as np

# Định nghĩa các tham số chung
num_qubits = 4
num_layers = 2 # Chúng ta sẽ có 2 lớp lặp lại

# 1. Định nghĩa thiết bị lượng tử
dev = qml.device("default.qubit", wires=num_qubits, shots=None)

# 2. Định nghĩa QNode với các tham số đầu vào
@qml.qnode(dev)
def create_parametric_circuit(params):
    """
    Một QNode tạo ra một mạch tham số.
    'params' là một mảng 2D: params[l][i] là tham số cho qubit i, lớp l.
    """
    # Lớp đầu tiên: Các cổng Hadamard để tạo chồng chập ban đầu
    for i in range(num_qubits):
        qml.Hadamard(wires=i)

    # Các lớp lặp lại (ansatz)
    for l in range(num_layers):
        # Lớp các cổng quay (tham số hóa)
        for i in range(num_qubits):
            # Mỗi qubit có 3 góc quay RX, RY, RZ
            # Chúng ta sẽ dùng 3 tham số liên tiếp trong mảng params
            qml.RX(params[l, i, 0], wires=i)
            qml.RY(params[l, i, 1], wires=i)
            qml.RZ(params[l, i, 2], wires=i)
        
        # Lớp các cổng vướng víu
        # Dùng CNOT theo kiểu chuỗi: 0-1, 1-2, 2-3
        for i in range(num_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
            
    # Trả về trạng thái lượng tử cuối cùng
    return qml.state()

# 3. Tạo một bộ tham số ngẫu nhiên để thử nghiệm
param_shape = (num_layers, num_qubits, 3)
random_params_1 = np.random.uniform(0, 2 * np.pi, size=param_shape)

# 4. Chạy mạch với bộ tham số đầu tiên
print("="*80)
print("Chạy mạch với bộ tham số ngẫu nhiên #1...")
output_state_1 = create_parametric_circuit(random_params_1)
print("Trạng thái đầu ra #1 (4 phần tử đầu):")
print(output_state_1[:4])


# 5. Tạo một bộ tham số ngẫu nhiên khác
random_params_2 = np.random.uniform(0, 2 * np.pi, size=param_shape)

print("\n" + "-"*80)
print("Chạy mạch với bộ tham số ngẫu nhiên #2...")
output_state_2 = create_parametric_circuit(random_params_2)
print("Trạng thái đầu ra #2 (4 phần tử đầu):")
print(output_state_2[:4])

# 6. Vẽ cấu trúc của mạch
print("\n" + "="*80)
print("Đây là cấu trúc của mạch tham số (Ansatz):")

# Xóa bỏ 'expansion_strategy="device"'
drawer = qml.draw(create_parametric_circuit)
print(drawer(random_params_1))
print("="*80)
