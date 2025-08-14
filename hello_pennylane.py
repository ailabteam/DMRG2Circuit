import pennylane as qml
from pennylane import numpy as np # PennyLane có phiên bản NumPy riêng hỗ trợ gradient

# 1. Định nghĩa một thiết bị lượng tử (ở đây là trình mô phỏng)
# 'shots=None' nghĩa là chúng ta muốn mô phỏng chính xác (statevector)
dev = qml.device("default.qubit", wires=2, shots=None)

# 2. Định nghĩa một "QNode" - đây là hàm liên kết mạch lượng tử với thiết bị
# Nó giống như việc biến một mạch thành một hàm Python có thể tính toán được
@qml.qnode(dev)
def create_bell_state():
    """Một QNode tạo ra Bell state."""
    # Thêm các cổng lượng tử
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    
    # Trả về trạng thái lượng tử cuối cùng
    return qml.state()

# 3. Chạy mạch và lấy kết quả
final_state = create_bell_state()

print("="*80)
print("Chạy mạch Bell state bằng PennyLane thành công!")
print(f"Trạng thái lượng tử đầu ra (statevector):")
print(final_state)

# Trạng thái Bell lý thuyết là 1/sqrt(2) * (|00> + |11>)
# Vector của nó là [1/sqrt(2), 0, 0, 1/sqrt(2)]
# ~ [0.707, 0, 0, 0.707]
print("\nSo sánh với giá trị lý thuyết: [0.707, 0, 0, 0.707]")

# 4. Vẽ mạch
# Chúng ta có thể lấy bản vẽ của mạch từ QNode
print("\n" + "="*80)
print("Đây là mạch lượng tử của bạn (vẽ bởi PennyLane):")
print(qml.draw(create_bell_state)())
print("="*80)
