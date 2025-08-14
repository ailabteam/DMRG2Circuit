import sys
import numpy as np

def check_qiskit_environment():
    """
    Một script kiểm tra toàn diện môi trường Qiskit, đã cập nhật cho Qiskit 1.0+.
    """
    print("="*80)
    print("BẮT ĐẦU KIỂM TRA MÔI TRƯỜNG QISKIT")
    print("="*80)

    # --- KIỂM TRA 1: IMPORT CÁC THƯ VIỆN ---
    print("\n[KIỂM TRA 1/3] Đang import các thư viện Qiskit...")
    try:
        import qiskit
        from qiskit import QuantumCircuit, transpile
        # SỬA LẠI Ở ĐÂY: Import AerSimulator trực tiếp từ qiskit_aer
        from qiskit_aer import AerSimulator, __version__ as aer_version
        from qiskit.visualization import circuit_drawer

        print(f"  Thành công! Phiên bản Qiskit: {qiskit.__version__}")
        print(f"  Thành công! Phiên bản Qiskit Aer: {aer_version}")

    except ImportError as e:
        print(f"\n[LỖI] Không thể import Qiskit hoặc Qiskit Aer.")
        print(f"  Lỗi chi tiết: {e}")
        print("  Hãy thử chạy lại các lệnh cài đặt:")
        print("  pip install qiskit")
        print("  pip install qiskit-aer")
        sys.exit(1)
    
    # --- KIỂM TRA 2: XÂY DỰNG VÀ VẼ MẠCH ---
    print("\n[KIỂM TRA 2/3] Đang xây dựng một mạch lượng tử (Bell State)...")
    try:
        # Trong Qiskit mới, nên khởi tạo mạch mà không có các bit cổ điển
        # và thêm chúng sau nếu cần
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        # `measure_all` sẽ tự động thêm các bit cổ điển cần thiết
        qc.measure_all()
        
        print("  Xây dựng mạch thành công.")
        
        # Dùng qc.draw() là cách mới và được khuyến khích hơn
        print("  Bản vẽ mạch:")
        print(qc.draw("text"))
        
    except Exception as e:
        print(f"\n[LỖI] Có lỗi xảy ra trong quá trình xây dựng hoặc vẽ mạch.")
        print(f"  Lỗi chi tiết: {e}")
        sys.exit(1)

    # --- KIỂM TRA 3: CHẠY MÔ PHỎNG ---
    print("\n[KIỂM TRA 3/3] Đang chạy mô phỏng trên AerSimulator...")
    try:
        simulator = AerSimulator()
        
        # Qiskit 1.0 không cần `transpile` cho simulator nữa, nhưng làm vậy vẫn tốt
        compiled_circuit = transpile(qc, simulator)
        
        result = simulator.run(compiled_circuit, shots=1024).result()
        
        counts = result.get_counts(compiled_circuit)
        
        print("  Chạy mô phỏng thành công!")
        print(f"  Kết quả đếm (counts): {counts}")
        
        # Sửa lại key trong counts cho Qiskit mới
        # Bây giờ nó có dạng "00 00" (classical_bits quantum_bits), ta cần '00'
        # Hoặc nó có thể trả về số nguyên, ví dụ 0 cho '00' và 3 cho '11'
        # Cách an toàn nhất là kiểm tra sự tồn tại của cả hai dạng
        counts_keys = counts.keys()
        success = ('00' in counts_keys and '11' in counts_keys) or \
                  (0 in counts_keys and 3 in counts_keys)

        if success:
            print("  -> Kết quả mô phỏng ĐÚNG như kỳ vọng cho Bell state.")
        else:
            print("  -> CẢNH BÁO: Kết quả mô phỏng không giống như kỳ vọng.")

    except Exception as e:
        print(f"\n[LỖI] Có lỗi xảy ra trong quá trình chạy mô phỏng.")
        print(f"  Lỗi chi tiết: {e}")
        sys.exit(1)

    print("\n" + "="*80)
    print("KIỂM TRA HOÀN TẤT: Môi trường Qiskit của bạn đã sẵn sàng!")
    print("="*80)

if __name__ == '__main__':
    check_qiskit_environment()
