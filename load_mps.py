import tenpy
import tenpy.tools.hdf5_io as hdf5_io
import numpy as np

def load_mps_from_file(filename):
    """
    Đọc dữ liệu từ file HDF5 và khôi phục lại đối tượng MPS cùng với model.
    """
    print(f"Đang đọc dữ liệu từ file: '{filename}'")
    
    # hdf5_io.load nhận vào tên file và trả về một dictionary
    # giống như cái chúng ta đã lưu
    try:
        data = hdf5_io.load(filename)
    except FileNotFoundError:
        print(f"\n!!! Lỗi: Không tìm thấy file '{filename}'.")
        print("!!! Hãy chắc chắn rằng bạn đã chạy file 'run_dmrg_ising.py' trước.")
        exit()
        
    # Lấy đối tượng psi từ dictionary
    if "psi" not in data:
        print(f"\n!!! Lỗi: Không tìm thấy đối tượng 'psi' trong file '{filename}'.")
        exit()
        
    psi = data["psi"]
    
    print("\nKhôi phục đối tượng MPS thành công!")
    print(psi)
    
    # Kiểm tra xem có đúng không bằng cách tính lại năng lượng
    # (nếu model cũng được lưu)
    if "model" in data:
        model = data["model"]
        energy_recalculated = model.H_MPO.expectation_value(psi)
        print(f"\nKiểm tra: Năng lượng tính lại từ MPS đã load: {energy_recalculated:.10f}")
    
    return psi

# --- Chạy chương trình chính ---
if __name__ == '__main__':
    # Tên file chúng ta đã tạo ở Bài học 2
    mps_filename = "gs_ising_L10_J1.0_g0.5.h5"
    
    # Gọi hàm để load MPS
    loaded_psi = load_mps_from_file(mps_filename)
    
    # Bây giờ, chúng ta có thể làm việc với loaded_psi
    # như một đối tượng MPS bình thường
    print("\nBond dimensions của MPS đã load:", loaded_psi.chi)

    # ==========================================================
    # PHẦN MỚI: TRÍCH XUẤT TENSOR
    # ==========================================================
    print("\n" + "="*80)
    print("Trích xuất các tensor B từ MPS...")
    
    # Đảm bảo MPS ở dạng chính tắc để các tensor có ý nghĩa
    # Đây là một bước quan trọng trước khi làm việc với các tensor B
    loaded_psi.canonical_form() 
    
    # Các tensor được lưu trong thuộc tính ._B
    # Đây là một danh sách các mảng numpy
    B_tensors = loaded_psi._B
    
    # In ra thông tin về các tensor
    for i, B in enumerate(B_tensors):
        # B là một mảng numpy. .shape cho biết kích thước của nó
        print(f"  Tensor B tại site {i}: có shape = {B.shape}")
        
    print("\nCác tensor này là 'đáp án' cho bộ biên dịch lượng tử của chúng ta.")
    print("Shape của một tensor B[i] là: (chi_left, phys_dim, chi_right)")
    print("  - chi_left: kích thước bond dimension bên trái")
    print("  - phys_dim: kích thước không gian vật lý (2 cho spin-1/2)")
    print("  - chi_right: kích thước bond dimension bên phải")
    print("="*80)
