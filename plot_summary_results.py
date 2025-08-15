import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def analyze_campaign(campaign_name, experiments):
    """
    Đọc tất cả các file metrics.json từ một chiến dịch,
    tổng hợp dữ liệu và vẽ các biểu đồ so sánh.
    """
    print(f"\n--- Analyzing Campaign: {campaign_name} ---")
    
    results_data = []
    for exp_params in experiments:
        # Xây dựng tên thư mục dựa trên tham số
        exp_name_variational = f"L{exp_params['L']}_g{exp_params['g']}_layers{exp_params['layers']}_alpha{exp_params['alpha']}"
        exp_name_baseline = f"L{exp_params['L']}_g{exp_params['g']}_baseline"
        
        # Đọc kết quả của phương pháp variational
        try:
            with open(f"results/{exp_name_variational}/summary_metrics.json", 'r') as f:
                data = json.load(f)
                # Thêm một cột để phân biệt
                data['label'] = f"L={exp_params['L']}, g={exp_params['g']}"
                results_data.append(data)
        except FileNotFoundError:
            print(f"Warning: Results not found for {exp_name_variational}")

        # Đọc kết quả của baseline (nếu có)
        try:
            with open(f"results/{exp_name_baseline}/metrics.json", 'r') as f:
                data = json.load(f)
                data['label'] = f"L={exp_params['L']}, g={exp_params['g']}"
                results_data.append(data)
        except FileNotFoundError:
            # Hoàn toàn bình thường nếu baseline không chạy xong
            pass

    if not results_data:
        print("No data found to analyze.")
        return

    # Chuyển dữ liệu thành DataFrame của pandas để dễ xử lý
    df = pd.DataFrame(results_data)
    
    # --- FIGURE 1: BOX PLOT SO SÁNH FIDELITY ---
    # Chỉ vẽ cho phương pháp variational
    df_variational = df[df['method'] == 'Variational (Ours)'].copy()
    # Cần "explode" danh sách fidelities để mỗi lần chạy là một hàng riêng
    df_exploded = df_variational.explode('fidelities_all_runs')
    
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='label', y='fidelities_all_runs', data=df_exploded, palette="viridis")
    sns.stripplot(x='label', y='fidelities_all_runs', data=df_exploded, color=".25", alpha=0.6) # Thêm các điểm dữ liệu
    plt.ylabel("Final Global Fidelity")
    plt.xlabel("Experiment Configuration")
    plt.title(f"Distribution of Fidelities across Multiple Runs ({campaign_name})")
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(f"results/summary_{campaign_name}_fidelity_boxplot.png")
    plt.close()
    print(f"Saved fidelity boxplot for {campaign_name}")

    # --- FIGURE 2: BAR PLOT SO SÁNH CNOT COUNT ---
    plt.figure(figsize=(12, 7))
    sns.barplot(x='label', y='num_cnots', hue='method', data=df, palette="muted")
    plt.ylabel("Number of CNOT Gates (log scale)")
    plt.xlabel("Experiment Configuration")
    plt.yscale('log')
    plt.title(f"CNOT Count Comparison ({campaign_name})")
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(f"results/summary_{campaign_name}_cnot_barplot.png")
    plt.close()
    print(f"Saved CNOT count barplot for {campaign_name}")

def main():
    # Định nghĩa các chiến dịch tương ứng với file main_experiment.py
    campaign1_params = [
        {'L': 10, 'g': 0.5, 'layers': 8, 'alpha': 0.1},
        {'L': 10, 'g': 1.0, 'layers': 8, 'alpha': 0.1},
        {'L': 10, 'g': 1.5, 'layers': 8, 'alpha': 0.1},
    ]
    analyze_campaign("Physical_Regimes", campaign1_params)

    campaign2_params = [
        {'L': 6, 'g': 0.8, 'layers': 4, 'alpha': 0.1},
        {'L': 8, 'g': 0.8, 'layers': 8, 'alpha': 0.1},
        {'L': 10, 'g': 0.8, 'layers': 8, 'alpha': 0.1},
    ]
    analyze_campaign("Scalability", campaign2_params)

if __name__ == '__main__':
    # Cần cài đặt pandas: pip install pandas
    main()
