import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def analyze_and_plot_results():
    """
    Đọc tất cả các file metrics.json, tổng hợp dữ liệu,
    in ra bảng tóm tắt và vẽ các biểu đồ so sánh.
    """
    results_dir = "results"
    all_metrics = []

    # --- 1. Thu thập tất cả dữ liệu ---
    print("--- Reading all metric files ---")
    for subdir, _, files in os.walk(results_dir):
        for file in files:
            if file == "summary_metrics.json" or (file == "metrics.json" and "baseline" in subdir):
                filepath = os.path.join(subdir, file)
                with open(filepath, 'r') as f:
                    try:
                        data = json.load(f)
                        all_metrics.append(data)
                        print(f"  Loaded: {filepath}")
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON from {filepath}")
    
    if not all_metrics:
        print("Fatal: No metric files found. Please run main_experiment.py and run_baseline.py first.")
        return

    # Chuyển dữ liệu thành DataFrame của pandas để dễ xử lý
    df = pd.DataFrame(all_metrics)

    # --- 2. Xử lý dữ liệu và In ra Bảng Tóm tắt ---
    
    # Chiến dịch 1: Physical Regimes
    df_campaign1 = df[df['L'] == 10].copy()
    print("\n\n" + "="*80)
    print("TABLE 1: COMPREHENSIVE COMPARISON (L=10)")
    print("="*80)
    # Sắp xếp để dễ nhìn
    df_campaign1 = df_campaign1.sort_values(by=['g', 'method'], ascending=[True, False])
    # Tạo cột Fidelity (Mean ± Std) cho phương pháp của chúng ta
    df_campaign1['Fidelity (Mean ± Std)'] = df_campaign1.apply(
        lambda row: f"{row.get('fidelity_mean', 0):.3f} ± {row.get('fidelity_std', 0):.3f}" 
                    if row['method'] == 'Variational (Ours)' else '~1.0',
        axis=1
    )
    print(df_campaign1[[
        'g', 'method', 'Fidelity (Mean ± Std)', 'depth', 'num_cnots', 'run_time_seconds'
    ]].to_string(index=False))
    print("="*80)

    # Chiến dịch 2: Scalability
    df_campaign2 = df[df['g'] == 0.8].copy()
    print("\n\n" + "="*80)
    print("TABLE 2: SCALABILITY ANALYSIS (g=0.8)")
    print("="*80)
    df_campaign2 = df_campaign2.sort_values(by=['L', 'method'], ascending=[True, False])
    print(df_campaign2[[
        'L', 'method', 'best_run_fidelity', 'depth', 'num_cnots'
    ]].to_string(index=False))
    print("="*80)


    # --- 3. Vẽ và Lưu các Figures Tổng hợp ---
    print("\n\n--- Generating Summary Figures ---")

    # FIGURE 1: BOX PLOT SO SÁNH FIDELITY (Chiến dịch 1)
    df_c1_variational = df_campaign1[df_campaign1['method'] == 'Variational (Ours)'].copy()
    if not df_c1_variational.empty:
        df_c1_exploded = df_c1_variational.explode('fidelities_all_runs')
        df_c1_exploded['g_label'] = df_c1_exploded['g'].apply(lambda g: f"g = {g}")

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='g_label', y='fidelities_all_runs', data=df_c1_exploded, palette="viridis", order=[f"g = {g}" for g in sorted(df_c1_exploded['g'].unique())])
        sns.stripplot(x='g_label', y='fidelities_all_runs', data=df_c1_exploded, color=".25", alpha=0.7, order=[f"g = {g}" for g in sorted(df_c1_exploded['g'].unique())])
        plt.ylabel("Final Global Fidelity")
        plt.xlabel("Transverse Field Strength (g)")
        plt.title("Fidelity Distribution Across Physical Regimes (L=10)")
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()
        output_filename = "results/summary_Physical_Regimes_fidelity_boxplot.png"
        plt.savefig(output_filename, dpi=300)
        plt.close()
        print(f"Saved figure: {output_filename}")

    # FIGURE 2: BAR PLOT SO SÁNH TÀI NGUYÊN (Chiến dịch 2)
    df_c2_plot = df_campaign2.melt(
        id_vars=['L', 'method'], 
        value_vars=['depth', 'num_cnots'], 
        var_name='Resource', 
        value_name='Count'
    )
    # Chuyển các giá trị 'Intractable' thành NaN để không vẽ, nhưng vẫn giữ trong legend
    df_c2_plot['Count'] = pd.to_numeric(df_c2_plot['Count'], errors='coerce')

    plt.figure(figsize=(12, 7))
    g = sns.catplot(
        data=df_c2_plot, x='L', y='Count', hue='method', col='Resource',
        kind='bar', palette="muted", sharey=False
    )
    g.set_axis_labels("Number of Qubits (L)", "Count (log scale)")
    g.set_titles("{col_name}")
    g.fig.suptitle("Scalability of Circuit Resources (g=0.8)", y=1.03)
    # Dùng thang log cho cả hai
    for ax in g.axes.flat:
        ax.set_yscale('log')
        ax.grid(axis='y', linestyle='--')
    
    output_filename = "results/summary_Scalability_comparison.png"
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Saved figure: {output_filename}")


if __name__ == '__main__':
    # Cần cài đặt pandas: pip install pandas
    analyze_and_plot_results()
