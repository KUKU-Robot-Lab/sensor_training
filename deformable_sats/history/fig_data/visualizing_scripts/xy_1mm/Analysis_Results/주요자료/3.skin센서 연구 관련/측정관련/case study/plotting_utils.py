import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def plot_csv_data(filepath):
    """
    주어진 CSV 파일 경로를 읽어 5개의 그래프를 생성하고 PNG 파일로 저장합니다.
    원점 복귀 전까지만의 데이터를 사용합니다.
    """
    # --- 1. 파일 로딩 ---
    try:
        df = pd.read_csv(filepath)
        print(f"'{os.path.basename(filepath)}' 파일 로딩 완료.")
    except Exception as e:
        print(f"오류: CSV 파일을 읽는 중 문제가 발생했습니다 - {e}")
        return

    # --- 2. 데이터 전처리 ---
    if df.empty:
        print("오류: CSV 파일에 데이터가 없습니다.")
        return

    if 'Motor_Position_mm' in df.columns:
        df.rename(columns={'Motor_Position_mm': 'Depth_mm'}, inplace=True)

    non_zero_data = df[df['Sensor_Raw_Value'] != 0]
    if not non_zero_data.empty:
        first_press_index = non_zero_data.index[0]
        temp_df = df.loc[first_press_index:]
        first_zero_indices_after_press = temp_df.index[temp_df['Sensor_Raw_Value'] == 0].tolist()
        
        if first_zero_indices_after_press:
            end_index = first_zero_indices_after_press[0]
            df = df.iloc[:end_index + 1]
            print(f"데이터를 첫 0 감지 지점(인덱스 {end_index})까지 잘라 분석합니다.")

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Elapsed_Time_s'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds()

    # --- 3. 그래프 생성 및 저장 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    directory_path = os.path.dirname(filepath)
    folder_name = os.path.basename(directory_path)
    base_filename = os.path.join(directory_path, folder_name)

    # 그래프 1, 2, 3 (기존과 동일)
    try:
        fig1, ax1 = plt.subplots(figsize=(12, 7))
        ax1.plot(df['Elapsed_Time_s'], df['Sensor_Raw_Value'], label='Polymer Sensor', color='blue')
        ax1.set_title('Time vs. Polymer Sensor Value', fontsize=16)
        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('Sensor Raw Value', fontsize=12)
        ax1.legend()
        ax1.grid(True)
        output_filename_1 = f"{base_filename}_time_vs_polymer.png"
        plt.savefig(output_filename_1)
        plt.close(fig1)
        print(f"그래프 저장 완료: {os.path.basename(output_filename_1)}")
    except Exception as e:
        print(f"그래프 1 생성 실패: {e}")

    try:
        fig2, ax2 = plt.subplots(figsize=(12, 7))
        ax2.plot(df['Elapsed_Time_s'], df['AFD_Fz'], label='AFD Fz', color='green')
        ax2.set_title('Time vs. AFD50 Sensor Value', fontsize=16)
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Force (N)', fontsize=12)
        ax2.legend()
        ax2.grid(True)
        output_filename_2 = f"{base_filename}_time_vs_force.png"
        plt.savefig(output_filename_2)
        plt.close(fig2)
        print(f"그래프 저장 완료: {os.path.basename(output_filename_2)}")
    except Exception as e:
        print(f"그래프 2 생성 실패: {e}")

    try:
        fig3, ax3 = plt.subplots(figsize=(12, 7))
        scatter = ax3.scatter(df['Depth_mm'], df['Sensor_Raw_Value'], c=df['Elapsed_Time_s'], cmap='viridis', s=10)
        ax3.set_title('Depth vs. Polymer Sensor Value', fontsize=16)
        ax3.set_xlabel('Depth (mm)', fontsize=12)
        ax3.set_ylabel('Sensor Raw Value', fontsize=12)
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Elapsed Time (s)')
        ax3.grid(True)
        output_filename_3 = f"{base_filename}_depth_vs_polymer.png"
        plt.savefig(output_filename_3)
        plt.close(fig3)
        print(f"그래프 저장 완료: {os.path.basename(output_filename_3)}")
    except Exception as e:
        print(f"그래프 3 생성 실패: {e}")

    # 그래프 4: 통합 그래프 (듀얼 Y축, 색상 및 스케일 조정)
    try:
        fig4, ax4_1 = plt.subplots(figsize=(12, 7))
        
        # 첫번째 Y축 (센서 값)
        ax4_1.set_xlabel('Time (s)', fontsize=12)
        ax4_1.set_ylabel('Sensor Value', fontsize=12)
        ln1 = ax4_1.plot(df['Elapsed_Time_s'], df['Sensor_Raw_Value'], color='blue', label='Polymer Sensor')
        ln2 = ax4_1.plot(df['Elapsed_Time_s'], df['AFD_Fz'], color='orange', linestyle='--', label='AFD Fz')
        ax4_1.tick_params(axis='y')
        ax4_1.grid(True, which='major', axis='x')

        # 두번째 Y축 (Depth, 녹색)
        ax4_2 = ax4_1.twinx()
        color = 'green'
        ax4_2.set_ylabel('Depth (mm)', fontsize=12, color=color)
        ln3 = ax4_2.plot(df['Elapsed_Time_s'], df['Depth_mm'], color=color, label='Depth')
        ax4_2.tick_params(axis='y', labelcolor=color)

        # "일자" 문제 해결: Depth 데이터에 맞게 Y축 범위를 명시적으로 설정
        min_depth = df['Depth_mm'].min()
        max_depth = df['Depth_mm'].max()
        # 데이터에 약간의 여백(padding)을 주어 보기 좋게 만듭니다.
        padding = (max_depth - min_depth) * 0.1
        if padding < 1: padding = 1.0 # 최소 여백 보장
        ax4_2.set_ylim(min_depth - padding, max_depth + padding)

        ax4_1.set_title('Combined Sensor and Depth Data over Time', fontsize=16)
        
        # 범례 합치기
        lns = ln1 + ln2 + ln3
        labs = [l.get_label() for l in lns]
        ax4_1.legend(lns, labs, loc='best')

        fig4.tight_layout()
        output_filename_4 = f"{base_filename}_combined.png"
        plt.savefig(output_filename_4)
        plt.close(fig4)
        print(f"그래프 저장 완료: {os.path.basename(output_filename_4)}")
    except Exception as e:
        print(f"그래프 4 생성 실패: {e}")

    # 그래프 5: 3D 플롯 (추가됨)
    try:
        fig5 = plt.figure(figsize=(10, 8))
        ax5 = fig5.add_subplot(111, projection='3d')
        
        sc = ax5.scatter(df['Elapsed_Time_s'], df['Depth_mm'], df['AFD_Fz'], c=df['Elapsed_Time_s'], cmap='viridis')
        
        ax5.set_title('3D Plot: Time, Depth, and Force', fontsize=16)
        ax5.set_xlabel('Time (s)', fontsize=12)
        ax5.set_ylabel('Depth (mm)', fontsize=12)
        ax5.set_zlabel('Force (N)', fontsize=12)
        
        cbar = plt.colorbar(sc, ax=ax5, shrink=0.6)
        cbar.set_label('Elapsed Time (s)')
        
        output_filename_5 = f"{base_filename}_3d_plot.png"
        plt.savefig(output_filename_5)
        plt.close(fig5)
        print(f"그래프 저장 완료: {os.path.basename(output_filename_5)}")
    except Exception as e:
        print(f"그래프 5 생성 실패: {e}")
