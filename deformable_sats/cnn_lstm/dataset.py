
import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

# 현재 파일(cnn_lstm) 바로 아래의 logs 폴더를 사용
LOG_DIR = str(Path(__file__).resolve().parent / "logs")

class TactileDataset(Dataset):
    def __init__(self, p_i, seq_len):
        self.seq_len = seq_len
        self.sensors = [f's{i}' for i in range(1, 17)]
        
        # 요청에 따라 ground_truth_cols를 6개로 수정
        self.ground_truth_cols = ['x_c', 'y_c', 'z', 'fx', 'fy', 'fz']
        self.force_cols = ['fx', 'fy', 'fz']

        # 1. p_i에 해당하는 모든 데이터 로드 및 통합 (예: p1_..., p1_com_...)
        search_pattern = os.path.join(LOG_DIR, f"p{p_i}_*", "merged_data.csv")
        data_files = glob.glob(search_pattern)
        if not data_files:
            raise FileNotFoundError(f"경고: P{p_i}에 대한 'merged_data.csv' 파일을 찾을 수 없습니다. 검색 패턴: {search_pattern}")

        df_list = []
        for file_path in sorted(data_files):
            df = self._load_and_prepare_df(file_path)
            if df is not None: df_list.append(df)
        
        if not df_list:
            raise ValueError(f"오류: P{p_i}에 대한 유효한 데이터가 없습니다.")

        self.full_df = pd.concat(df_list, ignore_index=True)

        # 2. Baseline 계산 및 적용
        baseline = self._compute_baseline(self.full_df)
        self.s_prime_df = self.full_df[self.sensors].subtract(baseline, axis=1)

        # 3. 데이터 스케일링
        self.scaler = StandardScaler()
        self.s_prime_df[self.sensors] = self.scaler.fit_transform(self.s_prime_df[self.sensors])

        print(f"정보: P{p_i}에 대한 총 {len(self.full_df)}개의 데이터 포인트 로드 및 전처리 완료.")

    def _load_and_prepare_df(self, file_path):
        try:
            df = pd.read_csv(file_path)
            df.rename(columns=lambda c: c.replace('_afd50', '').replace('_motor', ''), inplace=True)
            
            # z 컬럼 이름 통일
            if 'z_displacement_mm_laser' in df.columns:
                df.rename(columns={'z_displacement_mm_laser': 'z'}, inplace=True)
            elif 'z_laser' in df.columns:
                df.rename(columns={'z_laser': 'z'}, inplace=True)
            
            df = df.loc[:, ~df.columns.duplicated()]
            
            # 필요한 모든 컬럼이 있는지 확인
            all_needed = self.sensors + self.ground_truth_cols
            # ground_truth_cols에 이미 force_cols가 포함되어 있으므로 중복 제거
            all_needed = list(dict.fromkeys(all_needed))

            for col in all_needed:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=all_needed, inplace=True)
            return df if not df.empty else None
        except Exception as e:
            print(f"오류: '{os.path.basename(file_path)}' 로드 실패: {e}")
            return None

    def _compute_baseline(self, df, eps=0.1):
        try:
            no_load_condition = (df['fx'].abs() < eps) & (df['fy'].abs() < eps) & (df['fz'].abs() < eps)
            if no_load_condition.any():
                return df.loc[no_load_condition, self.sensors].mean()
        except KeyError as e:
            print(f"베이스라인 계산 오류: 필요한 힘 데이터({e})가 없습니다. 베이스라인을 0으로 설정합니다.")
        except Exception as e:
            print(f"베이스라인 계산 중 예외 발생: {e}. 베이스라인을 0으로 설정합니다.")
        return pd.Series(0, index=self.sensors)

    def __len__(self):
        return len(self.full_df) - self.seq_len

    def __getitem__(self, idx):
        s_prime_seq = self.s_prime_df.iloc[idx:idx + self.seq_len].values
        target = self.full_df.iloc[idx + self.seq_len - 1][self.ground_truth_cols].values.astype(np.float32)

        s_prime_seq_reshaped = s_prime_seq.reshape(self.seq_len, 4, 4)[:, np.newaxis, :, :]

        return torch.from_numpy(s_prime_seq_reshaped).float(), torch.from_numpy(target).float()
