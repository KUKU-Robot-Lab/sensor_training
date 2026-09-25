
import argparse
import glob
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from dataset import TactileDataset
from model_def import CNN_LSTM
torch.backends.cudnn.enabled = False

# 평가 스크립트가 위치한 폴더 내의 'logs' 폴더를 모델 경로로 사용
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')

def evaluate_single_model(model_path):
    """단일 모델 파일을 평가하고 결과를 딕셔너리 리스트로 반환합니다."""
    results = []
    # 1. 모델 파일 로드
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        print(f"\n--- 모델 평가 시작: {os.path.basename(model_path)} ---")
    except FileNotFoundError:
        print(f"오류: 모델 파일 '{model_path}'을(를) 찾을 수 없습니다.")
        return

    config = checkpoint['config']
    p_i = config['target_point_id']
    ground_truth_cols = checkpoint['ground_truth_cols']

    # 2. 데이터셋 재생성 및 테스트 데이터셋 분리
    print(f"정보: P{p_i}에 대한 테스트 데이터셋 준비 중...")
    torch.manual_seed(config['random_seed']) # 학습과 동일한 시드로 분리
    try:
        full_dataset = TactileDataset(p_i=p_i, seq_len=config['seq_len'])
        full_dataset.scaler = checkpoint['scaler'] # 학습 시 사용한 스케일러 적용
    except (FileNotFoundError, ValueError) as e:
        print(f"데이터셋 생성 오류: {e}")
        return []

    test_size = int(config['test_split_ratio'] * len(full_dataset))
    train_size = len(full_dataset) - test_size
    _, test_dataset = random_split(full_dataset, [train_size, test_size])

    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    print(f"테스트 데이터: {len(test_dataset)}개")

    # 3. 모델 초기화 및 상태 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_LSTM(output_size=len(ground_truth_cols)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 4. 평가 수행
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            outputs = model(sequences)
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(outputs.cpu().numpy())

    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)

    # 5. 성능 지표 계산 및 출력
    for i, label in enumerate(ground_truth_cols):
        mse = mean_squared_error(all_targets[:, i], all_predictions[:, i])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_targets[:, i], all_predictions[:, i])
        r2 = r2_score(all_targets[:, i], all_predictions[:, i])

        results.append({
            'model': os.path.basename(model_path),
            'point_id': p_i,
            'output': label,
            'r2': r2,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
        })
    print(f"정보: P{p_i} 모델 평가 완료.")
    return results

def evaluate_all_models():
    """logs 폴더의 모든 모델을 평가하고 결과를 CSV로 저장합니다."""
    model_paths = glob.glob(os.path.join(LOG_DIR, 'p*_cnn_lstm_model.pth'))
    if not model_paths:
        print(f"오류: '{LOG_DIR}'에서 평가할 모델 파일(*_cnn_lstm_model.pth)을 찾을 수 없습니다.")
        return

    print(f"정보: 총 {len(model_paths)}개의 모델에 대한 평가를 시작합니다.")

    all_results = []
    for model_path in sorted(model_paths):
        results = evaluate_single_model(model_path)
        all_results.extend(results)

    if not all_results:
        print("오류: 평가를 완료한 모델이 없습니다.")
        return

    summary_df = pd.DataFrame(all_results)
    output_path = os.path.join(LOG_DIR, 'evaluation_summary.csv')
    summary_df.to_csv(output_path, index=False, float_format='%.6f')

    print(f"\n{'='*50}\n평가 요약 결과:\n")
    print(summary_df)
    print(f"\n성공: 전체 평가 결과가 '{output_path}'에 저장되었습니다.")

if __name__ == '__main__':
    evaluate_all_models()
