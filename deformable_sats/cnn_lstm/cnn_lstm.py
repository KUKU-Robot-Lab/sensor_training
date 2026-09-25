
import os
import re
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import TactileDataset, LOG_DIR
from model_def import CNN_LSTM

#torch.backends.cudnn.enabled = False
# --- 모델 및 학습 하이퍼파라미터 ---
config = {
    'seq_len': 50,
    'batch_size': 512,
    'epochs': 50,
    'learning_rate': 0.001,
    'test_split_ratio': 0.2,
    'random_seed': 42
}


def _log_cuda_status(device: torch.device) -> None:
    if device.type != "cuda":
        print("정보: CUDA 장치를 찾지 못해 CPU로 학습을 진행합니다.")
        return

    try:
        index = device.index or 0
        props = torch.cuda.get_device_properties(index)
        print(f"정보: GPU '{torch.cuda.get_device_name(index)}' (SM {props.major}.{props.minor}) 사용")
        print(f"       총 메모리: {props.total_memory / (1024 ** 3):.2f} GiB")
    except Exception as exc:  # pragma: no cover - 진단용 출력
        print(f"경고: GPU 정보 조회 실패 - {exc}")

    print(f"정보: Torch CUDA 버전: {torch.version.cuda}")
    cudnn_enabled = torch.backends.cudnn.enabled
    cudnn_version = torch.backends.cudnn.version()
    print(f"정보: cuDNN enabled={cudnn_enabled}, version={cudnn_version}")
    print(f"정보: CUDA memory allocated={torch.cuda.memory_allocated()/ (1024 ** 2):.2f} MiB, reserved={torch.cuda.memory_reserved()/ (1024 ** 2):.2f} MiB")


def train_all_points():
    # 1. 학습할 모든 포인트 ID 자동 감지
    try:
        all_log_folders = glob.glob(os.path.join(LOG_DIR, "p*_*_*"))
        p_ids_str = sorted(list(set([re.search(r'p(\d+)', os.path.basename(f.rstrip(os.sep))).group(1) for f in all_log_folders])))
        p_ids = [int(p) for p in p_ids_str]
        if not p_ids:
            print("오류: 학습할 p* 폴더를 찾을 수 없습니다.")
            return
        print(f"감지된 포인트 ID: {p_ids}")
    except Exception as e:
        print(f"포인트 ID 감지 중 오류 발생: {e}")
        return

    # 2. 각 포인트에 대해 학습 수행
    for p_i in p_ids:
        print(f"\n{'='*50}\n--- 포인트 P{p_i} 학습 시작 ---\n{'='*50}")

        try:
            # 3. 데이터셋 생성 및 분리
            torch.manual_seed(config['random_seed']) # 재현성을 위한 랜덤 시드 고정
            full_dataset = TactileDataset(p_i=p_i, seq_len=config['seq_len'])

            test_size = int(config['test_split_ratio'] * len(full_dataset))
            train_size = len(full_dataset) - test_size
            train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True ,num_workers=8)

            # 4. 모델, 손실 함수, 옵티마이저 정의
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _log_cuda_status(device)
            model = CNN_LSTM(output_size=len(full_dataset.ground_truth_cols)).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

            print(f"정보: P{p_i} 모델 학습 시작 (Device: {device})")
            print(f"학습 데이터: {len(train_dataset)}개, 테스트 데이터: {len(test_dataset)}개")

            # 5. 학습 루프
            for epoch in range(config['epochs']):
                model.train()
                epoch_loss = 0.0
                progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}", unit="batch")

                for sequences, targets in progress_bar:
                    sequences, targets = sequences.to(device), targets.to(device)
                    outputs = model(sequences)
                    loss = criterion(outputs, targets)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    progress_bar.set_postfix(loss=f'{loss.item():.6f}')

                avg_loss = epoch_loss / len(train_loader)
                tqdm.write(f"Epoch [{epoch+1}/{config['epochs']}] - Avg Loss: {avg_loss:.6f}")

            print(f"\n정보: P{p_i} 모델 학습 완료.")

            # 6. 모델 및 스케일러 저장
            output_data = {
                'model_state_dict': model.state_dict(),
                'scaler': full_dataset.scaler,
                'config': {**config, 'target_point_id': p_i}, # 현재 포인트 ID를 config에 추가
                'ground_truth_cols': full_dataset.ground_truth_cols
            }
            model_filename = os.path.join(LOG_DIR, f'p{p_i}_cnn_lstm_model.pth')
            torch.save(output_data, model_filename)
            print(f"성공: 학습된 모델을 '{model_filename}'에 저장했습니다.")

        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"오류: P{p_i} 학습 실패 - {e}")
            continue # 다음 포인트로 넘어감

if __name__ == '__main__':
    train_all_points()
