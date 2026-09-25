import os
import glob
from plotting_utils import plot_csv_data

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    search_paths = [
        os.path.join(script_dir, 'data', '*.csv'),
        os.path.join(script_dir, '*.csv')
    ]
    
    csv_files = []
    for path in search_paths:
        csv_files.extend(glob.glob(path))
        
    if not csv_files:
        print("오류: 현재 폴더 또는 'data' 하위 폴더에 분석할 CSV 파일이 없습니다.")
        return
        
    try:
        latest_file = max(csv_files, key=os.path.getmtime)
        print(f"가장 최근 CSV 파일 발견: {os.path.basename(latest_file)}")
        plot_csv_data(latest_file)
    except Exception as e:
        print(f"파일 처리 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
