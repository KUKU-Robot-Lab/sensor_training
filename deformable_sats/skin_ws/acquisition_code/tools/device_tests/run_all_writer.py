import subprocess
import sys
import time
import os

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    scripts = [
        os.path.join(current_dir, "afd50_writer.py"),
        os.path.join(current_dir, "due_writer.py"),
        os.path.join(current_dir, "ethermotion_writer.py")
    ]

    processes = []

    print("🚀 모든 데이터 수집기를 동시에 실행합니다...")
    print("🛑 종료하려면 이 터미널에서 'Ctrl + C'를 누르세요.\n")

    try:
        for script in scripts:
            script_name = os.path.basename(script)
            print(f"[{script_name}] 실행 중...")
            p = subprocess.Popen([sys.executable, script])
            processes.append(p)

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n🛑 종료 신호(Ctrl+C) 수신됨. 모든 수집기를 안전하게 종료합니다...")
        for p in processes:
            p.terminate()
            p.wait()
        print("✅ 모든 수집기 종료 완료.")

if __name__ == "__main__":
    main()