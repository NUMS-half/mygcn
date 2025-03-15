import subprocess
import sys

# 按顺序执行三个脚本
def run_scripts():
    # 定义要执行的脚本列表
    scripts = [
        "data_generate.py",
        "label_generate.py",
        "data_process.py"
    ]

    # 依次执行每个脚本
    for script in scripts:
        print(f"正在执行: {script}")
        subprocess.run([sys.executable, script])
        print(f"执行完成: {script}")

if __name__ == "__main__":
    run_scripts()