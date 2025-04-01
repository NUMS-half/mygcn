import sys
import subprocess
from helper import get_logger

logger = get_logger("Data")

def run_scripts():
    logger.info("批量执行数据生成与预处理脚本...")
    # 要执行的脚本列表
    scripts = [
        "data_generate.py",
        "label_generate.py",
        "data_process.py"
    ]

    # 依次执行每个脚本
    for i, script in enumerate(scripts):
        logger.info(f"正在执行: {script} ({i+1}/{len(scripts)})")
        subprocess.run([sys.executable, script])
        logger.info(f"执行完成: {script}")

if __name__ == "__main__":
    run_scripts()