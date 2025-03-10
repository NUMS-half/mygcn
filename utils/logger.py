import os
import logging
from datetime import datetime

# 全局日志配置
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s-%(name)s-%(levelname)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_DIR = "../logs"

# 确保日志目录存在
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 是否已初始化
_is_initialized = False


def setup_logging(log_to_file=True, log_level=None):
    """
    初始化日志系统

    Args:
        log_to_file: 是否将日志输出到文件
        log_level: 日志级别，默认为INFO
    """
    global _is_initialized

    if _is_initialized:
        return

    level = log_level if log_level is not None else LOG_LEVEL

    # 配置根日志记录器
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT
    )

    # 添加文件处理器（如果需要）
    if log_to_file:
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(LOG_DIR, f"app_{today}.log")

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

        # 添加到根日志记录器
        logging.getLogger().addHandler(file_handler)

    _is_initialized = True


def get_logger(name):
    """
    获取指定名称的日志记录器（懒加载）

    Args:
        name: 日志记录器名称，通常为类名或模块名

    Returns:
        Logger: 配置好的日志记录器
    """
    if not _is_initialized:
        setup_logging()

    return logging.getLogger(name)
