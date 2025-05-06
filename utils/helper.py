import os

import sklearn
import yaml
import torch
import random
import logging
import numpy as np
from gnn.models import *
import torch.optim as optim
from datetime import datetime
from torch.optim import lr_scheduler

__all__ = [
    "setup_logging",
    "get_logger",
    "set_seed",
    "load_config",
    "save_config",
    "get_model",
    "get_model_out",
    "get_optimizer",
    "get_scheduler",
    "get_config_value"
]

# 全局日志配置
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s-%(name)s-%(levelname)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_DIR = "./logs"
CONFIG_PATH = "./config.yml"

# 确保日志目录存在
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 定义：是否已初始化
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

    # 添加文件处理器
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


def set_seed(seed=42):
    """
    设置随机种子，确保实验可复现性

    Args:
        seed: 随机种子值，默认为42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    sklearn.utils.check_random_state(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# ===== 配置文件相关功能 =====

def load_config(config_path=CONFIG_PATH):
    """
    加载YAML配置文件

    Args:
        config_path: 配置文件路径，默认为"config.yml"

    Returns:
        dict: 加载的配置字典
    """
    logger = get_logger("Config")

    if not os.path.exists(config_path):
        logger.warning(f"配置文件 {config_path} 不存在")
        return None

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        logger.info(f"成功加载配置文件: {config_path}")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return None


def save_config(config, config_path=CONFIG_PATH):
    """
    保存配置到YAML文件

    Args:
        config: 配置字典
        config_path: 保存路径，默认为"config.yml"
    """
    logger = get_logger("Config")

    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        logger.info(f"配置已保存到: {config_path}")
    except Exception as e:
        logger.error(f"保存配置失败: {e}")


def get_model(config, num_features, device):
    """
    根据配置创建模型实例

    Args:
        config: 配置字典
        num_features: 输入特征维度
        device: 设备(CPU/GPU)

    Returns:
        torch.nn.Module: 创建的模型实例
    """
    logger = get_logger("ModelBuilder")

    model_config = config["model"]
    model_type = model_config["type"]
    hidden_dim = model_config["hidden_dim"]
    output_dim = model_config["output_dim"]
    dropout = model_config["dropout"]
    edge_dropout = model_config["edge_dropout"]

    try:
        if model_type == "GCN":
            model = GCN(num_features, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout, edge_dropout=edge_dropout)
            logger.info(f"创建GCN模型: hidden_dim={hidden_dim}, output_dim={output_dim}, dropout={dropout}, edge_dropout={edge_dropout}")
        elif model_type == "UserBehaviorGCN":
            model = UserBehaviorGCN(num_features, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout, edge_dropout=edge_dropout)
            logger.info(f"UserBehaviorGCN: hidden_dim={hidden_dim}, output_dim={output_dim}, dropout={dropout}, edge_dropout={edge_dropout}")
        elif model_type == "GAT":
            model = GAT(num_features, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout, edge_dropout=edge_dropout)
            logger.info(f"创建GAT模型: hidden_dim={hidden_dim}, output_dim={output_dim}, dropout={dropout}, edge_dropout={edge_dropout}")
        elif model_type == "GraphSAGE":
            model = GraphSAGE(num_features, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout, edge_dropout=edge_dropout)
            logger.info(f"创建GraphSAGE模型: hidden_dim={hidden_dim}, output_dim={output_dim}, dropout={dropout}, edge_dropout={edge_dropout}")
        elif model_type == "RGCN":
            model = RGCN(num_features, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout, edge_dropout=edge_dropout)
            logger.info(f"创建RGCN模型: hidden_dim={hidden_dim}, output_dim={output_dim}, dropout={dropout}, edge_dropout={edge_dropout}")
        elif model_type == "CausalRGCN":
            model = CausalRGCN(num_features, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout, edge_dropout=edge_dropout)
            logger.info(f"创建CausalRGCN模型: hidden_dim={hidden_dim}, output_dim={output_dim}, dropout={dropout}, edge_dropout={edge_dropout}")
        else:
          logger.error(f"不支持的模型类型: {model_type}")
          raise ValueError(f"不支持的模型类型: {model_type}")
    except ImportError:
        logger.error(f"无法导入 {model_type} 模型，请确保 models.py 文件存在")
        raise
    return model.to(device)

def get_model_out(model_type, model, data):
    """
    根据配置获取模型输出维度

    参数:
        config: 配置字典
    返回:
        out: 模型输出
    """

    if model_type in ["UserBehaviorGCN", "RGCN", "CausalRGCN"]:
        return model(data.x, data.edge_index, data.edge_type)
    else:
        return model(data.x, data.edge_index)


def get_optimizer(config, model_parameters):
    """
    根据配置创建优化器

    Args:
        config: 配置字典
        model_parameters: 模型参数

    Returns:
        torch.optim.Optimizer: 创建的优化器
    """
    logger = get_logger("OptimizerBuilder")

    optimizer_config = config["optimizer"]
    optimizer_type = optimizer_config["type"]
    lr = optimizer_config["lr"]
    weight_decay = optimizer_config["weight_decay"]

    if optimizer_type == "AdamW":
        optimizer = optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)
        logger.info(f"创建AdamW优化器: lr={lr}, weight_decay={weight_decay}")
    else:
        logger.error(f"不支持的优化器类型: {optimizer_type}")
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")

    return optimizer


def get_scheduler(config, optimizer, epoch_num):
    """
    根据配置创建学习率调度器

    Args:
        config: 配置字典
        optimizer: 优化器实例
        epoch_num: 总训练轮数

    Returns:
        torch.optim.lr_scheduler._LRScheduler: 创建的学习率调度器
    """
    logger = get_logger("SchedulerBuilder")

    scheduler_config = config["scheduler"]
    scheduler_type = scheduler_config["type"]

    if scheduler_type == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get("mode", "max"),          # 默认监控F1分数提高
            factor=scheduler_config.get("factor", 0.5),        # 学习率减少因子
            patience=scheduler_config.get("patience", 10),     # 等待多少个epoch无改善后降低学习率
            min_lr=scheduler_config.get("min_lr", 1e-6)        # 最小学习率
        )
        logger.info(
            f"创建ReduceLROnPlateau调度器: mode={scheduler_config.get('mode', 'max')}, "
            f"factor={scheduler_config.get('factor', 0.5)}, patience={scheduler_config.get('patience', 10)}, "
            f"min_lr={scheduler_config.get('min_lr', 1e-6)}")
    elif scheduler_type == "OneCycleLR":
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=scheduler_config["max_lr"],
            total_steps=epoch_num,
            pct_start=scheduler_config["pct_start"],
            anneal_strategy=scheduler_config["anneal_strategy"],
            div_factor=scheduler_config["div_factor"],
            final_div_factor=scheduler_config["final_div_factor"]
        )
        logger.info(
            f"创建OneCycleLR调度器: max_lr={scheduler_config['max_lr']}, pct_start={scheduler_config['pct_start']}, "
            f"total_steps={epoch_num}, strategy={scheduler_config['anneal_strategy']}, div={scheduler_config['div_factor']}, final_div={scheduler_config['final_div_factor']}")
    else:
        logger.error(f"不支持的调度器类型: {scheduler_type}")
        raise ValueError(f"不支持的调度器类型: {scheduler_type}")

    return scheduler


def get_config_value(config, path, default=None):
    """
    安全地从配置字典中获取嵌套值

    Args:
        config: 配置字典
        path: 键路径，例如 "model.hidden_dim"
        default: 如果路径不存在，返回的默认值

    Returns:
        获取的值或默认值
    """
    keys = path.split('.')
    value = config

    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default
