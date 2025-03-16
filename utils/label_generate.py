import random
import pandas as pd
from utils.helper import set_seed
from utils.data_config import LABELS_MAPPING
from utils.logger import get_logger

logger = get_logger("LabelGenerator")

# 读取数据
INPUT_FILE = "../data/raw/generated_user_behavior_data.csv"
OUTPUT_FILE = "../data/raw/labeled_user_behavior_data.csv"

noise = 0.025  # 2.5% 概率随机改变标签

def add_noise(label):
    """
    给定真实标签，随机调整一定比例，使数据更加贴合实际
    """
    noise_probability = noise
    if random.random() < noise_probability:
        return random.choice(list(LABELS_MAPPING.keys()))  # 随机分配一个新标签
    return label


def predict_behavior(row):
    """
    根据用户行为特征预测其下一步可能的行为类别 (BEHAVIOR_LABEL)
    """
    user_type = row["DYPT.LAB_NAM_COMBINED"]
    feedback = row["DYPT.MS"]

    # 获取关键特征
    is_complaint = "投诉用户" in user_type
    is_blacklist = "黑名单用户" in user_type
    is_high_risk = "高风险用户" in user_type
    is_high_value = "高价值用户" in user_type
    is_low_value = "低价值用户" in user_type
    is_silent = "沉默反馈用户" in user_type
    high_arpu = any(x in user_type for x in ["上月折后ARPU在(100,200]", "上月折后ARPU在(200,inf]"])
    low_arpu = "上月折后ARPU在(0,30]" in user_type
    high_traffic = any(x in user_type for x in ["本月截至当前使用总流量(500,1000]", "本月截至当前使用总流量(1000,inf]"])
    low_traffic = "本月截至当前使用总流量(0,10]" in user_type
    high_package = any(x in user_type for x in ["主套餐月租费(169,269]", "主套餐月租费(269,inf]"])
    low_package = "主套餐月租费(0,19]" in user_type

    # 逻辑判断
    if is_blacklist or is_high_risk:
        label = 5  # 信用恶化（黑名单或高风险用户）
    elif "账单问题" in feedback:
        if is_complaint and is_low_value and low_arpu:
            label = 2  # 投诉 + 低价值用户 + 低消费 -> 套餐降级
        elif is_complaint and is_high_value and high_arpu:
            label = 1  # 投诉 + 高价值用户 + 高消费 -> 套餐升级
        else:
            label = 6  # 账单问题但影响不大，保持现状
    elif "流量" in feedback:
        if is_complaint and high_traffic and high_arpu:
            label = 3  # 高流量 + 高消费 + 投诉 -> 流量超套投诉
        elif is_complaint and low_traffic and low_arpu:
            label = 2  # 低流量 + 低消费 + 投诉 -> 降级
        elif is_high_value and high_arpu:
            label = 1  # 高价值用户 + 高消费 -> 可能升级
        else:
            label = 6  # 影响不大，保持现状
    elif "套餐问题" in feedback:
        if is_high_value and high_package:
            label = 1  # 高套餐 + 高价值用户可能升级
        elif is_low_value and low_package:
            label = 2  # 低套餐 + 低价值用户可能降级
        elif is_complaint:
            label = 2  # 投诉用户可能降级
        else:
            label = 6  # 影响不大，保持现状
    elif is_silent:
        if high_arpu and is_high_value:
            label = 4  # 高消费用户可能绑定融合业务
        elif low_arpu and is_low_value:
            label = 6  # 低消费用户更可能保持现状
        elif is_blacklist or is_high_risk:
            label = 5  # 信用恶化
        else:
            label = 6  # 保持现状
    elif "网络问题" in feedback:
        if is_complaint and low_package:
            label = 0  # 低套餐 + 投诉用户 -> 可能流失
        elif is_complaint and high_package:
            label = 1  # 高套餐 + 投诉用户 -> 可能升级
        else:
            label = 6  # 影响不大，保持现状
    else:
        label = 6  # 其他情况保持现状

    return add_noise(label)  # 添加随机噪声，使数据更真实
    # return label


def process_data():
    """
    读取数据、预测 BEHAVIOR_LABEL 并保存
    """
    set_seed()
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    # 预测每个用户的行为标签
    df["BEHAVIOR_LABEL"] = df.apply(predict_behavior, axis=1)

    # 保存带标签的数据
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    logger.info(f"已生成 {len(df)} 条数据（噪声：{noise * 100}%），保存至 {OUTPUT_FILE}")


if __name__ == "__main__":
    process_data()
