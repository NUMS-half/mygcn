import pandas as pd
import random
from utils.data_config import (LABELS_MAPPING, VALUE_LABELS, CREDIT_LABELS, FEEDBACK_LABELS,
                               TRAFFIC_LABELS, PACKAGE_LABELS, VOICE_EXCEED_LABELS,
                               TRAFFIC_EXCEED_LABELS, ARPU_LABELS, MOU_LABELS)
from utils.helper import set_seed, get_logger

logger = get_logger("LabelGenerator")

# 读取数据
INPUT_FILE = "../data/raw/generated_user_behavior_data.csv"
OUTPUT_FILE = "../data/raw/labeled_user_behavior_data.csv"


def extract_dimension_features(row):
    """
    详细提取用户的42个维度特征
    """
    user_type = row["DYPT.LAB_NAM_COMBINED"]
    feedback = row["DYPT.MS"]

    features = {}

    # 1. 用户价值维度 (3)
    for label in VALUE_LABELS:
        features[f"value_{label}"] = label in user_type

    # 2. 用户信用维度 (3)
    for label in CREDIT_LABELS:
        features[f"credit_{label}"] = label in user_type

    # 3. 用户反馈维度 (3)
    for label in FEEDBACK_LABELS:
        features[f"feedback_{label}"] = label in user_type

    # 4. 流量使用维度 (7)
    for label in TRAFFIC_LABELS:
        features[f"traffic_{label}"] = label in user_type

    # 5. 月租套餐资费维度 (7)
    for label in PACKAGE_LABELS:
        features[f"package_{label}"] = label in user_type

    # 6. 语音超套费用维度 (6)
    for label in VOICE_EXCEED_LABELS:
        features[f"voice_exceed_{label}"] = label in user_type

    # 7. 流量超套费用维度 (4)
    for label in TRAFFIC_EXCEED_LABELS:
        features[f"traffic_exceed_{label}"] = label in user_type

    # 8. 上月折后ARPU维度 (5)
    for label in ARPU_LABELS:
        features[f"arpu_{label}"] = label in user_type

    # 9. 上月MOU维度 (4)
    for label in MOU_LABELS:
        features[f"mou_{label}"] = label in user_type

    # 额外提取反馈内容
    feedback_types = ["账单问题", "流量", "套餐问题", "网络问题", "服务问题", "其他"]
    for fb_type in feedback_types:
        features[f"ms_{fb_type}"] = fb_type in feedback

    return features


def determine_label_by_rules(features):
    """
    基于明确的规则判断用户行为标签
    """
    score = {
        0: 0,  # 流失离网
        1: 0,  # 套餐升级
        2: 0,  # 套餐降级
        3: 0,  # 流量超套投诉
        4: 0,  # 信用恶化
        5: 0  # 保持现状
    }

    # --- 标签0：流失离网 ---

    # 黑名单/高风险用户 + 投诉用户 有最高流失风险
    if features.get("credit_黑名单用户", False) and features.get("feedback_投诉用户", False):
        score[0] += 40
    elif features.get("credit_高风险用户", False) and features.get("feedback_投诉用户", False):
        score[0] += 30

    # 网络问题投诉是流失的主要原因
    if features.get("feedback_投诉用户", False) and features.get("ms_网络问题", False):
        score[0] += 25

    # 高价值投诉用户流失风险
    if features.get("value_高价值用户", False) and features.get("feedback_投诉用户", False):
        score[0] += 20

    # 低价值用户本身流失风险就较高
    if features.get("value_低价值用户", False):
        score[0] += 15

    # ARPU低且套餐高的用户往往在考虑流失
    arpu_low = any(features.get(f"arpu_{label}", False) for label in ARPU_LABELS[:2])
    package_high = any(features.get(f"package_{label}", False) for label in PACKAGE_LABELS[5:])
    if arpu_low and package_high:
        score[0] += 20

    # --- 标签1：套餐升级 ---

    # 高价值用户 + 高流量使用 是升级套餐的主力
    high_traffic = any(features.get(f"traffic_{label}", False) for label in TRAFFIC_LABELS[5:])
    if features.get("value_高价值用户", False) and high_traffic:
        score[1] += 30

    # 流量超套费用高的用户很可能升级套餐
    high_traffic_exceed = any(features.get(f"traffic_exceed_{label}", False) for label in TRAFFIC_EXCEED_LABELS[2:])
    if high_traffic_exceed:
        score[1] += 35

    # 中价值用户 + 流量咨询 也可能升级
    if (features.get("value_中价值用户", False) and
            features.get("feedback_咨询用户", False) and
            features.get("ms_流量", False)):
        score[1] += 25

    # 高MOU + 中等流量 的用户可能需要更全面的套餐
    high_mou = any(features.get(f"mou_{label}", False) for label in MOU_LABELS[2:])
    mid_traffic = any(features.get(f"traffic_{label}", False) for label in TRAFFIC_LABELS[3:5])
    if high_mou and mid_traffic:
        score[1] += 20

    # --- 标签2：套餐降级 ---

    # 低价值用户 + 高套餐资费 是降级的典型情况
    high_package = any(features.get(f"package_{label}", False) for label in PACKAGE_LABELS[4:])
    if features.get("value_低价值用户", False) and high_package:
        score[2] += 35

    # 账单问题投诉的用户可能降级
    if features.get("feedback_投诉用户", False) and features.get("ms_账单问题", False):
        score[2] += 25

    # 低ARPU + 套餐咨询 可能导致降级
    low_arpu = any(features.get(f"arpu_{label}", False) for label in ARPU_LABELS[:2])
    if low_arpu and features.get("ms_套餐问题", False):
        score[2] += 30

    # 低MOU + 高套餐 的用户浪费套餐资源，可能降级
    low_mou = any(features.get(f"mou_{label}", False) for label in MOU_LABELS[:2])
    if low_mou and high_package:
        score[2] += 20

    # --- 标签3：流量超套投诉 ---

    # 高流量使用 + 低套餐 是超套投诉的高风险组合
    low_package = any(features.get(f"package_{label}", False) for label in PACKAGE_LABELS[:3])
    if high_traffic and low_package:
        score[3] += 45

    # 历史流量超套费用高的用户更易再次投诉
    if high_traffic_exceed:
        score[3] += 35

    # 中等流量 + 流量咨询 可能预示超套问题
    if mid_traffic and features.get("ms_流量", False):
        score[3] += 25

    # --- 标签4：信用恶化 ---

    # 已经在高风险名单上的用户信用恶化风险高
    if features.get("credit_高风险用户", False):
        score[4] += 50

    # 账单问题投诉 + 高额费用 可能导致信用问题
    high_arpu = any(features.get(f"arpu_{label}", False) for label in ARPU_LABELS[3:])
    if features.get("ms_账单问题", False) and high_arpu:
        score[4] += 30

    # 高套餐消费 + 高超套费用 + 投诉 可能导致拒付
    high_voice_exceed = any(features.get(f"voice_exceed_{label}", False) for label in VOICE_EXCEED_LABELS[3:])
    if (high_package and (high_voice_exceed or high_traffic_exceed) and
            features.get("feedback_投诉用户", False)):
        score[4] += 40

    # --- 标签5：保持现状 ---

    # 中价值 + 非投诉 + 信用良好 的用户最稳定
    if (features.get("value_中价值用户", False) and
            not features.get("feedback_投诉用户", False) and
            features.get("credit_信用良好用户", False)):
        score[5] += 40

    # 沉默反馈用户天然倾向于保持现状
    if features.get("feedback_沉默反馈用户", False):
        score[5] += 30

    # 中等ARPU + 中等套餐 的用户往往满意当前状态
    mid_arpu = any(features.get(f"arpu_{label}", False) for label in ARPU_LABELS[2:3])
    mid_package = any(features.get(f"package_{label}", False) for label in PACKAGE_LABELS[2:4])
    if mid_arpu and mid_package:
        score[5] += 25

    # 流量使用与套餐匹配度高的用户也倾向于保持
    low_traffic = any(features.get(f"traffic_{label}", False) for label in TRAFFIC_LABELS[:3])
    low_package = any(features.get(f"package_{label}", False) for label in PACKAGE_LABELS[:3])
    mid_package = any(features.get(f"package_{label}", False) for label in PACKAGE_LABELS[3:5])
    high_package = any(features.get(f"package_{label}", False) for label in PACKAGE_LABELS[5:])

    if (low_traffic and low_package) or (mid_traffic and mid_package) or (high_traffic and high_package):
        score[5] += 30

    # 无投诉也是保持现状的重要指标
    if not features.get("feedback_投诉用户", False) and not features.get("feedback_咨询用户", False):
        score[5] += 20

    # 找到得分最高的标签
    max_score = -1
    best_label = 5  # 默认为保持现状

    for label, label_score in score.items():
        if label_score > max_score:
            max_score = label_score
            best_label = label

    return best_label


def process_data():
    """
    读取数据、预测 BEHAVIOR_LABEL 并保存
    """
    set_seed()
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    # 提取特征并预测标签
    labels = []
    for _, row in df.iterrows():
        features = extract_dimension_features(row)
        label = determine_label_by_rules(features)
        labels.append(label)

    df["BEHAVIOR_LABEL"] = labels

    # 计算标签分布并打印
    label_counts = df["BEHAVIOR_LABEL"].value_counts(normalize=True).sort_index() * 100
    distribution_info = "\n".join([f"    标签 {label}: {LABELS_MAPPING[label]['name']} - {count:.1f}%"
                                   for label, count in label_counts.items()])

    logger.info(f"生成的标签分布:\n{distribution_info}")

    # 保存带标签的数据
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    logger.info(f"已生成 {len(df)} 条带标签数据，保存至 {OUTPUT_FILE}")


if __name__ == "__main__":
    process_data()