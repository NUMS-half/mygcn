import os
import random
import logging
import pandas as pd
from datetime import datetime, timedelta
from utils.data_config import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s-%(name)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("DataGenerator")

# 设置标签的权重（可修改）
value_weights = USER_TYPE_WEIGHTS[0]
credit_weights = USER_TYPE_WEIGHTS[1]
feedback_weights = USER_TYPE_WEIGHTS[2]

# 定义反馈流程
feedback_process = FEEDBACK_PROCESS


# 随机日期生成
def random_date(start_date, end_date):
    return start_date + timedelta(seconds=random.randint(0, int((end_date - start_date).total_seconds())))


def generate_data(num_samples=1000):
    """模拟数据生成函数（默认情况下1000条）"""
    data = []
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2024, 12, 31)

    for sn in range(1, num_samples + 1):
        # 1. 随机选择每个用户类型维度的标签（根据权重）
        value_label = random.choices(DIMENSION_MAPPINGS["value"], value_weights)[0]
        credit_label = random.choices(DIMENSION_MAPPINGS["credit"], credit_weights)[0]
        feedback_label = random.choices(DIMENSION_MAPPINGS["feedback"], feedback_weights)[0]

        # 2. 随机选择用户行为数据取值
        def get_weighted_label(dimension_key):
            """从维度映射中生成一个加权的随机选择，最后一项（inf）的权重较低"""
            weights = [5] * (len(DIMENSION_MAPPINGS[dimension_key]) - 1) + [1]
            return random.choices(DIMENSION_MAPPINGS[dimension_key], weights)[0]

        traffic_label = get_weighted_label("traffic")
        package_label = get_weighted_label("package")
        voice_exceed_label = get_weighted_label("voice_exceed")
        traffic_exceed_label = get_weighted_label("traffic_exceed")
        arpu_label = get_weighted_label("arpu")
        mou_label = get_weighted_label("mou")

        # 3. 将所有行为标签拼接为一个字符串
        behavior_str = ",".join([
            f'"{traffic_label}"',
            f'"{package_label}"',
            f'"{voice_exceed_label}"',
            f'"{traffic_exceed_label}"',
            f'"{arpu_label}"',
            f'"{mou_label}"'
        ])

        # 随机选择反馈流程
        ms_process = random.choice(feedback_process)

        # 生成随机日期
        date = random_date(start_date, end_date)
        accept_dt = date.strftime('%Y/%m/%d %H:%M:%S')
        order_id = date.strftime("%Y%m%d%H%M%S") + ''.join(random.choices('0123456789', k=11))

        # 收集数据
        row = [
            "深圳",  # DYPT.BRANCH_NAM
            accept_dt,  # DYPT.ACCEPT_DT
            order_id,  # DYPT.ORDER_ID
            sn,  # DYPT.SN
            f'"{feedback_label}","{credit_label}","{value_label}",{behavior_str}',  # DYPT.LAB_NAM_COMBINED
            ms_process  # DYPT.MS
        ]

        data.append(row)

    return data


# 生成模拟数据
simulated_data = generate_data(DATA_SIZE)

# 创建DataFrame
columns = ["DYPT.BRANCH_NAM", "DYPT.ACCEPT_DT", "DYPT.ORDER_ID", "DYPT.SN", "DYPT.LAB_NAM_COMBINED", "DYPT.MS"]
df = pd.DataFrame(simulated_data, columns=columns)

# 保存为CSV文件
csv_file = "../data/raw/generated_user_behavior_data.csv"
output_dir = os.path.dirname(csv_file)
try:
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"已创建输出目录: {output_dir}")
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
except Exception as e:
    logger.error(f"保存编码映射信息失败: {e}")
logger.info(f"已生成 {DATA_SIZE} 条模拟数据，并保存为：{csv_file}")
