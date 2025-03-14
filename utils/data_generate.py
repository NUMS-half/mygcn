import os
import random
import pandas as pd
from utils.data_config import *
from utils.logger import get_logger
from datetime import datetime, timedelta


class DataGenerator:
    """用户行为数据生成器，用于生成模拟的用户行为数据"""

    def __init__(self, output_dir="../data/raw"):
        """
        初始化数据生成器

        参数:
            output_dir: 输出目录路径
        """
        self.output_dir = output_dir
        self.logger = get_logger("DataGenerator")

        # 设置标签的权重
        self.value_weights = USER_TYPE_WEIGHTS[0]
        self.credit_weights = USER_TYPE_WEIGHTS[1]
        self.feedback_weights = USER_TYPE_WEIGHTS[2]

        # 定义反馈流程
        self.feedback_process = FEEDBACK_PROCESS

    @staticmethod
    def random_date(start_date, end_date):
        """生成两个日期之间的随机日期"""
        return start_date + timedelta(seconds=random.randint(0, int((end_date - start_date).total_seconds())))

    @staticmethod
    def get_weighted_label(dimension_key):
        """从维度映射中生成一个加权的随机选择，最后一项权重较低"""
        weights = [5] * (len(DIMENSION_MAPPINGS[dimension_key]) - 1) + [1]
        return random.choices(DIMENSION_MAPPINGS[dimension_key], weights)[0]

    def generate_data(self, num_samples=1000):
        """
        生成模拟用户行为数据

        参数:
            num_samples: 要生成的数据样本数量

        返回:
            list: 生成的数据行列表
        """
        data = []
        start_date = datetime(2021, 1, 1)
        end_date = datetime(2024, 12, 31)

        for sn in range(1, num_samples + 1):
            # 1. 随机选择用户类型维度的标签
            value_label = random.choices(DIMENSION_MAPPINGS["value"], self.value_weights)[0]
            credit_label = random.choices(DIMENSION_MAPPINGS["credit"], self.credit_weights)[0]
            feedback_label = random.choices(DIMENSION_MAPPINGS["feedback"], self.feedback_weights)[0]

            # 2. 随机选择用户行为数据
            traffic_label = self.get_weighted_label("traffic")
            package_label = self.get_weighted_label("package")
            voice_exceed_label = self.get_weighted_label("voice_exceed")
            traffic_exceed_label = self.get_weighted_label("traffic_exceed")
            arpu_label = self.get_weighted_label("arpu")
            mou_label = self.get_weighted_label("mou")

            # 3. 拼接行为标签字符串
            behavior_str = ",".join([
                f'"{traffic_label}"',
                f'"{package_label}"',
                f'"{voice_exceed_label}"',
                f'"{traffic_exceed_label}"',
                f'"{arpu_label}"',
                f'"{mou_label}"'
            ])

            # 随机选择反馈流程
            ms_process = random.choice(self.feedback_process)

            # 生成随机日期和订单ID
            date = self.random_date(start_date, end_date)
            accept_dt = date.strftime('%Y/%m/%d %H:%M:%S')
            order_id = date.strftime("%Y%m%d%H%M%S") + ''.join(random.choices('0123456789', k=11))

            # 构建数据行
            row = [
                "",  # LABEL
                "深圳",  # DYPT.BRANCH_NAM
                accept_dt,  # DYPT.ACCEPT_DT
                order_id,  # DYPT.ORDER_ID
                sn,  # DYPT.SN
                f'"{feedback_label}","{credit_label}","{value_label}",{behavior_str}',  # DYPT.LAB_NAM_COMBINED
                ms_process  # DYPT.MS
            ]

            data.append(row)

        return data

    def save_to_csv(self, data, filename="generated_user_behavior_data.csv"):
        """
        将生成的数据保存为CSV文件

        参数:
            data: 要保存的数据
            filename: 文件名

        返回:
            str: 保存的文件路径，失败则返回None
        """
        # 创建DataFrame
        columns = ["BEHAVIOR_LABEL", "DYPT.BRANCH_NAM", "DYPT.ACCEPT_DT", "DYPT.ORDER_ID",
                   "DYPT.SN", "DYPT.LAB_NAM_COMBINED", "DYPT.MS"]
        df = pd.DataFrame(data, columns=columns)

        # 保存为CSV文件
        csv_file = os.path.join(self.output_dir, filename)

        try:
            # 确保输出目录存在
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                self.logger.info(f"已创建输出目录: {self.output_dir}")

            df = df.sort_values(by='DYPT.ACCEPT_DT')
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            self.logger.info(f"已生成 {len(data)} 条模拟数据，并保存为：{csv_file}")
            return csv_file
        except Exception as e:
            self.logger.error(f"保存数据失败: {e}")
            return None

    def process(self, num_samples=DATA_SIZE, filename="generated_user_behavior_data.csv"):
        """
        执行完整的数据生成流程

        参数:
            num_samples: 要生成的样本数量
            filename: 输出文件名

        返回:
            str: 保存的文件路径，失败则返回None
        """
        try:
            data = self.generate_data(num_samples)
            return self.save_to_csv(data, filename)
        except Exception as e:
            self.logger.error(f"数据生成过程中出错: {e}")
            return None


def main():
    """主函数，用于直接执行脚本时运行"""
    generator = DataGenerator()
    generator.process()


if __name__ == "__main__":
    main()