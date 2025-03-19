"""生成模拟数据配置文件"""

__all__ = [
    "DATA_SIZE",
    "DIMENSION_MAPPINGS",
    "DIMENSION_TO_RELATION",
    "USER_TYPE_WEIGHTS",
    "FEEDBACK_PROCESS",
    "LABELS_MAPPING"
]

# 标签
LABELS_MAPPING = {
    0: {"name": "流失离网", "description": "用户在接下来30天内将会主动销户或转网至其他运营商"},
    1: {"name": "套餐升级", "description": "用户将升级至更高档位套餐或叠加增值服务"},
    2: {"name": "套餐降级", "description": "用户将降低套餐档位或取消增值服务"},
    3: {"name": "流量超套投诉", "description": "用户将可能因为流量超额产生高费用并提交投诉工单"},
    4: {"name": "信用恶化", "description": "用户信用将会变差，考虑加入黑名单或灰名单"},
    5: {"name": "保持现状", "description": "用户将不会发生上述任何行为变化"}
}

# 数据量设置
DATA_SIZE = 30000

# 维度标签配置
# 1. 用户价值维度
VALUE_LABELS = ["高价值用户", "中价值用户", "低价值用户"]
# 2. 用户信用维度
CREDIT_LABELS = ["黑名单用户", "高风险用户", "信用良好用户"]
# 3. 用户反馈维度
FEEDBACK_LABELS = ["咨询用户", "投诉用户", "沉默反馈用户"]
# 4. 流量使用维度
TRAFFIC_LABELS = [f"本月截至当前使用总流量({start},{end}]" for start, end in
                  [(0, 10),
                   (10, 50),
                   (50, 100),
                   (100, 250),
                   (250, 500),
                   (500, 1000),
                   (1000, 'inf')]]
# 5. 月租套餐资费维度
PACKAGE_LABELS = [f"主套餐月租费({start},{end}]" for start, end in
                  [(0, 19),
                   (19, 59),
                   (59, 99),
                   (99, 139),
                   (139, 169),
                   (169, 269),
                   (269, 'inf')]]
# 6. 语音超套费用维度
VOICE_EXCEED_LABELS = [f"上月超套语音费用({start},{end}]" for start, end in
                       [(0, 10),
                        (10, 30),
                        (30, 60),
                        (60, 100),
                        (100, 200),
                        (200, 'inf')]]
# 7. 流量超套费用维度
TRAFFIC_EXCEED_LABELS = [f"上月超套流量费用({start},{end}]" for start, end in
                         [(0, 30),
                          (30, 60),
                          (60, 100),
                          (100, 'inf')]]
# 8. 上月折后ARPU维度
ARPU_LABELS = [f"上月折后ARPU在({start},{end}]" for start, end in
               [(0, 30),
                (30, 50),
                (50, 100),
                (100, 200),
                (200, 'inf')]]
# 9. 上月MOU维度
MOU_LABELS = [f"上月MOU在({start},{end}]" for start, end in
              [(0, 30),
               (30, 60),
               (60, 100),
               (100, 'inf')]]

# 维度标签映射
DIMENSION_MAPPINGS = {
    "value": VALUE_LABELS,
    "credit": CREDIT_LABELS,
    "feedback": FEEDBACK_LABELS,
    "traffic": TRAFFIC_LABELS,
    "package": PACKAGE_LABELS,
    "voice_exceed": VOICE_EXCEED_LABELS,
    "traffic_exceed": TRAFFIC_EXCEED_LABELS,
    "arpu": ARPU_LABELS,
    "mou": MOU_LABELS
}

# 边类型映射
DIMENSION_TO_RELATION = {
    # 用户特征关系
    'value': 0,
    'credit': 0,
    'feedback': 0,

    # 用户套餐关系
    'package': 1,
    'arpu': 1,

    # 用户行为关系
    'traffic': 2,
    'voice_exceed': 2,
    'traffic_exceed': 2,
    'mou': 2
}

# 用户类型比例
USER_TYPE_WEIGHTS = [
    [0.2, 0.5, 0.3],  # value维度：高价值用户占20%，中价值用户占50%，低价值用户占30%
    [0.2, 0.1, 0.7],  # credit维度：黑名单用户占20%，高风险用户占10%，信用良好用户占70%
    [0.2, 0.6, 0.2],  # feedback维度：咨询用户占20%，投诉用户占60%，沉默反馈用户占20%
]

# 反馈流程列表
FEEDBACK_PROCESS = [
    "移动服务→基础服务→资费套餐→全局流转→业务规则→对业务规定/流程不满→全局流转",
    "套餐问题→用户反映套餐收费不明确→业务规则→流程不清晰，未得到及时响应",
    "语音和流量问题→语音费用高，流量包不清晰→服务无法满足需求→用户希望改进",
    "网络问题→网络不稳定，无法正常上网→技术支持没有及时处理→全局流转",
    "账单问题→上月账单计算不准确，产生意外费用→用户服务不佳→全局流转"
]
