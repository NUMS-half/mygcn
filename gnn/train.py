import os
import time
import torch
import numpy as np
from gnn.gcn import GCN, RGCN
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.helper import set_seed, get_logger
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# 设置日志记录器
logger = get_logger("Train")

output_dir = "../output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    logger.info(f"已创建输出目录: {output_dir}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"current training device: {device}")


def load_data(split):
    """加载指定分割的数据"""
    try:
        path = f"../data/processed/{split}/knowledge_graph.pt"
        logger.info(f"正在从 {path} 加载 {split} 数据")
        return torch.load(path, weights_only=False)
    except Exception as e:
        logger.error(f"加载 {split} 数据时出错: {e}")
        raise


def preprocess_data(data):
    """预处理数据并记录统计信息"""
    data = data.to(device)

    # 只检查标签范围，但不修改无效标签
    valid_mask = (data.y >= 0) & (data.y < 6)
    invalid_count = (~valid_mask).sum().item()

    if invalid_count > 0:
        logger.info(f"发现 {invalid_count} 个非预测节点（标签超出0-5范围）")

        # 确保这些节点不在任何掩码中
        if hasattr(data, "train_mask"):
            data.train_mask = data.train_mask & valid_mask
        if hasattr(data, "val_mask"):
            data.val_mask = data.val_mask & valid_mask
        if hasattr(data, "test_mask"):
            data.test_mask = data.test_mask & valid_mask

    logger.info(f"总节点数: {data.num_nodes}, 有效标签节点数: {valid_mask.sum().item()}")

    # 检查数据是否包含掩码
    if not hasattr(data, "train_mask") or not hasattr(data, "val_mask") or not hasattr(data, "test_mask"):
        logger.warning("数据缺少训练/验证/测试掩码，请先运行数据预处理脚本生成")

    return data


def hybrid_loss(pred, target, class_weights=None, gamma=2.0, label_smoothing=0.1, balance=0.8):
    """
    混合损失函数：结合Focal Loss和标签平滑

    参数:
        pred: 模型预测 [batch_size, num_classes]
        target: 真实标签 [batch_size]
        class_weights: 类别权重 [num_classes]
        gamma: Focal Loss聚焦参数，越大越关注难分样本
        label_smoothing: 标签平滑系数
        balance: Focal Loss比重 (1-balance为标签平滑损失比重)
    """
    num_classes = pred.size(1)
    # 1. Focal Loss部分
    log_prob = F.log_softmax(pred, dim=1)
    prob = torch.exp(log_prob)
    target_prob = prob.gather(1, target.unsqueeze(1)).squeeze(1)  # 获取目标类别的概率
    focal_weight = (1 - target_prob) ** gamma  # 计算focal权重
    # 应用类别权重
    if class_weights is not None:
        per_sample_weights = class_weights.gather(0, target)
        focal_weight = focal_weight * per_sample_weights

    focal_loss_val = -focal_weight * log_prob.gather(1, target.unsqueeze(1)).squeeze(1)
    # 2. 标签平滑部分
    target_one_hot = F.one_hot(target, num_classes).float()  # 将目标转换为one-hot编码
    soft_target = target_one_hot * (1 - label_smoothing) + label_smoothing / num_classes  # 应用标签平滑
    smooth_loss_val = -torch.sum(soft_target * log_prob, dim=1)  # 计算交叉熵
    # 3. 组合损失
    combined_loss = balance * focal_loss_val + (1 - balance) * smooth_loss_val
    return combined_loss.mean()


def calculate_class_weights(class_counts, beta=0.9999, adjustment=2.0):
    """
    计算优化的类别权重

    参数:
        class_counts: 各类别样本数
        beta: 有效数量参数
        adjustment: 平衡因子调整参数

    返回:
        normalized_weights: 归一化后的类别权重
    """
    # 有效数量计算 (来自论文 "Class-Balanced Loss Based on Effective Number of Samples")
    effective_num = 1.0 - torch.pow(beta, class_counts)
    weights = (1.0 - beta) / effective_num

    # 应用平衡因子调整
    weights = torch.pow(weights, 1 / adjustment)

    # 归一化权重
    normalized_weights = weights / weights.sum() * len(weights)

    return normalized_weights


def train():
    """训练 GCN 模型，使用优化的损失函数和权重计算"""

    # 记录开始时间
    start_time = time.time()

    epoch_num = 1500
    train_data = preprocess_data(load_data("train"))
    val_data = preprocess_data(load_data("val"))

    # ===== 关键修改: 将验证数据映射到与训练数据相同的节点空间 =====
    if train_data.num_nodes != val_data.num_nodes:
        logger.warning(f"训练节点数({train_data.num_nodes})与验证节点数({val_data.num_nodes})不匹配，进行调整...")

        # 创建新的验证掩码(全False，长度与训练节点数一致)
        new_val_mask = torch.zeros(train_data.num_nodes, dtype=torch.bool, device=device)
        # 如果验证集节点少于训练集，使用前n个
        if val_data.num_nodes <= train_data.num_nodes:
            # 将原始验证掩码中的True值保留
            valid_indices = torch.where(val_data.val_mask)[0]
            new_val_mask[valid_indices] = True
        # 如果验证集节点多于训练集(不太可能)，仅保留训练集大小
        else:
            # 取前train_data.num_nodes个节点的掩码
            new_val_mask[:] = val_data.val_mask[:train_data.num_nodes]

        # 替换验证数据为训练数据，但使用验证掩码
        val_data = train_data.clone()
        val_data.val_mask = new_val_mask
        val_data.train_mask.fill_(False)  # 确保训练掩码为False

    # 确定关系数量

    # model = GCN(train_data.num_features, 128, 6).to(device)
    model = RGCN(train_data.num_features, hidden_dim=128, output_dim=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)  # 超参数 [lr: 0.001-0.005]

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.02,  # 超参数 [max_lr: 0.005-0.01]
        total_steps=epoch_num,
        pct_start=0.1,  # 预热时间 [pct_start: 0.1-0.3]
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=100
    )

    # 计算类别权重 - 改进版
    # 计算类别数量
    class_counts = torch.zeros(6)
    for label in train_data.y[train_data.train_mask]:
        if 0 <= label < 6:
            class_counts[label.item()] += 1

    logger.info(f"各类别样本数量: {class_counts.tolist()}")

    # 调用工具函数计算权重
    beta = 0.9999  # 超参数 [beta: 0.9-0.9999]
    cb_adjustment = 2.0  # 超参数 [cb_adjustment: 1.5-3.0]
    class_weights = calculate_class_weights(class_counts, beta, cb_adjustment)
    class_weights = class_weights.to(device)

    logger.info(f"优化后类别权重: {class_weights.tolist()}")

    # 早停参数
    patience = 150  # 超参数 [patience: 50-150]
    counter = 0
    best_val_acc = 0.0
    best_f1 = 0.0
    best_model_path = os.path.join(output_dir, "model")
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)

    # 跟踪指标列表
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1s = []
    epochs = []

    # 设置混合损失参数
    focal_gamma = 2.5  # 超参数 [focal_gamma: 1.0-2.5]
    label_smoothing = 0.2  # 超参数 [label_smoothing: 0.05-0.2]
    loss_balance = 0.5  # 超参数 [loss_balance: 0.7-0.9]


    for epoch in range(epoch_num):
        model.train()
        optimizer.zero_grad()

        # 动态调整focal_gamma - 随训练进程逐渐降低
        current_gamma = max(1.0, focal_gamma * (1 - epoch / epoch_num))

        # out = model(train_data.x, train_data.edge_index)
        out = model(train_data.x, train_data.edge_index, train_data.edge_type)

        # 使用混合损失函数
        loss = hybrid_loss(
            out[train_data.train_mask],
            train_data.y[train_data.train_mask],
            class_weights=class_weights,
            gamma=current_gamma,
            label_smoothing=label_smoothing,
            balance=loss_balance
        )

        loss.backward()

        # 梯度裁剪避免爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.75)  # 超参数 [max_norm: 0.5-2.0]

        optimizer.step()

        # 评估验证集性能
        model.eval()
        with torch.no_grad():
            # out = model(val_data.x, val_data.edge_index)
            out = model(train_data.x, train_data.edge_index, train_data.edge_type)

            # 计算验证损失
            val_loss = hybrid_loss(
                out[val_data.val_mask],
                val_data.y[val_data.val_mask],
                class_weights=class_weights,
                gamma=current_gamma,
                label_smoothing=label_smoothing,
                balance=loss_balance
            )

            # 获取预测标签和真实标签
            pred = out.argmax(dim=1)
            y_pred = pred[val_data.val_mask].cpu().numpy()
            y_true = val_data.y[val_data.val_mask].cpu().numpy()

            # 计算验证指标
            correct = (y_pred == y_true).sum()
            val_acc = correct / len(y_true)
            val_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            val_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            val_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # 记录指标
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        val_accuracies.append(val_acc)
        val_f1s.append(val_f1)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        epochs.append(epoch)

        # 更新学习率
        scheduler.step()

        # 每30个epoch打印指标
        if epoch % 30 == 0:
            logger.info(
                f"Epoch {epoch}: 【Loss】 Train: {loss.item():.4f}, Val: {val_loss.item():.4f} "
                f"【Val Metrics】 Acc: {val_acc:.4f}, P: {val_precision:.4f}, R: {val_recall:.4f}, F1: {val_f1:.4f}"
            )

        # 使用F1分数作为早停标准
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(best_model_path, "best_model.pth"))
            counter = 0
            logger.info(f"模型改进，当前最佳F1分数: {best_f1:.4f}, 验证准确率: {val_acc:.4f}")
        else:
            counter += 1

        # 早停检查
        if counter >= patience:
            logger.info(f"早停触发，停止训练! 最佳F1分数: {best_f1:.4f}, 验证准确率: {best_val_acc:.6f}")
            break

    # 计算并输出总训练时间
    total_time = time.time() - start_time
    formatted_time = time.strftime("%M:%S", time.gmtime(total_time))
    logger.info(f"训练完成，总耗时: {formatted_time}")

    # 绘制混淆矩阵
    final_confusion_matrix = confusion_matrix(y_true, y_pred)
    logger.info(f"最终混淆矩阵:\n{final_confusion_matrix}")

    # 添加可视化代码
    visualize_training_process(epochs, train_losses, val_losses, val_accuracies,
                               val_precisions, val_recalls, val_f1s)

    return best_model_path


def visualize_training_process(epochs, train_losses, val_losses, val_accuracies,
                               val_precisions, val_recalls, val_f1s):
    """可视化训练过程中的损失和多种评估指标变化"""

    # 创建输出目录
    output_path = os.path.join(output_dir, "figures")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        logger.info(f"已创建图片输出目录: {output_path}")

    # 创建图像 - 2x2布局用于显示所有指标
    plt.figure(figsize=(16, 12))

    # 1. 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2.5)
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2.5)
    plt.title('Loss Curves', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    # 2. 准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(epochs, val_accuracies, 'g-', linewidth=2.5)
    plt.title('Validation Accuracy', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 3. 精确率和召回率曲线
    plt.subplot(2, 2, 3)
    plt.plot(epochs, val_precisions, 'm-', label='Precision', linewidth=2.5)
    plt.plot(epochs, val_recalls, 'c-', label='Recall', linewidth=2.5)
    plt.title('Precision & Recall', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)

    # 4. F1分数曲线
    plt.subplot(2, 2, 4)
    plt.plot(epochs, val_f1s, 'y-', linewidth=2.5)
    plt.title('F1 Score', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('F1', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 调整布局并保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "all_metrics.png"), dpi=300, bbox_inches='tight')
    logger.info(f"所有评估指标可视化已保存至 {output_path}/all_metrics.png")

    # 创建简洁版图表 - 仅包含损失和F1分数
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2.5)
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2.5)
    plt.title('Loss Curves', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    # 综合指标曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, 'g-', label='Accuracy', linewidth=2.5)
    plt.plot(epochs, val_f1s, 'y-', label='F1 Score', linewidth=2.5)
    plt.title('Model Performance', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "training_summary.png"), dpi=300, bbox_inches='tight')
    logger.info(f"训练摘要可视化已保存至 {output_path}/training_summary.png")

    # 关闭图表以释放内存
    plt.close('all')


def evaluate(model, split, train_data=None):
    """
    在验证集或测试集上评估模型，计算多种性能指标

    Args:
        model: 要评估的模型
        split: 数据集分割("val"或"test")
        train_data: 训练数据，用于维度对齐(可选)
    """
    data = preprocess_data(load_data(split))

    # ===== 打印诊断信息 =====
    if train_data is not None:
        logger.debug(f"训练数据: 节点数={train_data.num_nodes}, 特征维度={train_data.num_features}")
        logger.debug(f"{split}数据: 节点数={data.num_nodes}, 特征维度={data.num_features}")

    # ===== 关键修改: 将评估数据映射到与训练数据相同的节点空间 =====
    if train_data is not None and train_data.num_nodes != data.num_nodes:
        logger.warning(f"训练节点数({train_data.num_nodes})与{split}节点数({data.num_nodes})不匹配，进行调整...")

        # 确定要使用的掩码
        if split == "val":
            eval_mask = data.val_mask
        elif split == "test":
            eval_mask = data.test_mask
        else:
            raise ValueError(f"不支持的数据集类型: {split}")

        # 创建新的评估掩码(全False，长度与训练节点数一致)
        new_eval_mask = torch.zeros(train_data.num_nodes, dtype=torch.bool, device=device)

        # 如果评估集节点少于训练集，使用有效的索引
        if data.num_nodes <= train_data.num_nodes:
            # 将原始评估掩码中的True值映射到新掩码
            valid_indices = torch.where(eval_mask)[0]
            if len(valid_indices) > 0:
                new_eval_mask[valid_indices] = True
        # 如果评估集节点多于训练集(不太可能)，仅保留训练集大小
        else:
            # 取前train_data.num_nodes个节点的掩码
            new_eval_mask[:] = eval_mask[:train_data.num_nodes]

        # 使用训练数据作为基础，但替换为评估掩码
        aligned_data = train_data.clone()
        if split == "val":
            aligned_data.val_mask = new_eval_mask
            aligned_data.train_mask.fill_(False)  # 确保训练掩码为False
            aligned_data.test_mask.fill_(False)  # 确保测试掩码为False
        else:  # test
            aligned_data.test_mask = new_eval_mask
            aligned_data.train_mask.fill_(False)  # 确保训练掩码为False
            aligned_data.val_mask.fill_(False)  # 确保验证掩码为False

        # 使用对齐后的数据
        data = aligned_data
        logger.info(f"成功将{split}数据对齐到训练数据维度")

    model.eval()
    results = {}

    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_type)
        pred = out.argmax(dim=1)

        # 根据数据集类型选择正确的掩码
        if split == "val":
            mask = data.val_mask
        elif split == "test":
            mask = data.test_mask
        else:
            raise ValueError(f"不支持的数据集类型: {split}")

        # 检查掩码维度与模型输出是否匹配
        if len(mask) != len(out):
            logger.error(f"掩码维度({len(mask)})与模型输出维度({len(out)})不匹配!")
            raise ValueError("掩码维度与模型输出不匹配")

        # 获取预测结果和真实标签（使用CPU以便用于sklearn）
        mask_indices = torch.where(mask)[0]
        if len(mask_indices) == 0:
            logger.warning(f"在{split}集中没有标记为True的样本!")
            results['accuracy'] = 0
            results['precision'] = 0
            results['recall'] = 0
            results['f1'] = 0
            results['confusion_matrix'] = np.array([[0]])
            return results

        y_pred = pred[mask].cpu().numpy()
        y_true = data.y[mask].cpu().numpy()

        # 计算准确率
        correct = (y_pred == y_true).sum()
        results['accuracy'] = correct / len(y_true)

        # 计算精确率、召回率、F1分数（使用宏平均以适应多分类）
        results['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        results['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        results['f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # 计算混淆矩阵
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred)

    return results


def test():
    """测试模型"""
    # 加载训练数据用于对齐
    train_data = preprocess_data(load_data("train"))

    # 加载模型
    model = RGCN(train_data.num_features, hidden_dim=128, output_dim=6).to(device)
    model.load_state_dict(
        torch.load(os.path.join(output_dir, "model/best_model.pth"), map_location=device, weights_only=True))

    # 传入训练数据进行对齐评估
    test_results = evaluate(model, "test", train_data)

    # 输出测试结果
    logger.info(f"{'=' * 15} 测试集评估结果 {'=' * 15}")
    logger.info(f"准确率 (Accuracy): {test_results['accuracy']:.6f}")
    logger.info(f"精确率 (Precision): {test_results['precision']:.6f}")
    logger.info(f"召回率 (Recall): {test_results['recall']:.6f}")
    logger.info(f"F1分数: {test_results['f1']:.6f}")

    # 输出混淆矩阵
    conf_matrix = test_results['confusion_matrix']
    logger.info(f"混淆矩阵:")
    logger.info(f"\n{conf_matrix}")

    # 计算每个类别的预测准确率
    logger.info(f"\n{'=' * 15} 各类别预测准确率 {'=' * 15}")
    for i in range(conf_matrix.shape[0]):
        row_sum = conf_matrix[i].sum()
        if row_sum > 0:  # 避免除零错误
            class_accuracy = conf_matrix[i, i] / row_sum
            logger.info(f"类别 {i}: {class_accuracy * 100:.2f}% ({conf_matrix[i, i]}/{row_sum})")
        else:
            logger.info(f"类别 {i}: 无样本")

    return test_results


# if __name__ == "__main__":
#     try:
#         # 存储多轮测试的结果
#         rounds = 5
#         all_results = []
#
#         logger.info(f"开始执行{rounds}轮训练与评估...")
#
#         for round_num in range(1, rounds + 1):
#             logger.info(f"\n{'=' * 20} 第 {round_num}/{rounds} 轮训练 {'=' * 20}")
#             set_seed(42 + round_num)  # 每轮使用不同种子
#
#             best_model_path = train()
#             if best_model_path:
#                 test_results = test()
#                 all_results.append(test_results)
#                 logger.info(f"第 {round_num} 轮完成，F1: {test_results['f1']:.6f}")
#             else:
#                 logger.error(f"第 {round_num} 轮训练失败，跳过测试")
#
#         # 计算统计数据
#         if all_results:
#             logger.info(f"\n{'=' * 20} {round} 轮训练统计结果 {'=' * 20}")
#
#             # 提取各项指标
#             accuracies = [r['accuracy'] for r in all_results]
#             precisions = [r['precision'] for r in all_results]
#             recalls = [r['recall'] for r in all_results]
#             f1s = [r['f1'] for r in all_results]
#
#             # 计算平均值和标准差
#             import numpy as np
#
#             metrics = {
#                 "准确率 (Accuracy)": accuracies,
#                 "精确率 (Precision)": precisions,
#                 "召回率 (Recall)": recalls,
#                 "F1分数": f1s
#             }
#
#             for name, values in metrics.items():
#                 mean = np.mean(values)
#                 std = np.std(values)
#                 logger.info(f"{name}: {mean:.6f} ± {std:.6f}")
#
#             # 打印所有轮次的原始数据
#             logger.info("\n各轮次详细指标:")
#             for i, r in enumerate(all_results):
#                 logger.info(
#                     f"轮次 {i + 1}: Acc={r['accuracy']:.4f}, P={r['precision']:.4f}, R={r['recall']:.4f}, F1={r['f1']:.4f}")
#
#     except Exception as e:
#         logger.error(f"发生错误: {e}")
#         import traceback
#
#         logger.error(traceback.format_exc())
if __name__ == "__main__":
    try:
        set_seed()
        best_model_path = train()
        if best_model_path:
            test()
        else:
            logger.error("训练失败，跳过测试")
    except Exception as e:
        logger.error(f"发生错误: {e}")
        import traceback

        logger.error(traceback.format_exc())
