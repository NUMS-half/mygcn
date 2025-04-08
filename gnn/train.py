import os
import time
import torch
import numpy as np

from utils.helper import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gnn.layers import ImprovedCausalRGCNConv
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score

# 设置日志记录器
logger = get_logger("Train")

# 创建输出目录
output_dir = "../output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    logger.info(f"已创建输出目录: {output_dir}")

# 检测设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"current training device: {device}")

# 全局变量，缓存图谱数据
_cached_data = None


def load_data():
    """加载图谱数据（懒加载）"""
    global _cached_data

    # 若数据已缓存，则直接返回
    if _cached_data is not None:
        return _cached_data
    try:
        path = f"../data/processed/knowledge_graph.pt"
        logger.info(f"正在从 {path} 加载图谱数据")
        _cached_data = torch.load(path, weights_only=False)

        # 检查边类型的实际数量
        unique_edge_types = torch.unique(_cached_data.edge_type)
        num_relations = unique_edge_types.max().item() + 1
        logger.info(f"数据中的关系类型数量与分布: {num_relations}: {unique_edge_types.tolist()}")

        return _cached_data
    except Exception as e:
        logger.error(f"加载图谱数据时出错: {e}")
        raise


def preprocess_data(data):
    """预处理数据并记录统计信息"""
    data = data.to(device)

    # 只检查标签范围，但不修改无效标签
    valid_mask = (data.y >= 0) & (data.y < 6)
    invalid_count = torch.logical_not(valid_mask).sum().item()

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


def hybrid_loss(pred, target, class_weights=None, gamma=2.0, label_smoothing=0.1, balance=0.8, alpha=0.3):
    """
    混合损失函数：结合Focal Loss和标签平滑

    参数:
        pred: 模型预测
        target: 真实标签
        class_weights: 类别权重
        gamma: Focal Loss聚焦参数，越大越关注难分样本
        label_smoothing: 标签平滑系数
        balance: Focal Loss比重 (1-balance为标签平滑损失比重)
    """
    # 获取类别数量
    num_classes = pred.size(1)

    # 1. 计算概率和对数概率
    # 对预测logits应用log_softmax得到对数概率
    log_prob = F.log_softmax(pred, dim=1)
    # 计算预测概率（指数变换）
    prob = torch.exp(log_prob)

    # 2. Focal Loss计算
    # 提取每个样本对应目标类别的预测概率
    target_prob = prob.gather(1, target.unsqueeze(1)).squeeze(1)

    # Focal Loss的核心: (1-p_t)^γ使模型关注难分样本
    # 当γ>0时，易分样本(p_t接近1)的权重变小，难分样本(p_t接近0)的权重变大
    focal_weight = alpha * (1 - target_prob) ** gamma
    # focal_weight = (1 - target_prob) ** gamma


    # 应用类别权重以处理类别不平衡问题
    if class_weights is not None:
        # 为每个样本获取对应类别的权重
        per_sample_weights = class_weights.gather(0, target)
        # 将类别权重应用到focal权重上
        focal_weight = focal_weight * per_sample_weights

    # 计算加权的Focal Loss值
    focal_loss_val = -focal_weight * log_prob.gather(1, target.unsqueeze(1)).squeeze(1)

    # 3. 标签平滑损失计算
    # 将目标标签转换为one-hot编码
    target_one_hot = F.one_hot(target, num_classes).float()

    # 应用标签平滑: 主类别概率为(1-α)，其余类别共享α概率
    soft_target = target_one_hot * (1 - label_smoothing) + label_smoothing / num_classes

    # 计算平滑标签的交叉熵损失
    smooth_loss_val = -torch.sum(soft_target * log_prob, dim=1)

    # 4. 加权组合两种损失
    combined_loss = balance * focal_loss_val + (1 - balance) * smooth_loss_val

    # 返回批次平均损失
    return combined_loss.mean()


def emphasis_ignore_loss(model, data, out, similarity_threshold=0.9, max_pairs=100,
                         use_cache=True, feature_cache=None, max_total_pairs=300):
    """
    高效版的强调损失(Lp)和忽略损失(Ln)计算，增加提前停止功能

    参数:
        model: GNN模型
        data: 图数据
        out: 模型输出
        similarity_threshold: 相似度阈值，默认0.9
        max_pairs: 每个类别最大考虑的节点对数量，防止过度计算
        sample_classes: 是否采样部分类别，默认False
        use_cache: 是否使用特征缓存，默认True
        feature_cache: 特征缓存对象，当use_cache=True时需提供
        max_total_pairs: 收集的强调/忽略损失节点对的最大总数，达到后停止收集
    """
    device = data.x.device

    # 0. 缓存特征表示（可选）
    if use_cache and feature_cache is not None:
        model_hash = hash(tuple(p.sum().item() for p in model.parameters()))
        data_hash = hash(str(data.x.shape)) + hash(str(data.edge_index.shape))
        cached_features = feature_cache.get(model_hash, data_hash)

        if cached_features is not None:
            Hi = cached_features
        else:
            Hi = model.get_node_reps(data.x, data.edge_index, data.edge_type)
            feature_cache.put(model_hash, data_hash, Hi)
    else:
        # 1. 得到 GNN 输出的节点表示矩阵 Hi
        Hi = model.get_node_reps(data.x, data.edge_index, data.edge_type)

    # 2. 只处理训练集节点
    mask = data.train_mask
    Hi_train = Hi[mask]
    labels = data.y[mask]
    predicted_labels = out[mask].argmax(dim=1)

    # 3. 提前归一化所有特征向量
    Hi_norm = F.normalize(Hi_train, p=2, dim=1)

    # 4. 计算预测正确的样本
    is_correct = predicted_labels == labels

    # 获取类别数
    num_classes = out.size(1)

    # 初始化损失值和计数器
    Lp_losses = []
    Ln_losses = []

    # 设置提前退出标志
    lp_collection_complete = False
    ln_collection_complete = False

    # 对每个类别计算损失
    for c in range(num_classes):
        # 如果已收集足够的节点对，提前结束循环
        if lp_collection_complete and ln_collection_complete:
            break

        # 获取类别c的正确预测样本
        c_correct_mask = (labels == c) & is_correct
        if c_correct_mask.sum() == 0:
            continue  # 跳过没有正确预测的类别

        c_correct_indices = torch.where(c_correct_mask)[0]

        c_correct_features = Hi_norm[c_correct_indices]

        # === 强调损失(Lp) - 向量化实现 ===
        # 只有在尚未收集足够的强调对时进行收集
        if not lp_collection_complete:
            non_c_incorrect_mask = (labels != c) & (~is_correct)
            if non_c_incorrect_mask.sum() > 0:
                non_c_incorrect_indices = torch.where(non_c_incorrect_mask)[0]
                non_c_incorrect_features = Hi_norm[non_c_incorrect_indices]

                # 批量计算相似度矩阵
                sim_matrix = torch.mm(c_correct_features, non_c_incorrect_features.t())

                # 找出高相似度对
                high_sim_mask = sim_matrix > similarity_threshold

                if high_sim_mask.sum() > 0:
                    # 向量化处理：获取所有高相似度对的索引
                    row_indices, col_indices = torch.where(high_sim_mask)

                    # 按行(正确分类样本)分组
                    unique_rows = torch.unique(row_indices)

                    for row_idx in unique_rows:
                        # 检查是否已达到目标对数量
                        if len(Lp_losses) >= max_total_pairs:
                            lp_collection_complete = True
                            break

                        # 找出当前样本的所有高相似度目标
                        target_cols = col_indices[row_indices == row_idx]

                        # 限制处理的对数
                        if len(target_cols) > max_pairs:
                            target_cols = target_cols[:max_pairs]

                        # 批量计算相似度
                        target_features = non_c_incorrect_features[target_cols].mean(0)
                        source_features = c_correct_features[row_idx]

                        sim = F.cosine_similarity(target_features.unsqueeze(0), source_features.unsqueeze(0))
                        Lp_losses.append(-sim)  # 负相似度

                        # 实时检查是否达到目标
                        if len(Lp_losses) >= max_total_pairs:
                            lp_collection_complete = True
                            break

        # === 忽略损失(Ln) - 向量化实现 ===
        # 只有在尚未收集足够的忽略对时进行收集
        if not ln_collection_complete:
            c_incorrect_mask = (labels == c) & (~is_correct)
            if c_incorrect_mask.sum() > 0:
                c_incorrect_indices = torch.where(c_incorrect_mask)[0]
                c_incorrect_features = Hi_norm[c_incorrect_indices]

                # 批量计算相似度矩阵
                sim_matrix = torch.mm(c_correct_features, c_incorrect_features.t())

                # 找出高相似度对
                high_sim_mask = sim_matrix > similarity_threshold

                if high_sim_mask.sum() > 0:
                    # 向量化处理：获取所有高相似度对的索引
                    row_indices, col_indices = torch.where(high_sim_mask)

                    # 按行(正确分类样本)分组
                    unique_rows = torch.unique(row_indices)

                    for row_idx in unique_rows:
                        # 检查是否已达到目标对数量
                        if len(Ln_losses) >= max_total_pairs:
                            ln_collection_complete = True
                            break

                        # 找出当前样本的所有高相似度目标
                        target_cols = col_indices[row_indices == row_idx]

                        # 限制处理的对数
                        if len(target_cols) > max_pairs:
                            target_cols = target_cols[:max_pairs]

                        # 批量计算相似度
                        target_features = c_incorrect_features[target_cols].mean(0)
                        source_features = c_correct_features[row_idx].detach()  # 注意这里使用detach()

                        sim = F.cosine_similarity(target_features.unsqueeze(0), source_features.unsqueeze(0))
                        Ln_losses.append(sim)  # 正相似度

                        # 实时检查是否达到目标
                        if len(Ln_losses) >= max_total_pairs:
                            ln_collection_complete = True
                            break

    # 计算最终损失
    Lp = torch.tensor(0.0, device=device)
    Ln = torch.tensor(0.0, device=device)

    if Lp_losses:
        Lp = torch.stack(Lp_losses).mean()

    if Ln_losses:
        Ln = torch.stack(Ln_losses).mean()

    # 记录找到的准线节点对数量，便于调试
    num_lp_pairs = len(Lp_losses)
    num_ln_pairs = len(Ln_losses)

    return Lp, Ln, num_lp_pairs, num_ln_pairs

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
    """训练 GNN 模型，使用YAML配置中的参数"""

    # 加载参数配置
    config = load_config()

    # 记录开始时间
    start_time = time.time()

    model_type = config["model"]["type"]
    epoch_num = config["training"]["epoch_num"]
    patience = config["training"]["patience"]
    print_interval = config["training"]["print_interval"]
    enable_early_stopping = config["training"]["enable_early_stopping"]

    data = preprocess_data(load_data())

    # 创建模型
    model = get_model(config, data.num_features, device)

    # 创建优化器
    optimizer = get_optimizer(config, model.parameters())

    # 创建学习率调度器
    scheduler = get_scheduler(config, optimizer, epoch_num)

    # 计算类别数量
    class_counts = torch.zeros(6)
    for label in data.y[data.train_mask]:
        if 0 <= label < 6:
            class_counts[label.item()] += 1

    logger.info(f"各类别样本数量: {class_counts.tolist()}")

    # 计算类别权重
    beta = config["class_weights"]["beta"]
    cb_adjustment = config["class_weights"]["cb_adjustment"]
    class_weights = calculate_class_weights(class_counts, beta, cb_adjustment)
    class_weights = class_weights.to(device)

    logger.info(f"优化后类别权重: {class_weights.tolist()}")

    # 早停参数
    counter = 0
    best_macro_f1 = 0.0
    best_model_path = os.path.join(output_dir, "model")
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)

    # 跟踪指标列表
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_macro_f1s = []
    val_weighted_f1s = []
    epochs = []
    learning_rates = []

    # 设置混合损失参数
    focal_gamma = config["loss"]["focal_gamma"]
    label_smoothing = config["loss"]["label_smoothing"]
    loss_balance = config["loss"]["loss_balance"]

    # 设置Lp和Ln的权重系数
    r_cam_config = config["loss"]["r_cam"]
    enable_r_cam = r_cam_config["enable"]
    lp_weight = r_cam_config["lp_weight"]
    ln_weight = r_cam_config["ln_weight"]
    similarity_threshold = r_cam_config["similarity_threshold"]
    max_pairs = r_cam_config["max_pairs"]
    max_total_pairs = r_cam_config["max_total_pairs"]
    if enable_r_cam:
        logger.info(f"训练已启用强调损失(lp)和忽略损失(ln)计算： Lp权重={lp_weight}, Ln权重={ln_weight}, 相似度阈值={similarity_threshold}, 单点最大对数={max_pairs}, 最大总对数={max_total_pairs}")

    # 梯度裁剪参数
    clip_norm = config["gradient"]["clip_norm"]

    # 可视化参数
    visualize = config["evaluation"]["visualization"]

    for epoch in range(epoch_num):
        model.train()

        # 动态调整focal_gamma - 随训练进程逐渐降低
        # current_gamma = max(1.0, focal_gamma * (1 - epoch / epoch_num))

        # 前向传播
        out = get_model_out(model_type, model, data)

        # 使用混合损失函数计算主损失
        loss = hybrid_loss(
            out[data.train_mask],
            data.y[data.train_mask],
            class_weights=class_weights,
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
            balance=loss_balance,
            # class_counts=class_counts
        )

        # 记录主损失
        main_loss = loss.clone()

        # 计算强调损失和忽略损失
        if enable_r_cam:
            Lp, Ln, num_lp_pairs, num_ln_pairs = emphasis_ignore_loss(
                model, data, out,
                similarity_threshold=similarity_threshold,
                max_pairs=max_pairs,
                max_total_pairs=max_total_pairs
            )

            # 记录找到的准线节点对数量
            if epoch % print_interval == 0:
                logger.info(f"找到的准线节点对: Lp={num_lp_pairs}, Ln={num_ln_pairs}")

            # 动态调整损失
            if num_lp_pairs > 0 and num_ln_pairs > 0:
                # 有效对数足够时才应用损失
                loss = loss + lp_weight * Lp + ln_weight * Ln

        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪避免爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

        optimizer.step()
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        # 评估验证集性能
        model.eval()
        with torch.no_grad():
            out = get_model_out(model_type, model, data)

            # 计算验证损失
            val_loss = hybrid_loss(
                out[data.val_mask],
                data.y[data.val_mask],
                class_weights=class_weights,
                gamma=focal_gamma,
                label_smoothing=label_smoothing,
                balance=loss_balance,
                # class_counts=class_counts
            )

            # 获取预测标签和真实标签
            pred = out.argmax(dim=1)
            y_pred = pred[data.val_mask].cpu().numpy()
            y_true = data.y[data.val_mask].cpu().numpy()

            # 计算验证指标
            correct = np.sum(y_pred == y_true)
            val_acc = correct / len(y_true)
            val_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            val_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            val_macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            val_weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # 记录指标
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        val_accuracies.append(val_acc)
        val_macro_f1s.append(val_macro_f1)
        val_weighted_f1s.append(val_weighted_f1)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        epochs.append(epoch)

        # 更新学习率
        scheduler.step(val_loss.item())

        # 每interval个epoch打印指标
        if epoch % print_interval == 0:
            if enable_r_cam:
                logger.info(
                    f"Epoch {epoch}: 【Loss】 Main: {main_loss.item():.4f}, Lp: {Lp.item():.4f}, Ln: {Ln.item():.4f}, Total: {loss.item():.4f} "
                    f"【Val】 Loss: {val_loss.item():.4f}, Acc: {val_acc:.4f}, Macro-F1: {val_macro_f1:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch}: 【Loss】 Main: {main_loss.item():.4f}, Total: {loss.item():.4f} "
                    f"【Val】 Loss: {val_loss.item():.4f}, Acc: {val_acc:.4f}, Macro-F1: {val_macro_f1:.4f}"
                )

        # 使用F1分数作为早停标准
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            torch.save(model.state_dict(), os.path.join(best_model_path, "best_model.pth"))
            counter = 0
            logger.info(f"✅模型改进! 当前Epoch: {epoch}, 最佳Macro-F1: {best_macro_f1:.4f}, 验证准确率: {val_acc:.4f}")
        else:
            counter += 1

        # 早停检查
        if enable_early_stopping and counter >= patience:
            logger.info(f"⚠️触发早停，停止训练! 当前最佳验证Macro-F1分数: {best_macro_f1:.4f}")
            break

    # 计算并输出总训练时间
    total_time = time.time() - start_time
    formatted_time = time.strftime("%H:%M:%S", time.gmtime(total_time))
    logger.info(f"⏱️训练完成，耗时: {formatted_time}")

    # 可视化训练过程
    if visualize:
        visualize_training_process(epochs, train_losses, val_losses, val_accuracies,
                                   val_precisions, val_recalls, val_macro_f1s, learning_rates)

    return best_model_path


def visualize_training_process(epochs, train_losses, val_losses, val_accuracies,
                               val_precisions, val_recalls, val_f1s, learning_rates):
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

    # 学习率变化曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, learning_rates, linewidth=2.5)
    plt.title('Learning Rate Schedule in Training', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.yscale('log')  # 使用对数刻度以更好地显示变化

    # 保存学习率曲线
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "learning_rate.png"), dpi=300, bbox_inches='tight')
    logger.info(f"学习率变化曲线已保存至 {output_path}/learning_rate.png")

    # 关闭图表以释放内存
    plt.close('all')


def analyze_interventions(model, data, config):
    """分析特征节点(0-41)的因果影响"""
    model.eval()
    intervention_logger = get_logger("Intervention")

    # 指定要分析的特征节点范围(0-41)
    feature_nodes = list(range(42))  # 0-41号节点

    intervention_logger.info(f"执行特征节点(0-41)因果干预分析")

    # 执行初始的前向传播获取隐藏表示
    hidden_features = {}
    handles = []

    def get_input_hook(name):
        def hook(module, inputs, outputs):
            hidden_features[name] = {
                'x': inputs[0].detach(),
                'edge_index': inputs[1],
                'edge_type': inputs[2]
            }

        return hook

    # 为每个ImprovedCausalRGCNConv层注册钩子
    for name, module in model.named_modules():
        if isinstance(module, ImprovedCausalRGCNConv):
            handles.append(module.register_forward_hook(get_input_hook(name)))

    # 执行一次前向传播
    with torch.no_grad():
        model(data.x, data.edge_index, data.edge_type)

    # 移除钩子
    for handle in handles:
        handle.remove()

    # 收集所有因果层的分析结果
    causal_influence = {}
    for name, module in model.named_modules():
        if isinstance(module, ImprovedCausalRGCNConv) and name in hidden_features:
            layer_results = {}

            # 对每个特征节点单独进行干预分析
            for node_id in feature_nodes:
                with torch.no_grad():
                    inputs = hidden_features[name]
                    results = module.do_intervention(
                        inputs['x'], inputs['edge_index'], inputs['edge_type'],
                        target_nodes=[node_id]
                    )

                if 'effect' in results:
                    # 计算该节点的整体因果影响力
                    effect = results['effect']
                    total_effect = torch.abs(effect).sum().item()
                    layer_results[node_id] = total_effect

            # 对节点影响力排序
            sorted_nodes = sorted(layer_results.items(), key=lambda x: x[1], reverse=True)
            top_nodes = sorted_nodes[:10]  # 选择影响最大的10个特征节点

            intervention_logger.info(f"层 {name} 特征节点因果影响排名:")
            for node, effect in top_nodes:
                intervention_logger.info(f" 节点{node}: {effect:.2f}")

            causal_influence[name] = {
                'node_effects': dict(sorted_nodes)
            }

    return causal_influence

def evaluate(model, split, data=None, config=None):
    """在验证集或测试集上评估模型，计算多种性能指标

    Args:
        model: 要评估的模型
        split: 评估集类型，"val"或"test"
        data: 预处理好的数据，如果为None则重新加载
        config: 配置参数，如果为None则使用默认配置
    """
    # 如果没有传入数据，则加载数据
    if data is None:
        data = preprocess_data(load_data())

    # 如果没有传入配置，则加载默认配置
    if config is None:
        config = load_config()

    val_logger = get_logger("Evaluate")
    model.eval()
    results = {}

    with torch.no_grad():
        out = get_model_out(config["model"]["type"], model, data)

        pred = out.argmax(dim=1)

        # 根据数据集类型选择正确的掩码
        if split == "val":
            mask = data.val_mask
            val_logger.info("在验证集上评估模型...")
        elif split == "test":
            mask = data.test_mask
            val_logger.info("在测试集上评估模型...")
        else:
            raise ValueError(f"不支持的数据集类型: {split}")

        # 获取预测结果和真实标签
        mask_indices = torch.where(mask)[0]
        if len(mask_indices) == 0:
            val_logger.warning(f"在{split}集中没有标记为True的样本!")
            results['accuracy'] = 0
            results['precision'] = 0
            results['recall'] = 0
            results['f1-macro'] = 0
            results['f1-weighted'] = 0
            results['kappa'] = 0
            results['confusion_matrix'] = np.array([[0]])
            return results

        y_pred = pred[mask].cpu().numpy()
        y_true = data.y[mask].cpu().numpy()

        # 计算评估指标
        correct = np.sum(y_pred == y_true)
        results['accuracy'] = correct / len(y_true)
        results['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        results['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        results['f1-macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        results['f1-weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        results['kappa'] = cohen_kappa_score(y_true, y_pred)
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred)

    return results


def test(model_path=None, config_path="config.yml"):
    """测试模型

    Args:
        model_path: 模型权重路径，如果为None则使用默认路径
        config_path: 配置文件路径
    """
    # 加载配置
    config = load_config(config_path)
    test_logger = get_logger("Test")

    test_logger.info(f"{'=' * 15} 测试集进行评估 {'=' * 15}")

    # 加载测试数据
    data = preprocess_data(load_data())

    # 使用配置创建模型
    model = get_model(config, data.num_features, device)

    # 确定模型权重路径
    if model_path is None:
        model_path = os.path.join(config["paths"]["model_dir"], "best_model.pth")

    # 加载最佳模型权重
    if os.path.exists(model_path):
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True))
        test_logger.info(f"成功加载模型权重: {model_path}")
    else:
        test_logger.error(f"模型权重文件不存在: {model_path}")
        return None

    # 评估模型
    test_results = evaluate(model, "test", data, config)

    # 添加因果干预分析
    test_logger.info(f"{'=' * 15} 执行因果干预分析 {'=' * 15}")
    analyze_interventions(model, data, config)

    # 输出测试结果
    print_metrics_table(test_results, logger)

    return test_results


def print_metrics_table(results, logger, title="测试集评估结果"):
    """
    以表格形式打印评估指标

    Args:
        results: 包含评估指标的字典
        logger: 日志记录器
        title: 表格标题
    """
    table_header = f"{'=' * 15} {title} {'=' * 15}"
    table_divider = "+-------------+----------+"
    table_format = "| {metric:<11} | {value:>8.6f} |"

    logger.info(table_header)
    logger.info(table_divider)
    logger.info("| Metric      | Value    |")
    logger.info(table_divider)
    logger.info(table_format.format(metric="Accuracy", value=results['accuracy']))
    logger.info(table_format.format(metric="Precision", value=results['precision']))
    logger.info(table_format.format(metric="Recall", value=results['recall']))
    logger.info(table_format.format(metric="Kappa-Score", value=results['kappa']))
    logger.info(table_format.format(metric="Macro-F1", value=results['f1-macro']))
    logger.info(table_format.format(metric="Weighted-F1", value=results['f1-weighted']))
    logger.info(table_divider)

    # 输出混淆矩阵
    conf_matrix = results['confusion_matrix']
    logger.info(f"Confusion Matrix:\n{conf_matrix}")

    # 计算每个类别的预测准确率
    logger.info(f"\n{'=' * 15} 各类别预测准确率 {'=' * 15}")

    # 直接从混淆矩阵计算每个类别的准确率
    for i in range(conf_matrix.shape[0]):
        row_sum = conf_matrix[i].sum()
        if row_sum > 0:  # 避免除零错误
            class_accuracy = conf_matrix[i, i] / row_sum
            logger.info(f"类别 {i}: {class_accuracy * 100:.2f}% ({conf_matrix[i, i]}/{row_sum})")
        else:
            logger.info(f"类别 {i}: 无样本")


if __name__ == "__main__":
    try:
        set_seed(2345)  # 设置随机种子
        best_model_path = train()  # 训练模型
        if best_model_path:
            test()  # 测试模型
        else:
            logger.error("训练失败，跳过测试")
    except Exception as e:
        logger.error(f"训练或测试过程发生错误: {e}")
        import traceback

        logger.error(traceback.format_exc())
