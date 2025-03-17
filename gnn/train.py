import os
import time
import torch
from gnn.model import StableGAT
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils.helper import set_seed, get_logger

# 设置日志记录器
logger = get_logger("Train")

output_dir = "../output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    logger.info(f"已创建输出目录: {output_dir}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    valid_mask = (data.y >= 0) & (data.y < 7)
    invalid_count = (~valid_mask).sum().item()

    if invalid_count > 0:
        logger.info(f"发现 {invalid_count} 个非预测节点（标签超出0-6范围）")

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


def train():
    """训练 GCN 模型"""

    # 记录开始时间
    start_time = time.time()

    train_data = preprocess_data(load_data("train"))
    val_data = preprocess_data(load_data("val"))

    model = StableGAT(train_data.num_features, 128, 7).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    # 早停参数
    patience = 100
    counter = 0
    best_val_acc = 0.0
    best_model_path = os.path.join(output_dir, "model")
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
        logger.info(f"已创建模型输出目录: {best_model_path}")

    # 添加用于跟踪指标的列表
    train_losses = []
    val_losses = []  # 新增：记录验证损失
    val_accuracies = []
    epochs = []

    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()

        out = model(train_data.x, train_data.edge_index)

        # NLL损失
        loss = F.nll_loss(out[train_data.train_mask], train_data.y[train_data.train_mask])

        loss.backward()
        optimizer.step()

        # 评估验证集性能
        model.eval()
        with torch.no_grad():
            out = model(val_data.x, val_data.edge_index)

            # 计算验证损失
            val_loss = F.nll_loss(out[val_data.val_mask], val_data.y[val_data.val_mask])

            # 计算验证准确率
            pred = out.argmax(dim=1)
            correct = pred[val_data.val_mask].eq(val_data.y[val_data.val_mask]).sum().item()
            val_acc = correct / val_data.val_mask.sum().item()

        # 记录每个epoch的指标
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        val_accuracies.append(val_acc)
        epochs.append(epoch)

        # 每10个epoch打印一次训练和验证准确率
        if epoch % 10 == 0:
            logger.info(
                f"Epoch {epoch}: Train Loss {loss.item():.4f}, Val Loss {val_loss.item():.4f}, Val Acc {val_acc:.4f}")

        # 记录最佳模型（基于验证准确率）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(best_model_path, "best_model.pth"))
            counter = 0
            logger.info(f"模型改进，当前最佳验证准确率: {best_val_acc:.4f}")
        else:
            counter += 1
            # 每10个epoch打印一次无改进信息
            if counter % 10 == 0:
                logger.info(f"无改进 {counter}/{patience}")

        # 早停检查
        if counter >= patience:
            logger.info(f"早停触发，停止训练! 最佳验证准确率: {best_val_acc:.6f}")
            break

    # 计算并输出总训练时间
    total_time = time.time() - start_time
    seconds = int(total_time % 60)
    milliseconds = int((total_time - int(total_time)) * 1000)
    formatted_time = f"{seconds:02d}\"{milliseconds:02d}"
    logger.info(f"训练完成，总耗时: {formatted_time}")

    # 添加可视化代码 - 更新函数参数
    visualize_training_process(epochs, train_losses, val_losses, val_accuracies)

    return best_model_path


def visualize_training_process(epochs, train_losses, val_losses, val_accuracies):
    """可视化训练过程中的损失和准确率变化"""

    # 创建输出目录
    output_path = os.path.join(output_dir, "figures")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        logger.info(f"已创建图片输出目录: {output_path}")

    # 创建图像 - 使用2个子图而不是3个
    plt.figure(figsize=(12, 5))

    # 绘制训练和验证损失子图（合并在一起）
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2.5)
    plt.plot(epochs, val_losses, 'r-', label='Validate Loss', linewidth=2.5)
    plt.title('Loss Curves', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    # 绘制验证准确率子图
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, 'g-', linewidth=2.5)
    plt.title('Validate Accuracy', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 调整布局并保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "training.png"), dpi=300, bbox_inches='tight')
    logger.info(f"训练过程可视化已保存至 {output_path}/training.png")

    # 关闭图表以释放内存
    plt.close('all')

def evaluate(model, split):
    """在验证集或测试集上评估模型"""
    data = preprocess_data(load_data(split))
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

        # 根据数据集类型选择正确的掩码
        if split == "val":
            mask = data.val_mask
        elif split == "test":
            mask = data.test_mask
        else:
            raise ValueError(f"不支持的数据集类型: {split}")

        correct = pred[mask].eq(data.y[mask]).sum().item()
        acc = correct / mask.sum().item()
    return acc


def test():
    """测试模型"""
    test_data = preprocess_data(load_data("test"))
    model = StableGAT(test_data.x.shape[1], 128, 7).to(device)
    model.load_state_dict(torch.load(os.path.join(output_dir, "model/best_model.pth"), weights_only=True))
    test_acc = evaluate(model, "test")
    logger.info(f"测试集准确率: {test_acc:.6f}")
    return test_acc


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
