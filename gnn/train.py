import torch
import numpy as np
from gnn.gcn import GCN
import torch.nn.functional as F
from utils.logger import get_logger

# 设置日志记录器
logger = get_logger("Train")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 固定随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(split):
    """加载指定分割的数据"""
    try:
        path = f"../data/processed/{split}/knowledge_graph.pt"
        logger.info(f"正在从 {path} 加载 {split} 数据")
        return torch.load(path)
    except Exception as e:
        logger.error(f"加载 {split} 数据时出错: {e}")
        raise


def preprocess_data(data):
    """预处理数据并记录统计信息"""
    data = data.to(device)

    # 确保 y 值在 0-6 之间
    valid_mask = (data.y >= 0) & (data.y < 7)
    invalid_count = (~valid_mask).sum().item()

    if invalid_count > 0:
        logger.warning(f"发现 {invalid_count} 个超出0-6范围的标签，设为类别0")
        data.y[~valid_mask] = 0

    logger.info(f"总节点数: {data.num_nodes}, 有标签节点数: {valid_mask.sum().item()}")

    # 检查数据是否包含掩码
    if not hasattr(data, "train_mask") or not hasattr(data, "val_mask") or not hasattr(data, "test_mask"):
        logger.warning("数据缺少训练/验证/测试掩码，请先运行数据预处理脚本生成")

    return data


def train():
    """训练 GCN 模型"""
    train_data = preprocess_data(load_data("train"))
    val_data = preprocess_data(load_data("val"))

    model = GCN(train_data.num_features, 128, 7).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                           factor=0.5, patience=10,
                                                           min_lr=1e-6)

    # 早停参数
    patience = 50
    counter = 0
    best_val_acc = 0.0
    best_model_path = "best_model.pth"

    for epoch in range(500):
        model.train()
        optimizer.zero_grad()

        out = model(train_data.x, train_data.edge_index)
        loss = F.nll_loss(out[train_data.train_mask], train_data.y[train_data.train_mask])
        loss.backward()
        optimizer.step()

        # 评估验证集性能
        model.eval()
        with torch.no_grad():
            out = model(val_data.x, val_data.edge_index)
            pred = out.argmax(dim=1)
            correct = pred[val_data.val_mask].eq(val_data.y[val_data.val_mask]).sum().item()
            val_acc = correct / val_data.val_mask.sum().item()

        logger.info(f"Epoch {epoch}: Loss {loss.item():.8f}, Val Acc {val_acc:.8f}")

        # 学习率调度器更新
        scheduler.step(val_acc)



        # 记录最佳模型（基于验证准确率）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            counter = 0
            logger.info(f"模型改进，当前最佳验证准确率: {best_val_acc:.8f}")
        else:
            counter += 1
            logger.info(f"无改进 {counter}/{patience}")

        # 早停检查
        if counter >= patience:
            logger.info(f"早停触发，停止训练! 最佳验证准确率: {best_val_acc:.8f}")
            break

    return best_model_path


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
    model = GCN(test_data.x.shape[1], 128, 7).to(device)
    model.load_state_dict(torch.load("best_model.pth"))
    test_acc = evaluate(model, "test")
    logger.info(f"测试集准确率: {test_acc:.8f}")
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
