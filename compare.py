import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.helper import load_config


def load_metrics_data(filepath):
    """加载模型性能指标数据"""
    if not os.path.exists(filepath):
        print(f"错误：找不到文件 {filepath}")
        return None

    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def get_model_types(output_dir):
    """获取所有已训练的模型类型"""
    possible_models = ["GCN", "GraphSAGE", "GAT", "RGCN", "CausalRGCN"]
    existing_models = []

    for model_type in possible_models:
        model_dir = os.path.join(output_dir, f"model_{model_type}")
        metrics_file = os.path.join(model_dir, "all_metrics_data.pkl")
        if os.path.exists(metrics_file):
            existing_models.append(model_type)

    return existing_models


def create_performance_table(models_data):
    """为不同模型创建性能比较表"""
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1-macro', 'macro-auc-roc', 'kappa']

    table_data = []

    for model_type, data in models_data.items():
        model_metrics = {'Model': model_type}

        for metric in metrics_to_compare:
            if metric in data:
                values = data[metric]
                mean_value = np.mean(values) * 100
                std_value = np.std(values) * 100
                model_metrics[f'{metric}'] = f"{mean_value:.2f}±{std_value:.2f}"

        table_data.append(model_metrics)

    return pd.DataFrame(table_data)


def plot_performance_comparison(models_data, output_dir):
    """为不同模型创建性能比较图表"""
    metrics = ['accuracy', 'f1-macro', 'macro-auc-roc']

    # 使用专业配色方案
    colors = sns.color_palette("colorblind", len(metrics))

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    model_names = list(models_data.keys())
    x_pos = np.arange(len(model_names))
    width = 0.2

    for i, metric in enumerate(metrics):
        means = []
        std_devs = []

        for model in model_names:
            if metric in models_data[model] and len(models_data[model][metric]) > 0:
                means.append(np.mean(models_data[model][metric]))
                std_devs.append(np.std(models_data[model][metric]))
            else:
                means.append(0)
                std_devs.append(0)

        offset = width * (i - len(metrics) / 2 + 0.5)
        plt.bar(x_pos + offset, means, width, yerr=std_devs, label=metric.upper(),
                color=colors[i], capsize=4, alpha=0.8, edgecolor='black', linewidth=0.5)

    plt.xlabel('Model Type', fontsize=12)
    plt.ylabel('Average Performance', fontsize=12)
    plt.title('Performance Comparison Across Different Models', fontsize=14)
    plt.xticks(x_pos, model_names, rotation=30)
    plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    output_path = os.path.join(output_dir, 'model_performance_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"性能比较图表已保存至：{output_path}")


def plot_convergence_comparison(models_data, output_dir):
    """为不同模型创建收敛速度比较图表"""
    # 使用专业配色方案
    colors = sns.color_palette("colorblind", len(models_data))

    # 训练损失曲线
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    i = 0
    has_plot = False
    for model_type, data in models_data.items():
        if 'epoch' in data and 'train_loss' in data:
            epochs = data['epoch'][:400]  # 限制为前400个epoch以便更好地可视化

            # 计算平均训练损失
            mean_losses = np.zeros(len(epochs))
            count = 0

            for losses in data['train_loss']:
                if len(losses) >= len(epochs):
                    mean_losses += np.array(losses[:len(epochs)])
                    count += 1

            if count > 0:
                mean_losses /= count
                plt.plot(epochs, mean_losses, label=f"{model_type}",
                         linewidth=3, color=colors[i])
                has_plot = True
                i += 1

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Training Loss', fontsize=12)
    plt.title('Convergence Effect Comparison Across Models', fontsize=14)

    if has_plot:
        plt.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'model_convergence_comparison.png')
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"收敛速度比较图表已保存至：{output_path}")


def compare_models():
    """主函数：比较不同模型的性能"""
    config = load_config()
    output_dir = config["paths"]["output_dir"]

    model_types = get_model_types(output_dir)
    print(f"找到以下模型类型进行比较：{model_types}")

    # 创建可视化输出目录
    vis_output_dir = os.path.join(output_dir, "cmp_experiment")
    os.makedirs(vis_output_dir, exist_ok=True)

    # 加载每个模型的性能指标数据
    models_data = {}
    for model_type in model_types:
        metrics_file = os.path.join(output_dir, f"model_{model_type}", "all_metrics_data.pkl")
        data = load_metrics_data(metrics_file)
        if data:
            models_data[model_type] = data

    if not models_data:
        print("未找到有效的模型性能数据，无法进行比较")
        return

    # 创建性能比较表
    comparison_table = create_performance_table(models_data)
    print("模型性能比较：")
    print(comparison_table.to_string(index=False))

    # 保存表格到CSV文件
    table_path = os.path.join(vis_output_dir, "model_performance_comparison.csv")
    comparison_table.to_csv(table_path, index=False)
    print(f"性能比较表已保存至：{table_path}")

    # 创建性能比较图表
    plot_performance_comparison(models_data, vis_output_dir)

    # 创建收敛速度比较图表
    plot_convergence_comparison(models_data, vis_output_dir)

    print(f"所有比较图表已保存至：{vis_output_dir}")


if __name__ == "__main__":
    compare_models()