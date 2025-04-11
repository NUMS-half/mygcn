import os
import yaml
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_config(config_path="../gnn/config.yml"):
    """Load configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_metrics_data(filepath):
    """Load model performance metrics data"""
    if not os.path.exists(filepath):
        print(f"Error: File not found {filepath}")
        return None

    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def get_model_types(output_dir):
    """Get all trained model types"""
    possible_models = ["GCN", "GraphSAGE", "GAT", "RGCN", "CausalRGCN"]
    existing_models = []

    for model_type in possible_models:
        model_dir = os.path.join(output_dir, f"model_{model_type}")
        metrics_file = os.path.join(model_dir, "all_metrics_data.pkl")
        if os.path.exists(metrics_file):
            existing_models.append(model_type)

    return existing_models


def create_performance_table(models_data):
    """Create performance comparison table for different models"""
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
    """Create performance comparison chart for different models"""
    metrics = ['accuracy', 'f1-macro', 'macro-auc-roc']

    # Use professional color palette
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
    print(f"Performance comparison chart saved to: {output_path}")


def plot_convergence_comparison(models_data, output_dir):
    """Create convergence speed comparison chart for different models"""
    # Use professional color palette
    colors = sns.color_palette("colorblind", len(models_data))

    # Training loss curve
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    i = 0
    for model_type, data in models_data.items():
        if 'epoch' in data and 'train_loss' in data:
            epochs = data['epoch'][:400] # Limit to first 400 epochs for better visualization

            # Calculate average training loss
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
                i += 1

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Training Loss', fontsize=12)
    plt.title('Convergence Effect Comparison Across Models', fontsize=14)
    plt.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'model_convergence_comparison.png')
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Convergence speed comparison chart saved to: {output_path}")

    # Validation metrics curves
    validation_metrics = ['val_accuracy', 'val_f1_macro', 'val_macro-auc-roc']
    metric_display_names = {
        'val_accuracy': 'Accuracy',
        'val_f1_macro': 'F1-Macro',
        'val_macro-auc-roc': 'Macro AUC-ROC'
    }

    for metric in validation_metrics:
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")

        i = 0
        for model_type, data in models_data.items():
            if 'epoch' in data and metric in data:
                epochs = data['epoch']

                # Calculate average validation metrics
                mean_values = np.zeros(len(epochs))
                count = 0

                for values in data[metric]:
                    if len(values) >= len(epochs):
                        mean_values += np.array(values[:len(epochs)])
                        count += 1

                if count > 0:
                    mean_values /= count
                    plt.plot(epochs, mean_values, label=f"{model_type}",
                             linewidth=2, color=colors[i])
                    i += 1

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(f'Average {metric_display_names[metric]}', fontsize=12)
        plt.title(f'{metric_display_names[metric]} Progression Across Models', fontsize=14)
        plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        output_path = os.path.join(output_dir, f'model_{metric}_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{metric_display_names[metric]} comparison chart saved to: {output_path}")


def compare_models():
    """Main function: Compare performance of different models"""
    config = load_config()
    output_dir = config["paths"]["output_dir"]

    model_types = get_model_types(output_dir)
    print(f"Found following model types for comparison: {model_types}")

    # Create visualization output directory
    vis_output_dir = os.path.join(output_dir, "cmp_experiment")
    os.makedirs(vis_output_dir, exist_ok=True)

    # Load performance metrics data for each model
    models_data = {}
    for model_type in model_types:
        metrics_file = os.path.join(output_dir, f"model_{model_type}", "all_metrics_data.pkl")
        data = load_metrics_data(metrics_file)
        if data:
            models_data[model_type] = data

    if not models_data:
        print("No valid model performance data found, cannot proceed with comparison")
        return

    # Create performance comparison table
    comparison_table = create_performance_table(models_data)
    print("\nModel Performance Comparison:")
    print(comparison_table.to_string(index=False))

    # Save table to CSV file
    table_path = os.path.join(vis_output_dir, "model_performance_comparison.csv")
    comparison_table.to_csv(table_path, index=False)
    print(f"Performance comparison table saved to: {table_path}")

    # Create performance comparison charts
    plot_performance_comparison(models_data, vis_output_dir)

    # Create convergence speed comparison charts
    plot_convergence_comparison(models_data, vis_output_dir)

    print(f"All comparison charts saved to: {vis_output_dir}")


if __name__ == "__main__":
    compare_models()