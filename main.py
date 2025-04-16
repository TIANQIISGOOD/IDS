import numpy as np
from matplotlib import pyplot as plt

from src.data_loader import DataLoader
from src.gru import train_improved_gru
from src.model import BiLSTM_CNN
from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator
from src.bilstm import BiLSTM, train_bilstm
from src.cnn import CNN, train_cnn
from src.rf import RF, train_rf
from configs.params import config
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from datetime import datetime

from src.ablation_models import SpatialOnlyModel, TemporalOnlyModel


def measure_detection_time(model, X_test):
    """测量模型的检测时间"""
    start_time = time.time()
    _ = model.predict(X_test)
    end_time = time.time()
    return (end_time - start_time) / len(X_test)


def print_metrics(model_name, metrics):
    """打印模型评估指标"""
    print("{:<25} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<12}".format(
        "模型", "Accuracy", "F1", "FPR", "FNR", "AUC", "Running Time/s"))
    print("-" * 90)
    print("{:<25} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<12.6f}".format(
        model_name,
        metrics['accuracy'],
        metrics['f1'],
        metrics['fpr'],
        metrics['fnr'],
        metrics['auc'],
        metrics['running_time']
    ))


def main():
    # 设置随机种子
    tf.random.set_seed(42)
    np.random.seed(42)

    # 记录执行信息
    current_time = "2025-04-15 06:18:38"
    current_user = "TIANQIISGOOD"
    print(f"Execution Time (UTC): {current_time}")
    print(f"User: {current_user}")
    print("=" * 50)

    # 初始化组件
    loader = DataLoader("data/OPCUA_dataset_public.csv")
    evaluator = ModelEvaluator()

    try:
        # 加载数据
        X_train_full, X_test, y_train_full, y_test = loader.load_data()
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42
        )

        # 调整数据形状
        X_train_dl = X_train.reshape(-1, config.FEATURE_DIM, 1)
        X_val_dl = X_val.reshape(-1, config.FEATURE_DIM, 1)
        X_test_dl = X_test.reshape(-1, config.FEATURE_DIM, 1)

        # 存储结果和训练历史
        results = {}
        histories = {}

        # 1. 训练和评估基准模型
        print("\n=== 训练基准模型 ===")

        # CNN-BiLSTM（完整模型）
        print("\nTraining CNN-BiLSTM model...")
        model_cnn_bilstm = BiLSTM_CNN()
        trainer = ModelTrainer(model_cnn_bilstm)
        histories['CNN-BiLSTM'] = trainer.train(X_train_dl, y_train, X_val_dl, y_val)
        results['CNN-BiLSTM'] = evaluator.evaluate(model_cnn_bilstm, X_test_dl, y_test)
        print_metrics('CNN-BiLSTM', results['CNN-BiLSTM'])

        # CNN
        print("\nTraining CNN model...")
        cnn_model, histories['CNN'] = train_cnn(X_train_dl, y_train, X_val_dl, y_val)
        results['CNN'] = evaluator.evaluate(cnn_model, X_test_dl, y_test)
        print_metrics('CNN', results['CNN'])

        # BiLSTM
        print("\nTraining BiLSTM model...")
        bilstm_model, histories['BiLSTM'] = train_bilstm(X_train_dl, y_train, X_val_dl, y_val)
        results['BiLSTM'] = evaluator.evaluate(bilstm_model, X_test_dl, y_test)
        print_metrics('BiLSTM', results['BiLSTM'])

        # GRU
        print("\nTraining GRU model...")
        gru_model, histories['GRU'] = train_improved_gru(X_train_dl, y_train, X_val_dl, y_val)
        results['GRU'] = evaluator.evaluate(gru_model, X_test_dl, y_test)
        print_metrics('GRU', results['GRU'])

        # 2. 消融实验
        print("\n=== 训练消融实验模型 ===")

        # 仅空间特征模型
        print("\nTraining Spatial-Only Model...")
        spatial_model = SpatialOnlyModel()
        trainer_spatial = ModelTrainer(spatial_model)
        histories['Spatial-Only'] = trainer_spatial.train(X_train_dl, y_train, X_val_dl, y_val)
        results['Spatial-Only'] = evaluator.evaluate(spatial_model, X_test_dl, y_test)
        print_metrics('Spatial-Only', results['Spatial-Only'])

        # 仅时间特征模型
        print("\nTraining Temporal-Only Model...")
        temporal_model = TemporalOnlyModel()
        trainer_temporal = ModelTrainer(temporal_model)
        histories['Temporal-Only'] = trainer_temporal.train(X_train_dl, y_train, X_val_dl, y_val)
        results['Temporal-Only'] = evaluator.evaluate(temporal_model, X_test_dl, y_test)
        print_metrics('Temporal-Only', results['Temporal-Only'])

        # 3. 结果综合展示
        print("\n=== 模型性能综合对比表 ===")
        print("{:<25} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<12}".format(
            "模型", "Accuracy", "F1", "FPR", "FNR", "AUC", "Running Time/s"))
        print("-" * 90)

        # 按照特定顺序展示结果
        model_order = [
            'CNN-BiLSTM',  # 完整模型
            'Spatial-Only',  # 仅空间特征（消融实验1）
            'Temporal-Only',  # 仅时间特征（消融实验2）
            'CNN',  # 基准模型1
            'BiLSTM',  # 基准模型2
            'GRU'  # 基准模型3
        ]

        for model_name in model_order:
            metrics = results[model_name]
            print("{:<25} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<12.6f}".format(
                model_name,
                metrics['accuracy'],
                metrics['f1'],
                metrics['fpr'],
                metrics['fnr'],
                metrics['auc'],
                metrics['running_time']
            ))

        # 4. 可视化训练历史
        plt.figure(figsize=(15, 5))

        # 训练准确率对比
        plt.subplot(1, 2, 1)
        for model_name, history in histories.items():
            plt.plot(history.history['accuracy'], label=model_name)
        plt.title('Model Accuracy Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        # 训练损失值对比
        plt.subplot(1, 2, 2)
        for model_name, history in histories.items():
            plt.plot(history.history['loss'], label=model_name)
        plt.title('Model Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('model_comparison_results.png')
        plt.close()

        # 5. 性能提升分析
        print("\n=== 性能提升分析 ===")
        base_metrics = results['CNN-BiLSTM']

        # 计算相对于其他模型的提升
        other_models = [
            'Spatial-Only',
            'Temporal-Only',
            'CNN',
            'BiLSTM',
            'GRU'
        ]

        for model_name in other_models:
            comp_metrics = results[model_name]
            print(f"\n相比于{model_name}模型的性能提升:")
            print(f"Accuracy: {(base_metrics['accuracy'] - comp_metrics['accuracy']) * 100:.2f}%")
            print(f"F1: {(base_metrics['f1'] - comp_metrics['f1']) * 100:.2f}%")
            print(f"FPR减少: {(comp_metrics['fpr'] - base_metrics['fpr']) * 100:.2f}%")
            print(f"FNR减少: {(comp_metrics['fnr'] - base_metrics['fnr']) * 100:.2f}%")
            print(f"AUC: {(base_metrics['auc'] - comp_metrics['auc']) * 100:.2f}%")
            print(
                f"检测时间提升: {((comp_metrics['running_time'] - base_metrics['running_time']) / comp_metrics['running_time']) * 100:.2f}%")

    except Exception as e:
        import traceback
        print(f"实验过程中发生错误: {str(e)}")
        print("\n详细错误信息:")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()