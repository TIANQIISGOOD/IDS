from data_loader import DataLoader
from gru import train_improved_gru
from model import BiLSTM_CNN
from trainer import ModelTrainer
from evaluator import ModelEvaluator
from bilstm import BiLSTM, train_bilstm
from cnn import CNN, train_cnn
from params import config
from ablation_models import SpatialOnlyModel, TemporalOnlyModel

import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.utils import to_categorical


def measure_detection_time(model, X_test):
    start_time = time.time()
    _ = model.predict(X_test)
    end_time = time.time()
    return (end_time - start_time) / len(X_test)


def print_model_results(name, metrics):
    print(f"\n=== {name} Results ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    print(f"FPR: {metrics['fpr']:.4f}")
    print(f"FNR: {metrics['fnr']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Running Time: {metrics['running_time']:.4f}s")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])


def train_models(X_train_dl, y_train, X_val_dl, y_val):
    trained_models = {}

    print("\nTraining CNN-BiLSTM model...")
    model_cnn_bilstm = BiLSTM_CNN()
    trainer = ModelTrainer(model_cnn_bilstm)
    history_cnn_bilstm = trainer.train(X_train_dl, y_train, X_val_dl, y_val)
    trained_models['BiLSTM-CNN'] = model_cnn_bilstm

    print("\nTraining CNN model...")
    cnn_model, history_cnn = train_cnn(X_train_dl, y_train, X_val_dl, y_val)
    trained_models['CNN'] = cnn_model

    print("\nTraining BiLSTM model...")
    bilstm_model, history_bilstm = train_bilstm(X_train_dl, y_train, X_val_dl, y_val)
    trained_models['BiLSTM'] = bilstm_model

    print("\nTraining GRU model...")
    gru_model, history_gru = train_improved_gru(X_train_dl, y_train, X_val_dl, y_val)
    trained_models['GRU'] = gru_model

    print("\nTraining Spatial-Only model...")
    spatial_model = SpatialOnlyModel()
    spatial_trainer = ModelTrainer(spatial_model)
    history_spatial = spatial_trainer.train(X_train_dl, y_train, X_val_dl, y_val)
    trained_models['Spatial-Only'] = spatial_model

    print("\nTraining Temporal-Only model...")
    temporal_model = TemporalOnlyModel()
    temporal_trainer = ModelTrainer(temporal_model)
    history_temporal = temporal_trainer.train(X_train_dl, y_train, X_val_dl, y_val)
    trained_models['Temporal-Only'] = temporal_model

    return trained_models


def evaluate_models(models, X_test_dl, y_test, evaluator):
    results = {}
    for name, model in models.items():
        results[name] = evaluator.evaluate(model, X_test_dl, y_test)
        results[name]['detection_time'] = measure_detection_time(model, X_test_dl)
    return results


def print_comparison_table(results):
    print("\n=== Model Performance Comparison ===")
    print("{:<15} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8}".format(
        "Model", "Accuracy", "F1", "FPR", "FNR", "AUC", "Time(s)"))
    print("-" * 80)
    for name, metrics in results.items():
        print("{:<15} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f}".format(
            name,
            metrics['accuracy'],
            metrics['f1'],
            metrics['fpr'],
            metrics['fnr'],
            metrics['auc'],
            metrics['running_time']
        ))


if __name__ == "__main__":
    # 记录执行信息
    current_time = "2025-04-16 14:22:38"
    current_user = "TIANQIISGOOD"
    print(f"Execution Time (UTC): {current_time}")
    print(f"User: {current_user}")
    print("=" * 50)

    try:
        # 数据加载
        print("\nLoading and preprocessing data...")
        loader = DataLoader("gas_final.csv")
        X_train_full, X_test, y_train_full, y_test = loader.load_data()

        # 数据分割
        print("Splitting data into train and validation sets...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
        )

        # 数据形状处理
        X_train_dl = X_train.reshape(-1, config.FEATURE_DIM, 1)
        X_val_dl = X_val.reshape(-1, config.FEATURE_DIM, 1)
        X_test_dl = X_test.reshape(-1, config.FEATURE_DIM, 1)

        print(f"Training data shape: {X_train_dl.shape}")
        print(f"Validation data shape: {X_val_dl.shape}")
        print(f"Test data shape: {X_test_dl.shape}")
        print(f"Training labels shape: {y_train.shape}")

        # 训练所有模型
        trained_models = train_models(X_train_dl, y_train, X_val_dl, y_val)
        print("\nAll models have been trained successfully!")

        # 询问是否进行测试集评估
        while True:
            response = input("\nWould you like to evaluate the models on the test set? (yes/no): ").lower()
            if response in ['yes', 'no']:
                break
            print("Please enter 'yes' or 'no'")

        if response == 'yes':
            # 初始化评估器
            evaluator = ModelEvaluator()

            # 评估所有模型
            print("\nEvaluating models on test set...")
            results = evaluate_models(trained_models, X_test_dl, y_test, evaluator)

            # 打印详细的分类报告
            print("\n=== 详细模型性能报告 ===")
            for model_name, metrics in results.items():
                print_model_results(model_name, metrics)

            # 打印比较表
            print_comparison_table(results)

        # 添加执行信息到结果底部
        print("\n" + "=" * 50)
        print(f"Results generated at: {current_time} UTC")
        print(f"Generated by: {current_user}")

    except Exception as e:
        print(f"训练或评估过程中发生错误: {str(e)}")
        print(f"Error occurred at: {current_time} UTC")
        print(f"User: {current_user}")