# main.py 修复后
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


def main():
    # 设置随机种子
    tf.random.set_seed(42)
    np.random.seed(42)

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
        results['CNN-BiLSTM']['detection_time'] = measure_detection_time(model_cnn_bilstm, X_test_dl)
        print("{:<25} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8}".format(
            "模型", "Accuracy", "Precision", "Recall", "F1", "AUC", "检测时间(ms)"
        ))
        print("-" * 90)
        model_name1='CNN-BiLSTM'
        metrics1 = results[model_name1]
        print("{:<25} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f}".format(
            model_name1,
            metrics1['accuracy'],
            metrics1['precision'],
            metrics1['recall'],
            metrics1['f1'],
            metrics1['auc'],
            metrics1['detection_time'] * 1000  # 转换为毫秒
        ))

        # CNN
        print("\nTraining CNN model...")
        cnn_model, histories['CNN'] = train_cnn(X_train_dl, y_train, X_val_dl, y_val)
        results['CNN'] = evaluator.evaluate(cnn_model, X_test_dl, y_test)
        results['CNN']['detection_time'] = measure_detection_time(cnn_model, X_test_dl)
        print("{:<25} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8}".format(
            "模型", "Accuracy", "Precision", "Recall", "F1", "AUC", "检测时间(ms)"
        ))
        print("-" * 90)
        model_name2='CNN'
        metrics2 = results[model_name2]
        print("{:<25} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f}".format(
            model_name2,
            metrics2['accuracy'],
            metrics2['precision'],
            metrics2['recall'],
            metrics2['f1'],
            metrics2['auc'],
            metrics2['detection_time'] * 1000  # 转换为毫秒
        ))

        # BiLSTM
        print("\nTraining BiLSTM model...")
        bilstm_model, histories['BiLSTM'] = train_bilstm(X_train_dl, y_train, X_val_dl, y_val)
        results['BiLSTM'] = evaluator.evaluate(bilstm_model, X_test_dl, y_test)
        results['BiLSTM']['detection_time'] = measure_detection_time(bilstm_model, X_test_dl)
        print("{:<25} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8}".format(
            "模型", "Accuracy", "Precision", "Recall", "F1", "AUC", "检测时间(ms)"
        ))
        print("-" * 90)
        model_name3='BiLSTM'
        metrics3 = results[model_name3]
        print("{:<25} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f}".format(
            model_name3,
            metrics3['accuracy'],
            metrics3['precision'],
            metrics3['recall'],
            metrics3['f1'],
            metrics3['auc'],
            metrics3['detection_time'] * 1000  # 转换为毫秒
        ))

        # GRU
        print("\nTraining GRU model...")
        gru_model, histories['GRU'] = train_improved_gru(X_train_dl, y_train, X_val_dl, y_val)
        results['GRU'] = evaluator.evaluate(gru_model, X_test_dl, y_test)
        results['GRU']['detection_time'] = measure_detection_time(gru_model, X_test_dl)
        print("{:<25} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8}".format(
            "模型", "Accuracy", "Precision", "Recall", "F1", "AUC", "检测时间(ms)"
        ))
        print("-" * 90)
        model_name4='GRU'
        metrics4 = results[model_name4]
        print("{:<25} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f}".format(
            model_name4,
            metrics4['accuracy'],
            metrics4['precision'],
            metrics4['recall'],
            metrics4['f1'],
            metrics4['auc'],
            metrics4['detection_time'] * 1000  # 转换为毫秒
        ))

        # 2. 消融实验
        print("\n=== 训练消融实验模型 ===")

        # 仅空间特征模型
        print("\nTraining Spatial-Only Model...")
        spatial_model = SpatialOnlyModel()
        trainer_spatial = ModelTrainer(spatial_model)
        histories['Spatial-Only'] = trainer_spatial.train(X_train, y_train, X_test, y_test)
        results['Spatial-Only'] = evaluator.evaluate(spatial_model, X_test, y_test)
        results['Spatial-Only']['detection_time'] = measure_detection_time(spatial_model, X_test)
        print("{:<25} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8}".format(
            "模型", "Accuracy", "Precision", "Recall", "F1", "AUC", "检测时间(ms)"
        ))
        print("-" * 90)
        model_name5='Spatial-Only'
        metrics5 = results[model_name5]
        print("{:<25} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f}".format(
            model_name5,
            metrics5['accuracy'],
            metrics5['precision'],
            metrics5['recall'],
            metrics5['f1'],
            metrics5['auc'],
            metrics5['detection_time'] * 1000  # 转换为毫秒
        ))

        # 仅时间特征模型
        print("\nTraining Temporal-Only Model...")
        temporal_model = TemporalOnlyModel()
        trainer_temporal = ModelTrainer(temporal_model)
        histories['Temporal-Only'] = trainer_temporal.train(X_train, y_train, X_test, y_test)
        results['Temporal-Only'] = evaluator.evaluate(temporal_model, X_test, y_test)
        results['Temporal-Only']['detection_time'] = measure_detection_time(temporal_model, X_test)
        print("{:<25} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8}".format(
            "模型", "Accuracy", "Precision", "Recall", "F1", "AUC", "检测时间(ms)"
        ))
        print("-" * 90)
        model_name6='Temporal-Only'
        metrics6 = results[model_name6]
        print("{:<25} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f}".format(
            model_name6,
            metrics6['accuracy'],
            metrics6['precision'],
            metrics6['recall'],
            metrics6['f1'],
            metrics6['auc'],
            metrics6['detection_time'] * 1000  # 转换为毫秒
        ))

        # 3. 结果展示
        print("\n=== 模型性能综合对比表 ===")
        print("{:<25} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8}".format(
            "模型", "Accuracy", "Precision", "Recall", "F1", "AUC", "检测时间(ms)"
        ))
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
            print("{:<25} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f}".format(
                model_name,
                metrics['accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['f1'],
                metrics['auc'],
                metrics['detection_time'] * 1000  # 转换为毫秒
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
        # 5. 性能提升分析
        print("\n=== 性能提升分析 ===")
        cnn_bilstm_acc = results['CNN-BiLSTM']['accuracy']

        # 计算相对于每个其他模型的提升
        other_models = [
            'Spatial-Only',  # 消融实验1
            'Temporal-Only',  # 消融实验2
            'CNN',  # 基准模型1
            'BiLSTM',  # 基准模型2
            'GRU'  # 基准模型3
        ]

    except Exception as e:
        import traceback
        print(f"实验过程中发生错误: {str(e)}")
        print("\n详细错误信息:")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()

