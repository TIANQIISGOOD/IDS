# evaluator.py 改进后的代码
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, recall_score, \
    f1_score
import matplotlib.pyplot as plt
import numpy as np


class ModelEvaluator:
    @staticmethod
    def evaluate(model, X_test, y_test, model_type='dl'):
        """
        支持评估深度学习模型和传统机器学习模型
        model_type: 'dl'表示深度学习模型，'ml'表示机器学习模型
        """
        # 统一概率预测逻辑
        if model_type == 'ml':
            # 机器学习模型需要二维输入
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            y_prob = model.predict_proba(X_test_flat)[:, 1]
            y_pred_class = model.predict(X_test_flat)
        else:
            # 深度学习模型保持三维输入结构
            if len(X_test.shape) == 2:
                X_test = X_test.reshape(*X_test.shape, 1)  # 添加通道维度
            y_prob = model.predict(X_test).flatten()
            y_pred_class = (y_prob > 0.5).astype(int)

        # 通用评估指标
        print("\n" + "=" * 50)
        print(f"Evaluating {model.__class__.__name__}:")
        print("Classification Report:")
        print(classification_report(y_test, y_pred_class))

        # 计算FPR
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_class).ravel()
        fpr = fp / (fp + tn)
        print(f"FPR: {fpr * 100:.2f}%")

        # 计算AUC-ROC
        print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")

        return {
            'accuracy': accuracy_score(y_test, y_pred_class),
            'recall': recall_score(y_test, y_pred_class),
            'f1': f1_score(y_test, y_pred_class),
            'fpr': fpr,
            'auc': roc_auc_score(y_test, y_prob)
        }

    @staticmethod
    def compare_models(results):
        """可视化对比结果"""
        # 绘制指标对比图
        metrics = ['accuracy', 'recall', 'f1', 'auc']
        plt.figure(figsize=(15, 10))

        for i, metric in enumerate(metrics, 1):
            plt.subplot(2, 2, i)
            plt.bar(results.keys(), [v[metric] for v in results.values()])
            plt.title(metric.upper())
            plt.ylim(0.5, 1.0)

        # 单独绘制FPR对比
        plt.figure(figsize=(8, 4))
        plt.bar(results.keys(), [v['fpr'] for v in results.values()], color='orange')
        plt.axhline(0.01, color='r', linestyle='--', label='1% Threshold')
        plt.title('False Positive Rate Comparison')
        plt.legend()
        plt.show()