import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
import time


class ModelEvaluator:
    def __init__(self):
        pass

    def calculate_metrics(self, y_true, y_pred, y_pred_proba, running_time):
        """
        计算所有评估指标

        参数:
        y_true: 真实标签
        y_pred: 预测标签 (0/1)
        y_pred_proba: 预测概率
        running_time: 运行时间

        返回:
        包含所有指标的字典
        """
        # 计算混淆矩阵的元素
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))

        # 计算基本指标
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        # 计算FPR和FNR
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        fnr = FN / (FN + TP) if (FN + TP) > 0 else 0

        # 计算AUC
        fpr_curve, tpr_curve, _ = roc_curve(y_true, y_pred_proba)
        auc_score = auc(fpr_curve, tpr_curve)

        # 返回所有指标
        return {
            'accuracy': accuracy,
            'f1': f1,
            'fpr': fpr,
            'fnr': fnr,
            'auc': auc_score,
            'running_time': running_time
        }

    def evaluate(self, model, X_test, y_test):
        """
        评估模型性能

        参数:
        model: 训练好的模型
        X_test: 测试数据
        y_test: 测试标签

        返回:
        包含所有评估指标的字典
        """
        # 测量运行时间
        start_time = time.time()
        y_pred_proba = model.predict(X_test)
        end_time = time.time()
        running_time = (end_time - start_time)  # 每个样本的平均运行时间

        # 将概率转换为类别
        y_pred = (y_pred_proba > 0.5).astype(int)

        # 计算所有指标
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba, running_time)

        return metrics