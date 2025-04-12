import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class ModelEvaluator:
    def evaluate(self, model, X_test, y_test):
        """评估模型性能"""
        # 获取预测结果
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # 处理预测结果维度
        if len(y_pred.shape) > 1:
            y_pred = y_pred.reshape(-1)
            y_pred_proba = y_pred_proba.reshape(-1)

        # 计算各项指标
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=1),
            'recall': recall_score(y_test, y_pred, zero_division=1),
            'f1': f1_score(y_test, y_pred, zero_division=1),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }

        return metrics