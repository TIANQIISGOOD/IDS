from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np
import time
from params import config


class ModelEvaluator:
    def evaluate(self, model, X_test, y_test):
        # 计算运行时间
        start_time = time.time()
        y_pred = model.predict(X_test)
        end_time = time.time()
        running_time = end_time - start_time

        # 转换为类别
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        # 计算混淆矩阵
        conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

        # 计算FPR和FNR
        fp = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
        fn = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
        tp = np.diag(conf_matrix)
        tn = conf_matrix.sum() - (fp + fn + tp)

        # 计算总体FPR和FNR
        fpr = np.sum(fp) / (np.sum(fp) + np.sum(tn))
        fnr = np.sum(fn) / (np.sum(fn) + np.sum(tp))

        # 计算其他指标
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true_classes, y_pred_classes, average='weighted'
        )

        # 计算平均AUC
        auc_scores = []
        for i in range(config.NUM_CLASSES):
            auc = roc_auc_score(y_test[:, i], y_pred[:, i])
            auc_scores.append(auc)
        mean_auc = np.mean(auc_scores)

        return {
            'accuracy': accuracy,
            'f1': f1,
            'fpr': fpr,
            'fnr': fnr,
            'auc': mean_auc,
            'running_time': running_time,
            'confusion_matrix': conf_matrix
        }