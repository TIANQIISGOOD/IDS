from sklearn.ensemble import RandomForestClassifier
from configs.params import config

def RF():
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,  # 添加随机种子
        oob_score=True    # 启用袋外评估
    )

# def train_rf(X_train, y_train):
#     model = RF()
#     model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
#     return model
def train_rf(X_train, y_train, X_val=None, y_val=None):
    """
    训练随机森林模型

    Parameters:
    -----------
    X_train : array-like
        训练数据
    y_train : array-like
        训练标签
    X_val : array-like, optional
        验证数据（对RF可选）
    y_val : array-like, optional
        验证标签（对RF可选）
    """
    model = RF()
    # 确保输入是2D
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    model.fit(X_train_2d, y_train)

    # 如果提供了验证集，计算验证集得分
    if X_val is not None and y_val is not None:
        X_val_2d = X_val.reshape(X_val.shape[0], -1)
        val_score = model.score(X_val_2d, y_val)
        print(f"Validation Score: {val_score:.4f}")

    return model