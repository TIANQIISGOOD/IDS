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

def train_rf(X_train, y_train):
    model = RF()
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    return model