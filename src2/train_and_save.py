import numpy as np
import tensorflow as tf
import pickle
from data_loader import DataLoader
from gru import train_improved_gru
from model import BiLSTM_CNN
from trainer import ModelTrainer
from evaluator import ModelEvaluator
from bilstm import BiLSTM, train_bilstm
from cnn import CNN, train_cnn
from ablation_models import SpatialOnlyModel, TemporalOnlyModel
from params import config
from sklearn.model_selection import train_test_split


def save_model_weights(model, name):
    """保存模型权重和结构"""
    # 保存模型权重
    model.save_weights(f'saved_models/{name}_weights.h5')
    # 保存模型配置
    model_config = model.get_config()
    with open(f'saved_models/{name}_config.pkl', 'wb') as f:
        pickle.dump(model_config, f)


if __name__ == "__main__":
    current_time = "2025-04-28 03:29:09"
    current_user = "TIANQIISGOOD"
    print(f"Execution Time (UTC): {current_time}")
    print(f"User: {current_user}")
    print("=" * 50)

    # 数据加载
    print("\nLoading and preprocessing data...")
    loader = DataLoader("OPCUA_dataset_public.csv")
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

    # 保存测试集数据
    print("Saving test data...")
    np.save('saved_models/X_test.npy', X_test_dl)
    np.save('saved_models/y_test.npy', y_test)

    try:
        # 训练和保存模型
        models = {
            'BiLSTM-CNN': BiLSTM_CNN(),
            'CNN': CNN(),
            'BiLSTM': BiLSTM(),
            'GRU': None,  # 将在训练时创建
            'Spatial-Only': SpatialOnlyModel(),
            'Temporal-Only': TemporalOnlyModel()
        }

        # 训练和保存每个模型
        for name, model in models.items():
            print(f"\nTraining {name} model...")

            if name == 'CNN':
                model, history = train_cnn(X_train_dl, y_train, X_val_dl, y_val)
            elif name == 'BiLSTM':
                model, history = train_bilstm(X_train_dl, y_train, X_val_dl, y_val)
            elif name == 'GRU':
                model, history = train_improved_gru(X_train_dl, y_train, X_val_dl, y_val)
            else:
                trainer = ModelTrainer(model)
                history = trainer.train(X_train_dl, y_train, X_val_dl, y_val)

            # 保存模型权重和配置
            print(f"Saving {name} model...")
            save_model_weights(model, name)

            # 保存训练历史
            with open(f'saved_models/{name}_history.pkl', 'wb') as f:
                pickle.dump(history.history, f)

        print("\nAll models have been trained and saved successfully!")

    except Exception as e:
        print(f"Error during training: {str(e)}")