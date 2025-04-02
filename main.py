# main.py 修复后
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

def measure_detection_time(model, X_test):
    start_time = time.time()
    _ = model.predict(X_test)
    end_time = time.time()
    return (end_time - start_time) / len(X_test)

if __name__ == "__main__":
    # 初始化结果字典
    results = {}

    # 数据加载
    loader = DataLoader("data/OPCUA_dataset_public.csv")
    X_train_full, X_test, y_train_full, y_test = loader.load_data()
    #X_train, X_test, y_train, y_test = loader.load_data()

    # # 统一调整输入形状
    # X_train_cnn = X_train.reshape(-1, config.FEATURE_DIM, 1)
    # X_test_cnn = X_test.reshape(-1, config.FEATURE_DIM, 1)
    # X_train_rf = X_train.reshape(X_train.shape[0], -1)
    # X_test_rf = X_test.reshape(X_test.shape[0], -1)
    #
    evaluator = ModelEvaluator()
    #
    # # 评估BiLSTM-CNN
    # model = BiLSTM_CNN()
    # trainer = ModelTrainer(model)
    # history = trainer.train(X_train_cnn, y_train, X_test_cnn, y_test)
    # results['BiLSTM-CNN'] = evaluator.evaluate(model, X_test_cnn, y_test)
    #
    # # 评估CNN
    # cnn_model, _ = train_cnn(X_train_cnn, y_train)
    # results['CNN'] = evaluator.evaluate(cnn_model, X_test_cnn, y_test)
    #
    # # 评估BiLSTM
    # bilstm_model, _ = train_bilstm(X_train_cnn, y_train)  # 正确接收模型
    # results['BiLSTM'] = evaluator.evaluate(bilstm_model, X_test_cnn, y_test)
    #
    # # 评估随机森林
    # rf_model = train_rf(X_train_rf, y_train)
    # results['RandomForest'] = evaluator.evaluate(rf_model, X_test_rf, y_test, model_type='ml')
    #
    #
    # try:
    #     # BiLSTM-CNN
    #     model = BiLSTM_CNN()
    #     trainer = ModelTrainer(model)
    #     history = trainer.train(...)
    #     results['BiLSTM-CNN'] = evaluator.evaluate(...)
    #
    #     # CNN
    #     cnn_model, _ = train_cnn(...)
    #     results['CNN'] = evaluator.evaluate(...)
    #
    #     # BiLSTM
    #     bilstm_model, _ = train_bilstm(...)
    #     results['BiLSTM'] = evaluator.evaluate(...)
    #
    #     # 随机森林
    #     rf_model = train_rf(...)
    #     results['RandomForest'] = evaluator.evaluate(...)
    # except Exception as e:
    #     print(f"评估过程中发生错误: {str(e)}")
    #
    #
    # # 可视化对比结果
    # evaluator.compare_models(results)
    #
    # # 打印对比表格
    # print("\n\n=== 模型性能对比表 ===")
    # print("{:<15} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8}".format(
    #     "模型名称", "Accuracy", "Recall", "F1", "FPR", "AUC"))
    # print("-" * 70)
    # for name, metrics in results.items():
    #     print("{:<15} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f}".format(
    #         name,
    #         metrics['accuracy'],
    #         metrics['recall'],
    #         metrics['f1'],
    #         metrics['fpr'],
    #         metrics['auc']
    #     ))

    # main.py

    # 1. 数据分割


    # 首先分割出测试集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )
    # 从训练集中再分割出验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    # 3. 数据形状处理
    # 深度学习模型需要3D数据
    X_train_dl = X_train.reshape(-1, config.FEATURE_DIM, 1)
    X_val_dl = X_val.reshape(-1, config.FEATURE_DIM, 1)
    X_test_dl = X_test.reshape(-1, config.FEATURE_DIM, 1)

    # 随机森林需要2D数据 (这里其实可以直接使用原始形状，因为已经是2D的了)
    X_train_rf = X_train
    X_val_rf = X_val
    X_test_rf = X_test

    # 初始化评估器
    evaluator = ModelEvaluator()
    results = {}

    try:
        # 4. 模型训练和评估
        #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             CNN-BiLSTM
        print("\nTraining CNN-BiLSTM model...")
        model_cnn_bilstm = BiLSTM_CNN()
        trainer = ModelTrainer(model_cnn_bilstm)
        history_cnn_bilstm = trainer.train(X_train_dl, y_train, X_val_dl, y_val)
        results['BiLSTM-CNN'] = evaluator.evaluate(model_cnn_bilstm, X_test_dl, y_test)
        results['BiLSTM-CNN']['detection_time'] = measure_detection_time(model_cnn_bilstm, X_test_dl)

        # CNN
        print("\nTraining CNN model...")
        cnn_model, history_cnn = train_cnn(X_train_dl, y_train, X_val_dl, y_val)
        results['CNN'] = evaluator.evaluate(cnn_model, X_test_dl, y_test)
        results['CNN']['detection_time'] = measure_detection_time(cnn_model, X_test_dl)

        # BiLSTM
        print("\nTraining BiLSTM model...")
        bilstm_model, history_bilstm = train_bilstm(X_train_dl, y_train, X_val_dl, y_val)
        results['BiLSTM'] = evaluator.evaluate(bilstm_model, X_test_dl, y_test)
        results['BiLSTM']['detection_time'] = measure_detection_time(bilstm_model, X_test_dl)

        # GRU
        print("\nTraining GRU model...")
        gru_model, history_gru = train_improved_gru(X_train_dl, y_train, X_val_dl, y_val)
        results['GRU'] = evaluator.evaluate(gru_model, X_test_dl, y_test)
        results['GRU']['detection_time'] = measure_detection_time(gru_model, X_test_dl)

        # 5. 显示结果
        print("\n=== 模型性能对比表 ===")
        print("{:<15} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8}".format(
            "模型名称", "Accuracy", "Recall", "F1", "FPR", "AUC", "检测时间(ms)"))
        print("-" * 80)
        for name, metrics in results.items():
            print("{:<15} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f}".format(
                name,
                metrics['accuracy'],
                metrics['recall'],
                metrics['f1'],
                metrics['fpr'],
                metrics['auc'],
                metrics['detection_time'] * 1000  # 转换为毫秒
            ))

    except Exception as e:
        print(f"训练或评估过程中发生错误: {str(e)}")