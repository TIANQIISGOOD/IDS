# main.py 修复后
from src.data_loader import DataLoader
from src.model import BiLSTM_CNN
from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator
from src.bilstm import BiLSTM, train_bilstm
from src.cnn import CNN, train_cnn
from src.rf import RF, train_rf
from configs.params import config
import tensorflow as tf

if __name__ == "__main__":
    # 初始化结果字典
    results = {}

    # 数据加载
    loader = DataLoader("data/OPCUA_dataset_public.csv")
    X_train, X_test, y_train, y_test = loader.load_data()

    # 统一调整输入形状
    X_train_cnn = X_train.reshape(-1, config.FEATURE_DIM, 1)
    X_test_cnn = X_test.reshape(-1, config.FEATURE_DIM, 1)
    X_train_rf = X_train.reshape(X_train.shape[0], -1)
    X_test_rf = X_test.reshape(X_test.shape[0], -1)

    evaluator = ModelEvaluator()

    # 评估BiLSTM-CNN
    model = BiLSTM_CNN()
    trainer = ModelTrainer(model)
    history = trainer.train(X_train_cnn, y_train, X_test_cnn, y_test)
    results['BiLSTM-CNN'] = evaluator.evaluate(model, X_test_cnn, y_test)

    # 评估CNN
    cnn_model, _ = train_cnn(X_train_cnn, y_train)
    results['CNN'] = evaluator.evaluate(cnn_model, X_test_cnn, y_test)

    # 评估BiLSTM
    bilstm_model, _ = train_bilstm(X_train_cnn, y_train)  # 正确接收模型
    results['BiLSTM'] = evaluator.evaluate(bilstm_model, X_test_cnn, y_test)

    # 评估随机森林
    rf_model = train_rf(X_train_rf, y_train)
    results['RandomForest'] = evaluator.evaluate(rf_model, X_test_rf, y_test, model_type='ml')


    try:
        # BiLSTM-CNN
        model = BiLSTM_CNN()
        trainer = ModelTrainer(model)
        history = trainer.train(...)
        results['BiLSTM-CNN'] = evaluator.evaluate(...)

        # CNN
        cnn_model, _ = train_cnn(...)
        results['CNN'] = evaluator.evaluate(...)

        # BiLSTM
        bilstm_model, _ = train_bilstm(...)
        results['BiLSTM'] = evaluator.evaluate(...)

        # 随机森林
        rf_model = train_rf(...)
        results['RandomForest'] = evaluator.evaluate(...)
    except Exception as e:
        print(f"评估过程中发生错误: {str(e)}")


    # 可视化对比结果
    evaluator.compare_models(results)

    # 打印对比表格
    print("\n\n=== 模型性能对比表 ===")
    print("{:<15} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8}".format(
        "模型名称", "Accuracy", "Recall", "F1", "FPR", "AUC"))
    print("-" * 70)
    for name, metrics in results.items():
        print("{:<15} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f} | {:<8.4f}".format(
            name,
            metrics['accuracy'],
            metrics['recall'],
            metrics['f1'],
            metrics['fpr'],
            metrics['auc']
        ))