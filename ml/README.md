# 机器学习模块

存放策略相关的机器学习训练、预测和验证代码。

## 内容

```
ml/
├── analyze_test_results.py      # 测试结果分析
├── lightgbm_brick_model.txt     # LightGBM模型（BRICK策略）
├── lightgbm_config.json         # LightGBM配置
├── ml_brick_train_dataset_build.py  # BRICK训练集构建
├── predict.py                   # 模型预测入口
├── test_models.py               # 模型测试
├── threshold_analysis.py        # 阈值分析
├── train_lightgbm.py            # 训练LightGBM
├── train_xgboost.py             # 训练XGBoost
├── validate_threshold.py         # 阈值验证
├── xgboost_brick_model.json     # XGBoost模型（BRICK策略）
└── xgboost_config.json          # XGBoost配置
```

## 用途说明

| 文件 | 说明 |
|------|------|
| `train_*.py` | 训练模型脚本 |
| `lightgbm_*.txt/json` | 训练好的模型文件 |
| `ml_*_dataset_build.py` | 构建训练数据集 |
| `predict.py` | 模型推理入口 |
| `analyze_test_results.py` | 分析模型表现 |
