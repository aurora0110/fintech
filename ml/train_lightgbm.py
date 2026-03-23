import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import lightgbm as lgb
import json

DATA_FILE = Path("/Users/lidongyang/Desktop/Qstrategy/ml/brick_train_dataset.csv")
OUTPUT_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/ml")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    'signal_ret', 'brick_red_len', 'brick_green_len_prev', 'rebound_ratio',
    'red_len_vs_1d', 'red_len_vs_3d', 'red_len_vs_5d', 'red_len_vs_10d',
    'green_len_prev_vs_1d', 'green_len_prev_vs_3d', 'green_len_prev_vs_5d', 'green_len_prev_vs_10d',
    'trend_spread', 'close_to_trend', 'close_to_long',
    'trend_slope_3', 'trend_slope_5', 'trend_slope_10',
    'ma10_slope_3', 'ma10_slope_5', 'ma10_slope_10',
    'ma20_slope_3', 'ma20_slope_5', 'ma20_slope_10',
    'signal_vs_ma5', 'ret1', 'ret5', 'ret10',
    'RSI14', 'MACD_DIF', 'MACD_DEA', 'MACD_hist',
    'KDJ_K', 'KDJ_D', 'KDJ_J',
    'body_ratio', 'close_location', 'upper_shadow_pct', 'lower_shadow_pct'
]

PARAMS = {
    'objective': 'binary',
    'metric': ['auc', 'binary_logloss'],
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 42,
    'n_jobs': -1
}

TRAIN_PARAMS = {
    'num_boost_round': 500,
    'early_stopping_rounds': 50
}


def load_data():
    df = pd.read_csv(DATA_FILE)
    print(f"数据集大小: {df.shape}")
    print(f"正样本: {(df['label'] == 1).sum()}, 负样本: {(df['label'] == 0).sum()}")
    return df


def prepare_features(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    y = df['label'].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    return X, y


def train_model(X_train, y_train, X_val, y_val):
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    print("\n开始训练LightGBM模型...")
    print(f"模型参数: {PARAMS}")
    print(f"训练参数: {TRAIN_PARAMS}")
    
    model = lgb.train(
        PARAMS,
        train_data,
        num_boost_round=TRAIN_PARAMS['num_boost_round'],
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=TRAIN_PARAMS['early_stopping_rounds']),
            lgb.log_evaluation(period=50)
        ]
    )
    
    return model


def evaluate_model(model, X_test, y_test):
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    print("\n" + "="*50)
    print("测试集评估结果")
    print("="*50)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"AUC:       {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['负样本', '正样本']))
    
    print("混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))
    
    return y_pred_proba, y_pred


def show_feature_importance(model, feature_names):
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    print("\n特征重要性 Top 20:")
    print(importance.head(20).to_string(index=False))
    
    importance.to_csv(OUTPUT_DIR / "lightgbm_feature_importance.csv", index=False)
    print(f"\n特征重要性已保存到: {OUTPUT_DIR / 'lightgbm_feature_importance.csv'}")
    
    return importance


def save_model_with_params(model):
    model_file = OUTPUT_DIR / "lightgbm_brick_model.txt"
    model.save_model(str(model_file))
    print(f"\n模型已保存到: {model_file}")
    
    config = {
        'model_params': PARAMS,
        'train_params': TRAIN_PARAMS,
        'best_iteration': int(model.best_iteration),
        'feature_cols': FEATURE_COLS,
        'num_features': len(FEATURE_COLS)
    }
    config_file = OUTPUT_DIR / "lightgbm_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"模型配置已保存到: {config_file}")


def main():
    df = load_data()
    X, y = prepare_features(df)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\n训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
    
    model = train_model(X_train, y_train, X_val, y_val)
    
    evaluate_model(model, X_test, y_test)
    
    show_feature_importance(model, FEATURE_COLS)
    
    save_model_with_params(model)


if __name__ == "__main__":
    main()
