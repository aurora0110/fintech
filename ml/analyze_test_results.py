import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from pathlib import Path

OUTPUT_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/ml")

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


def main():
    test_file = OUTPUT_DIR / "brick_test_dataset.csv"
    df = pd.read_csv(test_file)
    print(f"测试集样本数: {len(df)}")
    print(f"正样本: {len(df[df['label']==1])}, 负样本: {len(df[df['label']==0])}")
    
    X = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_true = df['label'].values
    
    print("\n" + "="*80)
    print("LightGBM 预测分析")
    print("="*80)
    lgb_model = lgb.Booster(model_file=str(OUTPUT_DIR / "lightgbm_brick_model.txt"))
    lgb_proba = lgb_model.predict(X, num_iteration=lgb_model.best_iteration)
    lgb_pred = (lgb_proba > 0.5).astype(int)
    
    lgb_correct = lgb_pred == y_true
    lgb_results = df[['code', 'name', 'date', 'close']].copy()
    lgb_results['true_label'] = y_true
    lgb_results['pred_proba'] = lgb_proba
    lgb_results['pred_label'] = lgb_pred
    lgb_results['correct'] = lgb_correct
    
    lgb_success = lgb_results[lgb_results['correct']]
    lgb_fail = lgb_results[~lgb_results['correct']]
    
    print(f"\n预测正确: {len(lgb_success)} / {len(df)} ({len(lgb_success)/len(df)*100:.1f}%)")
    print(f"预测错误: {len(lgb_fail)} / {len(df)} ({len(lgb_fail)/len(df)*100:.1f}%)")
    
    print("\n预测成功的样本 (部分):")
    print(lgb_success[['code', 'name', 'date', 'close', 'true_label', 'pred_proba']].head(20).to_string(index=False))
    
    print("\n预测失败的样本 (部分):")
    print(lgb_fail[['code', 'name', 'date', 'close', 'true_label', 'pred_proba']].head(20).to_string(index=False))
    
    lgb_success_file = OUTPUT_DIR / "lgb_test_success.csv"
    lgb_fail_file = OUTPUT_DIR / "lgb_test_fail.csv"
    lgb_success.to_csv(lgb_success_file, index=False, encoding='utf-8-sig')
    lgb_fail.to_csv(lgb_fail_file, index=False, encoding='utf-8-sig')
    print(f"\n成功样本已保存: {lgb_success_file}")
    print(f"失败样本已保存: {lgb_fail_file}")
    
    print("\n" + "="*80)
    print("XGBoost 预测分析")
    print("="*80)
    xgb_model = xgb.Booster()
    xgb_model.load_model(str(OUTPUT_DIR / "xgboost_brick_model.json"))
    dtest = xgb.DMatrix(X)
    xgb_proba = xgb_model.predict(dtest)
    xgb_pred = (xgb_proba > 0.5).astype(int)
    
    xgb_correct = xgb_pred == y_true
    xgb_results = df[['code', 'name', 'date', 'close']].copy()
    xgb_results['true_label'] = y_true
    xgb_results['pred_proba'] = xgb_proba
    xgb_results['pred_label'] = xgb_pred
    xgb_results['correct'] = xgb_correct
    
    xgb_success = xgb_results[xgb_results['correct']]
    xgb_fail = xgb_results[~xgb_results['correct']]
    
    print(f"\n预测正确: {len(xgb_success)} / {len(df)} ({len(xgb_success)/len(df)*100:.1f}%)")
    print(f"预测错误: {len(xgb_fail)} / {len(df)} ({len(xgb_fail)/len(df)*100:.1f}%)")
    
    print("\n预测成功的样本 (部分):")
    print(xgb_success[['code', 'name', 'date', 'close', 'true_label', 'pred_proba']].head(20).to_string(index=False))
    
    print("\n预测失败的样本 (部分):")
    print(xgb_fail[['code', 'name', 'date', 'close', 'true_label', 'pred_proba']].head(20).to_string(index=False))
    
    xgb_success_file = OUTPUT_DIR / "xgb_test_success.csv"
    xgb_fail_file = OUTPUT_DIR / "xgb_test_fail.csv"
    xgb_success.to_csv(xgb_success_file, index=False, encoding='utf-8-sig')
    xgb_fail.to_csv(xgb_fail_file, index=False, encoding='utf-8-sig')
    print(f"\n成功样本已保存: {xgb_success_file}")
    print(f"失败样本已保存: {xgb_fail_file}")
    
    print("\n" + "="*80)
    print("详细分析")
    print("="*80)
    print("\n正样本预测情况:")
    pos_mask = y_true == 1
    lgb_pos_correct = np.sum(lgb_correct[pos_mask])
    xgb_pos_correct = np.sum(xgb_correct[pos_mask])
    print(f"  LightGBM 正样本召回率: {lgb_pos_correct}/{np.sum(pos_mask)} ({lgb_pos_correct/np.sum(pos_mask)*100:.1f}%)")
    print(f"  XGBoost  正样本召回率: {xgb_pos_correct}/{np.sum(pos_mask)} ({xgb_pos_correct/np.sum(pos_mask)*100:.1f}%)")
    
    print("\n负样本预测情况:")
    neg_mask = y_true == 0
    lgb_neg_correct = np.sum(lgb_correct[neg_mask])
    xgb_neg_correct = np.sum(xgb_correct[neg_mask])
    print(f"  LightGBM 负样本准确率: {lgb_neg_correct}/{np.sum(neg_mask)} ({lgb_neg_correct/np.sum(neg_mask)*100:.1f}%)")
    print(f"  XGBoost  负样本准确率: {xgb_neg_correct}/{np.sum(neg_mask)} ({xgb_neg_correct/np.sum(neg_mask)*100:.1f}%)")


if __name__ == "__main__":
    main()
