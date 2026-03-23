import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from pathlib import Path

OUTPUT_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/ml")
THRESHOLD = 0.8

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
    
    X = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_true = df['label'].values
    
    print(f"测试集样本数: {len(df)}")
    print(f"使用阈值: {THRESHOLD}")
    
    print("\n加载模型...")
    lgb_model = lgb.Booster(model_file=str(OUTPUT_DIR / "lightgbm_brick_model.txt"))
    xgb_model = xgb.Booster()
    xgb_model.load_model(str(OUTPUT_DIR / "xgboost_brick_model.json"))
    
    lgb_proba = lgb_model.predict(X, num_iteration=lgb_model.best_iteration)
    dtest = xgb.DMatrix(X)
    xgb_proba = xgb_model.predict(dtest)
    
    results = df[['code', 'name', 'date', 'close', 'open', 'high', 'low']].copy()
    results['true_label'] = y_true
    results['lgb_proba'] = lgb_proba
    results['xgb_proba'] = xgb_proba
    results['lgb_pred'] = (lgb_proba > THRESHOLD).astype(int)
    results['xgb_pred'] = (xgb_proba > THRESHOLD).astype(int)
    results['signal_ret'] = df['signal_ret']
    
    print("\n" + "="*100)
    print(f"LightGBM 阈值 {THRESHOLD} 预测结果")
    print("="*100)
    
    lgb_pos = results[results['lgb_pred'] == 1]
    lgb_correct = lgb_pos[lgb_pos['true_label'] == 1]
    lgb_wrong = lgb_pos[lgb_pos['true_label'] == 0]
    
    print(f"\n预测为正样本数: {len(lgb_pos)}")
    print(f"预测正确: {len(lgb_correct)}")
    print(f"预测错误: {len(lgb_wrong)}")
    print(f"精确率: {len(lgb_correct)/len(lgb_pos)*100:.1f}%" if len(lgb_pos) > 0 else "精确率: N/A")
    
    print(f"\n预测为正样本的股票列表:")
    print(f"{'代码':<8} {'名称':<10} {'日期':<12} {'收盘价':<10} {'信号日涨幅':<12} {'预测概率':<10} {'实际标签':<10} {'预测结果'}")
    print("-"*100)
    for _, row in lgb_pos.iterrows():
        actual = "正样本(成功)" if row['true_label'] == 1 else "负样本(失败)"
        result = "✓ 正确" if row['true_label'] == 1 else "✗ 错误"
        print(f"{row['code']:<8} {row['name']:<10} {str(row['date'])[:10]:<12} {row['close']:<10.2f} {row['signal_ret']*100:<11.2f}% {row['lgb_proba']:<10.4f} {actual:<10} {result}")
    
    print("\n" + "="*100)
    print(f"XGBoost 阈值 {THRESHOLD} 预测结果")
    print("="*100)
    
    xgb_pos = results[results['xgb_pred'] == 1]
    xgb_correct = xgb_pos[xgb_pos['true_label'] == 1]
    xgb_wrong = xgb_pos[xgb_pos['true_label'] == 0]
    
    print(f"\n预测为正样本数: {len(xgb_pos)}")
    print(f"预测正确: {len(xgb_correct)}")
    print(f"预测错误: {len(xgb_wrong)}")
    print(f"精确率: {len(xgb_correct)/len(xgb_pos)*100:.1f}%" if len(xgb_pos) > 0 else "精确率: N/A")
    
    print(f"\n预测为正样本的股票列表:")
    print(f"{'代码':<8} {'名称':<10} {'日期':<12} {'收盘价':<10} {'信号日涨幅':<12} {'预测概率':<10} {'实际标签':<10} {'预测结果'}")
    print("-"*100)
    for _, row in xgb_pos.iterrows():
        actual = "正样本(成功)" if row['true_label'] == 1 else "负样本(失败)"
        result = "✓ 正确" if row['true_label'] == 1 else "✗ 错误"
        print(f"{row['code']:<8} {row['name']:<10} {str(row['date'])[:10]:<12} {row['close']:<10.2f} {row['signal_ret']*100:<11.2f}% {row['xgb_proba']:<10.4f} {actual:<10} {result}")
    
    lgb_output = OUTPUT_DIR / f"lgb_threshold_{THRESHOLD}_predictions.csv"
    xgb_output = OUTPUT_DIR / f"xgb_threshold_{THRESHOLD}_predictions.csv"
    
    lgb_pos.to_csv(lgb_output, index=False, encoding='utf-8-sig')
    xgb_pos.to_csv(xgb_output, index=False, encoding='utf-8-sig')
    
    print(f"\n预测结果已保存:")
    print(f"  LightGBM: {lgb_output}")
    print(f"  XGBoost: {xgb_output}")


if __name__ == "__main__":
    main()
