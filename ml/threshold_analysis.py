import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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


def evaluate_threshold(y_true, y_proba, threshold):
    y_pred = (y_proba > threshold).astype(int)
    
    total = len(y_true)
    correct = np.sum(y_pred == y_true)
    accuracy = correct / total
    
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    pos_recall = tp / np.sum(pos_mask) if np.sum(pos_mask) > 0 else 0
    neg_accuracy = tn / np.sum(neg_mask) if np.sum(neg_mask) > 0 else 0
    
    pred_pos_count = np.sum(y_pred == 1)
    pred_neg_count = np.sum(y_pred == 0)
    
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pos_recall': pos_recall,
        'neg_accuracy': neg_accuracy,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'pred_pos_count': pred_pos_count,
        'pred_neg_count': pred_neg_count
    }


def main():
    test_file = OUTPUT_DIR / "brick_test_dataset.csv"
    df = pd.read_csv(test_file)
    
    X = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_true = df['label'].values
    
    print("加载模型...")
    lgb_model = lgb.Booster(model_file=str(OUTPUT_DIR / "lightgbm_brick_model.txt"))
    xgb_model = xgb.Booster()
    xgb_model.load_model(str(OUTPUT_DIR / "xgboost_brick_model.json"))
    
    lgb_proba = lgb_model.predict(X, num_iteration=lgb_model.best_iteration)
    dtest = xgb.DMatrix(X)
    xgb_proba = xgb_model.predict(dtest)
    
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    
    print("\n" + "="*100)
    print("LightGBM 不同阈值效果分析")
    print("="*100)
    print(f"{'阈值':<8} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1':<10} {'正样本召回':<12} {'负样本准确':<12} {'预测正样本数':<12}")
    print("-"*100)
    
    lgb_results = []
    for t in thresholds:
        r = evaluate_threshold(y_true, lgb_proba, t)
        lgb_results.append(r)
        print(f"{t:<8.2f} {r['accuracy']:<10.2%} {r['precision']:<10.2%} {r['recall']:<10.2%} {r['f1']:<10.2%} {r['pos_recall']:<12.2%} {r['neg_accuracy']:<12.2%} {r['pred_pos_count']:<12}")
    
    print("\n" + "="*100)
    print("XGBoost 不同阈值效果分析")
    print("="*100)
    print(f"{'阈值':<8} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1':<10} {'正样本召回':<12} {'负样本准确':<12} {'预测正样本数':<12}")
    print("-"*100)
    
    xgb_results = []
    for t in thresholds:
        r = evaluate_threshold(y_true, xgb_proba, t)
        xgb_results.append(r)
        print(f"{t:<8.2f} {r['accuracy']:<10.2%} {r['precision']:<10.2%} {r['recall']:<10.2%} {r['f1']:<10.2%} {r['pos_recall']:<12.2%} {r['neg_accuracy']:<12.2%} {r['pred_pos_count']:<12}")
    
    lgb_df = pd.DataFrame(lgb_results)
    xgb_df = pd.DataFrame(xgb_results)
    
    lgb_df.to_csv(OUTPUT_DIR / "lgb_threshold_analysis.csv", index=False)
    xgb_df.to_csv(OUTPUT_DIR / "xgb_threshold_analysis.csv", index=False)
    
    print("\n" + "="*100)
    print("推荐阈值")
    print("="*100)
    
    best_f1_idx = lgb_df['f1'].idxmax()
    best_lgb = lgb_df.loc[best_f1_idx]
    print(f"\nLightGBM 最佳F1阈值: {best_lgb['threshold']:.2f}")
    print(f"  准确率: {best_lgb['accuracy']:.2%}, 精确率: {best_lgb['precision']:.2%}, 召回率: {best_lgb['recall']:.2%}, F1: {best_lgb['f1']:.2%}")
    
    best_f1_idx = xgb_df['f1'].idxmax()
    best_xgb = xgb_df.loc[best_f1_idx]
    print(f"\nXGBoost 最佳F1阈值: {best_xgb['threshold']:.2f}")
    print(f"  准确率: {best_xgb['accuracy']:.2%}, 精确率: {best_xgb['precision']:.2%}, 召回率: {best_xgb['recall']:.2%}, F1: {best_xgb['f1']:.2%}")
    
    print("\n" + "="*100)
    print("高精确率阈值 (预测正样本更可靠)")
    print("="*100)
    
    for t in [0.7, 0.75, 0.8, 0.85, 0.9]:
        lgb_r = [r for r in lgb_results if r['threshold'] == t][0]
        xgb_r = [r for r in xgb_results if r['threshold'] == t][0]
        print(f"\n阈值 {t}:")
        print(f"  LightGBM: 精确率 {lgb_r['precision']:.2%}, 预测正样本 {lgb_r['pred_pos_count']} 个, 其中正确 {lgb_r['tp']} 个")
        print(f"  XGBoost:  精确率 {xgb_r['precision']:.2%}, 预测正样本 {xgb_r['pred_pos_count']} 个, 其中正确 {xgb_r['tp']} 个")
    
    print(f"\n阈值分析结果已保存到:")
    print(f"  {OUTPUT_DIR / 'lgb_threshold_analysis.csv'}")
    print(f"  {OUTPUT_DIR / 'xgb_threshold_analysis.csv'}")


if __name__ == "__main__":
    main()
