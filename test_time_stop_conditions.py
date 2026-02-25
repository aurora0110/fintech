"""
时间止损条件测试工具

功能说明：
1. 测试不同时间止损条件的策略表现
2. 支持测试多种时间周期和涨幅要求的组合，包括：
   - 3天、4天、5天未上涨1%
   - 3天、4天、5天未上涨2%
   - 3天、4天、5天未上涨3%
3. 计算并比较各条件下的策略绩效指标
4. 自动找出年化收益率最高的最佳策略
5. 保存详细测试结果到CSV文件

测试流程：
1. 遍历所有时间止损条件组合
2. 对每个条件运行短期交易策略回测
3. 收集所有股票的回测结果
4. 按条件分组分析绩效指标
5. 输出各条件的表现并找出最佳策略
"""
import os
import sys
import pandas as pd

# 添加项目根目录到 Python 路径
sys.path.append('/Users/lidongyang/Desktop/Qstrategy')
from utils.backtest_short_term import ShortTermBacktest

def test_time_stop_conditions():
    """
    测试不同时间止损条件的策略表现
    """
    # 配置参数
    data_dir = "/Users/lidongyang/Desktop/Qstrategy/data/20260207/normal/"
    
    # 定义需要测试的时间止损条件
    time_stop_conditions_list = [
        # 3天后未上涨1%就平仓
        [(3, 1)],
        # 4天后未上涨1%就平仓
        [(4, 1)],
        # 5天后未上涨1%就平仓
        [(5, 1)],
        # 3天后未上涨2%就平仓
        [(3, 2)],
        # 4天后未上涨2%就平仓
        [(4, 2)],
        # 5天后未上涨2%就平仓
        [(5, 2)],
        # 3天后未上涨3%就平仓
        [(3, 3)],
        # 4天后未上涨3%就平仓
        [(4, 3)],
        # 5天后未上涨3%就平仓
        [(5, 3)],
    ]
    
    # 条件名称
    condition_names = [
        "3天未涨1%",
        "4天未涨1%",
        "5天未涨1%",
        "3天未涨2%",
        "4天未涨2%",
        "5天未涨2%",
        "3天未涨3%",
        "4天未涨3%",
        "5天未涨3%",
    ]
    
    # 存储每个条件的结果
    all_results = []
    
    # 获取所有股票文件
    stock_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    print(f"找到 {len(stock_files)} 个股票文件")
    
    # 遍历每个时间止损条件
    for i, (time_stop_conditions, condition_name) in enumerate(zip(time_stop_conditions_list, condition_names)):
        print(f"\n测试条件 {i+1}/{len(time_stop_conditions_list)}: {condition_name}")
        
        # 创建回测器
        backtest = ShortTermBacktest(data_dir)
        
        # 遍历所有股票
        for j, file_name in enumerate(stock_files):
            if j % 100 == 0:
                print(f"处理进度: {j}/{len(stock_files)}")
            
            # 提取股票代码
            stock_code = file_name.split('#')[-1].replace('.txt', '')
            
            # 加载数据
            file_path = os.path.join(data_dir, file_name)
            df = backtest.load_stock_data(file_path)
            
            if df is not None and len(df) > 60:  # 确保数据足够长
                # 运行策略
                result = backtest.run_strategy(df, stock_code, time_stop_conditions)
                # 添加条件名称
                result['condition'] = condition_name
                all_results.append(result)
    
    # 分析结果
    analyze_results(all_results)

def analyze_results(all_results):
    """
    分析回测结果
    """
    if not all_results:
        print("没有回测结果")
        return
    
    results_df = pd.DataFrame(all_results)
    
    # 清理数据，移除无效的年化收益率
    results_df['annual_return'] = pd.to_numeric(results_df['annual_return'], errors='coerce')
    results_df = results_df.dropna(subset=['annual_return'])
    
    # 按条件分组分析
    condition_groups = results_df.groupby('condition')
    
    print("\n各条件策略表现:")
    print("-" * 80)
    print(f"{'条件':<12} {'股票数':<8} {'总交易数':<10} {'成功率':<10} {'年化收益率':<15} {'平均每笔盈利':<15}")
    print("-" * 80)
    
    best_condition = None
    best_annual_return = -float('inf')
    
    for condition, group in condition_groups:
        # 计算统计数据
        num_stocks = len(group)
        total_trades = group['num_trades'].sum()
        success_rate = group['success_rate'].mean()
        annual_return = group['annual_return'].mean()
        avg_profit_per_trade = group['avg_profit_per_trade'].mean()
        
        # 打印结果
        print(f"{condition:<12} {num_stocks:<8} {total_trades:<10} {success_rate:.2%} {annual_return:.4f} {avg_profit_per_trade:.4f}")
        
        # 找出年化收益率最高的条件
        if annual_return > best_annual_return:
            best_annual_return = annual_return
            best_condition = condition
    
    print("-" * 80)
    print(f"\n最佳策略: {best_condition}")
    print(f"年化收益率: {best_annual_return:.4f}")
    
    # 保存详细结果
    results_df.to_csv("/Users/lidongyang/Desktop/Qstrategy/time_stop_results.csv", index=False, encoding='utf-8-sig')
    print("\n详细结果已保存到 time_stop_results.csv")

if __name__ == "__main__":
    test_time_stop_conditions()
