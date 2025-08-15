import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

class InteractiveStockCalculator:
    def __init__(self, initial_cash=100000.0):
        self.cash = initial_cash
        self.shares = 0
        self.history = []
        self.initial_cash = initial_cash  # 记录初始资金
        
    def run(self):
        print("""
        🚀 股票交易计算器（交互模式）
        ----------------------------
        输入格式: <日期> <价格> <数量> <操作(buy/sell)>
        示例: 20230801 150.5 100 buy
        输入 'q' 退出 | 'h' 查看历史 | 'c' 当前状态 | 'r' 收益率报告 | 'p' 绘制总资产随时间变化图
        """)
        
        while True:
            try:
                user_input = input(">>>>>>> ").strip()
                
                # 处理空输入
                if not user_input:
                    print("⚠️ 请输入命令或交易数据")
                    continue
                    
                # 处理特殊命令
                if user_input.lower() == 'q':
                    print("退出计算器")
                    break
                elif user_input.lower() == 'h':
                    self.show_history()
                    continue
                elif user_input.lower() == 'c':
                    self.show_status()
                    continue
                elif user_input.lower() == 'r':
                    self.show_returns()
                    continue
                elif user_input.lower() == 'p':
                    self.plot_portfolio_value()
                    continue
                # 解析交易指令（严格检查参数数量）
                parts = user_input.split()
                if len(parts) != 4:
                    print("❌ 输入格式错误，需要4个参数：<日期> <价格> <数量> <操作>")
                    print("   示例: 20230801 150.5 100 buy")
                    continue
                
                date, price, quantity, action = parts
                self.execute_trade(date, float(price), int(quantity), action.lower())
                
            except ValueError as e:
                print(f"❌ 错误: {e}")
            except Exception as e:
                print(f"❌ 系统错误: {e}")
    def execute_trade(self, date, price, quantity, action):
        """执行交易并更新状态"""
        try:
            datetime.strptime(date, "%Y%m%d")
        except ValueError:
            raise ValueError("日期格式应为 YYYYMMDD")
        
        if action not in ['buy', 'sell']:
            raise ValueError("操作必须是 'buy' 或 'sell'")
        
        if action == 'buy':
            cost = price * quantity
            if cost > self.cash:
                raise ValueError(f"资金不足，需要 ¥{cost:.2f}，当前可用 ¥{self.cash:.2f}")
            self.cash -= cost
            self.shares += quantity
        else:
            if quantity > self.shares:
                raise ValueError(f"持股不足，当前持有 {self.shares} 股")
            self.cash += price * quantity
            self.shares -= quantity
        
        trade = {
            'date': date,
            'price': price,
            'quantity': quantity,
            'action': action,
            'cash': self.cash,
            'shares': self.shares
        }
        self.history.append(trade)
        
        print(f"✅ {date} {action.upper()} {quantity}股 @¥{price:.2f}")
        self.show_status()

    def calculate_returns(self, current_price=None):
        """
        计算收益率
        :param current_price: 当前股价（如果未提供则使用最近交易价格）
        :return: (current_return, annualized_return)
        """
        if not self.history:
            return 0.0, 0.0
        
        # 获取当前股价（默认使用最近一次交易价格）
        if current_price is None:
            current_price = self.history[-1]['price']
        
        # 计算总投入成本（所有买入操作的总和）
        total_invested = sum(
            t['price'] * t['quantity'] for t in self.history 
            if t['action'] == 'buy'
        )
        
        # 计算当前持仓价值
        current_value = self.cash + (self.shares * current_price)
        
        # 当前收益率 = (当前价值 - 初始资金) / 初始资金
        current_return = (current_value - self.initial_cash) / self.initial_cash
        
        # 计算年化收益率（需计算投资时长）
        if len(self.history) >= 2:
            start_date = datetime.strptime(self.history[0]['date'], "%Y%m%d")
            end_date = datetime.strptime(self.history[-1]['date'], "%Y%m%d")
            days = (end_date - start_date).days
            years = max(days / 365.0, 0.001)  # 避免除以零
            
            annualized_return = (1 + current_return) ** (1 / years) - 1
        else:
            annualized_return = 0.0
        
        return current_return * 100, annualized_return * 100  # 转换为百分比

    def show_status(self, current_price=None):
        """显示当前持仓状态（含收益率）"""
        print("\n📊 当前状态")
        print(f"现金: ¥{self.cash:.2f}")
        print(f"持股: {self.shares}股")
        
        if self.shares > 0:
            avg_price = sum(
                t['price'] * t['quantity'] for t in self.history 
                if t['action'] == 'buy'
            ) / sum(
                t['quantity'] for t in self.history 
                if t['action'] == 'buy'
            )
            print(f"平均成本: ¥{avg_price:.2f}/股")
            
            # 显示当前收益率（如果提供当前股价）
            if current_price is not None:
                current_ret, annual_ret = self.calculate_returns(current_price)
                print(f"当前收益率: {current_ret:.2f}%")
                print(f"年化收益率: {annual_ret:.2f}%")
        
        print("-" * 30)

    def show_returns(self):
        """显示详细的收益率报告"""
        if not self.history:
            print("暂无交易记录，无法计算收益率")
            return
        
        # 获取最近一次交易价格作为当前股价
        current_price = self.history[-1]['price']
        current_ret, annual_ret = self.calculate_returns(current_price)
        
        print("\n📈 收益率报告")
        print(f"初始资金: ¥{self.initial_cash:.2f}")
        print(f"当前总价值: ¥{self.cash + (self.shares * current_price):.2f}")
        print(f"当前收益率: {current_ret:.2f}%")
        print(f"年化收益率: {annual_ret:.2f}%")
        
        # 显示时间跨度（如果有多笔交易）
        if len(self.history) >= 2:
            start_date = self.history[0]['date']
            end_date = self.history[-1]['date']
            print(f"投资周期: {start_date} 至 {end_date}")
        
        print("-" * 50)

    def show_history(self):
        """显示交易历史"""
        if not self.history:
            print("暂无交易记录")
            return
            
        print("\n📜 交易历史")
        print("{:<12} {:<6} {:<8} {:<6} {:<10} {:<6}".format(
            "日期", "操作", "价格", "数量", "现金", "持股"))
        for trade in self.history:
            print("{date:<12} {action:<6} ¥{price:<7.2f} {quantity:<6} ¥{cash:<9.2f} {shares:<6}".format(**trade))
        print("-" * 50)

    def plot_portfolio_value(self):
        """绘制总资产随时间变化图（修正计算逻辑）"""
        if not self.history:
            print("暂无交易记录，无法绘图")
            return

        dates = []
        values = []

        for i, trade in enumerate(self.history):
            date = datetime.strptime(trade['date'], "%Y%m%d")
            cash = trade['cash']
            shares = trade['shares']
            price = trade['price']  # 注意：这里是当时的买入/卖出价

            # ✅ 正确计算总资产，不重复使用 price
            total_value = cash + shares * price

            dates.append(date)
            values.append(total_value)

        # 只显示 4 个平均分布的日期作为 x 轴标签
        total_points = len(dates)
        if total_points <= 4:
            xticks = dates
        else:
            indices = [int(total_points * i / 3) for i in range(4)]
            xticks = [dates[i] for i in indices]

        plt.figure(figsize=(10, 5))
        plt.plot(dates, values, marker='o', linestyle='-', color='green')
        plt.title("📈 Total Assets")
        plt.xlabel("date")
        plt.ylabel("RMB")
        plt.grid(True)

        plt.xticks(xticks, [d.strftime("%Y-%m-%d") for d in xticks], rotation=30)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    calculator = InteractiveStockCalculator(initial_cash=100000)
    try:
        calculator.run()
    except KeyboardInterrupt:
        print("\n强制退出程序")
        sys.exit(0)