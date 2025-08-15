import sys
from datetime import datetime

class InteractiveStockCalculator:
    def __init__(self, initial_cash=100000.0):
        self.cash = initial_cash
        self.shares = 0
        self.history = []
        
    def run(self):
        print("""
        🚀 股票交易计算器（交互模式）
        ----------------------------
        输入格式: <日期> <价格> <数量> <操作(buy/sell)>
        示例: 2023-08-01 150.5 100 buy
        输入 'q' 退出 | 'h' 查看历史 | 'c' 当前状态
        """)
        
        while True:
            try:
                user_input = input(">>> ").strip()
                
                # 退出命令
                if user_input.lower() == 'q':
                    print("退出计算器")
                    break
                    
                # 查看历史
                elif user_input.lower() == 'h':
                    self.show_history()
                    continue
                    
                # 查看当前状态
                elif user_input.lower() == 'c':
                    self.show_status()
                    continue
                
                # 解析交易指令
                parts = user_input.split()
                if len(parts) != 4:
                    raise ValueError("输入格式错误，需要4个参数")
                
                date, price, quantity, action = parts
                self.execute_trade(date, float(price), int(quantity), action.lower())
                
            except ValueError as e:
                print(f"❌ 错误: {e}")
            except Exception as e:
                print(f"❌ 系统错误: {e}")

    def execute_trade(self, date, price, quantity, action):
        """执行交易并更新状态"""
        # 验证日期格式
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("日期格式应为 YYYY-MM-DD")
        
        # 验证操作类型
        if action not in ['buy', 'sell']:
            raise ValueError("操作必须是 'buy' 或 'sell'")
        
        # 执行交易
        if action == 'buy':
            cost = price * quantity
            if cost > self.cash:
                raise ValueError(f"资金不足，需要 ¥{cost:.2f}，当前可用 ¥{self.cash:.2f}")
            self.cash -= cost
            self.shares += quantity
        else:  # sell
            if quantity > self.shares:
                raise ValueError(f"持股不足，当前持有 {self.shares} 股")
            self.cash += price * quantity
            self.shares -= quantity
        
        # 记录交易
        trade = {
            'date': date,
            'price': price,
            'quantity': quantity,
            'action': action,
            'cash': self.cash,
            'shares': self.shares
        }
        self.history.append(trade)
        
        # 打印结果
        print(f"✅ {date} {action.upper()} {quantity}股 @¥{price:.2f}")
        self.show_status()

    def show_status(self):
        """显示当前持仓状态"""
        print("\n📊 当前状态")
        print(f"现金: ¥{self.cash:.2f}")
        print(f"持股: {self.shares}股")
        if self.shares > 0:
            avg_price = sum(t['price']*t['quantity'] for t in self.history if t['action']=='buy') / \
                       sum(t['quantity'] for t in self.history if t['action']=='buy')
            print(f"平均成本: ¥{avg_price:.2f}/股")
        print("-"*30)

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
        print("-"*50)

if __name__ == "__main__":
    calculator = InteractiveStockCalculator(initial_cash=50000)
    try:
        calculator.run()
    except KeyboardInterrupt:
        print("\n强制退出程序")
        sys.exit(0)