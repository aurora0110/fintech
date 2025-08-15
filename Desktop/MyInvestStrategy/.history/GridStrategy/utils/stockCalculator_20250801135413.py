import datetime
from typing import List, Dict

class StockPortfolioCalculator:
    def __init__(self, initial_cash: float = 100000.0):
        """
        初始化投资组合计算器
        :param initial_cash: 初始资金（默认10万元）
        """
        self.cash = initial_cash
        self.shares_held = 0  # 当前持有股数
        self.transaction_history: List[Dict] = []  # 交易历史记录
        self.portfolio_value_history = []  # 投资组合价值历史

    def execute_trade(self, date: str, price: float, quantity: int, action: str):
        """
        执行交易操作
        :param date: 交易日期（格式：YYYY-MM-DD）
        :param price: 每股价格
        :param quantity: 操作份数（必须是整数）
        :param action: 操作类型（'buy' 或 'sell'）
        """
        # 验证输入
        if action.lower() not in ['buy', 'sell']:
            raise ValueError("操作类型必须是 'buy' 或 'sell'")
        if quantity <= 0:
            raise ValueError("操作份数必须是正整数")
        if price <= 0:
            raise ValueError("股价必须大于0")

        total_cost = price * quantity

        if action.lower() == 'buy':
            if total_cost > self.cash:
                raise ValueError("资金不足，无法购买")
            self.cash -= total_cost
            self.shares_held += quantity
        else:  # sell
            if quantity > self.shares_held:
                raise ValueError("持有股数不足，无法卖出")
            self.cash += total_cost
            self.shares_held -= quantity

        # 记录交易
        transaction = {
            'date': date,
            'price': price,
            'quantity': quantity,
            'action': action,
            'remaining_cash': self.cash,
            'shares_held': self.shares_held
        }
        self.transaction_history.append(transaction)
        self.portfolio_value_history.append({
            'date': date,
            'value': self.cash + (self.shares_held * price)
        })

        return transaction

    def get_current_status(self, current_price: float = None):
        """
        获取当前投资组合状态
        :param current_price: 当前股价（用于计算总价值）
        :return: 当前状态字典
        """
        portfolio_value = self.cash
        if current_price and self.shares_held > 0:
            portfolio_value += self.shares_held * current_price

        return {
            'remaining_cash': round(self.cash, 2),
            'shares_held': self.shares_held,
            'portfolio_value': round(portfolio_value, 2) if current_price else None
        }

    def print_transaction_history(self):
        """打印交易历史"""
        print("\n交易历史记录：")
        print("{:<12} {:<8} {:<10} {:<8} {:<12} {:<10}".format(
            "日期", "操作", "价格", "数量", "剩余现金", "持有股数"))
        for tx in self.transaction_history:
            print("{date:<12} {action:<8} ¥{price:<9.2f} {quantity:<8} ¥{remaining_cash:<11.2f} {shares_held:<10}".format(
                **tx))

# 使用示例
if __name__ == "__main__":
    calculator = StockPortfolioCalculator(initial_cash=50000)
    
    # 示例交易
    trades = [
        ("2023-01-05", 150.50, 100, "buy"),
        ("2023-02-15", 165.75, 50, "buy"),
        ("2023-03-20", 180.25, 30, "sell"),
        ("2023-04-10", 175.30, 70, "sell")
    ]
    
    for date, price, qty, action in trades:
        try:
            result = calculator.execute_trade(date, price, qty, action)
            print(f"{date} {action.upper()} {qty}股 @¥{price:.2f} | 现金: ¥{result['remaining_cash']:.2f} | 股数: {result['shares_held']}")
        except ValueError as e:
            print(f"交易失败: {e}")
    
    # 获取当前状态（假设当前股价为185元）
    current_status = calculator.get_current_status(current_price=185)
    print("\n当前投资组合状态：")
    print(f"剩余现金: ¥{current_status['remaining_cash']:.2f}")
    print(f"持有股数: {current_status['shares_held']}")
    print(f"投资组合总价值: ¥{current_status['portfolio_value']:.2f}")
    
    # 打印交易历史
    calculator.print_transaction_history()