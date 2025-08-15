import json
import os
from datetime import datetime

class StockCalculator:
    def __init__(self, data_file="stock_data.json"):
        self.data_file = data_file
        self.remaining_amount = 0.0  # 剩余金额
        self.holding_shares = 0       # 持有股数
        self.transaction_history = []  # 交易历史
        
        # 加载已保存的数据
        self.load_data()
    
    def load_data(self):
        """从文件加载之前的计算数据"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.remaining_amount = float(data.get('remaining_amount', 0.0))
                    self.holding_shares = int(data.get('holding_shares', 0))
                    self.transaction_history = data.get('transaction_history', [])
                print("成功加载历史数据")
            except Exception as e:
                print(f"加载数据失败: {e}，将从头开始计算")
    
    def save_data(self):
        """保存当前数据到文件"""
        try:
            data = {
                'remaining_amount': self.remaining_amount,
                'holding_shares': self.holding_shares,
                'transaction_history': self.transaction_history
            }
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存数据失败: {e}")
    
    def validate_date(self, date_str):
        """验证日期格式是否正确 (YYYY-MM-DD)"""
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False
    
    def get_valid_price(self):
        """获取并验证每股价格输入"""
        while True:
            price_input = input("请输入当前每股价格 (元): ").strip()
            try:
                price = float(price_input)
                if price > 0:
                    return price
                else:
                    print("价格必须大于0，请重新输入")
            except ValueError:
                print("输入无效，请输入有效的数字")
    
    def get_valid_shares(self):
        """获取并验证操作份数输入"""
        while True:
            shares_input = input("请输入操作份数 (股，正数买入，负数卖出): ").strip()
            try:
                shares = int(shares_input)
                # 检查卖出是否超过持有数量
                if shares < 0 and abs(shares) > self.holding_shares:
                    print(f"卖出失败，当前仅持有{self.holding_shares}股，无法卖出{abs(shares)}股")
                    print("请重新输入操作份数")
                else:
                    return shares
            except ValueError:
                print("输入无效，请输入有效的整数")
    
    def get_valid_date(self):
        """获取并验证日期输入"""
        while True:
            date = input("请输入日期 (YYYY-MM-DD): ").strip()
            if self.validate_date(date):
                return date
            else:
                print("日期格式错误，请使用YYYY-MM-DD格式（例如：2023-10-01）")
    
    def execute_transaction(self):
        """执行买卖操作（分步获取输入并验证）"""
        # 分步获取并验证输入
        date = self.get_valid_date()
        price = self.get_valid_price()
        shares = self.get_valid_shares()
        
        # 计算交易金额 (买入为负，卖出为正)
        transaction_amount = -shares * price
        
        # 更新资产状态
        self.remaining_amount += transaction_amount
        self.holding_shares += shares
        
        # 记录交易历史
        self.transaction_history.append({
            'date': date,
            'price': price,
            'shares': shares,
            'amount': transaction_amount
        })
        
        # 保存数据
        self.save_data()
        return True
    
    def show_status(self):
        """显示当前状态"""
        print("\n===== 当前状态 =====")
        print(f"剩余金额: {self.remaining_amount:.2f} 元")
        print(f"持有股数: {self.holding_shares} 股")
        print("====================")
    
    def show_history(self):
        """显示交易历史"""
        if not self.transaction_history:
            print("暂无交易历史")
            return
            
        print("\n===== 交易历史 =====")
        print(f"{'日期':<12} {'价格(元)':<10} {'操作份数':<10} {'金额(元)':<10}")
        print("-" * 50)
        for trans in self.transaction_history:
            shares_str = f"+{trans['shares']}" if trans['shares'] > 0 else f"{trans['shares']}"
            amount_str = f"+{trans['amount']:.2f}" if trans['amount'] > 0 else f"{trans['amount']:.2f}"
            print(f"{trans['date']:<12} {trans['price']:<10.2f} {shares_str:<10} {amount_str:<10}")
        print("-" * 50)

def main():
    calculator = StockCalculator()
    print("欢迎使用股票买卖计算器")
    print("提示: 操作份数为正数表示买入，负数表示卖出")
    calculator.show_status()
    
    while True:
        print("\n请选择操作:")
        print("1. 执行买卖操作")
        print("2. 查看当前状态")
        print("3. 查看交易历史")
        print("4. 退出程序")
        
        choice = input("请输入选项 (1-4): ")
        
        if choice == '1':
            if calculator.execute_transaction():
                print("操作成功!")
                calculator.show_status()
        
        elif choice == '2':
            calculator.show_status()
        
        elif choice == '3':
            calculator.show_history()
        
        elif choice == '4':
            print("感谢使用，再见!")
            break
        
        else:
            print("无效选项，请重新输入1-4之间的数字")

if __name__ == "__main__":
    main()
    