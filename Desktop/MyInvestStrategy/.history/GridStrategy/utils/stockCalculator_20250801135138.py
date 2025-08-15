import json
import os
from datetime import datetime

class StockCalculator:
    def __init__(self):
        self.remaining_amount = 0.0
        self.holding_shares = 0
        self.history = []
        self.load_data()

    def load_data(self):
        """加载历史数据"""
        if os.path.exists("stock_data.json"):
            try:
                with open("stock_data.json", "r") as f:
                    data = json.load(f)
                    self.remaining_amount = float(data["amount"])
                    self.holding_shares = int(data["shares"])
                    self.history = data["history"]
                print("已加载历史数据")
            except:
                print("历史数据加载失败，将重新开始")

    def save_data(self):
        """保存当前数据"""
        data = {
            "amount": self.remaining_amount,
            "shares": self.holding_shares,
            "history": self.history
        }
        with open("stock_data.json", "w") as f:
            json.dump(data, f)

    def print_status(self):
        """显示当前状态"""
        print(f"\n当前剩余金额: {self.remaining_amount:.2f}元")
        print(f"当前持有股数: {self.holding_shares}股\n")

    def add_transaction(self):
        """添加交易记录"""
        # 输入日期
        while True:
            date = input("请输入日期(YYYY-MM-DD): ")
            try:
                datetime.strptime(date, "%Y-%m-%d")
                break
            except:
                print("日期格式错误，请重新输入")

        # 输入价格
        while True:
            try:
                price = float(input("请输入每股价格(元): "))
                if price > 0:
                    break
                else:
                    print("价格必须大于0，请重新输入")
            except:
                print("请输入有效的数字")

        # 输入数量
        while True:
            try:
                shares = int(input("请输入操作数量(股，正数买入，负数卖出): "))
                # 检查卖出是否合法
                if shares < 0 and abs(shares) > self.holding_shares:
                    print(f"无法卖出，当前仅持有{self.holding_shares}股")
                else:
                    break
            except:
                print("请输入有效的整数")

        # 计算并更新
        cost = price * abs(shares)
        if shares > 0:  # 买入
            self.remaining_amount -= cost
            self.holding_shares += shares
            print(f"买入成功，花费{cost:.2f}元")
        else:  # 卖出
            self.remaining_amount += cost
            self.holding_shares += shares  # shares是负数，相当于减去
            print(f"卖出成功，获得{cost:.2f}元")

        # 记录历史
        self.history.append({
            "date": date,
            "price": price,
            "shares": shares,
            "cost": cost
        })
        self.save_data()
        self.print_status()

    def show_history(self):
        """显示交易历史"""
        if not self.history:
            print("暂无交易记录")
            return
            
        print("\n交易历史:")
        print(f"{'日期':<12} {'价格':<8} {'数量':<8} {'金额':<8}")
        for item in self.history:
            typ = "买入" if item["shares"] > 0 else "卖出"
            print(f"{item['date']:<12} {item['price']:<8.2f} {item['shares']:<8} {item['cost']:<8.2f} {typ}")

def main():
    calc = StockCalculator()
    calc.print_status()
    
    while True:
        print("1. 进行交易")
        print("2. 查看历史")
        print("3. 退出")
        
        try:
            choice = int(input("请选择(1-3): "))
            if choice == 1:
                calc.add_transaction()
            elif choice == 2:
                calc.show_history()
            elif choice == 3:
                print("程序结束")
                break
            else:
                print("请输入1-3之间的数字")
        except:
            print("请输入有效的数字选项")

if __name__ == "__main__":
    main()
    