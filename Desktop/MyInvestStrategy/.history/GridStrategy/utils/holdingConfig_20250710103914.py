from datetime import datetime

now = datetime.now()
now_date = now.strftime("%Y%m%d")

stock_codes = ['600519', '000001', '002415', '300750', 'cash']
holdings = [1500, 1000, 300, 500, 200]