import yfinance as yf

#start_date = '2024-01-01'
#end_date = '2025-01-27'

#stock_data = yf.download('512980.SS', start=start_date, end=end_date)
#print(stock_data)

ticker = 'AAPL'
stock = yf.Ticker(ticker)
data = stock.history(period='1d')
print(data)
