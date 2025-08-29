import yfinance as yf

ticker = 'AAPL'
stock = yf.Ticker(ticker)
data = stock.history(period='1d')
print(data)


