from datetime import datetime

now = datetime.now()
now_date = now.strftime("%Y%m%d")

stock_codes = ['600019','000858', '601628', '002074', '000969', '002537','600886','600795','601988','000776'
               ,'sz159980','sh520990','cash']
holdings = [14140, 12541, 12000, 9072, 9021,8440,7420,7125,5750,5076,4885,3554,9504.3]