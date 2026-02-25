# J值小于13
J_threshold = 13
# 一致涨幅小于0.1
consensus_gain_threshold = 0.1 
# 一致振幅小于0.2
consensus_range_threshold = 0.2 
# 长阴短柱
price_range_threshold = 1.2 # 长阴线定义为比前一日价格实体大于等于1.2
svolume_day_threshold = 1 # 短柱定义为比前一日缩量（2日中成交量最低）
# 阳线反包
b1volume_day_threshold = 1 # 阳线反包前1日交易量
ma5 = 5 #回踩5日线
ma10 = 10 #回踩10日线
ma20 = 20 #回踩20日线
ma60 = 60 #回踩60日线
# 回踩黄线
# 回踩白线
# SB1是跌破之前平台的B1，放量上涨缩量下跌，然后横盘3天再来个下跌造成恐慌的B1，一般都会放量
sb1_days = 3