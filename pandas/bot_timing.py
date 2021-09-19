import yfinance as yf
from pprint import pprint
import time
from datetime import datetime

# время
def __now():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

for i in range(10):
    cur = yf.Ticker("BTC-USD")
    print(f"{__now()} Текущая цена: {cur.info['regularMarketPrice']:.3f}")
    time.sleep(10)

#pprint(cur.info)
#print(cur.info['open'], cur.info['regularMarketPrice'])