import configparser
import tkinter as tk
from tkinter import ttk
from tkinter import W, N, E, S
from tkinter import messagebox as mb
import os
import telepot as tp
from telepot.loop import MessageLoop
import datetime
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
from pmdarima.arima import ndiffs
from sklearn.metrics import mean_squared_error
from pmdarima.metrics import smape
import numpy as np
from ta import volatility as vlt
from ta import add_volatility_ta
from ta import add_trend_ta
from ta import momentum as mm
import warnings
from dateutil.parser import parse

warnings.filterwarnings('ignore')

# Переменные
train_data = []
test_data = [] 
y_train = [] 
y_test = [] 
chat_id__   = ''
username__  = ''
token__     = ''
currlist__  = []
len_pb = 200

# зачитать параметры
def __read_set():
    global username__, chat_id__, token__, currlist__, conf
    
    username__ = conf['Telegram']['username']
    chat_id__ = conf['Telegram']['chatid']
    token__ = conf['Telegram']['token']
    currlist__ = conf['Yfinance']['currlist'].split()

# время
def __now():
    return datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')

# телеграм чат    
def __handle(msg):
    global row_n, chat_id__
    
    if msg['chat']['username'] == username__:
        if not chat_id__:
            conf['Telegram']['chatid'] = str(tp.glance(msg)[2])
            with open(os.path.join(loc, 'settings.ini'), 'w') as configfile:
                conf.write(configfile)
            
        chat_id__ = tp.glance(msg)[2]
    
        row_n += 1
        tel_txt.insert(row_n, '{} << {}\n'.format(__now(), msg['text']))
        if msg['text'].lower() == '/start':
            __bStart()
        elif msg['text'].lower() == '/stop':
            __bStop()
    
# запуск
def __bStart():
    global b_lbl
    if chat_id__:
        b_lbl['text'] = 'Запущен'
        b_lbl['fg'] = '#2ea116'
        tel_bot.sendMessage(chat_id__, 'Бот запущен')
    else:
        mb.showerror('Ошибка', 'Надо отправить боту команду СТАРТ')

# остановка
def __bStop():
    global b_lbl
    if chat_id__:
        b_lbl['text'] = 'Остановлен'
        b_lbl['fg'] = '#ff0000'
        tel_bot.sendMessage(chat_id__, 'Бот остановлен')
    else:
        mb.showerror('Ошибка', 'Надо отправить боту команду СТОП')

# маска условия для DataFrame
def mask(df, curr, lag_days):
    m = ((df.index >= max(df.index)-datetime.timedelta(days=lag_days)) & 
         (df.index <= max(df.index)) & 
         (df['ticker'] == curr))
    return m

# Анализ волатильности всех котировок
def analize():
    # стили и цвета
    txt_con.tag_config('title', font=("Verdana", 10, 'bold'))
    txt_con.tag_config('curr', font=("Verdana", 9, 'bold'))
    
    txt_con.delete("1.0","end")
    rown = 1.0
    rown = put_consol(txt_con, rown, f'Теханализ по всем валютным парам из списка за период 2 мес')
    txt_con.tag_add('title', rown-1, f'{int(rown-1)}.end')
    
    df_list = []
    i = 0
    pb_load['maximum'] = len(currlist__)
    for curr in currlist__:
        data = yf.download(tickers = curr, 
                           period = '2mo', 
                           interval = '1d', 
                           group_by = 'ticker')
        data['ticker'] = curr
        data = add_volatility_ta(data, high = 'High', low = 'Low', close = 'Close', fillna=False)
        data = add_trend_ta(data, high = 'High', low = 'Low', close = 'Close', fillna=False)
        data['ATRp'] = data['volatility_atr']/data['Close']*100
        data['BB_wprc'] = (data['volatility_bbh']-data['volatility_bbl'])/data['Close']*100
        data['RSI'] = mm.RSIIndicator(close = data['Close'], window = 14, fillna=False).rsi()
        
        df_list.append(data)         
        df = pd.concat(df_list)
        
        # вывод в консоль теханализ
        rown = put_consol(txt_con, rown, f'{curr}')
        txt_con.tag_add('curr', rown-1, f'{int(rown-1)}.end')
        
        rown = put_consol(txt_con, rown, f'   Volume')
        rown = put_consol(txt_con, rown, f'      Среднее за 30 дней: {df.loc[mask(df, curr, 30), "Volume"].mean()/1000000:.0f}m')
        rown = put_consol(txt_con, rown, f'      Среднее за 7 дней: {df.loc[mask(df, curr, 7), "Volume"].mean()/1000000:.0f}m')
        rown = put_consol(txt_con, rown, f'      Текущее: {df.loc[mask(df, curr, 0), "Volume"].mean()/1000000:.0f}m')
        
        rown = put_consol(txt_con, rown, f'   Spot close price')
        rown = put_consol(txt_con, rown, f'      Среднее за 30 дней: {df.loc[mask(df, curr, 30), "Close"].mean():.3f}')
        rown = put_consol(txt_con, rown, f'      Среднее за 7 дней: {df.loc[mask(df, curr, 7), "Close"].mean():.3f}')
        rown = put_consol(txt_con, rown, f'      Текущее: {df.loc[mask(df, curr, 0), "Close"].mean():.3f}')
        
        rown = put_consol(txt_con, rown, f'   Average True Range (ATR), %')
        rown = put_consol(txt_con, rown, f'      Среднее за 30 дней: {df.loc[mask(df, curr, 30), "ATRp"].mean():.3f}')
        rown = put_consol(txt_con, rown, f'      Среднее за 7 дней: {df.loc[mask(df, curr, 7), "ATRp"].mean():.3f}')
        rown = put_consol(txt_con, rown, f'      Текущее: {df.loc[mask(df, curr, 0), "ATRp"].mean():.3f}')
        
        rown = put_consol(txt_con, rown, f'   Relative Strength Index (RSI), %')
        rown = put_consol(txt_con, rown, f'      Среднее за 30 дней: {df.loc[mask(df, curr, 30), "RSI"].mean():.3f}')
        rown = put_consol(txt_con, rown, f'      Среднее за 7 дней: {df.loc[mask(df, curr, 7), "RSI"].mean():.3f}')
        rown = put_consol(txt_con, rown, f'      Текущее: {df.loc[mask(df, curr, 0), "RSI"].mean():.3f}')
        
        rown = put_consol(txt_con, rown, f'   Bollinger Bands (BB), SMA')
        rown = put_consol(txt_con, rown, f'      Среднее за 30 дней: {df.loc[mask(df, curr, 30), "volatility_bbm"].mean():.3f}')
        rown = put_consol(txt_con, rown, f'      Среднее за 7 дней: {df.loc[mask(df, curr, 7), "volatility_bbm"].mean():.3f}')
        rown = put_consol(txt_con, rown, f'      Текущее: {df.loc[mask(df, curr, 0), "volatility_bbm"].mean():.3f}')
        
        
        rown = put_consol(txt_con, rown, f'   Bollinger Bands (BB) width, %')
        rown = put_consol(txt_con, rown, f'      Среднее за 30 дней: {df.loc[mask(df, curr, 30), "BB_wprc"].mean():.3f}')
        rown = put_consol(txt_con, rown, f'      Среднее за 7 дней: {df.loc[mask(df, curr, 7), "BB_wprc"].mean():.3f}')
        rown = put_consol(txt_con, rown, f'      Текущее: {df.loc[mask(df, curr, 0), "BB_wprc"].mean():.3f}')
        
        rown = put_consol(txt_con, rown, f'   Commodity Channel Index [-100, 100]')
        rown = put_consol(txt_con, rown, f'      Среднее за 30 дней: {df.loc[mask(df, curr, 30), "trend_cci"].mean():.2f}')
        rown = put_consol(txt_con, rown, f'      Среднее за 7 дней: {df.loc[mask(df, curr, 7), "trend_cci"].mean():.2f}')
        rown = put_consol(txt_con, rown, f'      Текущее: {df.loc[mask(df, curr, 0), "trend_cci"].mean():.2f}')
        
        i = i+ 1
        pb_load['value'] = i
        wnd.update_idletasks()
    
    df = df.rename(columns={'volatility_atr': 'ATR', 
                            'volatility_bbl': 'BB_low', 
                            'volatility_bbm': 'BB_mid', 
                            'volatility_bbh': 'BB_high',
                            'trend_cci'     : 'CCI_trend'})
    df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'ticker', 
             'ATR', 'ATRp', 'BB_low', 'BB_mid', 'BB_high', 'BB_wprc', 'CCI_trend', 'RSI']].copy()
    
    df.to_csv(os.path.join(loc, 'analize_tick.csv'), sep=';')

# Путь к файлу котировок
def data_loc(ticker):
    return os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), ticker)

# Расчет скользящего среднего. n - размер окна
def sma(df, n):
    sma = df.rolling(window = n).mean() #min_periods = 1
    return pd.DataFrame(sma)

# Вывод строки в консоль
def put_consol(txt_con, rown, text):
    txt_con.insert(rown, f'{text}\n')
    return rown + 1
    
# Нарисуем график
def mplot():
    global test_data, y_test
    
    if len(test_data) == 0:
        return 0
    
    #plt.plot(train_data.index, y_train, color='blue', label='Тренировочный набор 80%')
    plt.plot(test_data.index, y_test, color='red', label='Актуальная цена')
    plt.title(f'Валюта: {cmb_pairs.get()}')
    plt.legend()
    plt.show()
    
    return 0

# Сколько купили    
def buy_val(pClose, price):
    return round(price / pClose, 8)
    
# За сколько продали    
def sell_val(pClose, cty):
    return cty * pClose

# Формирование сетки ордеров
def orders_calc(close, count, prc, price, mg):
    order_list = []
    
    for i in range(count):
        order = close - (close * (prc*(i+1)/100))
        amount = price + (price * (mg*i/100))
        order_list.append((order, amount))
        
    #print(close, order_list[0][0], order_list[0][1])
    return order_list

# загрузка котировок        
def __bYload(): 
    global train_data, test_data, y_train, y_test
    df_list = []
    data = yf.download(tickers = cmb_pairs.get(), 
                       period = cmb_per.get(), 
                       interval = cmb_int.get(), 
                       group_by = 'ticker')
                           
    data['ticker'] = cmb_pairs.get()
    
    df_list.append(data)         
    df = pd.concat(df_list)    
    df.to_csv(os.path.join(loc, 'ticker.csv'), sep=';')
    
    df = read_csv(data_loc('ticker.csv'), sep= ';')
    
    txt_con.delete("1.0","end")
    rown = 1.0
    
    # Алгоритм Martingale
    rown = simulate_MG(df, rown)
    
    # Алгоритм SMA_4_10
    test_data, y_test, rown = simulate_SMA(df, rown, 4, 10)
    rown = put_consol(txt_con, rown, '')
    
    # Алгоритм SMA_20_50
    test_data, y_test, rown = simulate_SMA(df, rown, 20, 50)

# Симуляция по алгоритму Martingale. Вход по осциллятору 40 < RSI < 70, направление LONG 
def simulate_MG(df, rown):
    df_new = df[['Datetime', 'Close']].copy()
    #print(df_new['Datetime'])
    #df_new.set_index('Datetime')
    df_new['RSI'] = mm.RSIIndicator(close = df_new['Close'], window = 14, fillna=False).rsi()
    
    order_list = []
    amount_list = []
    buy_list = []
    total_sell = []
    total_buy = []
    total_comm = []
    start_date = None
    cycle_n = 0
    order_n = 0
    last_close = 0
    cap_sum = 0
    count_buy = 0
    count_sell = 0
    trail_stop = 0
    txt_ord.delete("1.0","end")
    rown1 = 1.0
    
    txt_con.tag_config('curr', font=("Verdana", 9, 'bold'))
    txt_con.tag_config('profit', foreground='green')
    txt_con.tag_config('loss', foreground='red')
    
    rown = put_consol(txt_con, rown, f'{cmb_pairs.get()}')
    txt_con.tag_add('curr', rown-1, f'{int(rown-1)}.end')
    rown = put_consol(txt_con, rown, f'Результат тестового набора по стратегии Martingale:')
    
    pb_load['maximum'] = len(df_new.index)
    for i in df_new.index:
        # определим точку входа и сформируем сетку ордеров
        if (df_new.loc[i, 'RSI'] > 40 and 
         df_new.loc[i, 'RSI'] < 70 and 
         len(order_list) == 0 and
         len(amount_list) == 0):
            start_date = parse(df_new.loc[i, 'Datetime'])
            
            order_list = orders_calc(df_new.loc[i, 'Close'], 
                                     int(cmb_count.get()), 
                                     float(cmb_step.get()),
                                     float(entr_price.get()),
                                     int(cmb_mart.get()))
            
            cycle_n = cycle_n + 1
            txt_ord.tag_config('curr', font=("Verdana", 9, 'bold'))
            rown1 = put_consol(txt_ord, rown1, f'CurrPrice = {df_new.loc[i, "Close"]:.3f}')
            txt_ord.tag_add('curr', rown1-1, f'{int(rown1-1)}.end')
            
            rown = put_consol(txt_con, rown, f'# цикла: {cycle_n} старт в: {start_date.strftime("%d.%m.%Y %H:%M:%S")}')
            
            cap_sum = 0
            for j in range(len(order_list)):
                rown1 = put_consol(txt_ord, rown1, f'  {j+1}: {order_list[j][0]:.3f} x {order_list[j][1]:.3f}')
                cap_sum = cap_sum + order_list[j][1]
            last_close = order_list[len(order_list)-1][0]
            trail_stop = 0
            
        # последовательно реализуем ордера
        if len(order_list) > 0:
            if df_new.loc[i, 'Close'] <= order_list[0][0]:
                order_n = order_n + 1
                rown = put_consol(txt_con, rown, f'  {order_n}) {order_list[0][0]:.3f} x {order_list[0][1]:.3f}')
                
                df_new.loc[i, 'Price'] = order_list[0][1]
                amount = buy_val(df_new.loc[i, 'Close'], order_list[0][1])
                df_new.loc[i, 'Amount'] = amount
                
                amount_list.append(amount)
                buy_list.append(order_list[0][1])
                
                comm = order_list[0][1]*float(cmb_comm.get())/100
                df_new.loc[i, 'Comm'] = comm
                total_comm.append(comm)
                
                #print(sum(amount_list), sum(buy_list), sum(buy_list)+(sum(buy_list)*float(cmb_prof.get())/100), sum(amount_list)*df_new.loc[i, 'Close'])
                order_list.pop(0)
        
        # зафиксируем профит + завершающий цикл ордер
        if len(amount_list) > 0 or (len(amount_list) > 0 and len(order_list) == 0):
            if sum(amount_list)*df_new.loc[i, 'Close'] >= sum(buy_list) + (sum(buy_list)*float(cmb_prof.get())/100):
                if trail_stop == 0 and df_new.loc[i, 'RSI'] < 70 and float(cmb_trail.get()) > 0:
                    trail_stop = df_new.loc[i, 'Close']
                    df_new.loc[i, 'Trail'] = 0
                elif (trail_stop > 0 
                  and df_new.loc[i, 'Close'] >= trail_stop - (trail_stop*float(cmb_trail.get())/100) 
                  and float(cmb_trail.get()) > 0):
                    #print(df_new.loc[i, 'Close'], trail_stop, trail_stop - (trail_stop*float(cmb_trail.get())/100))
                    trail_stop = df_new.loc[i, 'Close']
                    df_new.loc[i, 'Trail'] = 1
                else:
                    sell_price = sum(amount_list)*df_new.loc[i, 'Close']
                    df_new.loc[i, 'Profit'] = sell_price
                    rown = put_consol(txt_con, rown, f'Покупка: {sum(buy_list):.3f}'
                                                     f' Продажа: {sell_price:.3f}'
                                                     f' Профит: {sell_price - sum(buy_list):.3f}')
                    txt_con.tag_add('profit', rown-1, f'{int(rown-1)}.end')
                    rown = put_consol(txt_con, rown, f'')
                
                    total_buy.append(sum(buy_list))
                    total_sell.append(sell_price)
                
                    comm = sell_price*float(cmb_comm.get())/100
                    df_new.loc[i, 'Comm'] = comm
                    total_comm.append(comm)
                
                    count_buy = count_buy + len(amount_list)
                    count_sell = count_sell + 1
                    order_list = []
                    amount_list = []
                    buy_list = []
                    order_n = 0
                    last_close = 0
                    trail_stop = 0
        
        # Выход из цикла, если сработал stop loss после реализации последнего ордера
        if len(amount_list) > 0 and len(order_list) == 0:
            if df_new.loc[i, 'Close'] < last_close - (last_close*float(cmb_stop.get())/100):
                sell_price = sum(amount_list)*df_new.loc[i, 'Close']
                df_new.loc[i, 'Loss'] = sell_price
                rown = put_consol(txt_con, rown, f'Покупка: {sum(buy_list):.3f}'
                                                 f' Продажа: {sell_price:.3f}'
                                                 f' Потеря: {sell_price - sum(buy_list):.3f}')
                txt_con.tag_add('loss', rown-1, f'{int(rown-1)}.end')
                rown = put_consol(txt_con, rown, f'')
                
                total_buy.append(sum(buy_list))
                total_sell.append(sell_price)
                
                comm = sell_price*float(cmb_comm.get())/100
                df_new.loc[i, 'Comm'] = comm
                total_comm.append(comm)
                
                count_buy = count_buy + len(amount_list)
                count_sell = count_sell + 1
                order_list = []
                amount_list = []
                buy_list = []
                order_n = 0
                last_close = 0
                
        # Выход из цикла по аномально низкому RSI (< 30)
        """if len(amount_list) > 6 and df_new.loc[i, 'RSI'] < 30:
            sell_price = sum(amount_list)*df_new.loc[i, 'Close']
            df_new.loc[i, 'Loss'] = sell_price
            rown = put_consol(txt_con, rown, f'Покупка: {sum(buy_list):.3f}'
                                             f' Продажа: {sell_price:.3f}'
                                             f' Потеря: {sell_price - sum(buy_list):.3f}')
            txt_con.tag_add('loss', rown-1, f'{int(rown-1)}.end')
            rown = put_consol(txt_con, rown, f'')
                
            total_buy.append(sum(buy_list))
            total_sell.append(sell_price)
                
            comm = sell_price*float(cmb_comm.get())/100
            df_new.loc[i, 'Comm'] = comm
            total_comm.append(comm)
                
            count_buy = count_buy + len(amount_list)
            count_sell = count_sell + 1
            order_list = []
            amount_list = []
            buy_list = []
            order_n = 0
            last_close = 0"""
                
        # перезапуск алгоритма по таймеру
        if len(order_list) == int(cmb_count.get()):
            cur_date = parse(df_new.loc[i, 'Datetime'])
            diff_min = divmod((cur_date - start_date).total_seconds(), 60)[0]
            
            if diff_min >= int(cmb_rest.get()):
                rown = put_consol(txt_con, rown, f'------ Перезапуск в {cur_date.strftime("%d.%m.%Y %H:%M:%S")}')
                
                order_list = []
                amount_list = []
                buy_list = []
                order_n = 0
                last_close = 0
    
        pb_load['value'] = i
        wnd.update_idletasks()
        
    # Продаем что осталось
    if len(amount_list) > 0:
        i = max(df_new.index)
        sell_price = sum(amount_list)*df_new.loc[i, 'Close']
        df_new.loc[i, 'Profit'] = sell_price
        rown = put_consol(txt_con, rown, f'Покупка: {sum(buy_list):.3f}'
                                         f' Продажа: {sell_price:.3f}'
                                         f' Профит: {sell_price - sum(buy_list):.3f}')
        if sell_price - sum(buy_list) >= 0:
            txt_con.tag_add('profit', rown-1, f'{int(rown-1)}.end')
        else:
            txt_con.tag_add('loss', rown-1, f'{int(rown-1)}.end')
        rown = put_consol(txt_con, rown, f'')
        
        total_buy.append(sum(buy_list))
        total_sell.append(sell_price)
        
        comm = sell_price*float(cmb_comm.get())/100
        df_new.loc[i, 'Comm'] = comm
        total_comm.append(comm)
        count_buy = count_buy + len(amount_list)
        count_sell = count_sell + 1
    
    rown = put_consol(txt_con, rown, f'Количество покупок: {count_buy} Количество продаж: {count_sell}')
    rown = put_consol(txt_con, rown, f'Сумма покупок: {sum(total_buy):.3f} Сумма продаж: {sum(total_sell):.3f} Комиссия: {sum(total_comm):.3f}')
    rown = put_consol(txt_con, rown, f'Результат: {sum(total_sell) - sum(total_buy) - sum(total_comm):.3f}')
    rown = put_consol(txt_con, rown, f'Суммарный депозит: {cap_sum:.2f} (профит {(sum(total_sell) - sum(total_buy) - sum(total_comm))/cap_sum*100:.1f}%)')
    rown = put_consol(txt_con, rown, f'')
    
    df_new.to_csv(data_loc('ticker_MG.csv'), sep= ';', index=False)
    
    #print(df_new.index.values)    
    return rown
    
# Симуляция по алгоритму простого скользящего среднего (SMA_4_10, SMA_20_50)    
def simulate_SMA(df, rown, __low, __high):
    global txt_con
    
    #train_len = int(len(df) * 0.80)
    #train_data, test_data = df[:train_len].copy(), df[train_len:].copy()
    #test_data['sma_4'] = sma(test_data['Close'], 4)
    #test_data['sma_10'] = sma(test_data['Close'], 10)
    #y_train = train_data['Close'].values
    #y_test = test_data['Close'].values
    
    test_data = df.copy()
    test_data[f'sma_{__low}'] = sma(test_data['Close'], __low)
    test_data[f'sma_{__high}'] = sma(test_data['Close'], __high)
    y_test = test_data['Close'].values
    df_new = test_data[['Datetime', f'sma_{__low}', f'sma_{__high}', 'Close']].copy()
    
    signal = 0
    buy_sum = 0
    sell_sum = 0
    buy_price = []
    sell_price = []
    df_new['Signal_SMA'] = 0
    df_new['Buy_price_SMA'] = 0
    df_new['Sell_price_SMA'] = 0
    
    txt_con.tag_config('curr', font=("Verdana", 9, 'bold'))
    rown = put_consol(txt_con, rown, f'{cmb_pairs.get()}')
    txt_con.tag_add('curr', rown-1, f'{int(rown-1)}.end')
    rown = put_consol(txt_con, rown, f'Результат тестового набора по стратегии SMA_{__low}_{__high}:')
    
    pb_load['maximum'] = len(test_data.index)
    for i in test_data.index:
        if df_new.loc[i, f'sma_{__low}'] > df_new.loc[i, f'sma_{__high}']:
            if signal != 1:
                if len(buy_price) <= len(sell_price):
                    #price = df_new.loc[i, 'Close']
                    price = __price.get()
                    buy_sum = buy_val(df_new.loc[i, 'Close'], price)
                    #rown = put_consol(txt_con, rown, f'     Покупка единиц валюты: {buy_sum:.3f} Сумма покупки: {price}$')
                    
                    df_new.loc[i, 'Buy_price_SMA'] = price
                    buy_price.append(price)
                    df_new.loc[i, 'Sell_price_SMA'] = np.nan
                    
                df_new.loc[i, 'Signal_SMA'] = 1                   
                signal = 1
            else:
                df_new.loc[i, 'Buy_price_SMA'] = np.nan
                df_new.loc[i, 'Sell_price_SMA'] = np.nan
                df_new.loc[i, 'Signal_SMA'] = 0
        elif df_new.loc[i, f'sma_{__high}'] > df_new.loc[i, f'sma_{__low}']:
            if signal != -1:
                if len(buy_price) > len(sell_price):
                    #price = df_new.loc[i, 'Close']
                    price = __price.get()
                    sell_sum = sell_val(df_new.loc[i, 'Close'], buy_sum)
                    #rown = put_consol(txt_con, rown, f'     Продажа единиц валюты: {buy_sum:.3f} Сумма продажи: {sell_sum:.3f}$')
                    
                    df_new.loc[i, 'Buy_price_SMA'] = np.nan
                    df_new.loc[i, 'Sell_price_SMA'] = sell_sum
                    sell_price.append(sell_sum)
                   
                df_new.loc[i, 'Signal_SMA'] = -1                    
                signal = -1
            else:
                df_new.loc[i, 'Buy_price_SMA'] = np.nan
                df_new.loc[i, 'Sell_price_SMA'] = np.nan
                df_new.loc[i, 'Signal_SMA'] = 0
        else:
            df_new.loc[i, 'Buy_price_SMA'] = np.nan
            df_new.loc[i, 'Sell_price_SMA'] = np.nan
            df_new.loc[i, 'Signal_SMA'] = 0

        pb_load['value'] = i
        wnd.update_idletasks()

    # Продаем, если что-то осталось
    if len(buy_price) > len(sell_price):
        sell_sum = sell_val(df_new.loc[max(test_data.index), 'Close'], buy_sum)
        #rown = put_consol(txt_con, rown, f'     Продажа единиц валюты: {buy_sum:.3f} Сумма продажи: {sell_sum:.3f}$')
        
        #price = df_new.loc[max(test_data.index), 'Close']
        df_new.loc[max(test_data.index), 'Sell_price'] = sell_sum
        sell_price.append(sell_sum)
        signal = 0    
    
    df_new.to_csv(data_loc(f'ticker_sma{__low}_{__high}.csv'), sep= ';')
    
    rown = put_consol(txt_con, rown, f'Покупок: {len(buy_price)} Продаж: {len(sell_price)}')
    rown = put_consol(txt_con, rown, f'Сумма покупок: {sum(buy_price):.3f} Сумма продаж: {sum(sell_price):.3f}')
    rown = put_consol(txt_con, rown, f'Результат: {sum(sell_price) - sum(buy_price):.3f}')
    #rown = put_consol(txt_con, rown, f'{__price.get()}')
    
    return test_data, y_test, rown

# Инициализация окна
wnd = tk.Tk()
wnd.geometry('815x680')

loc = os.path.realpath(os.path.join(os.getcwd(), 
                       os.path.dirname(__file__)))
conf = configparser.ConfigParser()
conf.read(os.path.join(loc, 'settings.ini'))
__read_set()

if username__ == '':
    wnd.title('PYTRADE <Надо ввести username для Telegram в settings.ini>')
else:
    wnd.title('PYTRADE @'+ username__)
#print(conf['Telegram']['token'])

# Статус программы
b_lbl = tk.Label(wnd, text='Остановлен', 
                      font=("Verdana", 20, 'bold'), 
                      fg='#ff0000')  
b_lbl.grid(row=0, column=0, sticky=W, pady=4, padx=5)

# Виджет телеги
tel_frm = tk.Frame(wnd, 
                   width = 400, 
                   height = 100, 
                   relief = tk.RAISED, 
                   borderwidth = 1)
tel_frm.grid(row=1, column = 0, padx=5, pady=5, columnspan=2, rowspan=8, sticky=W)

tel_txt = tk.Text(tel_frm, width=50, height=8)
tel_txt.grid(row=0, column=0, padx=2, pady=2)

scrollb = tk.Scrollbar(tel_frm, command=tel_txt.yview)
scrollb.grid(row=0, column=1, sticky='nsew')
tel_txt['yscrollcommand'] = scrollb.set

tel_token = token__
tel_bot = tp.Bot(tel_token)

tel_txt.insert(1.0, 'Bot name:   {}\nstart time: {}\n'.format(tel_bot.getMe()['username'], __now()))
tel_txt.tag_add('title', 1.0, '3.end')
tel_txt.tag_config('title', font=("Verdana", 10, 'bold'))
tel_txt.insert(3.0, '---------------------------------\n')
row_n = 3.0

# кнопки старт/стоп
btn_start = tk.Button(wnd, text ="Старт", height = 1, width = 10, command = lambda: __bStart())
btn_start.grid(row=1, column=3, sticky=E, pady=4, padx=5)
btn_stop = tk.Button(wnd, text ="Стоп", height = 1, width = 10, command = lambda: __bStop())
btn_stop.grid(row=1, column=4, sticky=W, pady=4, padx=5)
btn_start['state'] = tk.DISABLED
btn_stop['state'] = tk.DISABLED

# блок 2 - настройки алгоритма
# кнопка загрузки котировок (stock) пока yfinance
b_lbl1 = tk.Label(wnd, text='Симуляция торгов на исторических данных YAHOO! FINANCE.', font=("Verdana", 8, 'bold'))
b_lbl1.grid(row=9, column=0, sticky=W, padx=5)

# Виджет элементов управления
frm_par = tk.Frame(wnd, 
                   width = 790, 
                   height = 60, 
                   relief = tk.RAISED, 
                   borderwidth = 1)
frm_par.grid(row=10, column = 0, padx=5, pady=5, columnspan=10, rowspan=3, sticky=W)

# строка 1 - котировки
# комбобоксы для загрузки котировок
lbl_pairs = tk.Label(frm_par, text='Котировки')
lbl_pairs.grid(row=0, column=0, sticky=E, pady=4, padx=5)

l_pairs = currlist__
cmb_pairs = ttk.Combobox(frm_par, values=l_pairs, state='readonly')
cmb_pairs.grid(row=0, column=1, sticky=W, pady=4, padx=5)
cmb_pairs.current(0)

lbl_per = tk.Label(frm_par, text='Период')
lbl_per.grid(row=0, column=2, sticky=E, pady=4, padx=5)
    
l_per = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
cmb_per = ttk.Combobox(frm_par, values=l_per, state='readonly')
cmb_per.grid(row=0, column=3, sticky=W, pady=4, padx=5)
cmb_per.current(0)
#cmb_per.bind('<<ComboboxSelected>>', lambda x : chg_cmb_per())

lbl_int = tk.Label(frm_par, text='Интервал')
lbl_int.grid(row=0, column=4, sticky=E, pady=4, padx=5)

l_int = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
cmb_int = ttk.Combobox(frm_par, values=l_int, state='readonly')
cmb_int.grid(row=0, column=5, sticky=W, pady=4, padx=5)
cmb_int.current(0)

# строка 2 - настройки
lbl_sma = tk.Label(frm_par, text='Алгоритм:')
lbl_sma.grid(row=2, column=0, sticky=E, pady=4, padx=5)
__alg = ['Martingale LONG', 'SMA_4_10', 'SMA_20_50']
cmb_sma = ttk.Combobox(frm_par, values=__alg, state='readonly')
cmb_sma.grid(row=2, column=1, sticky=W, pady=4, padx=5)
cmb_sma.current(0)

lbl_price = tk.Label(frm_par, text='1й ордер, $:')
lbl_price.grid(row=2, column=2, sticky=E, pady=4, padx=5)
__price = tk.IntVar()
__price.set(100)
entr_price = tk.Entry(frm_par, textvariable=__price)
entr_price.grid(row=2, column=3, sticky=W, pady=4, padx=5)

lbl_step = tk.Label(frm_par, text='Отступ ордера, %:')
lbl_step.grid(row=2, column=4, sticky=E, pady=4, padx=5)
__step = [0.5, 1, 2, 3, 4, 5]
cmb_step = ttk.Combobox(frm_par, values=__step, state='readonly')
cmb_step.grid(row=2, column=5, sticky=W, pady=4, padx=5)
cmb_step.current(0)

# строка 3 - настройки
lbl_count = tk.Label(frm_par, text='Количество ордеров:')
lbl_count.grid(row=3, column=0, sticky=E, pady=4, padx=5)
__count = [5, 7, 10, 15, 20]
cmb_count = ttk.Combobox(frm_par, values=__count, state='readonly')
cmb_count.grid(row=3, column=1, sticky=W, pady=4, padx=5)
cmb_count.current(2)

lbl_mart = tk.Label(frm_par, text='Matringale, %:')
lbl_mart.grid(row=3, column=2, sticky=E, pady=4, padx=5)
__mart = [5, 7, 10, 15, 20, 25]
cmb_mart = ttk.Combobox(frm_par, values=__mart, state='readonly')
cmb_mart.grid(row=3, column=3, sticky=W, pady=4, padx=5)
cmb_mart.current(3)

lbl_prof = tk.Label(frm_par, text='TakeProfit, %:')
lbl_prof.grid(row=3, column=4, sticky=E, pady=4, padx=5)
__prof = [0.7, 1, 2, 3, 5]
cmb_prof = ttk.Combobox(frm_par, values=__prof, state='readonly')
cmb_prof.grid(row=3, column=5, sticky=W, pady=4, padx=5)
cmb_prof.current(1)

# строка 4 - настройки
lbl_stop = tk.Label(frm_par, text='StopLoss, %:')
lbl_stop.grid(row=4, column=0, sticky=E, pady=4, padx=5)
__stop = [3, 5, 7, 10]
cmb_stop = ttk.Combobox(frm_par, values=__stop, state='readonly')
cmb_stop.grid(row=4, column=1, sticky=W, pady=4, padx=5)
cmb_stop.current(1)

lbl_comm = tk.Label(frm_par, text='Комиссия, %:')
lbl_comm.grid(row=4, column=2, sticky=E, pady=4, padx=5)
__comm = [0.075, 0.1]
cmb_comm = ttk.Combobox(frm_par, values=__comm, state='readonly')
cmb_comm.grid(row=4, column=3, sticky=W, pady=4, padx=5)
cmb_comm.current(0)

lbl_rest = tk.Label(frm_par, text='Перезапуск, мин:')
lbl_rest.grid(row=4, column=4, sticky=E, pady=4, padx=5)
__rest = [60, 120, 180, 240, 300]
cmb_rest = ttk.Combobox(frm_par, values=__rest, state='readonly')
cmb_rest.grid(row=4, column=5, sticky=W, pady=4, padx=5)
cmb_rest.current(0)

# строка 5 - настройки
lbl_trail = tk.Label(frm_par, text='TrailingStop, %:')
lbl_trail.grid(row=5, column=0, sticky=E, pady=4, padx=5)
__trail = [0, 0.1, 0.3, 0.5, 1]
cmb_trail = ttk.Combobox(frm_par, values=__trail, state='readonly')
cmb_trail.grid(row=5, column=1, sticky=W, pady=4, padx=5)
cmb_trail.current(1)

# строка 6 - кнопки
btn_analize = tk.Button(frm_par, text ="Анализ", height = 1, width = 10, command = lambda: analize())
btn_analize.grid(row=6, column=0, sticky=W, pady=4, padx=5)
btn_yload = tk.Button(frm_par, text ="Симуляция", height = 1, width = 10, command = lambda: __bYload())
btn_yload.grid(row=6, column=1, sticky=W, pady=4, padx=5)
btn_graf = tk.Button(frm_par, text ="График", height = 1, width = 10, command = lambda: mplot())
btn_graf.grid(row=6, column=2, sticky=W, pady=4)

# блок 3 - консоль
# Консольный вывод
pb_load = ttk.Progressbar(wnd, orient = tk.HORIZONTAL, length = len_pb, mode = 'determinate')
pb_load.grid(row=13, column=0, sticky=W, pady=4, padx=5)

frm_con = tk.Frame(wnd, 
                   width = 790, 
                   height = 60, 
                   relief = tk.RAISED, 
                   borderwidth = 1)
frm_con.grid(row=14, column = 0, padx=5, pady=5, columnspan=8, rowspan=3, sticky=W)

txt_ord = tk.Text(frm_con, width=28, height=14)
txt_ord.grid(row=0, column=0, padx=2, pady=2)

scrollb2 = tk.Scrollbar(frm_con, command=txt_ord.yview)
scrollb2.grid(row=0, column=1, sticky='nsew')
txt_ord['yscrollcommand'] = scrollb2.set

txt_con = tk.Text(frm_con, width=65, height=14)
txt_con.grid(row=0, column=2, padx=4, pady=2)

scrollb1 = tk.Scrollbar(frm_con, command=txt_con.yview)
scrollb1.grid(row=0, column=3, sticky='nsew')
txt_con['yscrollcommand'] = scrollb1.set

#MessageLoop(tel_bot, __handle).run_as_thread()

wnd.mainloop()
#MessageLoop(tel_bot, __handle).run_forever()