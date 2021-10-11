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
df = []

gr_buy = []
gr_in = []
gr_sell = []
gr_loss = []

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
def mplot(df, gr_in, gr_buy, gr_sell, gr_loss):
    if len(df) == 0:
        return 0
    
    plt.plot(df.index, df['Close'], color='gray', label='Актуальная цена')
    
    plt.plot(gr_in.index, gr_in, 'bx', label='Точка входа в цикл')
    plt.plot(gr_buy.index, gr_buy, 'g^', label='Ордера на покупку')
    plt.plot(gr_sell.index, gr_sell, 'rv', label='Ордер на продажу')
    plt.plot(gr_loss.index, gr_loss, 'mo', label='StopLoss')
    
    plt.title(f'Валюта: {cmb_pairs.get()}')
    plt.legend()
    plt.grid()
    plt.show()
    
    return 0

# Сколько купили    
def buy_val(pClose, price):
    return round(price / pClose, 8)
    
# За сколько продали    
def sell_val(pClose, cty):
    return cty * pClose

# Формирование сетки ордеров
def orders_calc(close, count, prc, prc1, price, mg):
    order_list = []
    
    __prc = 0
    for i in range(count):
        if i == 0:
            __prc = __prc + prc
            order = close - (close * (__prc/100))
        else:
            __prc = __prc + prc1
            order = close - (close * (__prc/100))
        amount = price + (price * (mg*i/100))
        order_list.append((order, amount))
        #print(__prc)
        
    #print(close, order_list[0][0], order_list[0][1])
    return order_list

# Симуляция по алгоритму       
def __Simulate(): 
    global train_data, test_data, y_train, y_test, df, gr_in, gr_buy, gr_sell, gr_loss
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
    if cmb_alg.get() == 'Martingale LONG':
        df, gr_in, gr_buy, gr_sell, gr_loss, rown = simulate_MG(df, rown)
    
    # Алгоритм SMA(5)+RSI
    elif cmb_alg.get() == 'SMA(5)+RSI':
        df,gr_in, gr_buy, gr_sell, gr_loss, rown = simulate_SMA(df, rown, 5)
    
    # Алгоритм SMA(10)+RSI
    elif cmb_alg.get() == 'SMA(10)+RSI':
        df, gr_in, gr_buy, gr_sell, gr_loss, rown = simulate_SMA(df, rown, 10)
    
    # Алгоритм SMA(20)+RSI
    elif cmb_alg.get() == 'SMA(20)+RSI':
        df, gr_in, gr_buy, gr_sell, gr_loss, rown = simulate_SMA(df, rown, 20)
        
    # Алгоритм SMA(40)+RSI
    elif cmb_alg.get() == 'SMA(40)+RSI':
        df, gr_in, gr_buy, gr_sell, gr_loss, rown = simulate_SMA(df, rown, 40)
    
# Симуляция по алгоритму Martingale. Вход по осциллятору RSI < 75, направление LONG 
def simulate_MG(df, rown):
    df_new = df[['Datetime', 'Close']].copy()
    #print(df_new['Datetime'])
    #df_new.set_index('Datetime')
    df_new['RSI'] = mm.RSIIndicator(close = df_new['Close'], window = 14, fillna=False).rsi()
    
    x_buy = []
    y_buy = []
    x_in = []
    y_in = []
    x_sell = []
    y_sell = []
    x_loss = []
    y_loss = []
    
    order_list = []
    
    amount_list = []
    buy_list = []
    comm_list = []
    
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
        if (df_new.loc[i, 'RSI'] < 75 and #(df_new.loc[i, 'RSI'] > 40 and 
         len(order_list) == 0 and
         len(amount_list) == 0):
            x_in.append(i)
            y_in.append(df_new.loc[i, 'Close'])
            
            start_date = parse(df_new.loc[i, 'Datetime'])
            
            order_list = orders_calc(df_new.loc[i, 'Close'], 
                                     int(cmb_count.get()), 
                                     float(cmb_step.get()),
                                     float(cmb_step1.get()),
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
            
        # Выход из цикла, если сработал stop loss. Критерий - падение стоимости безубытка на заданный процент
        if len(amount_list) > 0: #and len(order_list) == 0:
            #print(f'#{cycle_n} price={sum(buy_list)} ' 
            #      f'amount={sum(amount_list):.3f} ' 
            #      f'close={df_new.loc[i, "Close"]:.3f} ' 
            #      f'sell={sum(amount_list)*df_new.loc[i, "Close"]:.3f} '
            #      f'loss%={sum(buy_list) - (sum(buy_list)*float(cmb_stop.get())/100):.3f}')
            if sum(amount_list)*df_new.loc[i, 'Close'] < (sum(buy_list) - (sum(buy_list)*float(cmb_stop.get())/100)):
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
                comm_list = []
                order_n = 0
                last_close = 0
                
                x_loss.append(i)
                y_loss.append(df_new.loc[i, 'Close'])
            
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
                comm_list.append(comm)
                
                x_buy.append(i)
                y_buy.append(df_new.loc[i, 'Close'])
                
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
                    comm_list.append(comm)
                    
                    total_comm.append(sum(comm_list))
                
                    count_buy = count_buy + len(amount_list)
                    count_sell = count_sell + 1
                    order_list = []
                    amount_list = []
                    buy_list = []
                    comm_list = []
                    order_n = 0
                    last_close = 0
                    trail_stop = 0
                    
                    x_sell.append(i)
                    y_sell.append(df_new.loc[i, 'Close'])
                
        # перезапуск алгоритма по таймеру
        if len(order_list) == int(cmb_count.get()):
            cur_date = parse(df_new.loc[i, 'Datetime'])
            diff_min = divmod((cur_date - start_date).total_seconds(), 60)[0]
            
            if diff_min >= int(cmb_rest.get()):
                rown = put_consol(txt_con, rown, f'------ Перезапуск в {cur_date.strftime("%d.%m.%Y %H:%M:%S")}')
                
                order_list = []
                amount_list = []
                buy_list = []
                comm_list = []
                order_n = 0
                last_close = 0
    
        pb_load['value'] = i
        wnd.update_idletasks()
        
    # Завершающий цикл. Нужно корректно прервать, в расчете не участвует
    if len(amount_list) > 0:
        rown = put_consol(txt_con, rown, f'<Цикл прерван на шаге {order_n}. В расчете не участвует>')
        txt_con.tag_add('loss', rown-1, f'{int(rown-1)}.end')
    
    rown = put_consol(txt_con, rown, f'Количество покупок: {count_buy} Количество продаж: {count_sell}')
    rown = put_consol(txt_con, rown, f'Сумма покупок: {sum(total_buy):.3f} Сумма продаж: {sum(total_sell):.3f} Комиссия: {sum(total_comm):.3f}')
    rown = put_consol(txt_con, rown, f'Результат: {sum(total_sell) - sum(total_buy) - sum(total_comm):.3f}')
    rown = put_consol(txt_con, rown, f'Суммарный депозит: {cap_sum:.2f} (профит {(sum(total_sell) - sum(total_buy) - sum(total_comm))/cap_sum*100:.1f}%)')
    
    df_new.to_csv(data_loc('ticker_MG.csv'), sep= ';', index=False)
    
    gr_buy = pd.Series(y_buy)
    gr_buy.index = x_buy
    
    gr_in = pd.Series(y_in)
    gr_in.index = x_in
    
    gr_sell = pd.Series(y_sell)
    gr_sell.index = x_sell
    
    gr_loss = pd.Series(y_loss)
    gr_loss.index = x_loss
    
    #print(df_new.index.values)    
    return df_new, gr_in, gr_buy, gr_sell, gr_loss, rown

    
# Симуляция по алгоритму простого скользящего среднего (SMA(N)+RSI)    
def simulate_SMA(df, rown, __val):
    df_new = df[['Datetime', 'Close']].copy()
    df_new['RSI'] = mm.RSIIndicator(close = df_new['Close'], window = __val, fillna=False).rsi()
    df_new[f'sma_{__val}'] = sma(df_new['Close'], __val)
    
    txt_con.tag_config('curr', font=("Verdana", 9, 'bold'))
    txt_con.tag_config('profit', foreground='green')
    txt_con.tag_config('loss', foreground='red')
    
    txt_ord.delete("1.0","end")
    rown = put_consol(txt_con, rown, f'{cmb_pairs.get()}')
    txt_con.tag_add('curr', rown-1, f'{int(rown-1)}.end')
    rown = put_consol(txt_con, rown, f'Результат тестового набора по стратегии SMA_({__val})+RSI:')
    
    pb_load['maximum'] = len(df_new.index)
    
    # посчитаем депозит
    cap_sum = 0
    for i in range(int(cmb_count.get())):
        price = float(entr_price.get()) + float(entr_price.get())*i*float(cmb_mart.get())/100
        cap_sum = cap_sum + price
        #print(i, cap_sum)
    
    start_date = None
    cycle_n = 0
    price = 0
    
    amount_list = []
    buy_list = []
    comm_list = []
    
    total_sell = []
    total_buy = []
    total_comm = []
    
    x_buy = []
    y_buy = []
    x_in = []
    y_in = []
    x_sell = []
    y_sell = []
    x_loss = []
    y_loss = []
    
    signal = 0
    start_price = 0
    n_order = 0
    trail_stop = 0
    count_buy = 0
    count_sell = 0
    for i in df_new.index:
        cur_date = parse(df_new.loc[i, 'Datetime'])
        
        # Начало цикла = начало восходящего тренда: цена пробивает SMA снизу. RSI > 50
        if (i > 0 and 
        df_new.loc[i, 'Close'] > df_new.loc[i, f'sma_{__val}'] and 
        df_new.loc[i-1, 'Close'] < df_new.loc[i-1, f'sma_{__val}'] and
        df_new.loc[i, 'RSI'] > 50 and
        signal == 0):
            cycle_n = cycle_n + 1
            signal = 1            
            df_new.loc[i, 'Signal'] = 1
            
            start_price = df_new.loc[i, 'Close']
            start_date = parse(df_new.loc[i, 'Datetime'])
            rown = put_consol(txt_con, rown, f'# цикла: {cycle_n} '
                                             f'цена: {float(entr_price.get()):.0f}+{n_order*float(cmb_mart.get()):.0f}% ' 
                                             f'старт в: {start_date.strftime("%d.%m.%Y %H:%M:%S")}')
            
            x_in.append(i)
            y_in.append(start_price)
        
        # сигнал на покупку: цикл начался и цена возросла на заданный %
        if (signal == 1 and
        len(amount_list) == 0 and
        df_new.loc[i, 'Close'] >= start_price + (start_price*float(cmb_step.get())/100)):
            price = float(entr_price.get()) + float(entr_price.get())*n_order*float(cmb_mart.get())/100
            df_new.loc[i, 'Price'] = price
            amount = buy_val(df_new.loc[i, 'Close'], price)
            df_new.loc[i, 'Amount'] = amount
            
            amount_list.append(amount)
            buy_list.append(price)
                
            comm = price*float(cmb_comm.get())/100
            df_new.loc[i, 'Comm'] = comm
            comm_list.append(comm)
            
            rown = put_consol(txt_con, rown, f'  Покупка в {cur_date.strftime("%d.%m.%Y %H:%M:%S")} : {sum(buy_list):.3f}')
            
            count_buy = count_buy + 1
            
            x_buy.append(i)
            y_buy.append(df_new.loc[i, 'Close'])
            
        # выход по StopLoss
        if len(amount_list) > 0 and sum(amount_list)*df_new.loc[i, 'Close'] < (sum(buy_list) - (sum(buy_list)*float(cmb_stop.get())/100)):
            n_order = n_order + 1
            
            # если превысили количество ордеров, начинаем сначала
            if n_order > int(cmb_count.get()):
                n_order = 0
            
            sell_price = sum(amount_list)*df_new.loc[i, 'Close']
            df_new.loc[i, 'Loss'] = sell_price
                
            comm = sell_price*float(cmb_comm.get())/100
            df_new.loc[i, 'Comm'] = comm
            comm_list.append(comm)
            
            total_buy.append(sum(buy_list))
            total_sell.append(sell_price)
            total_comm.append(sum(comm_list))
            
            rown = put_consol(txt_con, rown, f'  Продажа в {cur_date.strftime("%d.%m.%Y %H:%M:%S")} : {sell_price:.3f}')
            rown = put_consol(txt_con, rown, f'Потеря: {sell_price - sum(buy_list):.3f}')
            txt_con.tag_add('loss', rown-1, f'{int(rown-1)}.end')
            
            signal = 0
            amount_list = []
            buy_list = []
            comm_list = []
            
            count_sell = count_sell + 1
            
            x_loss.append(i)
            y_loss.append(df_new.loc[i, 'Close'])
            
        # сигнал на продажу - цена превысила ожидаемый профит
        if len(amount_list) > 0 and sum(amount_list)*df_new.loc[i, 'Close'] >= sum(buy_list) + (sum(buy_list)*float(cmb_prof.get())/100):
            if trail_stop == 0 and float(cmb_trail.get()) > 0:
                trail_stop = df_new.loc[i, 'Close']
                df_new.loc[i, 'Trail'] = 0
            elif (trail_stop > 0 
              and df_new.loc[i, 'Close'] >= trail_stop - (trail_stop*float(cmb_trail.get())/100) 
              and float(cmb_trail.get()) > 0):
                trail_stop = df_new.loc[i, 'Close']
                df_new.loc[i, 'Trail'] = 1
            else:
                sell_price = sum(amount_list)*df_new.loc[i, 'Close']
                df_new.loc[i, 'Profit'] = sell_price
                rown = put_consol(txt_con, rown, f'  Продажа в {cur_date.strftime("%d.%m.%Y %H:%M:%S")} : {sell_price:.3f}')
                rown = put_consol(txt_con, rown, f'Профит: {sell_price - sum(buy_list):.3f}')
                txt_con.tag_add('profit', rown-1, f'{int(rown-1)}.end')
                
                comm = sell_price*float(cmb_comm.get())/100
                df_new.loc[i, 'Comm'] = comm
                comm_list.append(comm)
                
                total_buy.append(sum(buy_list))
                total_sell.append(sell_price)
                total_comm.append(sum(comm_list))
                
                n_order = 0
                signal = 0
                amount_list = []
                buy_list = []
                comm_list = []
                
                count_sell = count_sell + 1
                
                x_sell.append(i)
                y_sell.append(df_new.loc[i, 'Close'])
        
        # перезапуск алгоритма по таймеру
        if signal == 1 and len(amount_list) == 0:
            diff_min = divmod((cur_date - start_date).total_seconds(), 60)[0]
            
            if diff_min >= int(cmb_rest.get()):
                rown = put_consol(txt_con, rown, f'------ Перезапуск в {cur_date.strftime("%d.%m.%Y %H:%M:%S")}')               
                signal = 0
        
        pb_load['value'] = i
        wnd.update_idletasks()
        
    # Завершающий цикл. Нужно корректно прервать, в расчете не участвует
    if len(amount_list) > 0:
        rown = put_consol(txt_con, rown, f'<Цикл прерван. В расчете не участвует>')
        txt_con.tag_add('loss', rown-1, f'{int(rown-1)}.end')
        count_buy = count_buy - 1
        
    rown = put_consol(txt_con, rown, f'Количество покупок: {count_buy} Количество продаж: {count_sell}')
    rown = put_consol(txt_con, rown, f'Сумма покупок: {sum(total_buy):.3f} Сумма продаж: {sum(total_sell):.3f} Комиссия: {sum(total_comm):.3f}')
    rown = put_consol(txt_con, rown, f'Результат: {sum(total_sell) - sum(total_buy) - sum(total_comm):.3f}')
    rown = put_consol(txt_con, rown, f'Суммарный депозит: {cap_sum:.2f} (профит {(sum(total_sell) - sum(total_buy) - sum(total_comm))/cap_sum*100:.1f}%)')
    
    df_new.to_csv(data_loc(f'ticker_sma_rsi.csv'), sep= ';', index=False)
    
    #rown = put_consol(txt_con, rown, f'Покупок: {len(buy_price)} Продаж: {len(sell_price)}')
    #rown = put_consol(txt_con, rown, f'Сумма покупок: {sum(buy_price):.3f} Сумма продаж: {sum(sell_price):.3f}')
    #rown = put_consol(txt_con, rown, f'Результат: {sum(sell_price) - sum(buy_price):.3f}')
    #rown = put_consol(txt_con, rown, f'{__price.get()}')
    
    gr_buy = pd.Series(y_buy)
    gr_buy.index = x_buy
    
    gr_in = pd.Series(y_in)
    gr_in.index = x_in
    
    gr_sell = pd.Series(y_sell)
    gr_sell.index = x_sell
    
    gr_loss = pd.Series(y_loss)
    gr_loss.index = x_loss
    
    return df_new, gr_in, gr_buy, gr_sell, gr_loss, rown

# Инициализация окна
wnd = tk.Tk()
wnd.geometry('835x680')

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
# __count = [5, 7, 10, 15, 20, 30, 40, 50]
# __step = [0.5, 1, 2, 3, 4, 5]
# __step1 = [0.5, 1, 2, 3, 4, 5]
# __mart = [5, 7, 10, 15, 20, 25]
# __prof = [0.7, 1, 2, 3, 5]
# __stop = [3, 5, 7, 10, 20, 30]
# __rest = [60, 120, 180, 240, 300]
# __trail = [0, 0.1, 0.3, 0.5, 1]
lbl_alg = tk.Label(frm_par, text='Алгоритм:')
lbl_alg.grid(row=2, column=0, sticky=E, pady=4, padx=5)
__alg = ['Martingale LONG', 'SMA(5)+RSI', 'SMA(10)+RSI', 'SMA(20)+RSI', 'SMA(40)+RSI']
cmb_alg = ttk.Combobox(frm_par, values=__alg, state='readonly')
cmb_alg.grid(row=2, column=1, sticky=W, pady=4, padx=5)
cmb_alg.current(0)

lbl_price = tk.Label(frm_par, text='1й ордер, $:')
lbl_price.grid(row=2, column=2, sticky=E, pady=4, padx=5)
__price = tk.IntVar()
__price.set(100)
entr_price = tk.Entry(frm_par, textvariable=__price)
entr_price.grid(row=2, column=3, sticky=W, pady=4, padx=5)

lbl_count = tk.Label(frm_par, text='Количество ордеров:')
lbl_count.grid(row=2, column=4, sticky=E, pady=4, padx=5)
__count = tk.IntVar()
__count.set(10)
cmb_count = tk.Entry(frm_par, textvariable=__count)
cmb_count.grid(row=2, column=5, sticky=W, pady=4, padx=5)

# строка 3 - настройки
lbl_step = tk.Label(frm_par, text='Отступ 1го ордера, %:')
lbl_step.grid(row=3, column=0, sticky=E, pady=4, padx=5)
__step = tk.IntVar()
__step.set(0.5)
cmb_step = tk.Entry(frm_par, textvariable=__step)
cmb_step.grid(row=3, column=1, sticky=W, pady=4, padx=5)

lbl_step1 = tk.Label(frm_par, text='Остальных, %:')
lbl_step1.grid(row=3, column=2, sticky=E, pady=4, padx=5)
__step1 = tk.IntVar()
__step1.set(1)
cmb_step1 = tk.Entry(frm_par, textvariable=__step1)
cmb_step1.grid(row=3, column=3, sticky=W, pady=4, padx=5)

lbl_mart = tk.Label(frm_par, text='Matringale, %:')
lbl_mart.grid(row=3, column=4, sticky=E, pady=4, padx=5)
__mart = tk.IntVar()
__mart.set(15)
cmb_mart = tk.Entry(frm_par, textvariable=__mart)
cmb_mart.grid(row=3, column=5, sticky=W, pady=4, padx=5)

# строка 4 - настройки
lbl_prof = tk.Label(frm_par, text='TakeProfit, %:')
lbl_prof.grid(row=4, column=0, sticky=E, pady=4, padx=5)
__prof = tk.IntVar()
__prof.set(1)
cmb_prof = tk.Entry(frm_par, textvariable=__prof)
cmb_prof.grid(row=4, column=1, sticky=W, pady=4, padx=5)

lbl_stop = tk.Label(frm_par, text='StopLoss, %:')
lbl_stop.grid(row=4, column=2, sticky=E, pady=4, padx=5)
__stop = tk.IntVar()
__stop.set(10)
cmb_stop = tk.Entry(frm_par, textvariable=__stop)
cmb_stop.grid(row=4, column=3, sticky=W, pady=4, padx=5)

lbl_comm = tk.Label(frm_par, text='Комиссия, %:')
lbl_comm.grid(row=4, column=4, sticky=E, pady=4, padx=5)
__comm = [0.075, 0.1]
cmb_comm = ttk.Combobox(frm_par, values=__comm, state='readonly')
cmb_comm.grid(row=4, column=5, sticky=W, pady=4, padx=5)
cmb_comm.current(0)

# строка 5 - настройки
lbl_rest = tk.Label(frm_par, text='Перезапуск, мин:')
lbl_rest.grid(row=5, column=0, sticky=E, pady=4, padx=5)
__rest = tk.IntVar()
__rest.set(60)
cmb_rest = tk.Entry(frm_par, textvariable=__rest)
cmb_rest.grid(row=5, column=1, sticky=W, pady=4, padx=5)

lbl_trail = tk.Label(frm_par, text='TrailingStop, %:')
lbl_trail.grid(row=5, column=2, sticky=E, pady=4, padx=5)
__trail = tk.IntVar()
__trail.set(0.1)
cmb_trail = tk.Entry(frm_par, textvariable=__trail)
cmb_trail.grid(row=5, column=3, sticky=W, pady=4, padx=5)

# строка 6 - кнопки
btn_analize = tk.Button(frm_par, text ="Анализ", height = 1, width = 10, command = lambda: analize())
btn_analize.grid(row=6, column=0, sticky=W, pady=4, padx=5)
btn_yload = tk.Button(frm_par, text ="Симуляция", height = 1, width = 10, command = lambda: __Simulate())
btn_yload.grid(row=6, column=1, sticky=W, pady=4, padx=5)
btn_graf = tk.Button(frm_par, text ="График", height = 1, width = 10, command = lambda: mplot(df, gr_in, gr_buy, gr_sell, gr_loss))
btn_graf.grid(row=6, column=2, sticky=W, pady=4)

# блок 3 - консоль
# Консольный вывод
pb_load = ttk.Progressbar(wnd, orient = tk.HORIZONTAL, length = len_pb, mode = 'determinate')
pb_load.grid(row=13, column=0, sticky=W, pady=4, padx=5)

frm_con = tk.Frame(wnd, 
                   width = 800, 
                   height = 60, 
                   relief = tk.RAISED, 
                   borderwidth = 1)
frm_con.grid(row=14, column = 0, padx=5, pady=5, columnspan=8, rowspan=3, sticky=W)

txt_ord = tk.Text(frm_con, width=28, height=14)
txt_ord.grid(row=0, column=0, padx=2, pady=2)

scrollb2 = tk.Scrollbar(frm_con, command=txt_ord.yview)
scrollb2.grid(row=0, column=1, sticky='nsew')
txt_ord['yscrollcommand'] = scrollb2.set

txt_con = tk.Text(frm_con, width=68, height=14)
txt_con.grid(row=0, column=2, padx=4, pady=2)

scrollb1 = tk.Scrollbar(frm_con, command=txt_con.yview)
scrollb1.grid(row=0, column=3, sticky='nsew')
txt_con['yscrollcommand'] = scrollb1.set

#MessageLoop(tel_bot, __handle).run_as_thread()

wnd.mainloop()
#MessageLoop(tel_bot, __handle).run_forever()