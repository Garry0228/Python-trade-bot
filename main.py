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
    return datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

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
        data['ATRp'] = data['volatility_atr']/data['Close']*100
        data['BB_wprc'] = (data['volatility_bbh']-data['volatility_bbl'])/data['Close']*100
        
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
        
        rown = put_consol(txt_con, rown, f'   Bollinger Bands (BB), SMA')
        rown = put_consol(txt_con, rown, f'      Среднее за 30 дней: {df.loc[mask(df, curr, 30), "volatility_bbm"].mean():.3f}')
        rown = put_consol(txt_con, rown, f'      Среднее за 7 дней: {df.loc[mask(df, curr, 7), "volatility_bbm"].mean():.3f}')
        rown = put_consol(txt_con, rown, f'      Текущее: {df.loc[mask(df, curr, 0), "volatility_bbm"].mean():.3f}')
        
        
        rown = put_consol(txt_con, rown, f'   Bollinger Bands (BB) width, %')
        rown = put_consol(txt_con, rown, f'      Среднее за 30 дней: {df.loc[mask(df, curr, 30), "BB_wprc"].mean():.3f}')
        rown = put_consol(txt_con, rown, f'      Среднее за 7 дней: {df.loc[mask(df, curr, 7), "BB_wprc"].mean():.3f}')
        rown = put_consol(txt_con, rown, f'      Текущее: {df.loc[mask(df, curr, 0), "BB_wprc"].mean():.3f}')
        
        i = i+ 1
        pb_load['value'] = i
        wnd.update_idletasks()
    
    df = df.rename(columns={'volatility_atr': 'ATR', 
                            'volatility_bbl': 'BB_low', 
                            'volatility_bbm': 'BB_mid', 
                            'volatility_bbh': 'BB_high'})
    df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'ticker', 
             'ATR', 'ATRp', 'BB_low', 'BB_mid', 'BB_high', 'BB_wprc']].copy()
    
    df.to_csv(os.path.join(loc, 'analize_tick.csv'), sep=';')

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
    train_data, test_data, y_train, y_test = test_models(df)

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
    global train_data, test_data, y_train, y_test
    
    if len(train_data) == 0:
        return 0
    
    plt.plot(train_data.index, y_train, color='blue', label='Тренировочный набор 80%')
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
    
# Построение тренирочного и тестового набора    
def test_models(df):
    global txt_con
    rown = 1.0
    
    train_len = int(len(df) * 0.80)
    train_data, test_data = df[:train_len].copy(), df[train_len:].copy()
    test_data['sma_4'] = sma(test_data['Close'], 4)
    test_data['sma_10'] = sma(test_data['Close'], 10)

    y_train = train_data['Close'].values
    y_test = test_data['Close'].values
    
    df_new = test_data[['Datetime', 'sma_4', 'sma_10', 'Close']].copy()
    
    # Торговая стратегия SMA_4_10 - Просто алготрейдинг, не предсказываем
    signal = 0
    buy_sum = 0
    sell_sum = 0
    buy_price = []
    sell_price = []
    df_new['Signal_SMA'] = 0
    df_new['Buy_price_SMA'] = 0
    df_new['Sell_price_SMA'] = 0
    
    rown = put_consol(txt_con, rown, f'Результат тестового набора по стратегии SMA_4_10:')
    
    for i in test_data.index:
        if df_new.loc[i, 'sma_4'] > df_new.loc[i, 'sma_10']:
            if signal != 1:
                if len(buy_price) <= len(sell_price):
                    #price = df_new.loc[i, 'Close']
                    price = __price.get()
                    buy_sum = buy_val(df_new.loc[i, 'Close'], price)
                    rown = put_consol(txt_con, rown, f'     Покупка единиц валюты: {buy_sum:.3f} Сумма покупки: {price}$')
                    
                    df_new.loc[i, 'Buy_price_SMA'] = price
                    buy_price.append(price)
                    df_new.loc[i, 'Sell_price_SMA'] = np.nan
                    
                df_new.loc[i, 'Signal_SMA'] = 1                   
                signal = 1
            else:
                df_new.loc[i, 'Buy_price_SMA'] = np.nan
                df_new.loc[i, 'Sell_price_SMA'] = np.nan
                df_new.loc[i, 'Signal_SMA'] = 0
        elif df_new.loc[i, 'sma_10'] > df_new.loc[i, 'sma_4']:
            if signal != -1:
                if len(buy_price) > len(sell_price):
                    #price = df_new.loc[i, 'Close']
                    price = __price.get()
                    sell_sum = sell_val(df_new.loc[i, 'Close'], buy_sum)
                    rown = put_consol(txt_con, rown, f'     Продажа единиц валюты: {buy_sum:.3f} Сумма продажи: {sell_sum:.3f}$')
                    
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

    # Продаем, если что-то осталось
    if len(buy_price) > len(sell_price):
        sell_sum = sell_val(df_new.loc[max(test_data.index), 'Close'], buy_sum)
        rown = put_consol(txt_con, rown, f'     Продажа единиц валюты: {buy_sum:.3f} Сумма продажи: {sell_sum:.3f}$')
        
        #price = df_new.loc[max(test_data.index), 'Close']
        df_new.loc[max(test_data.index), 'Sell_price'] = sell_sum
        sell_price.append(sell_sum)
        signal = 0

    df_new.to_csv(data_loc('ticker_sma.csv'), sep= ';')
    
    rown = put_consol(txt_con, rown, f'    Покупок: {len(buy_price)} Продаж: {len(sell_price)}')
    rown = put_consol(txt_con, rown, f'    Сумма покупок: {sum(buy_price):.3f} Сумма продаж: {sum(sell_price):.3f}')
    rown = put_consol(txt_con, rown, f'    Результат: {sum(sell_price) - sum(buy_price):.3f}')
    #rown = put_consol(txt_con, rown, f'{__price.get()}')
    
    return train_data, test_data, y_train, y_test

# Инициализация окна
wnd = tk.Tk()
wnd.geometry('800x600')

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

lbl_sum = tk.Label(frm_par, text='Шаг покупки, $:')
lbl_sum.grid(row=2, column=0, sticky=E, pady=4, padx=5)
__price = tk.IntVar()
__price.set(500)
entr_sum = tk.Entry(frm_par, textvariable=__price)
entr_sum.grid(row=2, column=1, sticky=W, pady=4, padx=5)

lbl_sma = tk.Label(frm_par, text='Алгоритм:')
lbl_sma.grid(row=2, column=2, sticky=E, pady=4, padx=5)
__alg = ['Martingale LONG', 'SMA_4_10', 'SMA_20_50']
cmb_sma = ttk.Combobox(frm_par, values=__alg, state='readonly')
cmb_sma.grid(row=2, column=3, sticky=W, pady=4, padx=5)
cmb_sma.current(0)

btn_analize = tk.Button(frm_par, text ="Анализ", height = 1, width = 10, command = lambda: analize())
btn_analize.grid(row=3, column=0, sticky=W, pady=4, padx=5)
btn_yload = tk.Button(frm_par, text ="Симуляция", height = 1, width = 10, command = lambda: __bYload())
btn_yload.grid(row=3, column=1, sticky=W, pady=4, padx=5)
btn_graf = tk.Button(frm_par, text ="График", height = 1, width = 10, command = lambda: mplot())
btn_graf.grid(row=3, column=2, sticky=W, pady=4)

# Консольный вывод
pb_load = ttk.Progressbar(wnd, orient = tk.HORIZONTAL, length = len_pb, mode = 'determinate')
pb_load.grid(row=13, column=0, sticky=W, pady=4, padx=5)

frm_con = tk.Frame(wnd, 
                   width = 790, 
                   height = 60, 
                   relief = tk.RAISED, 
                   borderwidth = 1)
frm_con.grid(row=14, column = 0, padx=5, pady=5, columnspan=8, rowspan=3, sticky=W)

txt_ord = tk.Text(frm_con, width=30, height=14)
txt_ord.grid(row=0, column=0, padx=2, pady=2)

txt_con = tk.Text(frm_con, width=63, height=14)
txt_con.grid(row=0, column=1, padx=4, pady=2)

scrollb1 = tk.Scrollbar(frm_con, command=txt_con.yview)
scrollb1.grid(row=0, column=2, sticky='nsew')
txt_con['yscrollcommand'] = scrollb1.set

#MessageLoop(tel_bot, __handle).run_as_thread()

wnd.mainloop()
#MessageLoop(tel_bot, __handle).run_forever()