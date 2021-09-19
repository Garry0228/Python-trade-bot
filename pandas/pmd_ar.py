import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm
from pmdarima.datasets.stocks import load_msft
from pmdarima.arima import ndiffs
from pandas import read_csv
import os
from pandas.plotting import lag_plot
from sklearn.metrics import mean_squared_error
from pmdarima.metrics import smape

def data_loc(ticker):
    return os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), ticker)

def sma(df, n):
    sma = df.rolling(window = n).mean() #min_periods = 1
    return pd.DataFrame(sma)
    
def forecast_one_step(ind):
    fc = model.predict(n_periods=1)
    return (fc.tolist()[0])

df = read_csv(data_loc('ticker.csv'), sep= ';')#, index_col=0)

train_len = int(len(df) * 0.80)
train_data, test_data = df[:train_len].copy(), df[train_len:].copy()
test_data['sma_4'] = sma(test_data['Close'], 4)
test_data['sma_10'] = sma(test_data['Close'], 10)

y_train = train_data['Close'].values
y_test = test_data['Close'].values

df_new = test_data[['Datetime', 'Close']].copy()

kpss_diffs = ndiffs(y_train, alpha=0.05, test='kpss', max_d=10)
adf_diffs = ndiffs(y_train, alpha=0.05, test='adf', max_d=10)
n_diffs = max(adf_diffs, kpss_diffs)

model = pm.auto_arima(y_train, 
                     d=n_diffs, 
                     seasonal=False, 
                     stepwise=True,
                     suppress_warnings=True, 
                     error_action="ignore", 
                     max_p=6,
                     lbfgs='lbfgs',
                     max_order=None, 
                     trace=True)

print(model.order, model.aic())
        
forecasts = []
#for i in test_data.index:print(i)
#print(len(test_data.index))
for i in test_data.index:
    fc = forecast_one_step(None)
    forecasts.append(fc)
    df_new.loc[i, 'Predict'] = fc
    model.update(y_test[i - test_data.index[0]])
    #print(f'Факт={y_test[i - test_data.index[0]]:.3f} Прогноз={fc:.3f}')

print(f'Mean squared error: {mean_squared_error(y_test, forecasts):.3f}')
print(f'SMAPE: {smape(y_test, forecasts):.3f}')

df_new['P_sma_4'] = sma(df_new['Predict'], 4)
df_new['P_sma_10'] = sma(df_new['Predict'], 10)

# Торговая стратегия ARIMA
signal = 0
l_ind = 0
buy_price = []
sell_price = []
df_new['Signal'] = 0
df_new['Buy_price'] = 0
df_new['Sell_price'] = 0
#print(f'min index = {min(test_data.index)} max = {max(test_data.index)}')
for i in test_data.index:
    if i > min(test_data.index) and df_new.loc[i-1, 'Signal'] == 1:
        if len(buy_price) <= len(sell_price):
            price = df_new.loc[i, 'Close']
            df_new.loc[i, 'Buy_price'] = price
            df_new.loc[i, 'Sell_price'] = np.nan
            buy_price.append(price)
    
        df_new.loc[i, 'Signal'] = 2
    elif i > min(test_data.index) and df_new.loc[i-1, 'Signal'] == -1:
        if len(buy_price) > len(sell_price):
            price = df_new.loc[i, 'Close']
            df_new.loc[i, 'Sell_price'] = price
            df_new.loc[i, 'Buy_price'] = np.nan
            sell_price.append(price)
    
        df_new.loc[i, 'Signal'] = -2
    # Сначала продаем, потом покупаем
    elif df_new.loc[i, 'P_sma_4'] > df_new.loc[i, 'P_sma_10']:
        if signal != 1:
            if len(buy_price) > len(sell_price):
                price = df_new.loc[i, 'Close']
                df_new.loc[i, 'Sell_price'] = price
                df_new.loc[i, 'Buy_price'] = np.nan
                sell_price.append(price)
            
            df_new.loc[i, 'Signal'] = 1
            signal = 1
        else:
            df_new.loc[i, 'Buy_price'] = np.nan
            df_new.loc[i, 'Sell_price'] = np.nan
            df_new.loc[i, 'Signal'] = 0
    # Сначала покупаем, потом продаем
    elif df_new.loc[i, 'P_sma_10'] > df_new.loc[i, 'P_sma_4']:
        if signal != -1:
            if len(buy_price) <= len(sell_price):
                price = df_new.loc[i, 'Close']
                df_new.loc[i, 'Buy_price'] = price
                df_new.loc[i, 'Sell_price'] = np.nan
                buy_price.append(price)
        
            df_new.loc[i, 'Signal'] = -1
            signal = -1
        else:
            df_new.loc[i, 'Buy_price'] = np.nan
            df_new.loc[i, 'Sell_price'] = np.nan
            df_new.loc[i, 'Signal'] = 0
    else:
        df_new.loc[i, 'Signal'] = 0
        
# Продаем, если что-то осталось
if len(buy_price) > len(sell_price):
    price = df_new.loc[max(test_data.index), 'Close']
    df_new.loc[max(test_data.index), 'Sell_price'] = price
    sell_price.append(price)
    signal = 0

df_new.to_csv(data_loc('ticker_sma.csv'), sep= ';')    

"""# Торговая стратегия ARIMA
signal = 0
l_ind = 0
buy_price = []
sell_price = []
df_new['Signal'] = 0
df_new['Buy_price'] = 0
df_new['Sell_price'] = 0
for i in test_data.index:
    if df_new.loc[i, 'P_sma_4'] > df_new.loc[i, 'P_sma_10']:
        if signal != 1:
            price = df_new.loc[i-1, 'Close']
            df_new.loc[i, 'Buy_price'] = price
            buy_price.append(price)
            df_new.loc[i, 'Sell_price'] = np.nan
            df_new.loc[i, 'Signal'] = 1
            signal = 1
        else:
            df_new.loc[i, 'Buy_price'] = np.nan
            df_new.loc[i, 'Sell_price'] = np.nan
            df_new.loc[i, 'Signal'] = 0
    elif df_new.loc[i, 'P_sma_10'] > df_new.loc[i, 'P_sma_4']:
        if signal != -1:
            price = df_new.loc[i-1, 'Close']
            df_new.loc[i, 'Buy_price'] = np.nan
            df_new.loc[i, 'Sell_price'] = price
            sell_price.append(price)
            df_new.loc[i, 'Signal'] = -1
            signal = -1
        else:
            df_new.loc[i, 'Buy_price'] = np.nan
            df_new.loc[i, 'Sell_price'] = np.nan
            df_new.loc[i, 'Signal'] = 0
    else:
        df_new.loc[i, 'Buy_price'] = np.nan
        df_new.loc[i, 'Sell_price'] = np.nan
        df_new.loc[i, 'Signal'] = 0
    l_ind = i

# Продаем, если что-то осталось
if signal == 1:
    price = df_new.loc[l_ind, 'Close']
    df_new.loc[l_ind, 'Sell_price'] = price
    sell_price.append(price)
    signal = 0

df_new.to_csv(data_loc('ticker_sma.csv'), sep= ';')
    
print(f'Результат тестового набора по стратегии ARIMA:')
print(f'    Покупок: {len(buy_price)} Продаж: {len(sell_price)}')
print(f'    Сумма покупок: {sum(buy_price):.3f} Сумма продаж: {sum(sell_price):.3f}')
print(f'    Результат: {sum(sell_price) - sum(buy_price):.3f}')

# Торговая стратегия SMA_4_10 - Просто алготрейдинг, не предсказываем
signal = 0
l_ind = 0
buy_price = []
sell_price = []
df_new['Signal_SMA'] = 0
df_new['Buy_price_SMA'] = 0
df_new['Sell_price_SMA'] = 0
for i in test_data.index:
    if df_new.loc[i, 'sma_4'] > df_new.loc[i, 'sma_10']:
        if signal != 1:
            price = df_new.loc[i, 'Close']
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
            price = df_new.loc[i, 'Close']
            df_new.loc[i, 'Buy_price_SMA'] = np.nan
            df_new.loc[i, 'Sell_price_SMA'] = price
            sell_price.append(price)
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
    l_ind = i

# Продаем, если что-то осталось
if signal == 1:
    price = df_new.loc[l_ind, 'Close']
    df_new.loc[l_ind, 'Sell_price_SMA'] = price
    sell_price.append(price)
    signal = 0

df_new.to_csv(data_loc('ticker_sma.csv'), sep= ';')
    
print(f'Результат тестового набора по стратегии SMA_4_10:')
print(f'    Покупок: {len(buy_price)} Продаж: {len(sell_price)}')
print(f'    Сумма покупок: {sum(buy_price):.3f} Сумма продаж: {sum(sell_price):.3f}')
print(f'    Результат: {sum(sell_price) - sum(buy_price):.3f}')

plt.plot(train_data.index, y_train, color='blue', label='Тренировочный набор 80%')
plt.plot(test_data.index, forecasts, color='green', marker='o', label='Прогнозы')
plt.plot(test_data.index, y_test, color='red', label='Актуальная цена')
plt.title(f'Валюта: BTC. MSE: {mean_squared_error(y_test, forecasts):.3f}\n'
          f'SMAPE: {smape(y_test, forecasts):.3f}')
plt.legend()
plt.show()"""

