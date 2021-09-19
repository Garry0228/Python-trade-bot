from pandas import read_csv
from pandas import DataFrame
import datetime
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
import os
 
def parser(x):
	return datetime.datetime.strptime('202'+x, '%Y-%m')
    
def calc_arima_model(X, arima_order):
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        aic = model_fit.aic
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    rmse = sqrt(mean_squared_error(test, predictions))
    return aic
    
def cicle_arima_model(DS, p_val, d_val, q_val):
    DS = DS.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_val:
        for d in d_val:
            for q in q_val:
                order = (p,d,q)
                try:
                    rmse = calc_arima_model(DS, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print('%.3f %s' % (rmse, order))
                except:
                    continue
    print('Лучшая конфигурация: %.3f %s' % (best_score, best_cfg))
 
data_loc = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'ticker.csv')
series = read_csv(data_loc, sep= ';', index_col=0) 
#series = read_csv('C:\Pyton396\Scripts\PTT\shampoo.csv', sep= ',', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
#series.index = series.index.to_period('M')
#X = series.values

"""p_val = [0, 1, 2, 4, 5, 6, 8, 10]
d_val = [0, 1, 2, 3]
q_val = [0, 1, 2, 3]"""
p_val = [2, 5]
d_val = [0, 1]
q_val = [0, 1]

warnings.filterwarnings("ignore")
cicle_arima_model(series['Open'].values, p_val, d_val, q_val)
#print(series.head)

#print('%.3f' % (calc_arima_model(X, (5,1,0))))
#pyplot.plot(test)
#pyplot.plot(predictions, color='red')
#pyplot.show()
#print(predictions)
