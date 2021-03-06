%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.api import qqplot
import warnings
warnings.filterwarnings("ignore")
# 필요한 라이브러리 import

exam = pd.read_csv('examSet.csv')
exam
new = exam.iloc[:,31: 61]
new = new.T
new
# 1번째 생산정부터 44번째 생산정까지 생산량 dataframe 생성.

from datetime import date
from dateutil.rrule import rrule, MONTHLY
start_date = date(2018, 2, 1)
end_date = date(2020, 7, 31)
date_ = []

for date in rrule(MONTHLY, dtstart=start_date, until=end_date):
    print(date.strftime("%Y-%m-%d"))
    date_.append(date.strftime("%Y-%m-%d"))
# 시계열 예측 모델링을 위해선 시간을 나타내는 column이 필요하기 때문에 임의의 data column 생성(모든 생산정에 똑같이 사용)
# 정확한 날짜는 생산량을 예측하는데 중요하지 않다고 판단함.

new['date'] = date_

col = new.columns[:-1]
col = col.insert(0, 'date')

new=new[col]
new
# data column을 생산량 dataframe과 합침.

df = pd.DataFrame({'value':new[24].values}, index= new['date']) # 44개의 생산정의 시계열 예측을 한 번에 할 수 없어서 column을 하나씩 수정하며 값을 예측했습니다.
# new[15] ~ new[43]까지 시계열 모델링을 반복한 후 직접 엑셀 파일에 예측값을 작성했습니다.
df

# 데이터 전처리
df = df.dropna() #null 값 삭제
df = df[df['value'] > 1000] # value가 1000 이하인 값들은 모두 drop했습니다. 
df

from statsmodels.tsa.stattools import adfuller

def stationarity_check(ts):
            
    # Calculate rolling statistics
    roll_mean = ts.rolling(window=8, center=False).mean()
    roll_std = ts.rolling(window=8, center=False).std()

    # Perform the Dickey Fuller test
    dftest = adfuller(ts) 
    
    # Plot rolling statistics:
    fig = plt.figure(figsize=(12,6))
    orig = plt.plot(ts, color='blue',label='Original')
    mean = plt.plot(roll_mean, color='red', label='Rolling Mean')
    std = plt.plot(roll_std, color='green', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    # Print Dickey-Fuller test results

    print('\nResults of Dickey-Fuller Test: \n')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', 
                                             '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
result = stationarity_check(df['value'])

# n 번째 생산정의 데이터가 stationary한 지 아닌 지 테스트하기 위해 ADF Test를 진행했습니다.

#!pip install pmdarima
# auto_arima를 사용하기 위한 라이브러리 설치
from pmdarima.arima import auto_arima

model = auto_arima(df, start_p=0, start_q=0) # p=0, q=0부터 숫자를 늘려가며 AIC value가 가장 작은 parameter를 찾았습니다.
model.summary()

model = auto_arima(df, start_p=0, start_q=0)
print(model.seasonal_order) # auto_arima가 SARIMAX 모델을 선택하여 seasonal_order를 확인한 뒤 계절성의 여부를 확인했습니다.

model.plot_diagnostics(figsize=(16,8)) # 잔차에 대한 진단

from statsmodels.tsa.statespace.sarimax import SARIMAX
best_model = SARIMAX(df,
                     order=model.order,
                     seasonal_order=model.seasonal_order).fit()
display(best_model.summary())

# auto_arima가 SARIMAX를 선택하여 SARIMAX 모델링을 진행하였습니다.

def forecast_to_df(model, steps=6): # SARIMAX 모델과 적절한 parameter값을 통해 향후 6개월치 생산량을 예측했습니다.
    forecast = model.get_forecast(steps=steps)
    pred_df = forecast.conf_int()
    pred_df['pred'] = forecast.predicted_mean
    pred_df.columns = ['lower', 'upper', 'pred'] #pred column이 예측값입니다.
    return pred_df
print(forecast_to_df(best_model, steps=6)) # result of forecasting
np.mean(forecast_to_df(best_model, steps=6)['pred']) # 예측된 향후 6개월치 생산량의 평균값
