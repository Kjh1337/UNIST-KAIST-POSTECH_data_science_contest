#catboost 설치
!pip install catboost

from catboost import CatBoostClassifier, Pool, cv
from catboost.utils import eval_metric
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

import matplotlib
from matplotlib import font_manager, rc
from matplotlib import pyplot as plt
import platform
import seaborn as sns

#train, exam 데이터 불러오기
train = pd.read_csv('trainSet.csv')
test = pd.read_csv('examSet.csv')

# train 데이터 중 GAS_MONTH가 NAN값인 48개의 생산정만 추출, NAN값 있는 columns 제외
train = train.iloc[:48, :32]
train

# train information 확인, 총 32개의 컬럼 중 수치형 / 범주형 변수 갯수 확인

train.info()

# test 데이터도 마찬가지로 GAS_MONTH가 NAN값인 15개의 생산정만 추출, NAN값 있는 columns 제외
test = test.iloc[:15, :31]

test

# test information 확인, 총 32개의 컬럼 중 수치형 / 범주형 변수 갯수 확인

test.info()

# train 의 범주형 변수들의 분포 확인
category_feature = [col for col in train.columns if train[col].dtypes =='object']
category_feature = list(set(category_feature))

for col in category_feature:
  train[col].value_counts().plot(kind='bar')
  plt.title(col)
  plt.show()

# prediction에 중요하지 않을 것으로 예상되는 변수들 제거 
# 생산정 번호, 생산정ID 제거
# 용어 설명 파일에서 수압파쇄시 사용한 유체 종류는 유효하지 않다고 명시되어 있으므로 제거
# 제 1 프로판트 이름, 프로판트 구성 제거
# 최초,마지막 생산 날짜 등은 교수님이 오차값이 많다고 하셔서 제거했습니다. 

train_agg = train.set_index('No')

test_agg = test.set_index('No')

train_agg = train_agg.drop(['CPA Pretty Well ID','On Prod YYYY/MM/DD', 'First Prod YYYY/MM','Last Prod. YYYY/MM',
                            'Stimulation Fluid','Proppant Composition','Proppant Name 1' ], axis=1,)
test_agg = test_agg.iloc[:, 1:]
test_agg = test_agg.drop(['On Prod YYYY/MM/DD', 'First Prod YYYY/MM','Last Prod. YYYY/MM','Stimulation Fluid','Proppant Composition','Proppant Name 1'], axis=1,)

train_agg.reset_index(drop=True, inplace=True)
test_agg.reset_index(drop=True, inplace=True)

test_agg

# train set에서 훈련시킨 결과를 임시로 테스트하기 위해 0.8:0.2 비율로 validation set 분할
train_set, valid_set = train_test_split(train_agg, test_size=0.2,random_state=42)

valid_set

# target y 값을 생산 후 첫 6개월 월 평균 가스 생산량인 'First 6 mo. Avg. GAS (Mcf)'로 설정
y_train = train_set['First 6 mo. Avg. GAS (Mcf)']
train_set.drop(['First 6 mo. Avg. GAS (Mcf)'], axis=1, inplace=True)
y_valid =valid_set['First 6 mo. Avg. GAS (Mcf)']
valid_set.drop(['First 6 mo. Avg. GAS (Mcf)'], axis=1, inplace=True)

# 범주형 변수를 'Proppant size 1'으로 설정, 나머지는 다 numerical한 변수이기 때문
cat_features = ['Proppant Size 1' ]
target_col ='First 6 mo. Avg. GAS (Mcf)'

train_dataset = Pool(train_set, y_train, cat_features= cat_features)
test_dataset = Pool(valid_set, y_valid, cat_features= cat_features)

# catboost modeling 
# hyperparameter : n_estimators (500~1000), learning_rate (0.05~0.09) 범위에서 조절해가며  SMAPE가 낮은 5개의 후보군 선정
# 최종 예측값은 5개의 후보군 값의 평균값을 사용했습니다. 

from catboost import CatBoostRegressor
catb = CatBoostRegressor(
         cat_features=cat_features,
         loss_function='RMSE',
         eval_metric = 'SMAPE',
         n_estimators= 1000, 
         learning_rate=0.09, 
         random_state=42,
         )
    
catb.fit(train_agg.drop(columns=[target_col]), train_agg[target_col],verbose=100, eval_set=(valid_set,y_valid))

# validation set로 예측 결과 테스트 
pred = catb.predict(valid_set)

plt.plot(list(y_valid), color='red')
plt.plot(pred)
plt.xlabel('Each shale gas well')
plt.ylabel('Predicted First 6 mo.Avg.Gas')
plt.show()

# Feature Importance 확인

plt.figure(figsize=(10,8))
sorted_feature_importance = catb.feature_importances_.argsort()
plt.barh(train_agg.columns[sorted_feature_importance], catb.feature_importances_[sorted_feature_importance], color='turquoise')
plt.xlabel("CatBoost Feature Importance")

# validation y 값으로 예측 결과 성능 테스트, RMSE와 R-squared score 계산
rmse = (np.sqrt(mean_squared_error(y_valid, pred)))
r2 = r2_score(y_valid, pred)

print("Test performance")
print('RMSE: {:.2f}'.format(rmse))
print('R2: {:.2f}.'.format(r2))

# validation y 값으로 예측 결과 성능 테스트, SMAPE 계산

def smape(a, f):
  return 1/len(a) *np.sum(2*np.abs(f-a)/ (np.abs(a)+np.abs(f))*100)

s= smape(y_valid, pred)

print("Test performance")
print('SMAPE: {:.2f}%'.format(s))

# validation y 값으로 예측 결과 성능 테스트, MPE 계산

def mpe(a,f):
  return np.mean((a-f) /a)*100

mpe(y_valid, pred)

# test set의 생산값 예측

pred_ =[]

prediction = catb.predict(test_agg)

pred_.append(prediction)

pred_ = pd.DataFrame(pred_).transpose()

pred_

# 위 결과는 5개의 후보군 모델 중 n_estimator= 1000, learning rate =0.09를 가지는 하나의 결과

pred_.to_csv('static_prediction_candidate1.csv', index=False)

# 이와 같은 방법으로 hyperparameter를 조정한 5개의 모델을 엑셀로 저장 후 병합하여 최종 예측값은 5개의 평균값으로 사용했습니다.

pred_final = pd.read_csv('static_prediction_final.csv')

pred_final = pred_final.iloc[:,6]

pred_final
