"""**Linear Programming, Convex optimization 문제를 풀어주는 cvxpy 라이브러리 설치**"""

!pip install cvxpy

import numpy as np 
import pandas as pd
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import scipy.optimize as spopt 
import scipy.fftpack as spfft 
import scipy.ndimage as spimg 
import cvxpy as cp

# 예측값 있는 데이터 불러오기
# 해당 데이터는 대회에서 주어진 examSet.csv 파일에 본 팀이 예측한 Static 데이터 예측값과 시계열 데이터 예측값을 'Predicted' 열로 추가한 데이터입니다.
df = pd.read_csv('examSet_Decision.csv')

df

# 그 중 optimization 문제 푸는데 필요한 가격, 월별운영비용, 예측값 열만 추출

df = df.iloc[:,121:124]

df

from cvxpy import constraints

# 각 생산정의 향후 6개월 월 평균 가스 생산량 실제값(실제값이 없으므로 예측값 사용)
a = np.array(df.iloc[:,2])
# 각 생산정의 가격
p = np.array(df.iloc[:,0])
# 각 생산정의 1개월당 고정 운영 비용
c = np.array(df.iloc[:,1])
# 셰일가스 기준 가격 (1 Mcf당 $7로 수정)
ps = 7
# 각 생산정의 구입 여부
x = cp.Variable((1,44), boolean=True)

# 제약식 : 각 생산정의 구입 여부 * 생산정의 가격의 합이 $15000000 이내
constraints = [sum(x*p) <= 15000000,]

# 목적함수 
obj = cp.Maximize(sum((6*a*ps-p-6*c)*x.T))


prob = cp.Problem(obj, constraints)

# maximize 한 profit
print('maximize 한 profit :' ,prob.solve())
print('status:', prob.status)

# 만약 모든 생산정을 다 구매할 경우 profit이 어떻게 나오는지 출력, 모든 생산정 구매하면 역시 수익이 적자가 나옴
print(sum(6*a*ps - p- 6*c))

# 각 생산정의 구매여부 출력
print("x decision:", x.T.value)

# 의사결정 여부를 Dataframe으로 변환
x_decision = pd.DataFrame(x.value)

x_decision = x_decision.transpose()

x_decision


# 원래 df에 의사결정 여부 열 추가
df['x_decision'] = x_decision

# 생산정을 구매할 경우 드는 총 비용 열 추가
df['total_exp'] = df['PRICE ($)']* df['x_decision']

# 총 비용이 $15000000 넘는지 체크
print('total expenditure (no more than $15,000,000 :', sum(df['total_exp']))

# 예측값과 의사결정 여부만 추출 후 csv로 저장
df2 = df.iloc[:,2:4]

df2.to_csv('Decision_final.csv', index=False)

df
