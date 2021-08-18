# 실습 : 다차원의 outlier가 출력되도록 함수 수정


import numpy as np

aaa = np.array([[1,     2, 10000, 3, 4,   6,   7,   8, 90, 100,  5000],
                [1000,2000,3,   4000,5000,6000,7000,8,9000,10000,1001]])

# (2, 10) -> (10, 2)

aaa = aaa.transpose()
print(aaa.shape)

def outliers(data_out):
    allout = []
    for i in range(data_out.shape[1]):
    
        quartile_1, q2, quartile_3 = np.percentile(data_out[:,i], [25, 50, 75])
        print("1사분위 : ", quartile_1)
        print("q2 : ", q2)
        print("3사분위 : ", quartile_3)
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        
        a = np.where((data_out[:,i]>upper_bound) | (data_out[:,i]<lower_bound)) 
        allout.append(a)
        
outliers_loc =  outliers(aaa)

print('이상치의 위치 :', outliers_loc)

# 시각화
# 위 데이터를 boxplot으로 그리시오!!!

import matplotlib.pyplot as plt

plt.boxplot(aaa)
plt.show()