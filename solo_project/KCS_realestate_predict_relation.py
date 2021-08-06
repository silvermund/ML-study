import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

plt.rcParams['font.family'] = 'NanumGothic'

#1. 데이터

# base_rate = pd.read_csv('./solo_project/real_estate_predict.csv', encoding='EUC-KR', usecols=[1])
# base_rate = base_rate[0:2256]
# # base_rate = np.array(base_rate)
# # print(base_rate.shape) #(2256, 1)

# bank_rate = pd.read_csv('./solo_project/real_estate_predict.csv', encoding='EUC-KR', usecols=[2])
# bank_rate = bank_rate[0:2256]
# # bank_rate = np.array(bank_rate)
# # print(bank_rate.shape) #(2256, 1)

# kospi = pd.read_csv('./solo_project/real_estate_predict.csv', encoding='EUC-KR', usecols=[3])
# kospi = kospi[0:2256]
# # kospi = np.array(kospi)
# # print(kospi.shape) #(2256, 1)

# cpi = pd.read_csv('./solo_project/real_estate_predict.csv', encoding='EUC-KR', usecols=[4])
# cpi = cpi[0:2256]
# # cpi = np.array(cpi)
# # print(cpi.shape) #(2256, 1)

# volume = pd.read_csv('./solo_project/real_estate_predict.csv', encoding='EUC-KR', usecols=[5])
# volume = volume[0:2256]
# # volume = np.array(volume)
# # print(volume.shape) #(2256, 1)

# m2 = pd.read_csv('./solo_project/real_estate_predict.csv', encoding='EUC-KR', usecols=[6])
# m2 = m2[0:2256]
# # m2 = np.array(m2)
# # print(m2.shape) #(2256, 1)

# price = pd.read_csv('./solo_project/real_estate_predict.csv', encoding='EUC-KR', usecols=[7])
# price = price[0:2256]
# # price = np.array(price)
# # print(price.shape) #(2256, 1)

df = pd.read_csv('./solo_project/real_estate_predict.csv', encoding='EUC-KR', usecols=[1,2,3,4,5,6,7,8])

corr = df.corr(method = 'pearson')
print(corr)

# df_heatmap = sns.heatmap(corr, cbar = True, annot = True, annot_kws={'size' : 20}, fmt = '.2f', square = True, cmap = 'Blues')

# X = df.base_rate.values
# Y = df.price.values

# plt.scatter(X, Y, alpha=0.5)
# plt.title('기준금리-가격 상관관계')
# plt.xlabel('base_rate')
# plt.ylabel('price')
# plt.show()


# cov = np.corrcoef(X, Y)[0,1]
# print(cov) #-0.7700500291296795

# print(stats.pearsonr(X,Y)) #(-0.7700500291296795, 0.0)

# # 뒤 결과 값이 p-value인데, 귀무가설 "상관관계가 없다"에 대한 검정 결과 p-value가 3.46e-42라는 0에 아주 매우 가까운 값이 나왔으므로 귀무가설을 기각할 수 있음을 알 수 있습니다.


# X = df.bank_rate.values
# Y = df.price.values

# plt.scatter(X, Y, alpha=0.5)
# plt.title('대출금리-가격 상관관계')
# plt.xlabel('bank_rate')
# plt.ylabel('price')
# plt.show()


# print(stats.pearsonr(X,Y)) #(-0.7921732907711457, 0.0)

# X = df.kospi.values
# Y = df.price.values

# plt.scatter(X, Y, alpha=0.5)
# plt.title('코스피-가격 상관관계')
# plt.xlabel('kospi')
# plt.ylabel('price')
# plt.show()


# print(stats.pearsonr(X,Y)) #(0.6228885609055522, 7.35151817220132e-245)


# X = df.cpi.values
# Y = df.price.values

# plt.scatter(X, Y, alpha=0.5)
# plt.title('소비자물가지수 CPI-가격 상관관계')
# plt.xlabel('cpi')
# plt.ylabel('price')
# plt.show()

# print(stats.pearsonr(X,Y)) #(0.9671938329949575, 0.0)

X = df.volume.values
Y = df.price.values

plt.scatter(X, Y, alpha=0.5)
plt.title('아파트 거래량-가격 상관관계')
plt.xlabel('volume')
plt.ylabel('price')
plt.show()


print(stats.pearsonr(X,Y)) #(0.2955604425745328, 3.923234782267607e-47)



# X = df.m2.values
# Y = df.price.values

# plt.scatter(X, Y, alpha=0.5)
# plt.title('광의 통화(M2)-가격 상관관계')
# plt.xlabel('m2')
# plt.ylabel('price')
# plt.show()


# print(stats.pearsonr(X,Y)) #(0.9671938329949575, 0.0)

# X = df.unsold.values
# Y = df.price.values

# plt.scatter(X, Y, alpha=0.5)
# plt.title('미분양수-가격 상관관계')
# plt.xlabel('unsold')
# plt.ylabel('price')
# plt.show()


# print(stats.pearsonr(X,Y)) #(-0.44962388478293774, 9.954467828303841e-114)

