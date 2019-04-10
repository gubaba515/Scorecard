# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 11:38:13 2019

@author: GKX
"""

import pandas as pd
import numpy as np
import re
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc,roc_curve,accuracy_score,roc_auc_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
import sys
sys.path.append('E:\\评分卡')
import ScoreCardFun as sc

#%% 数据预处理
dat = pd.read_csv(open("E:\\评分卡\\XXX.csv", encoding="UTF-8"),low_memory=False)
dat = dat.loc[(dat['AA'].notnull()) & (dat['BB'].notnull()) & (dat['CC']>20180401),:]
#手动删除异常样本(orderid重复或某些字段缺失)
dat = dat.loc[(dat['AA']!='T1') & (dat['T2']!=-1) &
              (dat['T3']!=-1),:]
#更改列名
dat.columns = dat.columns.map(lambda x: x[2:])
#更改行名
dat.index = dat['AA']


#删除无意义的列
dat_clean = dat.drop(dat.columns[[0,2,6,7,8,9,10,11,12]], axis=1).copy()
dis_col = pd.Series(["t1","t2","t3"])
dis_col = dis_col.append(pd.Series([dat_clean.columns.values[i] for i in range(dat_clean.shape[1]) if 'flag' in dat_clean.columns.values[i]]))
#删除失效指标
uni_col = [c for c in dat_clean.columns if "tt" in c]
uni_col = pd.Series(list(set(uni_col).difference(set(["aa","bb"]))))
dis_col = dis_col.append(uni_col)
dat_clean = dat_clean.loc[:,~dat_clean.columns.isin(dis_col)]

#将某些字符型转换为数值型
dat_clean.A.value_counts(dropna=False)
dat_clean.A.replace({"A":1, "B":2, "C":3, "D":4, "D":4, "-1":-1}, inplace=True)
dat_clean.S.value_counts(dropna=False)
dat_clean.S.replace({"C":3, "D":4, "-1":-1}, inplace=True)
dat_clean.D.replace({"星期一":1, "星期二":2, "星期三":3, "星期四":4, "星期五":5, "星期六":6, "星期日":7}, inplace=True)
dat_clean.contacts1.value_counts(dropna=False)
dat_clean.contacts1.replace({"父母":1, "兄弟姐妹":1, "配偶":2, "同事":2, "子女":2, "本人":3, "朋友":3, "其他":3, "物流公司":3, np.NaN:3}, inplace=True)
dat_clean.bank = 1*(dat_clean.bank == "XX银行")

#将缺失值统一为-1
type_col = dat_clean.dtypes
for i in range(dat_clean.shape[1]):
    if type_col[i]!="object":
        if pd.Series(-99999).isin(dat_clean.iloc[:,i])[0]:
            dat_clean.iloc[:,i].replace({-99999:-1}, inplace=True)
        if pd.Series(-9999).isin(dat_clean.iloc[:,i])[0]:
            dat_clean.iloc[:,i].replace({-9999:-1}, inplace=True)
    dat_clean.iloc[:,i].fillna(-1, inplace=True)

##将逻辑值转换为数值
#for i in range(dat_clean.shape[1]-1):   #最后一列为时间格式这里去掉
#    if any(dat_clean.iloc[:,i].drop_duplicates().isin(["false", "FALSE", False, "true", "TRUE", True])):
#        try:    #由于False和True会默认为bool类型故这里不打引号，但是这样会将0和1误判为False和True，故这里使用try
#            dat_clean.iloc[:,i].replace({"false":0, "FALSE":0, False:0, "true":1, "TRUE":1, True:1}, inplace=True)
#            dat_clean.iloc[:,i] = dat_clean.iloc[:,i].astype("int64")
#        except:
#            pass

#删除字符串字段
dat_clean = dat_clean.loc[:,dat_clean.dtypes!="Q"]


#选择入模数据
max_overdue_day = dat_clean.max_overdue
approv_date = dat_clean.approve_time.map(lambda x: datetime.datetime.strptime(str(x),"%Y%m%d"))
dat_model = dat_clean.drop(["TT","TA"], axis=1).copy()

#删除缺失过多的字段
dat_model = dat_model.loc[:,dat_model.apply(lambda x: sum(x==-1)<0.9*len(dat_model))]
#删除某取值比例过大的字段
dat_model = dat_model.loc[:,dat_model.apply(lambda x: max(x.value_counts()/len(dat_model))<=0.95)]
#删除手机号及时间
dat_model = dat_model.loc[:,dat_model.apply(lambda x: max(x)<10**10)]
#删除取值仅为-1,1或-1,0的字段
dat_model = dat_model.loc[:,dat_model.apply(lambda x: (len(x.drop_duplicates())==2)&((sum(x.drop_duplicates())==-1)|(sum(x.drop_duplicates())==0)))==False]
#删除缺失值除去后某取值比例过大的字段
dat_model = dat_model.loc[:,dat_model.apply(lambda x: max(x[x!=-1].value_counts()/len(x[x!=-1]))<=0.995)]
#删除命中黑名单相关变量
dat_model = dat_model.loc[:,dat_model.columns.map(lambda x: ("A" in x)|("B" in x))==False]
#删除高度相关的字段
var_cor1 = dat_model.columns.values[dat_model.corr().apply(lambda x: sum(x>0.9999)>1)]
# var_cor1
dat_model = dat_model.loc[:,~dat_model.columns.isin(var_cor1[[1,2,3,4,5,11,13]])]

#手动删除无用字段
dat_model.drop(["XX","XXX","XXXX"],axis=1,inplace=True)

#var1为分类型
var1 = set(dat_model.columns[dat_model.apply(lambda x: len(x.value_counts())<9)&dat_model.columns.map(lambda x: "A" not in x).values])
#var2为连续型
var2 = set(dat_model.columns).difference(var1)
var1 = var1.difference(['TT'])

#添加因变量
dat_model["TT"] = 1*(max_overdue_day>7)
np.mean(dat_model.overdue)



#%% 离散化及WOE变换函数
#bairongmultiapplyinfo_al_m12_cell_notbank_selfnum
dat_discrete, knot = sc.Discrete_fun(dat_model, var2, ["AA","SS","DD"])
dat_discrete_clean, term_wrong = sc.Merge_fun(dat_discrete, var1)
term_wrong   #若有特殊情况输出则需特殊处理

#部分数据分析
dat_per = dat_discrete_clean.drop(dat_discrete_clean.columns[[i for i in range(2,61)]],axis = 1).copy()
dat_per["QQ"] = dat_discrete_clean[dat_discrete_clean.columns[3]]
dat_per["WW"] = dat_discrete_clean.T
dat_per["EE"] = dat_discrete_clean.Z


#合并
dat_discrete_clean.A.replace({-1:1}, inplace=True)

woeTable_list0, IV_list0 = sc.WOEtable(dat_discrete_clean)
#选择IV>0.006的变量
IV_list = IV_list0[IV_list0[0]>0.006].copy()
#删除高度相关的字段
var_cor2 = dat_discrete_clean[IV_list.index].columns.values[dat_discrete_clean[IV_list.index].corr().apply(lambda x: sum(x>0.95)>1)]
#dat_discrete_clean[var_cor2].corr()
IV_list = IV_list[~IV_list.index.isin(var_cor2[[1,3,5,7,10,11,12]])]
woeTable_list = [x for x in woeTable_list0 if x.index.name in IV_list.index.values]

dat_woe = sc.WOE(dat_discrete_clean, woeTable_list, IV_list)

#dat_woe.to_csv("dat_woe.csv")

cut_date = '2018-06-15'
x_train = dat_woe.iloc[:,:-1].loc[approv_date<cut_date].copy()
y_train = dat_woe.iloc[:,-1].loc[approv_date<cut_date].copy()
x_test = dat_woe.iloc[:,:-1].loc[approv_date>=cut_date].copy()
y_test = dat_woe.iloc[:,-1].loc[approv_date>=cut_date].copy()
y_test.value_counts()

#x_train, x_test, y_train, y_test = train_test_split(dat_woe.iloc[:,:-1], dat_woe.iloc[:,-1], test_size = 0.3)

#%% 变量选择（LASSO）
param_grid = {'C': [0.1, 0.5, 1, 10, 100]}
alg1 = LogisticRegression(penalty="l1")
grid_search = GridSearchCV(alg1, param_grid, scoring = "roc_auc", cv=5, n_jobs = -1)
grid_search.fit(x_train, y_train)
grid_search.best_params_

alg1 = LogisticRegression(C=0.1, penalty="l1", class_weight="balanced")
alg1.fit(x_train, y_train)
alg1.coef_

var_s = x_train.columns[alg1.coef_[0]!=0].values
var_s

#%% 逻辑回归
var_s = ["A","B","C","D","E"]
x_train_s = x_train[var_s]
x_test_s = x_test[var_s]

logit = sm.Logit(y_train,add_constant(x_train_s))
lr = logit.fit()
lr.summary2()

#查看vif
X = add_constant(x_train_s)
vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
vif

test_prob = lr.predict(add_constant(x_test_s))
fpr, tpr, thresholds = roc_curve(y_test, test_prob, pos_label=1)
plt.plot(fpr, tpr);plt.text(0.5,0.5,"AUC="+str(np.round(auc(fpr, tpr),4)),fontsize=15)


ks,ks_plot = sc.KS_fun(test_prob,y_test)
ks

#%% 结果分析

woe_table = [x for x in woeTable_list if x.index.name in var_s]
dat_woe_s = dat_woe[var_s]
beta = lr.params
#得分矩阵
tol_score = sc.Score_fun(x_train_s,dat_woe_s,woe_table,beta,digit=2)
#测试集样本得分
sample_score = sc.Test_data_score_fun(x_test_s,tol_score,basescore=500)
#切分点
qq = np.r_[np.array([min(sample_score)-1]), np.round(np.unique(np.percentile(sample_score,np.arange(10,100,10))),2), np.array([max(sample_score)])]
#逾期情况
rate = sc.Rate_fun(sample_score, y_test, qq)
rate
sc.Rate_plot(rate)


#%% 验证集

dat_verify = dat_clean[np.r_[["A"], var_s]].copy()
dat_verify['B'] = 1*(dat_verify.max_overdue>7)
dat_verify.drop("T",axis=1,inplace=True)

#合并
dat_verify.applyinfo_islocation.replace({-1:1}, inplace=True)

#选中变量的切分点
knot_s = knot[knot[0].isin(var_s)]
#验证集样本得分
verify_data_score = sc.Verify_data_score_fun(dat_verify, knot_s, tol_score, 500)

#逾期情况
rate0 = sc.Rate_fun(verify_data_score, dat_verify['A'], qq)
rate0
sc.Rate_plot(rate0)


#%% 评分卡输出
card = sc.ScoreCard_fun(knot_s, tol_score, x_train_s)
