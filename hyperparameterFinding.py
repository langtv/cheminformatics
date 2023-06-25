#!/usr/bin/env python
# coding: utf-8

# In[1]:


#######################################
# Finding the hyperparameter          #
# @author: A.Prof. Tran Van Lang, PhD #
# File: hyperparameterFinding.py      #
#######################################

import pandas as pd
import numpy as np
from time import time
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from imblearn.ensemble import EasyEnsembleClassifier
from visual import visualization


# In[2]:


# Đọc dữ liệu từ tập tin csv
df_train = pd.read_csv('data/BioassayDatasets/AID456red_train.csv')
df_test  = pd.read_csv('data/BioassayDatasets/AID456red_test.csv')


# In[3]:


# Xóa các hàng có giá trị bị khuyết (missing values)
df_train.dropna(inplace=True)


# In[4]:


# Dữ liệu gốc dùng để huấn luyện và kiểm chứng
X_tr = df_train.drop('Outcome', axis=1).values
X_te = df_test.drop('Outcome', axis=1).values

y_tr = df_train['Outcome'].values
y_te = df_test['Outcome'].values


# In[5]:


X_train, X_test = X_tr, X_te
y_train, y_test = y_tr, y_te


# In[6]:


# Rút gọn thuộc tính dùng ma trận tương quan để chọn đặc trưng
corr_matrix = df_train.drop('Outcome', axis=1).corr()
threshold = 0.75

corr_features = set()  
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            colname = corr_matrix.columns[i]
            corr_features.add(colname)

selected_features = set(df_train.columns) - corr_features
corr_features.add('Outcome')
print( 'Thuộc tính loại bỏ:\n',corr_features )
print( '\nThuộc tính chọn :\n',selected_features )


# In[7]:


# Tạo dữ liệu huấn luyện và kiểm chứng sau khi đã rút gọn
X_train_se = df_train.drop(corr_features, axis=1).values
X_test_se  = df_test.drop(corr_features, axis=1).values


# In[8]:


# Xử lý dữ liệu, dùng Min-max scaling để chuyển đổi giá trị dữ liệu về khoảng mong muốn
minmax_scaler = MinMaxScaler()
X_train_mms = minmax_scaler.fit_transform(X_train_se)
X_test_mms = minmax_scaler.fit_transform(X_test_se)


# In[9]:


# Tạo mẫu để xử lý mất cân bằng dùng phương pháp SMOTE để tăng cường thêm
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_mms, y_tr)
print("Số lượng mẫu sau khi resample:", len(X_train_smote))


# In[ ]:


# Chọn các tham số từ GridSearchCV cho EasyEnsemble
X_train, X_test = X_train_smote, X_test_mms
y_train, y_test = y_train_smote, y_te

param_grid = {
    'n_estimators': np.arange(1,30,10),
    'random_state': np.arange(0,8,3)
}                       # cv = 2, Tham số tốt nhất được chọn:  {'n_estimators': 1, 'random_state': 0}
t0 = time()
gs = GridSearchCV( EasyEnsembleClassifier(n_jobs=-1),param_grid,cv=10,n_jobs=-1)
gs.fit(X_train, y_train)
print( 'Eslaped time: %5.2f seconds' %(time()-t0) )
visualization('EasyEnsemble_selectedfeatures_MMax_SMOTE_GS',gs,X_test,y_test)

print( 'Tham số tốt nhất được chọn: ',gs.best_params_ )
print( 'Kết quả\n: ',gs.cv_results_ )


# In[ ]:




