#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#########################################
# Main program for activity prediction  #
# @author: A.Prof. Tran Van Lang, PhD   #
# File: activityPrediction.py           #
#########################################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
from joblib import dump, load
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from visual import visualization


# In[ ]:


# Đọc dữ liệu từ tập tin csv
df_train = pd.read_csv('data/BioassayDatasets/AID456red_train.csv')
df_test  = pd.read_csv('data/BioassayDatasets/AID456red_test.csv')


# In[ ]:


print( df_train.columns )


# In[ ]:


# Xóa các hàng có giá trị bị khuyết (missing values)
df_train.dropna(inplace=True)


# In[ ]:


# Xem qua về độ chênh của dữ liệu
data = df_train.loc[:,'NEG_01_NEG':'BadGroup']
plt.figure(figsize=(10, 6))
for col in data.columns:
    plt.plot(data[col], label=col)
plt.xlabel('Index')
plt.ylabel('Value')
#plt.legend()
plt.show()


# In[ ]:


print( 'Số lượng mẫu\n- Tập huấn luyện')
print( '    Loại inactivity:',df_train['Outcome'].value_counts()[0] )
print( '    Loại activity  :',df_train['Outcome'].value_counts()[1] )
print( '- Tập kiểm chứng')
print( '    Loại inactivity:',df_test['Outcome'].value_counts()[0] )
print( '    Loại activity  :',df_test['Outcome'].value_counts()[1] )


# In[ ]:


# Dữ liệu gốc dùng để huấn luyện và kiểm chứng
X_tr = df_train.drop('Outcome', axis=1).values
X_te = df_test.drop('Outcome', axis=1).values

y_tr = df_train['Outcome'].values
y_te = df_test['Outcome'].values


# In[ ]:


# Ban đầu, huấn luyện thử nghiệm mô hình theo GradientBoosting với các hyperparameter tuỳ chọn
X_train, X_test = X_tr, X_te
y_train, y_test = y_tr, y_te

gb = GradientBoostingClassifier( n_estimators=150, max_depth=10,random_state=42 )
gb.fit(X_train, y_train)
visualization('GradientBoosting',gb,X_test,y_test )


# In[ ]:


visualization('GradientBoosting',gb,X_test,y_test )


# In[ ]:


# Dùng thêm EasyEnsemble cũng với tham số tuỳ chọn
X_train, X_test = X_tr, X_te
y_train, y_test = y_tr, y_te

ee = EasyEnsembleClassifier(n_estimators=100, random_state=0,n_jobs=-1)
ee.fit(X_train, y_train)
visualization('EasyEnsemble',ee,X_test,y_test)


# In[ ]:


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


# In[ ]:


# Tạo dữ liệu huấn luyện và kiểm chứng sau khi đã rút gọn
X_train_se = df_train.drop(corr_features, axis=1).values
X_test_se  = df_test.drop(corr_features, axis=1).values


# In[ ]:


# Dùng EasyEnsemble với dữ liệu đã rút gọn thuộc tính
X_train, X_test = X_train_se, X_test_se
y_train, y_test = y_tr, y_te

ee = EasyEnsembleClassifier(n_estimators=100, random_state=0,n_jobs=-1)
ee.fit(X_train, y_train)
visualization('EasyEnsemble_selectedfeature',ee,X_test,y_test)


# In[ ]:


# Xử lý dữ liệu, dùng Min-max scaling để chuyển đổi giá trị dữ liệu về khoảng mong muốn
minmax_scaler = MinMaxScaler()
X_train_mms = minmax_scaler.fit_transform(X_train_se)
X_test_mms = minmax_scaler.fit_transform(X_test_se)


# In[ ]:


# Huấn luyện lại mô hình trên dữ liệu đã xử lý
X_train, X_test = X_train_mms, X_test_mms
y_train, y_test = y_tr, y_te

ee = EasyEnsembleClassifier(n_estimators=100, random_state=0,n_jobs=-1)
ee.fit(X_train, y_train)
visualization('EasyEnsemble_selectedfeatures_MMax',ee,X_test,y_test )


# In[ ]:


# Tạo mẫu để xử lý mất cân bằng dùng phương pháp SMOTE để tăng cường thêm
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_mms, y_tr)
print("Số lượng mẫu sau khi resample:", len(X_train_smote))


# In[ ]:


X_train, X_test = X_train_smote, X_test_mms
y_train, y_test = y_train_smote, y_te

ee = EasyEnsembleClassifier(n_estimators=40, random_state=9,n_jobs=-1)
ee.fit(X_train, y_train)
visualization('EasyEnsemble_selectedfeatures_MMax_SMOTE',ee,X_test,y_test )


# In[ ]:




