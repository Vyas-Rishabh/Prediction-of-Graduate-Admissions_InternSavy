#!/usr/bin/env python
# coding: utf-8

# # Prediction of Graduate Admissions from an Indian perspective

# ### Importing Libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[3]:


#Load CSV here
df = pd.read_csv("Admission_Dataset.csv")


# In[7]:


df.sample(7)


# In[8]:


df.columns


# In[9]:


df.shape


# df.describe()

# In[12]:


#Find missing value
df.isnull().sum()


# In[13]:


#finf duplicate value
df.duplicated().sum()


# In[14]:


df.head()


# ### Creating a copy and removing the Sl.No column

# In[15]:


df1 = df.copy()
df1.drop(['Serial No.'], axis=1, inplace=True)


# In[16]:


df1.head()


# ### Identifying & Removing outliers

# In[19]:


df1.boxplot(column=['Chance of Admit '])
plt.show()


# In[20]:


df1.boxplot(column=['GRE Score', 'TOEFL Score', 'University Rating'])
plt.show()


# In[21]:


df1.boxplot(column=['SOP','LOR ', 'CGPA', 'Research'])
plt.show()


# #### we can see there are outliers in chance of admit & LOR columns.

# In[23]:


Q1 = df1.quantile(0.25)
Q3 = df1.quantile(0.75)
IQR=Q3-Q1
IQR


# In[28]:


#upper limit
UL=Q3+IQR*1.5
print("Upper Limit:")
print(UL,"\n\n\n")

#lower limit
print("Lower Limit:")
LL=Q1-IQR*1.5
print(LL)


# In[29]:


#remove outliers based on the lower and upper limits
df_outliers_removed = df1[(df1>LL) & (df1<UL)]
df_outliers_removed


# In[30]:


#checking null values
df_outliers_removed.isnull().sum()


# In[31]:


#Drop the null values
df_outliers_removed.dropna(inplace=True)


# In[32]:


df_outliers_removed.shape


# In[35]:


df_outliers_removed.boxplot(figsize=(10,5), fontsize=8)
plt.show()


# ### we can see there are no outliers anymore.

# In[36]:


df2 = df_outliers_removed.copy()


# ### Univariate analysis

# In[37]:


df2['Chance of Admit '].plot.hist()
plt.xlabel('Chance of Admit ')
plt.show()


# In[39]:


df2['University Rating'].plot.hist()
plt.xlabel('rating')
plt.show()


# see the maximun no.of students are getting rating from 3 to 3.5

# In[40]:


df2['Research'].value_counts()


# 277 students have research experience and 209 students have no experience

# ### Bi-variate analysis

# In[41]:


df2.plot.scatter('GRE Score','Chance of Admit ')
plt.show()


# In[42]:


df2['Chance of Admit '].corr(df2['GRE Score'])


# chance of admit and GRE score are positively correlated.
# if GRE score increases there is more chance of getting admission.

# In[43]:


df2.plot.scatter('TOEFL Score','Chance of Admit ')
plt.show()


# In[44]:


df2['TOEFL Score'].corr(df2['Chance of Admit '])


# chance of admit and TOEFL score are positively correlated.
# if TOEFL score increases there is more chance of getting admission.

# In[46]:


df2.plot.scatter('CGPA','Chance of Admit ')
plt.show()


# In[47]:


df2['CGPA'].corr(df2['Chance of Admit '])


# chance of admit and CGPA are positively correlated. if CGPA increases there is more chance of getting admission.

# In[48]:


df2.plot.scatter('CGPA','TOEFL Score')
plt.show()


# In[49]:


df2.plot.scatter('CGPA','GRE Score')
plt.show()


# In[50]:


df2['CGPA'].corr(df2['GRE Score'])


# In[51]:


df2['CGPA'].corr(df2['TOEFL Score'])


# Students who have good CGPA , will definitely get a good score in TOEFL and GRE exams.

# ### Now, we'll Separating x and y

# In[52]:


x=df2.drop(['Chance of Admit '],axis=1)
y=df2['Chance of Admit ']
x.shape,y.shape


# In[53]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LR

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=56)


# ### Fit data into linear model

# In[55]:


lr = LR()


# In[56]:


lr.fit(x_train, y_train)


# ### Predicting over train and test set

# In[59]:


from sklearn.metrics import mean_absolute_error as mae, r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

pre_train = lr.predict(x_train)
mae_train = mae(pre_train, y_train)
mae_train


# In[60]:


pre_test=lr.predict(x_test)
mae_test=mae(pre_test,y_test)
mae_test


# ## Model Evaluation

# In[61]:


n = len(x_train)
m=len(x_test)


# ### Train data

# In[63]:


RMSE = np.sqrt(mean_squared_error(y_train,pre_train))
MSE = mean_squared_error(y_train, pre_train)
MAE = mean_absolute_error(y_train, pre_train)
r2_train = r2_score(y_train, pre_train)
adj_r2 = 1-(1-r2_train)*(n-1)/(n-mae_train-1)
print(RMSE)
print(MSE)
print(MAE)
print(r2_train)
print(adj_r2)


# ### Test data

# In[64]:


RMSE_test = np.sqrt(mean_squared_error(y_test,pre_test))
MSE_test = mean_squared_error(y_test, pre_test)
MAE_test = mean_absolute_error(y_test, pre_test)
r2_test = r2_score(y_test, pre_test)
adj_r2_test = 1-(1-r2_test)*(m-1)/(m-mae_test-1)
print(RMSE_test)
print(MSE_test)
print(MAE_test)
print(r2_test)
print(adj_r2_test)


# ## Accuracy of the model

# In[65]:


print('Accuracy of train set :',r2_train)
print('Accuracy of test set :',r2_test)


# You can find this project on <a href="https://github.com/Vyas-Rishabh/Prediction-of-Graduate-Admissions_InternSavy"><b>GitHub.</b></a>
