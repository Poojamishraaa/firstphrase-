#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[2]:


insurance_databse=pd.read_csv("https://raw.githubusercontent.com/dsrscientist/dataset4/main/medical_cost_insurance.csv")
insurance_databse


# In[3]:


insurance_databse.head()


# In[4]:


insurance_databse.shape


# In[5]:


insurance_databse.info()


# In[6]:


insurance_databse.isnull().sum()


# In[7]:


insurance_databse.describe()


# In[8]:


sns.set()
plt.figure(figsize=(6,6))
sns.distplot(insurance_databse['age'])
plt.title('Age Distribution')
plt.show()


# In[9]:


plt.figure(figsize=(6,6))
sns.countplot(x='sex',data=insurance_databse)
plt.title('Sex Distribution')
plt.show


# In[10]:


insurance_databse['sex'].value_counts()


# In[11]:


plt.figure(figsize=(6,6))
sns.displot(insurance_databse['bmi'])
plt.title('Bmi Distribution')
plt.show()


# In[12]:


plt.figure(figsize=(6,6))
sns.countplot(x='children',data=insurance_databse)
plt.title('Children Distribution')
plt.show()


# In[13]:


insurance_databse['children'].value_counts()


# In[14]:


plt.figure(figsize=(6,6))
sns.countplot(x='smoker',data= insurance_databse)
plt.title('Smoker')
plt.show()


# In[15]:


insurance_databse['smoker'].value_counts()


# In[16]:


plt.figure(figsize=(6,6))
sns.countplot(x='region',data=insurance_databse)
plt.title('region')
plt.show()


# In[17]:


insurance_databse['region'].value_counts()


# In[18]:


plt.figure(figsize=(6,6))
sns.distplot(insurance_databse['charges'])
plt.title['Charges Distribution']
plt.show()


# In[19]:


insurance_databse.replace({'sex':{'male':0,'female':1}},inplace=True)

insurance_databse.replace({'smoker':{'yes':0,'no':1}},inplace=True)

insurance_databse.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}},inplace=True)


# In[20]:


x=insurance_databse.drop(columns='charges',axis=1)
y=insurance_databse['charges']

print(x)


# In[21]:


print(y)


# In[22]:


x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=2)


# In[23]:


print(x.shape,x_train.shape,x_test.shape)


# In[24]:


regressor=LinearRegression()

regressor.fit(x_train, y_train)


# In[25]:


training_data_prediction=regressor.predict(x_train)


# In[26]:


r2_train=metrics.r2_score(y_train,training_data_prediction)
print('R square vale:',r2_train)


# In[27]:


test_data_prediction=regressor.predict(x_test)


# In[28]:


r2_test=metrics.r2_score(y_test,test_data_prediction)
print('R square vale:',r2_test)


# In[29]:


input_data=(29,0,32.11,2,1,3)


# In[30]:


input_data_as_numpy_array=np.asarray(input_data)

input_data_reshaped=input_data_as_numpy_array.reshape (1,-1)

prediction=regressor.predict(input_data_reshaped)

print(prediction)

print('The insurance cost is USD',prediction[0])


# In[ ]:





# In[ ]:




