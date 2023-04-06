#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score
import warnings
warnings.filterwarnings('ignore')


# In[6]:


df=pd.read_csv("https://raw.githubusercontent.com/dsrscientist/DSData/master/winequality-red.csv")
df


# In[7]:


df.head()


# In[8]:


df.info()


# In[9]:


plt.figure(figsize=(100,70))
sns.set(font_scale=5)
plt.subplot(331)
plt.title("Quality vs Fixed Acidity",fontsize=70)
sns.barplot(x='quality',y = 'fixed acidity',data=df)

plt.subplot(332)
plt.title('Quality vs Volatile Acidity',fontsize=70)
sns.barplot(x='quality',y = 'volatile acidity',data=df)

plt.subplot(333)
plt.title('Quality vs Citric Acid',fontsize=70)
sns.barplot(x='quality',y = 'citric acid',data=df)

plt.subplot(334)
plt.title('Quality vs Residual Sugar', fontsize=70)
sns.barplot(x='quality',y = 'residual sugar',data=df)

plt.subplot(335)
plt.title('Quality vs Chlorides',fontsize=70)
sns.barplot(x='quality',y = 'chlorides',data=df)

plt.subplot(336)
plt.title('Quality vs Free Sulfur Dioxide',fontsize=70)
sns.barplot(x='quality',y = 'free sulfur dioxide',data=df)

plt.subplot(337)
plt.title('Quality vs Total Sulfur Dioxide', fontsize=70)
sns.barplot(x='quality',y = 'free sulfur dioxide',data=df)

plt.subplot(338)
plt.title('Quality vs Sulphates',fontsize=70)
sns.barplot(x='quality',y = 'sulphates',data=df)

plt.subplot(339)
plt.title('Quality vs Alcohol',fontsize=70 )
sns.barplot(x='quality',y = 'alcohol',data=df)


# In[11]:


bins=(2,6.5,8)
classes=['bad','good']
df['quality']=pd.cut(df['quality'],bins = bins,labels=classes)


# In[12]:


label_quality=LabelEncoder()


# In[13]:


df['quality']=label_quality.fit_transform(df['quality'])


# In[15]:


df['quality'].value_counts()


# In[16]:


plt.figure(figsize=(10,7))
sns.set(font_scale=3)
plt.title('Quality vs Count',fontsize=30)
sns.countplot(df['quality'])


# In[17]:


df.head()


# In[19]:


x=df.drop('quality',axis=1)
y=df['quality']


# In[20]:


x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2,random_state=21)


# In[22]:


sc=StandardScaler()


# In[23]:


x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


# In[24]:


RF=RandomForestClassifier(n_estimators=100)
RF.fit(x_train,y_train)
pred_RF=RF.predict(x_test)


# In[25]:


pred1=accuracy_score(y_test,pred_RF)
print(pred1*100)


# In[26]:


cml=confusion_matrix(y_test,pred_RF)
axl=sns.heatmap(cml,annot=True)
axl.set(xlabel='predict',ylabel='true')
plt.show()


# In[27]:


LR=LogisticRegression().fit(x_train,y_train)


# In[28]:


pred_LR=LR.predict(x_test)
cm2=confusion_matrix(y_test,pred_LR)
ax2=sns.heatmap(cm2,annot=True)
ax2.set(xlabel='predict',ylabel='true')
pred2=accuracy_score(y_test,pred_LR)
print('logistic regression accuracy score:',pred2*100)


# In[ ]:




