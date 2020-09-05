#!/usr/bin/env python
# coding: utf-8

# In[105]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[106]:


loan_data = pd.read_csv('loan_data.csv')


# In[107]:


loan_data.head()


# In[108]:


loan_data['purpose'].unique()


# In[109]:


loan_data.info()


# In[110]:


loan_data['not.fully.paid'].unique()


# In[111]:


loan_data['credit.policy'].unique()


# In[112]:


sns.set_palette('GnBu_r')
sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
loan_data[loan_data['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
loan_data[loan_data['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')


# In[113]:


plt.figure(figsize=(10,6))
loan_data[loan_data['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='not.fully.paid=1')
loan_data[loan_data['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')


# In[114]:


plt.figure(figsize=(11,7))
sns.lmplot(y='int.rate',x='fico',data=loan_data,hue='credit.policy',
           col='not.fully.paid',palette='Set1')


# In[115]:


pur =pd.get_dummies(loan_data['purpose'],drop_first=True)


# In[116]:


pur.head()


# In[117]:


loan_data = pd.concat([loan_data,pur],axis=1)


# In[118]:


loan_data.head()


# In[119]:


loan_data['credit.policy'].unique()


# In[120]:


loan_data.drop('purpose',axis=1,inplace=True)


# In[121]:


loan_data.head()


# In[122]:


loan_data.info()


# In[123]:


X = loan_data.drop('not.fully.paid',axis=1)


# In[124]:


y = loan_data['not.fully.paid']


# In[125]:


X.head()


# In[126]:


y.head()


# In[127]:


from sklearn.model_selection import train_test_split


# In[128]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[129]:


from sklearn.tree import DecisionTreeClassifier


# In[130]:


dtree = DecisionTreeClassifier()


# In[131]:


dtree.fit(X_train,y_train)


# In[132]:


predd = dtree.predict(X_test)


# In[133]:


from sklearn.metrics import confusion_matrix,classification_report


# In[134]:


print(confusion_matrix(y_test,predd))
print('\n')
print(classification_report(y_test,predd))


# In[135]:


from sklearn.ensemble import RandomForestClassifier


# In[136]:


rfc = RandomForestClassifier(n_estimators=600)


# In[137]:


rfc.fit(X_train,y_train)


# In[138]:


predd = rfc.predict(X_test)


# In[139]:


print(confusion_matrix(y_test,predd))
print('\n')
print(classification_report(y_test,predd))


# In[ ]:





# # the End
