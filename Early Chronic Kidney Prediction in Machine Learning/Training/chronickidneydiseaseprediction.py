#!/usr/bin/env python
# coding: utf-8

# In[95]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[96]:


pd.pandas.set_option('display.max_columns', None)


# In[97]:


dataset = pd.read_csv("kidney_disease.csv")
dataset.head()


# In[98]:


dataset = dataset.drop('id', axis=1)


# In[99]:


dataset.shape


# In[100]:


dataset.isnull().sum()


# In[101]:


dataset.describe()


# In[102]:


dataset.dtypes


# In[103]:


dataset.head()


# In[104]:


dataset['rbc'].value_counts()


# In[105]:


dataset['rbc'] = dataset['rbc'].replace(to_replace = {'normal' : 0, 'abnormal' : 1})


# In[106]:


dataset['pc'].value_counts()


# In[107]:


dataset['pc'] = dataset['pc'].replace(to_replace = {'normal' : 0, 'abnormal' : 1})


# In[108]:


dataset['pcc'].value_counts()


# In[109]:


dataset['pcc'] = dataset['pcc'].replace(to_replace = {'notpresent':0,'present':1})


# In[110]:


dataset['ba'].value_counts()


# In[111]:


dataset['ba'] = dataset['ba'].replace(to_replace = {'notpresent':0,'present':1})


# In[112]:


dataset['htn'].value_counts()


# In[113]:


dataset['htn'] = dataset['htn'].replace(to_replace = {'yes' : 1, 'no' : 0})


# In[114]:


dataset['dm'].value_counts()


# In[115]:


dataset['dm'] = dataset['dm'].replace(to_replace = {'\tyes':'yes', ' yes':'yes', '\tno':'no'})


# In[116]:


dataset['dm'] = dataset['dm'].replace(to_replace = {'yes' : 1, 'no' : 0})


# In[117]:


dataset['cad'].value_counts()


# In[118]:


dataset['cad'] = dataset['cad'].replace(to_replace = {'\tno':'no'})


# In[119]:


dataset['cad'] = dataset['cad'].replace(to_replace = {'yes' : 1, 'no' : 0})


# In[120]:


dataset['appet'].unique()


# In[121]:


dataset['appet'] = dataset['appet'].replace(to_replace={'good':1,'poor':0,'no':np.nan})


# In[122]:


dataset['pe'].value_counts()


# In[123]:


dataset['pe'] = dataset['pe'].replace(to_replace = {'yes' : 1, 'no' : 0})


# In[124]:


dataset['ane'].value_counts()


# In[125]:


dataset['ane'] = dataset['ane'].replace(to_replace = {'yes' : 1, 'no' : 0})


# In[126]:


dataset['classification'].value_counts()


# In[127]:


dataset['classification'] = dataset['classification'].replace(to_replace={'ckd\t':'ckd'})


# In[128]:


dataset["classification"] = [1 if i == "ckd" else 0 for i in dataset["classification"]]


# In[129]:


dataset.head()


# In[130]:


dataset.dtypes


# In[131]:


dataset['pcv'] = pd.to_numeric(dataset['pcv'], errors='coerce')
dataset['wc'] = pd.to_numeric(dataset['wc'], errors='coerce')
dataset['rc'] = pd.to_numeric(dataset['rc'], errors='coerce')


# In[132]:


dataset.dtypes


# In[133]:


dataset.describe()


# In[134]:


dataset.isnull().sum().sort_values(ascending=False)


# In[135]:


dataset.columns


# In[136]:


features = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu',
           'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad',
           'appet', 'pe', 'ane']


# In[137]:


for feature in features:
    dataset[feature] = dataset[feature].fillna(dataset[feature].median())


# In[138]:


dataset.isnull().any().sum()


# In[139]:


plt.figure(figsize=(24,14))
sns.heatmap(dataset.corr(), annot=True, cmap='YlGnBu')
plt.show()


# In[140]:


dataset.drop('pcv', axis=1, inplace=True)


# In[141]:


dataset.head()


# In[142]:


sns.countplot(dataset['classification'])


# In[143]:


X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]


# In[144]:


X.head()


# In[145]:


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model=ExtraTreesClassifier()
model.fit(X,y)

plt.figure(figsize=(8,6))
ranked_features=pd.Series(model.feature_importances_,index=X.columns)
ranked_features.nlargest(24).plot(kind='barh')
plt.show()


# In[146]:


ranked_features.nlargest(8).index


# In[147]:


X = dataset[['sg', 'htn', 'hemo', 'dm', 'al', 'appet', 'rc', 'pc']]
X.head()


# In[148]:


X.tail()


# In[149]:


y.head()


# In[150]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=33)


# In[151]:


print(X_train.shape)
print(X_test.shape)


# In[152]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[153]:


from sklearn.ensemble import RandomForestClassifier
RandomForest = RandomForestClassifier()
RandomForest = RandomForest.fit(X_train,y_train)

y_pred = RandomForest.predict(X_test)

print('Accuracy:', accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[154]:


from sklearn.ensemble import AdaBoostClassifier
AdaBoost = AdaBoostClassifier()
AdaBoost = AdaBoost.fit(X_train,y_train)

y_pred = AdaBoost.predict(X_test)

print('Accuracy:', accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[155]:


from sklearn.ensemble import GradientBoostingClassifier
GradientBoost = GradientBoostingClassifier()
GradientBoost = GradientBoost.fit(X_train,y_train)

y_pred = GradientBoost.predict(X_test)

print('Accuracy:', accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[ ]:




