#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


from sklearn.datasets import load_iris
iris = load_iris()
data1 = pd.DataFrame(data= np.c_[iris['data'], iris['target']],columns= iris['feature_names'] + ['target'])

data1.head()


# In[3]:


feature_columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)']
X = data1[feature_columns]
y = data1['target']


# In[4]:


X


# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:


from sklearn.neighbors import KNeighborsClassifier

# Instantiate learning model (k = 3)
classifier = KNeighborsClassifier(n_neighbors=6)

# Fitting the model
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# # 6/4/23
# 
# 

# In[7]:


from sklearn.metrics import confusion_matrix, accuracy_score
accuracy_score(y_test,y_pred)


# In[8]:


len(X)


# In[9]:


d1=pd.read_csv('Social_Network_Ads.csv')
d1.head(5)


# In[10]:


del d1['User ID']
d1.head(5)


# In[11]:



# Import label encoder
from sklearn import preprocessing
  
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
d1['Gender']= label_encoder.fit_transform(d1['Gender'])
  
d1['Gender'].unique()


# In[12]:


d1


# In[13]:



X1 = d1[['Age','EstimatedSalary','Gender']]
y1 = d1['Purchased']


# In[14]:


X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size = 0.25, random_state = 0)


# In[17]:


from sklearn.neighbors import KNeighborsClassifier

# Instantiate learning model (k = 3)
classifier = KNeighborsClassifier(n_neighbors=6)

# Fitting the model
classifier.fit(X_train1, y_train1)

# Predicting the Test set results
y_pred1 = classifier.predict(X_test1)


# In[18]:


from sklearn.metrics import confusion_matrix, accuracy_score

accuracy_score(y_test1,y_pred1)


# In[27]:


neighbors=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]

acc=[]
for i in neighbors:
    
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(X_train1, y_train1)
    y_pred1 = classifier.predict(X_test1)
    
    print(i)
    
    a = accuracy_score(y_test1,y_pred1)
    
    print(a)
        
    acc.append(a)
    


acc

    


# In[38]:


import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D


line1,=plt.plot(neighbors,acc,'b',label="Testing acuracy")
plt.legend(handler_map={line1:HandlerLine2D(numpoints=2)})

plt.ylabel('Accuracy score')
plt.xlabel('n_neighbors')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




