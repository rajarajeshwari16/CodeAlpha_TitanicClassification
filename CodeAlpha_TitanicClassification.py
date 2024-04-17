#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/SarveshThiru/CodeAlpha_Titanic_Classification/blob/main/Titanic_Classification.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ## Import Modules

# In[28]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Loading the dataset

# In[29]:


train_df = pd.read_csv(r"train.csv")
test_df =pd.read_csv(r"test.csv")


# ## BASIC INSPECTIONS ON THE DATASET:
# 

# In[30]:


train_df.head()


# ## Displaying first 5 examples in test dataframe

# In[31]:


test_df.head()


# ## checking for the information of the train dataset

# In[32]:


train_df.info()


# ## checking for the information of the test dataset

# In[33]:


test_df.info()


# ## Describing the train dataset

# In[34]:


train_df.describe()


# ## Describing the test dataset

# In[35]:


test_df.describe()


# ## Checking the datatypes of the Dataset

# In[36]:


train_df.dtypes


# ## Data Analysis:

# In[37]:


# categorial attributes:

sns.countplot(x='Survived', data=train_df)
plt.ylabel('Count')
plt.show()


# In[38]:


sns.countplot(x='Pclass', data=train_df)
plt.ylabel('Count')
plt.show()


# In[39]:


sns.countplot(x='Sex', data=train_df)
plt.ylabel('Count')
plt.show()


# In[40]:


sns.countplot(x='SibSp', data=train_df)
plt.ylabel('Count')
plt.show()


# In[41]:


sns.countplot(x='Parch', data=train_df)
plt.ylabel('Count')
plt.show()


# In[42]:


sns.countplot(x='Embarked', data=train_df)
plt.ylabel('Count')
plt.show()


# In[43]:


#Numerical Attributes:

sns.histplot(x='Age', data=train_df)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Distribution of Ages')
plt.show()


# In[44]:


sns.displot(x='Fare', data=train_df)
plt.ylabel('Count')
plt.show()


# In[45]:


class_fare = train_df.pivot_table(index='Pclass',values='Fare')
class_fare.plot(kind='bar')
plt.xlabel('Pclass')
plt.ylabel('Average Fare')
plt.xticks(rotation=0)
plt.show()


# In[46]:


class_fare = train_df.pivot_table(index='Pclass',values='Fare',aggfunc=np.sum)
class_fare.plot(kind='bar')
plt.xlabel('Pclass')
plt.ylabel('Total Fare')
plt.xticks(rotation=0)
plt.show()


# In[47]:


sns.barplot(data=train_df,x='Pclass',y='Fare',hue='Survived')
plt.show()


# In[48]:


sns.barplot(data=train_df,x='Survived',y='Fare',hue='Pclass')
plt.show()


# ## Data PreProcessing

# In[49]:


train_len=len(train_df)
df=pd.concat([train_df,test_df],axis=0)
df=df.reset_index(drop=True)
df


# ### Check for Null Values

# In[50]:


df.isnull().sum()


# In[51]:


df=df.drop(columns=['Cabin'],axis=1)


# In[52]:


df


# In[53]:


df.columns


# ## Filling missing values

# In[54]:


df['Age'].mean


# In[55]:


df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())


# In[56]:


df['Embarked'].mode()[0]


# In[57]:


df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])


# ## Log transformation for uniform data distribution:

# In[58]:


sns.distplot(train_df['Fare'])


# In[59]:


train_df['Fare'] = np.log(train_df['Fare']+1)


# In[60]:


sns.distplot(train_df['Fare'])


# ## Correlation matrix

# In[61]:


# Assuming 'Name' is a column containing strings, and you want to exclude it
numeric_columns = train_df.select_dtypes(include=[np.number]).columns
subset_df = train_df[numeric_columns]

# Now compute the correlation matrix
corr_matrix = subset_df.corr()
plt.figure(figsize=(15,9))
sns.heatmap(corr_matrix,annot=True,cmap='coolwarm')


# In[62]:


df.head()


# ## Drop Unnecessary Columns

# In[63]:


df=df.drop(['Name','Ticket'],axis=1)


# In[64]:


df


# In[65]:


df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)


# ## Spliting dataset

# In[66]:


train=df.iloc[:train_len:]
test=df.iloc[train_len:,:]


# In[67]:


train.head()


# In[68]:


test.head()


# ## Train-Test Split

# In[69]:


x=df.drop(columns=['PassengerId','Survived'],axis=1)
y=train['Survived']


# In[70]:


x.head()


# In[71]:


x.isnull().sum()


# In[72]:


df


# In[73]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    train.drop(['Survived', 'PassengerId'], axis=1),
    train['Survived'],
    test_size=0.2,
    random_state=42)


# ## Checking for the shape of the training features(x) in the dataset:

# In[74]:


x_train.shape


# ## Checking for the shape of the testing features(x) in the dataset:

# In[75]:


x_test.shape


# ## Checking for the shape of the testing Target values(y) in the dataset:

# In[76]:


y_test.shape


# ## Checking for the shape of the traning target values(y) in the dataset:

# In[77]:


y_train.shape


# ## Model Evaluation:

# ## 1) Logistic Regression

# In[78]:


from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression()


# In[79]:


model_lr.fit(x_train,y_train)
y_pred = model_lr.predict(x_test)


# ## Liogistic regression Model Evaluation:

# In[80]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)


# In[81]:


print(f"Accuracy: {accuracy}")


# ## 2) Decision Tree Classifier

# In[82]:


from sklearn.tree import DecisionTreeClassifier
model_dtc=DecisionTreeClassifier()


# In[83]:


model_lr.fit(x_train,y_train)
y_pred = model_lr.predict(x_test)


# ## Decision tree model Evaluation:

# In[84]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)


# In[85]:


print(f"Accuracy: {accuracy}")


# ## 3) Random Forest Classifier

# In[86]:


from sklearn.ensemble import RandomForestClassifier
model_rfc=RandomForestClassifier()


# In[87]:


model_lr.fit(x_train,y_train)
y_pred = model_lr.predict(x_test)


# ## Random Forest model evaluation:

# In[88]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)


# In[89]:


print(f"Accuracy: {accuracy}")


# ## Support Vector Machines (SVM):

# In[90]:


from sklearn.svm import SVC


# In[91]:


svm_model = SVC(kernel='linear')


# In[92]:


svm_model.fit(x_train, y_train)


# In[93]:


y_pred_svm = svm_model.predict(x_test)


# ## SVM Evaluation:

# In[94]:


accuracy_svm = accuracy_score(y_test, y_pred_svm)


# In[95]:


print("SVM Accuracy:", accuracy_svm)


# ## K-Nearest Neighbors (KNN):

# In[96]:


from sklearn.neighbors import KNeighborsClassifier


# In[97]:


knn_model = KNeighborsClassifier(n_neighbors=5)


# In[98]:


knn_model.fit(x_train, y_train)


# In[99]:


y_pred_knn = knn_model.predict(x_test)


# ## KNN Evaluation

# In[100]:


accuracy_knn = accuracy_score(y_test, y_pred_knn)


# In[101]:


print("KNN Accuracy:", accuracy_knn)


# ## XGBoost:

# In[102]:


get_ipython().system('pip install xgboost')


# In[103]:


from xgboost import XGBClassifier


# In[104]:


model_xgb = XGBClassifier()


# In[105]:


model_xgb.fit(x_train, y_train)


# In[106]:


y_pred = model_xgb.predict(x_test)


# ## Evaluation:

# In[107]:


accuracy = accuracy_score(y_test, y_pred)


# In[108]:


print(f"Accuracy: {accuracy}")


# ## Adaboost Classifier

# In[109]:


from sklearn.ensemble import AdaBoostClassifier


# In[110]:


model_adaboost = AdaBoostClassifier()


# In[111]:


model_adaboost.fit(x_train, y_train)


# In[112]:


y_pred_adaboost = model_adaboost.predict(x_test)


# ## Adaboost model evaluation

# In[113]:


accuracy_adaboost = accuracy_score(y_test, y_pred_adaboost)


# In[114]:


print(f"AdaBoost Accuracy: {accuracy_adaboost}")


# ## Catboost Classifier

# In[115]:


pip install catboost


# In[116]:


from catboost import CatBoostClassifier


# In[117]:


model_catboost = CatBoostClassifier()


# In[118]:


model_catboost.fit(x_train, y_train)
y_pred_catboost = model_catboost.predict(x_test)


# ## Evaluation

# In[119]:


accuracy_catboost = accuracy_score(y_test, y_pred_catboost)
print(f"CatBoost Accuracy: {accuracy_catboost}")


# In[119]:




