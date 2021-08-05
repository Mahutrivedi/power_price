#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[242]:


path = r'G:\kaggle\power_price\Electrical.csv'

df = pd.read_csv(path,index_col=0)
df.head()


# In[3]:




df.rename(columns={"SMPEA":"forecasted_price","SMPEP2":"actual_price","SystemLoadEA" : "forcasted_load","SystemLoadEP2":"actual_load"},inplace=True)
df.head()


# In[4]:


print(df.columns)


# In[5]:


print(df.info())


# In[6]:


from sklearn.preprocessing import LabelEncoder


# In[7]:


encoder = LabelEncoder()

df["Holiday"]= encoder.fit_transform(df["Holiday"])
df.head()


# In[8]:


df["Holiday"].unique()


# In[9]:


print(df.info())


# In[10]:


df.replace('?',method="ffill",inplace=True)
df.drop(["DateTime"],axis=1,inplace=True)
df= df.astype(np.float64)
df.dropna(inplace=True)


# In[11]:


df.info()


# In[12]:


df.columns


# In[13]:


df["Holiday"] =df["Holiday"].astype(np.int64)
df["HolidayFlag"] =df["HolidayFlag"].astype(np.int64)
df["DayOfWeek"] =df["DayOfWeek"].astype(np.int64)
df["WeekOfYear"] =df["WeekOfYear"].astype(np.int64)
df["Day"] =df["Day"].astype(np.int64)
df["Month"] =df["Month"].astype(np.int64)
df["Year"] =df["Year"].astype(np.int64)
df["PeriodOfDay"] =df["PeriodOfDay"].astype(np.int64)


# In[14]:


df.info()


# In[15]:


df.isnull().sum()


# In[16]:


corr= df.corr()
corr


# In[17]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(12,10))
sns.heatmap(corr,cmap="YlGnBu")


# In[18]:


print(df.shape)


# In[19]:


for column in df.columns:
    plt.figure(figsize=(10,12))
    sns.displot(df[column],bins=50)


# In[20]:


for column in df.columns:
    plt.figure(figsize=(10,12))
    sns.boxplot(df[column])


# In[ ]:





# In[21]:


for column in df.columns:
    plt.figure(figsize=(10,12))
    sns.displot(df[column],bins=50)


# In[191]:


df.head()


# # First without removing outliers we will check the results

# In[192]:


from sklearn.model_selection import train_test_split
x=df.drop(["actual_price"],axis=1)
y=df["actual_price"]
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=2021)


# In[193]:


from sklearn.ensemble import RandomForestRegressor


# In[194]:


forest= RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=50, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=True, n_jobs=-1, random_state=2021, verbose=2, warm_start=False, ccp_alpha=0.0, max_samples=None)


# In[195]:


forest.fit(xtrain,ytrain)


# In[196]:


forest.oob_score_


# In[197]:


feature_importance = pd.DataFrame()
feature_importance["importance"] =  forest.feature_importances_
feature_importance["Feature_name"] = df.columns.drop(["actual_price"])


# In[198]:


feature_importance=feature_importance[["Feature_name","importance"]]


# In[199]:


ans = forest.predict(xtrain)


# In[200]:


feature_importance.sort_values(by=["importance"],ascending=False)


# In[201]:


from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error


print("Training performance")

print(r2_score(ans,ytrain))
print(mean_squared_error(ans,ytrain))
print(mean_absolute_error(ans,ytrain))


# In[202]:


ans = forest.predict(xtest)


# In[203]:


print("Testing performance")

print(r2_score(ans,ytest))
print(mean_squared_error(ans,ytest))
print(mean_absolute_error(ans,ytest))


# # As the out of bag evaluation already predicted,  the test results are horrible compared to train results

# In[204]:


df.head()


# In[205]:


#Holiday and DayofWeek looks like the features which require one hot encoding
print(df["Holiday"].unique())
print(df["DayOfWeek"].unique())


# In[206]:


#Holiday has 13 different types and holidays doesn't have relation among them 
#Day of week is also similar case


# In[207]:


updated_df=pd.get_dummies(df,columns=["Holiday","DayOfWeek"],drop_first=True)


# In[208]:


updated_df.head()


# In[209]:


x=updated_df.drop(["actual_price"],axis=1)
y=updated_df["actual_price"]
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state=2021)


# In[210]:


forest.fit(xtrain,ytrain)


# In[211]:


forest.oob_score_


# In[212]:


feature_importance = pd.DataFrame()
feature_importance["importance"] =  forest.feature_importances_
feature_importance["Feature_name"] = updated_df.columns.drop(["actual_price"])

feature_importance=feature_importance[["Feature_name","importance"]]

feature_importance.sort_values(by=["importance"],ascending=0)


# In[213]:



ans = forest.predict(xtrain)


# In[214]:


print("Training performance")

print(r2_score(ans,ytrain))
print(mean_squared_error(ans,ytrain))
print(mean_absolute_error(ans,ytrain))


# In[215]:


ans = forest.predict(xtest)


# In[216]:


print("Testing performance")

print(r2_score(ans,ytest))
print(mean_squared_error(ans,ytest))
print(mean_absolute_error(ans,ytest))


# In[217]:


#There is no improvement in performance after one hot encoding


# # This is the power of outliers ( In a bad way),  they are in small size but makes the machine learning model worst

# In[218]:


for column in df.columns:
    plt.figure(figsize=(10,10))
    sns.boxplot(df[column])


# In[219]:


tf_per= np.percentile(df["forecasted_price"],25)
sf_per= np.percentile(df["forecasted_price"],75)
iqr = sf_per-tf_per
ll=tf_per-1.5*iqr
ul=sf_per+1.5*iqr



# In[220]:


df_no_outlier = pd.DataFrame(df)


# In[221]:


df_no_outlier.head()


# In[222]:


df_no_outlier=df_no_outlier[df_no_outlier["forecasted_price"]>ll]
df_no_outlier=df_no_outlier[df_no_outlier["forecasted_price"]<ul]


# In[223]:


tf_per= np.percentile(df["CO2Intensity"],25)
sf_per= np.percentile(df["CO2Intensity"],75)
iqr = sf_per-tf_per
ll=tf_per-1.5*iqr
ul=sf_per+1.5*iqr

df_no_outlier=df_no_outlier[df_no_outlier["CO2Intensity"]>ll]
df_no_outlier=df_no_outlier[df_no_outlier["CO2Intensity"]<ul]


# In[224]:


tf_per= np.percentile(df["actual_price"],25)
sf_per= np.percentile(df["actual_price"],75)
iqr = sf_per-tf_per
ll=tf_per-1.5*iqr
ul=sf_per+1.5*iqr

df_no_outlier=df_no_outlier[df_no_outlier["actual_price"]>ll]
df_no_outlier=df_no_outlier[df_no_outlier["actual_price"]<ul]


# In[225]:


tf_per= np.percentile(df["ORKWindspeed"],25)
sf_per= np.percentile(df["ORKWindspeed"],75)
iqr = sf_per-tf_per
ll=tf_per-1.5*iqr
ul=sf_per+1.5*iqr

df_no_outlier=df_no_outlier[df_no_outlier["ORKWindspeed"]>ll]
df_no_outlier=df_no_outlier[df_no_outlier["ORKWindspeed"]<ul]


# In[226]:


for column in df_no_outlier.columns:
    plt.figure(figsize=(10,10))
    sns.boxplot(df_no_outlier[column])


# In[227]:


df_no_outlier.describe()


# In[228]:


x=df_no_outlier.drop(["actual_price"],axis=1)
y=df_no_outlier["actual_price"]
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state=2021)


# In[229]:


forest.fit(xtrain,ytrain)


# In[230]:


forest.oob_score_


# In[231]:


#Drastic improvement in oob score after outlier removation with just 100 trees


# In[232]:



ans = forest.predict(xtrain)
print("Training performance")

print(r2_score(ans,ytrain))
print(mean_squared_error(ans,ytrain))
print(mean_absolute_error(ans,ytrain))


# In[233]:


ans = forest.predict(xtest)
print("Testing performance")

print(r2_score(ans,ytest))
print(mean_squared_error(ans,ytest))
print(mean_absolute_error(ans,ytest))


# In[238]:


# Now we will try to fine tune the forest model


# In[239]:


forest = RandomForestRegressor(n_estimators=1000, criterion='mse', max_depth=28, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=8, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=True, n_jobs=-1, random_state=2021, verbose=2, warm_start=False, ccp_alpha=0.0, max_samples=None)
forest.fit(xtrain,ytrain)
print(forest.oob_score_)


# In[240]:


ans = forest.predict(xtrain)
print("Training performance")

print(r2_score(ans,ytrain))
print(mean_squared_error(ans,ytrain))
print(mean_absolute_error(ans,ytrain))


# In[241]:


ans = forest.predict(xtest)
print("Testing performance")

print(r2_score(ans,ytest))
print(mean_squared_error(ans,ytest))
print(mean_absolute_error(ans,ytest))


# In[ ]:




