#!/usr/bin/env python
# coding: utf-8

# # Bit Lords

# ### Importing packages and reading dataset

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
import plotly.graph_objs as go
import plotly.offline as py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFECV
from sklearn import tree
warnings.filterwarnings("ignore", 'This pattern has match groups')


# In[37]:


x=pd.read_csv("supply_chain.csv",encoding='latin1')
print(x.shape)


# In[38]:


x.head(10)


# In[39]:


x.describe()


# In[40]:


plt.figure(figsize=(22,6))
sns.heatmap(x.corr(),annot=True)


# In[ ]:





# ## Data Cleaning

# In[41]:


# So here we have to focus on columns: Shipment Mode, Dosage ,Line Item Insurance (USD) 
x.isnull().sum()


# In[42]:


# Checking th epercentage of missing values
print("Null values in Percentage")
(x.isnull().sum()/x.shape[0])*100


# In[94]:


#### # Dropping Unnecessary columns


# In[44]:


x=x.drop(columns=['PQ #','PO / SO #','ASN/DN #','Managed By','Fulfill Via','Vendor INCO Term','PQ First Sent to Client Date','Product Group','Vendor','ï»¿ID'])
x.head()


# ##### 1) Line Item Insaurance

# In[45]:


sns.boxplot(y=x['Line Item Insurance (USD)'])


# In[46]:


x['Line Item Insurance (USD)']=x['Line Item Insurance (USD)'].fillna(x['Line Item Insurance (USD)'].median())
x['Line Item Insurance (USD)'].values[x['Line Item Insurance (USD)'].values>1000]=x['Line Item Insurance (USD)'].median()
x['Line Item Insurance (USD)'].values[x['Line Item Insurance (USD)'].values>200]=x['Line Item Insurance (USD)'].mean()


# In[47]:


sns.boxplot(y=x['Line Item Insurance (USD)'])


# In[ ]:





# ##### Feight Cost (USD)

# In[48]:


x['Freight Cost (USD)'].values


# In[49]:


x=x.loc[~x['Freight Cost (USD)'].str.contains('Freight Included in Commodity Cost')]
x=x.loc[~x['Freight Cost (USD)'].str.contains('Invoiced Separately')]
x=x.loc[~x['Freight Cost (USD)'].str.contains('See')]


# In[50]:


x['Freight Cost (USD)'].values


# In[52]:


x.shape


# In[54]:


x.head(10)


# In[ ]:





# #### Shipment mode

# In[55]:


x['Shipment Mode'].values


# In[56]:


x['Shipment Mode']=x['Shipment Mode'].fillna(x['Shipment Mode'].mode()[0])


# In[57]:


x['Shipment Mode'].isnull().sum()


# In[58]:


x['Shipment Mode']


# In[ ]:





# #### Dosage 
# 

# In[59]:


print(x['Dosage'].values)


# In[60]:


print(x['Dosage'].mode())


# In[61]:


print(x['Dosage'].mode())
x['Dosage']=x['Dosage'].fillna(x['Dosage'].mode()[0])


# In[62]:


x['Dosage'].isnull().sum()


# In[ ]:





# #### Weight (Kilograms)

# In[63]:


x['Weight (Kilograms)'].unique()


# In[64]:


x=x.loc[~x['Weight (Kilograms)'].str.contains('Weight Captured Separately')]


# In[65]:


x=x.loc[~x['Weight (Kilograms)'].str.contains('See')]


# In[66]:


x['Weight (Kilograms)']


# In[70]:


sns.boxplot(x['Weight (Kilograms)'])


# In[ ]:





# #### PO Sent to Vendor Date

# In[82]:


x['PO Sent to Vendor Date'].values


# In[93]:


x['PO Sent to Vendor Date'].unique()


# In[87]:


x=x.loc[~x['PO Sent to Vendor Date'].str.contains('Date Not Captured')]
x=x.loc[~x['PO Sent to Vendor Date'].str.contains('N/A - From RDC')]


# In[90]:


print(x['PO Sent to Vendor Date'])
print(x['PO Sent to Vendor Date'].unique)


# In[91]:


x.shape


# In[92]:


x.head()


# In[ ]:





# In[ ]:





# ## Feature Engineering
# 

# In[95]:


for column in ['Scheduled Delivery Date', 'Delivered to Client Date', 'Delivery Recorded Date']:
        x[column] = pd.to_datetime(x[column])
        x[column + ' Year'] = x[column].apply(lambda x: x.year)
        x[column + ' Month'] = x[column].apply(lambda x: x.month)
        x[column + ' Day'] = x[column].apply(lambda x: x.day)
        x = x.drop(column, axis=1)


# In[96]:


x


# In[ ]:





# In[360]:


x['Delivered to Client Date Day'].values[x['Shipment Mode'].values==0]+=2


# In[362]:


x['Delivered to Client Date Day'].values[x['Shipment Mode'].values==1]+=4


# In[365]:


x['Delivered to Client Date Day'].values[x['Shipment Mode'].values==2]+=1


# In[366]:


x['Delivered to Client Date Day'].values[x['Shipment Mode'].values==3]


# In[367]:


x['Delivered to Client Date Day'].values[x['Shipment Mode'].values==3]+=7


# In[361]:


x.head()


# ## Data Visualization

# In[97]:


#Total Pack Price for Top 15 Countries with graph
TotalPrice = x.groupby(['Country'])['Pack Price'].sum().nlargest(15)
print("Total Pack Price for Top 15 Countries\n")
print(TotalPrice)
plt.figure(figsize=(22,6))
GraphData=x.groupby(['Country'])['Pack Price'].sum().nlargest(15)
GraphData.plot(kind='bar')
plt.ylabel('Total Pack Price')
plt.xlabel('Country Name')


# In[98]:


# Pie Chart of column Shipment Mode
ShippingMode = x["Shipment Mode"].value_counts()
labels = (np.array(ShippingMode.index))
sizes = (np.array((ShippingMode / ShippingMode.sum())*100))
trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(title="Shipment Mode")
dat = [trace]
fig = go.Figure(data=dat, layout=layout)
py.iplot(fig, filename="Shipment Mode")


# In[99]:


# Count of Manufacturing Site name
plt.figure(figsize=(22,6))
TopFiveManufacturingSite=x.groupby('Manufacturing Site').size().nlargest(10)
print(TopFiveManufacturingSite)
TopFiveManufacturingSite.plot(kind='bar')
plt.ylabel('Total Count')
plt.xlabel('Manufacturing Site')


# In[100]:


#Shipment mode vs PackPrice
plt.figure(figsize = (9,5))
plt.xticks(rotation = 90)
sns.barplot(x='Shipment Mode',y='Pack Price', data = x)
plt.show()


# In[101]:


# Country Wise Count
ItemCount = x["Country"].value_counts().nlargest(10)
sns.set_context("talk",font_scale=1)
plt.figure(figsize=(22,16))
sns.countplot(y=x['Country'],order = x['Country'].value_counts().nlargest(10).index)
plt.title('Top 10 Countries Wise Count \n')
plt.xlabel('Total Count')
plt.ylabel('Country')


# #### Insights
# * From the pie chart we can observe that most used shipment mode is air(89.9%) followed by ocean(9.52%),Followed by Air Charter(0.584%),followed by Truck(0.0334%).
# 
# * From the barplot of the 'Shipment Mode' vs 'Pack Price' ,we can observe that people send their goods in large proportion through trucks.
# 
# * from the count plot of 'total count vs country' we can observe that vietnam has exported large number of drugs.
# 
# * from the bar plot of 'Manufacturing Site' vs 'Total Count' we can observe that top two manufacturing sites are in India

# In[ ]:





# ## Data preprocessing

# In[109]:


le=LabelEncoder()
x['Shipment Mode']=le.fit_transform(x['Shipment Mode'])
# air=0,ocean=2,air charter=1,truck=3


# In[110]:


x['Sub Classification']=le.fit_transform(x['Sub Classification'])


# In[111]:


x['First Line Designation']=le.fit_transform(x['First Line Designation'])


# In[112]:


x['Weight (Kilograms)'] = x['Weight (Kilograms)'].astype(int)


# In[113]:


x.columns


# In[ ]:





# In[ ]:





# ## ML Models
# * Using Decision trees to provide the most optimal path with respect to features like Deliverytime,cost,Pack Price etc.

# In[906]:


xtrain=x[['Sub Classification','Unit of Measure (Per Pack)','Scheduled Delivery Date Year','Scheduled Delivery Date Month','Scheduled Delivery Date Day','Delivered to Client Date Year','Delivered to Client Date Month','Delivered to Client Date Day','Delivery Recorded Date Year','Delivery Recorded Date Month','Delivery Recorded Date Day','Line Item Quantity','Line Item Value','Pack Price','First Line Designation','Weight (Kilograms)','Freight Cost (USD)']]
ytrain=x['Shipment Mode']


# In[907]:


x_train=xtrain.iloc[:, ]
y_train=ytrain.iloc[:,]


# In[908]:


xtrain,xtest,ytrain,ytest=train_test_split(x_train,y_train,test_size=0.25,random_state=42)


# In[909]:


model=DecisionTreeClassifier(max_depth=8,criterion='gini',splitter='best')
model.fit(xtrain,ytrain)


# In[910]:


print('Testing Accuracy:',model.score(xtest,ytest))


# In[911]:


predictions=model.predict(xtest)


# In[912]:


matrix=confusion_matrix(ytest,predictions)
matrix


# In[913]:


value_counts=dict(x['Shipment Mode'].value_counts())
print(value_counts)


# In[ ]:





# In[ ]:





# In[ ]:





# In[883]:





# In[914]:


y_pred=model.predict(xtest)


# In[915]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(ytest, y_pred))


# In[865]:


predictions=model.predict(xtest)


# In[866]:


matrix=confusion_matrix(ytest,predictions)
matrix


# In[867]:


# Visualizing tree
fig=plt.figure(figsize=(85,80))
_=tree.plot_tree(model,max_depth=5,feature_names=x_train.columns,class_names=['0','1','2','3'],filled=True)


# In[916]:


pickle.dump(model,open('model_mode.pkl','wb'))


# In[ ]:





# #### Random Forest
# 

# In[150]:


xtrain=x[['Sub Classification','Unit of Measure (Per Pack)','Scheduled Delivery Date Year','Scheduled Delivery Date Month','Scheduled Delivery Date Day','Delivered to Client Date Year','Delivered to Client Date Month','Delivered to Client Date Day','Delivery Recorded Date Year','Delivery Recorded Date Month','Delivery Recorded Date Day','Line Item Quantity','Line Item Value','Pack Price','First Line Designation','Weight (Kilograms)','Freight Cost (USD)']]
ytrain=x['Shipment Mode']


# In[151]:


x_train=xtrain.iloc[:, ]
y_train=ytrain.iloc[:,]


# In[226]:


xtrain,xtest,ytrain,ytest=train_test_split(x_train,y_train,test_size=0.20,random_state=42)


# In[229]:


from sklearn.ensemble import RandomForestRegressor
  
 # create regressor object
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 42)
  
# fit the regressor with x and y data
regressor.fit(xtrain,ytrain)  


# In[231]:


print("Testing Accuracy:",regressor.score(xtest,ytest))


# In[ ]:





# XG Boost 

# In[265]:


xtrain=x[['Sub Classification','Unit of Measure (Per Pack)','Scheduled Delivery Date Year','Scheduled Delivery Date Month','Scheduled Delivery Date Day','Delivered to Client Date Year','Delivered to Client Date Month','Delivered to Client Date Day','Delivery Recorded Date Year','Delivery Recorded Date Month','Delivery Recorded Date Day','Line Item Quantity','Line Item Value','Pack Price','First Line Designation','Weight (Kilograms)']]
ytrain=x['Shipment Mode']


# In[266]:


x_train=xtrain.iloc[:, ]
y_train=ytrain.iloc[:,]


# In[ ]:





# In[267]:


xtrain,xtest,ytrain,ytest=train_test_split(x_train,y_train,test_size=0.20,random_state=42)


# In[268]:


import xgboost as xgb
from sklearn.metrics import mean_squared_error


# In[300]:


xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 1.0, learning_rate = 0.1,
                max_depth = 8, alpha =10, n_estimators = 200)


# In[301]:


xg_reg.fit(xtrain,ytrain)
preds = xg_reg.predict(xtest)


# In[302]:


print("Testing Accuracy:",xg_reg.score(xtest,ytest))


# In[ ]:





# In[ ]:





# In[ ]:





# ## Regression Model
# #### Estimation of the Date on which the goods will be delievered

# In[586]:


trainx=x[['Shipment Mode','Scheduled Delivery Date Year','Scheduled Delivery Date Month','Scheduled Delivery Date Day']]
trainy=x['Delivered to Client Date Day']


# In[587]:


X=trainx.iloc[:, ].values
Y=trainy.iloc[:,].values


# In[588]:


print(X)
print(Y)


# In[589]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=25)


# In[590]:


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[591]:


model_date=LinearRegression(fit_intercept=True,positive=True)
model_date.fit(X_train,Y_train)


# In[592]:


p=model_date.predict(X_test).astype(int)
p


# In[593]:


model_date.score(X_test,Y_test)


# In[594]:


model_date.predict([[0,2022,1,7]]).reshape(-1,1).astype(int)


# In[917]:


pickle.dump(model_date,open('model_date.pkl','wb'))


# ## Regression
# #### Prediction of Weight (Kilograms) and through it we can take decisions to how to reduce cost and minimise time, also we will abe to predict the total path cost

# In[ ]:





# In[453]:


x.head(5)


# In[577]:


x.columns


# In[733]:


print(x['Unit Price'])


# In[744]:


trainx=x[['Pack Price','Line Item Insurance (USD)','Unit Price','Line Item Quantity','Line Item Value']]
trainy=x['Weight (Kilograms)']


# In[745]:


X=trainx.iloc[:, ].values.astype(int)
Y=trainy.iloc[:,].values.astype(int)


# In[746]:


print(X)
print(Y)


# In[747]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=42)


# In[748]:


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[ ]:





# In[749]:


# Linear Regression
model_weight=LinearRegression(fit_intercept=True,positive=True)
model_weight.fit(X_train,Y_train)
print(model_weight.coef_.astype(int))
model_weight.intercept_.astype(int)


# In[750]:


model_weight.score(X_test,Y_test)


# In[751]:


model_weight.predict([[9.98,47.04,0.6,712,9980.00]]).astype(int)


# In[918]:


pickle.dump(model_weight,open('model_weight.pkl','wb'))


# In[ ]:




