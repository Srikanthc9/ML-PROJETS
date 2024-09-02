#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_excel(r'D:\job\DATA SCIENTIST\Job\ML PROJECTS\Linear regression.xlsx')


# In[2]:


df.head()


# In[3]:


#Data cleaning checking for missing vals.
df.isnull().sum()


# In[4]:


df.describe()


# In[5]:


import matplotlib.pyplot as plt


# In[6]:


plt.figure(figsize=(4,4))
plt.scatter(df['Square Footage'],df['House Price ($)'],color = 'blue',marker = 'o')
plt.title("Figuring out Square Footage vs. House Price using Scatter Plot of ")
plt.xlabel("Square Footage")
plt.ylabel("House prices($)")


# In[7]:


df.dtypes


# In[8]:


correlation = df['Square Footage'].corr(df['House Price ($)'])


# In[9]:


print("Correlation between Square Footage and House Price: ", correlation)

FOUND A STRONG CORREALTION BETWEEN INPUT FEATURES AND TARGET VALUE.
# In[10]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[11]:


df.head()


# In[12]:


#Dependent and Independent Variable.
X = df[['Square Footage']]
y = df['House Price ($)']


# In[13]:


print('Independent feature:')
print(X)


# In[14]:


print("Dependent variable:")
print(y)


# In[15]:


#Splitting the data:
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=42)

Building the Linear Regression Model
# In[16]:


from sklearn.linear_model import LinearRegression


# In[17]:


model = LinearRegression()


# In[18]:


model.fit(X_train,y_train)


# 
# #Making predictions

# In[19]:


y_pred = model.predict(X_test)


# Evaluating the Model

# In[20]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score 


# In[21]:


# Calculate the evaluation metrics
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)


# In[22]:


print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared:", r2)


# In[23]:


# Plot the test data
plt.scatter(X_test,y_test,color = 'blue',label = 'Actual Prices($)')
plt.plot(X_test,y_pred,color = 'red',linewidth=2,label = 'Predicted Prices($)')
plt.title('Square Footage vs. House Price')
plt.xlabel('Square Footage')
plt.ylabel('House Price ($)')
plt.legend()
plt.show()


# Interpreting the Model

# In[24]:


print('Coeffient (Slope):',model.coef_[0])


# In[25]:


print("Intercept:", model.intercept_)


# print("Coefficient (Slope):", model.coef_[0])
# print("Intercept:", model.intercept_)
# ```
# 
# ### Summary of the Model Building Steps
# 1. **Split the data** into training and testing sets.
# 2. **Build and train the model** using the training data.
# 3. **Make predictions** on the test data.
# 4. **Evaluate the model** using metrics like MAE, MSE, and R-squared.
# 5. **Visualize the results** by plotting the regression line.
# 6. **Interpret the coefficients** to understand the relationship between square footage and house price.
# 
# This workflow will help you create a simple linear regression model that can predict house prices based on square footage.

# In[ ]:




