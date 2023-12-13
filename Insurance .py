#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[2]:


a = r"C:\Users\acer\Desktop\insurance_data.csv"
df= pd.read_csv(a)
df


# In[3]:


#Load data visualization packages
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[4]:


df.columns


# In[6]:


df.head()


# In[7]:


df.shape


# In[9]:


df.isnull().sum()


# In[10]:


df = df.dropna()


# In[11]:


df.shape


# In[12]:


df.isnull().sum()


# In[13]:


df.duplicated().sum()


# In[14]:


df = df.drop(['index','PatientID'], axis=1)
df.shape


# In[15]:


df['diabetic'] = df['diabetic'].replace({'Yes': 'diabetic', 'No': 'non-diabetic'})
df['children'] = df['children'].replace({0 : 'none', 1 : 'one', 2:'two', 3:'three', 4:'four', 5:'five', 6:'six'})
df['smoker'] = df['smoker'].replace({'Yes': 'smoker', 'No': 'non-smoker'})


# In[17]:


df.head()


# In[18]:


age_range = [13,28,46,58,61]
labels = ['Gen Z','Millennials', 'Gen X', 'Baby Boomers']
df['age_group'] = pd.cut(df['age'], age_range,labels=labels)
ageGroup = df[['age_group', 'claim']].groupby('age_group').mean().sort_values(by="claim", ascending=True)


# In[19]:


df.head()


# In[20]:


# Create column with bmi ranges corresponding to Healthy Weight, Overweight and Obese. 

#bmi_range = [15.5,24.9,29.9,60]
bmi_range = [15.5,18.5,24.9,29.9,60]

labels = ['Underweight','Healthy Weight', 'Overweight', 'Obese']
#df['bmi_group'] = pd.cut(df['bmi'], bins=bmi_range)
df['bmi_group'] = pd.cut(df['bmi'], bmi_range, labels=labels)
bmiGroup = df[['bmi_group', 'claim']].groupby('bmi_group').mean().sort_values(by="claim", ascending=True)
#labels = ['Healthy Weight', 'Overweight', 'Obese']


# In[21]:


#Create column with blood pressure ranges corresponding to Normal Blood Pressure, Elevated Blood Presure, Hypertension Stage 1 
# and Hypertension Stage 2

bloodpressure_range = [71,120,129,139,148]
labels = ['Normal Blood Pressure', 'Elevated Blood Pressure', 'Hypertension Stage 1','Hypertension Stage 2']
df['bloodpressure_group'] = pd.cut(df['bloodpressure'], bloodpressure_range, labels = labels)
bloodpressureGroup = df[['bloodpressure_group', 'claim']].groupby('bloodpressure_group').mean().sort_values(by="claim", ascending=True)


# In[39]:


bloodpressureGroup


# In[22]:


df


# In[23]:


plt.figure(figsize = (10,5))
plt.title("Claims by Region")
df['region'].value_counts().plot(kind='pie', title = "Claims per Region")


# In[25]:


plt.figure(figsize = (10,5))
#plt.subplot(1,2,1)

plt.title("Gender by Region")
sns.countplot(x = 'gender', hue = 'region', data = df)


# In[26]:


plt.figure(figsize = (10,5))

df['age_group'].value_counts().plot(kind='pie', title = "Claims per Age Group")


# In[27]:


plt.figure(figsize = (10,5))
df['bmi_group'].value_counts().plot(kind='pie', title = "BMI Ranges")


# In[28]:


# Set the width and height of the figure
plt.figure(figsize=(10,6))


# Add title
plt.title("BMI and Claim Amount")


sns.scatterplot(x=df['bmi'], y=df['claim'])


# In[29]:


# Set the width and height of the figure
plt.figure(figsize=(10,6))

# Add title
plt.title("BMI Claims")

# Bar chart showing average arrival delay for Spirit Airlines flights by month
sns.barplot(x=df.bmi_group, y=df['claim'])

# Add label for vertical axis
plt.ylabel("Claim Amounts")


# In[30]:


# Sub plot

# plt.subplot(#Total number of rows, total number of columns, plot number)
plt.figure(figsize = (10,5))
plt.subplot(1,2,1)

plt.title("Correlation between Generation and BMI")
sns.countplot(x ='age_group', hue = 'bmi_group', data = df)


plt.subplot(1,2,2)

plt.title("Correlation between Blood Pressure and Generation")
sns.countplot(x = 'age_group', hue = 'bloodpressure_group', data = df)


# In[31]:


# Sub plot

# plt.subplot(#Total number of rows, total number of columns, plot number)
plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
sns.countplot(x = 'age_group', hue = 'gender', data = df)

plt.subplot(1,2,2)
sns.countplot(x ='age_group', hue = 'diabetic', data = df)


# In[32]:


# Sub plot

# plt.subplot(#Total number of rows, total number of columns, plot number)
plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
sns.countplot(x = 'age_group', hue = 'region', data = df)

plt.subplot(1,2,2)
sns.countplot( x = 'age_group', hue = 'smoker', data = df)


# In[33]:


# plt.subplot(#Total number of rows, total number of columns, plot number)
plt.figure(figsize = (10,5))

# Add title 
plt.title("Correlation of Generations and Children")

# Bar chart showing correlation between age groups and generations
sns.countplot( x = 'age_group', hue = 'children', data = df)


# In[34]:


# Set the width and height of the figure
plt.figure(figsize=(10,6))

# Add title
plt.title("Average Claim Amount per Age Group")

# Bar chart showing average arrival delay for Spirit Airlines flights by month
sns.barplot(x=df.region, y=df['claim'])

# Add label for vertical axis
plt.ylabel("Claim Amount")


# In[38]:


# Set the width and height of the figure
plt.figure(figsize=(10,6))

# Add title
plt.title("Blood Pressure Claims per Region")

# Bar chart showing blood pressure claims by region
sns.barplot(x=df.bloodpressure, y=df['region'])

# Add label for vertical axis
plt.ylabel("Region")


# In[40]:


df


# In[41]:


df.tail(25)


# In[ ]:




