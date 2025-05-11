#!/usr/bin/env python
# coding: utf-8

# # House Prices in India 

# ## Problem Statement

# #### To develop a machine learning regression model that accurately predicts house prices in India based on key factors such as location, size and  number of bedrooms with the aim of assisting real estate agents, buyers, and investors in making informed decisions.

# ## Hypothesis

# #### The hypothesis is that we believe that there is a correlation between  the features of the houses like size,location,number of bedrooms.By analyzing these features  we can build a linear  regression model that accurately predicts house prices within a certain margin of error

# ## Data Collection

# ### An Already Existing Dataset was used in this model

# ## Data Preprocessing 

# ## Importing the  necessary libraries

# In[36]:


#!pip install mlxtend -- To install the mlxtend algorithm in my machine
import pandas as pd # for data preprocessing
import numpy as np #for dealing with data in a vectorized format
import seaborn as sns #for visualizations
import matplotlib.pyplot as plt # for visualiazations
from sklearn.linear_model import LinearRegression  # for the Linear Model
from sklearn.metrics import mean_absolute_error    #metric for Linear Regression
from sklearn.model_selection import train_test_split  #for Feature Selection
import plotly.express as px # For Visualizations
from ipywidgets import Dropdown, FloatSlider, IntSlider, interact # For Linear Model Deployment

 


# ## Exploratory Data Analysis:

# ### Reading in the  CSV file
# 

# In[37]:


df=pd.read_csv("houses.csv")


# In[38]:


df.head()


# In[39]:


df.info()


# In[40]:


df.tail()


# In[41]:


df.shape


# In[42]:


df.info()


# In[43]:


df.dtypes


# In[44]:


df.describe()


# In[45]:


df.corr()


# ## Cleaning the dataset

# In[60]:


# Droping unwanted columns
df.drop(columns=["id","Date","lot area","Built Year","Renovation Year",
                 "living_area_renov","lot_area_renov","Distance from the airport",
                 "No of schools nearby","house grade","Area of the basement",
                 "house area(excluding basement)"],inplace=True)


# In[12]:


# Renaming some columns
df.rename(columns={"Lattitude":"lat","Longitude":"lon",
                   "living area":"area","Price":"price",
                   "No of bathrooms":"bathrooms",
                   "No of bedrooms":"bedrooms",},inplace=True) 


# In[13]:


df.info()


# ## Explolatory Data Analysis

# In[14]:


# Scatter plot of Price Vs Area
plt.scatter(df["price"],df["area"])
plt.title("Price Vs Area"),
plt.xlabel("Price(USD)"),
plt.ylabel("Area(m2)");


# In[15]:


# Distribution Of house areas using a histogram
plt.hist(df["area"])
plt.title("Distribution of house areas")
plt.xlabel("Area(m2)");


# In[16]:


df["price"].describe()


# In[17]:


# Checking the distribution of house prices using a box plot
plt.boxplot(df["price"],vert=False)
plt.title("Distribution of house Prices")
plt.xlabel("Price");


# ### As indicated by the scatter plot, the area histogram and the prices box plot,there are outliers in our data set

# ## Handling outliers in prices and area

# In[18]:


# Handling outliers ie deleting houses in the upper and lower 20th percentile of the dataset
low,high=df["price"].quantile([0.2,0.8])


mask_price=df["price"].between(low,high)


df=df[mask_price]#mask_area]


# ## Displaying the distribution of prices after outlier cleaning

# In[19]:


plt.boxplot(df["price"],vert=False)
plt.title("Distribution of house Prices")
plt.xlabel("Price");


# ## Displaying the distribution of area after outlier cleaning

# In[20]:


# Distribution Of house areas using a histogram
plt.hist(df["area"])
plt.title("Distribution of house areas")
plt.xlabel("Area(m2)");


# ## As shown by the histogram and the boxplot  the data now follows some sort of "Normality"

# ## Displaying Correlations within the dataset features

# In[21]:


corr_matrix=df.corr()
plt.figure(figsize=(9,5))

# Create the heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

# Add a title
plt.title('Correlation Heatmap')

# Show the plot
plt.show()


# ###  As indicated in the correlation Heatmap Price and Area are positively Correlated

# In[22]:


df.describe()


# In[23]:


df.shape


# ## Mapbox showing the relationship between loaction and Price

# In[24]:


fig = px.scatter_mapbox(
    df,  # Our DataFrame
    lat="lat",
    lon="lon",
    width=600,  # Width of map
    height=600,  # Height of map
    color="price",
    hover_data=["price"],  # Display price when hovering mouse over house
)

fig.update_layout(mapbox_style="open-street-map")

fig.show()


# ## The Mapbox shows that as we move to the northern part of India ,House prices increase

# ## Subsetting Data to be used for Linear Regression

# In[25]:


df1=df[["area","lat","lon","bedrooms","bathrooms","price"]]
df1.info()


# ## Splitting the Data

# In[26]:


features=["area","lat","lon","bedrooms","bathrooms"]
target="price"
X=df1[features]
y=df1[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## Instantiate Model

# In[27]:


model=LinearRegression()


# ## Baseline Model

# In[28]:


y_mean=y_train.mean()
y_pred_baseline=[y_mean]*len(y_train)
y_baseline_mae=mean_absolute_error(y_train,y_pred_baseline)
print("The baseline Mean Absolute Error is:$",round(y_baseline_mae,2))


# In[29]:


model.fit(X_train,y_train)


# In[30]:


y_pred_test=model.predict(X_test)
y_pred_test


# In[31]:


y_pred_test_mae=mean_absolute_error(y_test,y_pred_test)
print("The mean absolute error for the model is:$",round(y_pred_test_mae,2))


# ### The MeanAbsoluteEror of the model is 75898.15 which basically means that our price predictions using the model will be off by  around 7600 which is an improvement from the baseline model with the MeanAbsoluteError at around  $9500.This means that our Linear model is perfoming very well

# ### Creating A Python Widget for Project Deployment

# In[32]:


def make_prediction(area, lat, lon, bedrooms,bathrooms):
    data={
    "area":area,
    "lat":lat,
    "lon":lon,
    "bedrooms":bedrooms,
    "bathrooms":bathrooms
        
    }
    df1=pd.DataFrame(data,index=[0])
    
    prediction = model.predict(df1).round(2)[0]
    return f"Predicted apartment price: ${prediction}"


# # Implementing a simple model Deployment:Python Widgets

# In[62]:


interact(
    make_prediction,
    area=IntSlider(
        min=X_train["area"].min(),
        max=X_train["area"].max(),
        value=X_train["area"].mean(),
    ),
    lat=FloatSlider(
        min=X_train["lat"].min(),
        max=X_train["lat"].max(),
        step=0.01,
        value=X_train["lat"].mean(),
    ),
    lon=FloatSlider(
        min=X_train["lon"].min(),
        max=X_train["lon"].max(),
        step=0.01,
        value=X_train["lon"].mean(),
    ),
    bedrooms=IntSlider(
        min=X_train["bedrooms"].min(),
        max=X_train["bedrooms"].max(),
        value=X_train["bedrooms"].mean(),
    ),
    bathrooms=IntSlider(
        min=X_train["bathrooms"].min(),
        max=X_train["bathrooms"].max(),
        value=X_train["bathrooms"].mean()
    )
);


# # Remarks
# ### The model effectively predicts Price, with a mean absolute error of $ 75898.15 
# ### Features like Area showed strong correlation with Price.
# ### Some features (like number of bedrooms and bathrooms) had minimal influence and could be reconsidered.
# ### Potential improvements include adding more data, tuning or testing different models.
# 

# In[ ]:




