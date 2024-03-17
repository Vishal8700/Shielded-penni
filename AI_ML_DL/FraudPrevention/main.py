#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import mysql.connector

# Connect to MySQL database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="sunDay01@new",
    database=" user90data"
)
cursor = conn.cursor()

# Function to fetch data from MySQL and create DataFrame
def fetch_data_to_dataframe(table_name):
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, conn)
    return df

# List of table names
table_names = [
    "supplydemand"
]

# Dictionary to store DataFrames
dataframes = {}

# Fetch data and create DataFrames for each table
for table_name in table_names:
    dataframes[table_name] = fetch_data_to_dataframe(table_name)

# Close the database connection
conn.close()


# In[ ]:


dataframes['SupplyDemand']


# In[19]:


def droper(data_frames, table_name, col_arr):
    data_frames[table_name] = data_frames[table_name].drop(columns=col_arr)


# In[20]:


# Define the columns to drop
columns_to_drop = ['Timestamp']

# Specify the name of the DataFrame in the 'dataframes' dictionary
table_update = 'SupplyDemand'

# Call the function to drop columns
droper(dataframes, table_update, columns_to_drop)
dataframes["SupplyDemand"].sample(10)


# In[6]:


data=dataframes


# In[5]:


from sklearn.preprocessing import StandardScaler

def numerical_preprocessing(data_frames, table_name, columns):
    if table_name in data_frames:
        scaler = StandardScaler()
        data_frames[table_name][columns] = scaler.fit_transform(data_frames[table_name][columns])
    else:
        print(f"Table '{table_name}' not found in the data_frames dictionary.")

# Assuming 'dataframes' is your dictionary of DataFrames
numerical_preprocessing(dataframes, "SupplyDemand", ['UserCount','penniPrice'])
df =dataframes['SupplyDemand']


# In[12]:



# In[14]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# Load the dataset
data = df

# Separate features and target variable
X = data[['UserCount']]  # Features
y = data['penniPrice']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the model
model = Sequential()
model.add(Dense(64, input_dim=1, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.1)

# Evaluate the model
train_loss = model.evaluate(X_train_scaled, y_train, verbose=0)
test_loss = model.evaluate(X_test_scaled, y_test, verbose=0)
print('Train Loss:', train_loss)
print('Test Loss:', test_loss)



# In[16]:


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
app = FastAPI()
# Define request body schema
class Item(BaseModel):
    user_count: float

# Prediction endpoint
@app.post("/predict")
async def predict_price(item: Item):
    user_count = item.user_count
    scaled_user_count = scaler.transform([[user_count]])
    prediction = model.predict(scaled_user_count)
    return {"predicted_price": prediction[0][0]}

# Evaluation endpoint
@app.get("/evaluate")
async def evaluate_model():
    train_loss = model.evaluate(X_train_scaled, y_train, verbose=0)
    test_loss = model.evaluate(X_test_scaled, y_test, verbose=0)
    return {"train_loss": train_loss, "test_loss": test_loss}

@app.get("/")
def read_root():
    return {"Hello": "World"}

import uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)



