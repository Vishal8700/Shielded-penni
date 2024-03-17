import pandas as pd
import numpy as np
import mysql.connector
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import traceback


# Connect to MySQL database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Vishal123",
    database="user90data"
)
cursor = conn.cursor()

# Function to fetch data from MySQL and create DataFrame
def fetch_data_to_dataframe(table_name):
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, conn)
    return df

# Define the function to drop columns
def drop_columns(data_frames, table_name, col_arr):
    data_frames[table_name] = data_frames[table_name].drop(columns=col_arr)

# Load the dataset
dataframes = {}
dataframes["SupplyDemand"] = fetch_data_to_dataframe("supplydemand")

# Close the database connection
conn.close()

# Drop unnecessary columns
drop_columns(dataframes, "SupplyDemand", ["Timestamp"])

# Preprocess numerical data
def numerical_preprocessing(data_frames, table_name, columns):
    if table_name in data_frames:
        scaler = StandardScaler()
        data_frames[table_name][columns] = scaler.fit_transform(data_frames[table_name][columns])
    else:
        print(f"Table '{table_name}' not found in the data_frames dictionary.")

numerical_preprocessing(dataframes, "SupplyDemand", ['UserCount', 'penniPrice'])

# Separate features and target variable
X = dataframes["SupplyDemand"][['UserCount']]  # Features
y = dataframes["SupplyDemand"]['penniPrice']   # Target variable

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

try:
    model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.1)
except Exception as e:
    print("An error occurred while training the model:", e)


# Prediction endpoint
app = FastAPI()

class Item(BaseModel):
    user_count: float

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

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
