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
    "SupplyDemand"
]

# Dictionary to store DataFrames
dataframes = {}

# Fetch data and create DataFrames for each table
for table_name in table_names:
    dataframes[table_name] = fetch_data_to_dataframe(table_name)

# Close the database connection
conn.close()
