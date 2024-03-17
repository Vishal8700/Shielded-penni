import database
def droper(data_frames, table_name, col_arr):
    data_frames[table_name] = data_frames[table_name].drop(columns=col_arr)

# Define the columns to drop
columns_to_drop = ['Timestamp']

# Specify the name of the DataFrame in the 'dataframes' dictionary
table_update = 'SupplyDemand'

# Call the function to drop columns
droper(dataframes, table_update, columns_to_drop)
dataframes["SupplyDemand"].sample(10)


from sklearn.preprocessing import StandardScaler

def numerical_preprocessing(data_frames, table_name, columns):
    if table_name in data_frames:
        scaler = StandardScaler()
        data_frames[table_name][columns] = scaler.fit_transform(data_frames[table_name][columns])
    else:
        print(f"Table '{table_name}' not found in the data_frames dictionary.")

# Assuming 'dataframes' is your dictionary of DataFrames
numerical_preprocessing(dataframes, "SupplyDemand", ['UserCount','penniPrice'])
dataframes['SupplyDemand']