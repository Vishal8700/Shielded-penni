import pandas as pd
import numpy as np
import mysql.connector

# Connect to MySQL database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Vishal123",
    database=" users_data_new"
)
cursor = conn.cursor()

# Function to fetch data from MySQL and create DataFrame
def fetch_data_to_dataframe(table_name):
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, conn)
    return df

# List of table names
table_names = [
    "BlockchainTransactions",
    "UserProfile",
    "BehavioralPatterns",
    "CreditAndFinancialHistory",
    "NetworkAnalysis",
    "SentimentAnalysis",
    "CommunityBehavior",
    "SystemAndPlatformScores",
    "DeviceAndIPInformation",
    "MachineLearningFeatures",
    "HistoricalFraudData",
    "ExternalDataSources"
]

# Dictionary to store DataFrames
dataframes = {}

# Fetch data and create DataFrames for each table
for table_name in table_names:
    dataframes[table_name] = fetch_data_to_dataframe(table_name)

# Close the database connection
conn.close()


dataframes


def droper(data_frames, table_name, col_arr):
    data_frames[table_name] = data_frames[table_name].drop(columns=col_arr)





import pandas as pd


def handle_dates(data, column_names):
    df = data.copy()
    for column_name in column_names:
        df[column_name] = pd.to_datetime(df[column_name])
    return df






# Define the columns to drop
columns_to_drop = ['Name', 'Address','Email','UserID','PhoneNumber']

# Specify the name of the DataFrame in the 'dataframes' dictionary
table_update = 'UserProfile'



# In[8]:


dataframes["UserProfile"] = handle_dates(dataframes["UserProfile"], ['AccountCreationDate'])
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk


def apply_nlp_label_encoding(df, column_names):
    # Tokenization and Stemming
    stemmer = PorterStemmer()
    for column in column_names:
        if column in df.columns:
            if df[column].dtype == 'object':  # Check if the column contains text data
                df[column] = df[column].apply(lambda x: ' '.join([stemmer.stem(word) for word in word_tokenize(x.lower())]))
    
    # Label Encoding
    label_encoder = LabelEncoder()
    for column in column_names:
        if column in df.columns:
            if df[column].dtype == 'object':  # Check if the column contains categorical data
                df[column] = label_encoder.fit_transform(df[column])
    return df

# Assuming 'dataframes' is your dictionary of DataFrames
# Assuming 'BehavioralPatterns' is the key for the DataFrame of interest
dataframes["UserProfile"] = apply_nlp_label_encoding(dataframes["UserProfile"], ['KYCStatus'])



dataframes['BlockchainTransactions'] = handle_dates(dataframes['BlockchainTransactions'], ['Timestamp'])

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def apply_feature_scaling(dataframe, column_names):


    # Extract the specified columns from the DataFrame
    X = dataframe[column_names].values

    # Split the data into training and testing sets
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    # Generate placeholder y_train and y_test for demonstration
    y_train = np.random.rand(X_train.shape[0])
    y_test = np.random.rand(X_test.shape[0])

    # Define a dictionary to store the results of different scaling methods
    scaling_results = {}

    # Apply different scaling techniques
    for scaler in [StandardScaler(), MinMaxScaler(), RobustScaler()]:
        scaled_X_train = scaler.fit_transform(X_train)
        scaled_X_test = scaler.transform(X_test)

        # Train a linear regression model
        model = LinearRegression()
        model.fit(scaled_X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(scaled_X_test)
        mse = mean_squared_error(y_test, y_pred)

        # Store the results
        scaling_results[str(scaler)] = mse

    # Find the best scaling method with the lowest MSE
    best_scaling = min(scaling_results, key=scaling_results.get)

    return best_scaling

# Assuming 'dataframes' is your dictionary of DataFrames
best_scaling = apply_feature_scaling(dataframes['BlockchainTransactions'], ['AmountTransferred', 'TransactionFee', 'BlockHeight'])

print("Best scaling method:", best_scaling)





from sklearn.preprocessing import MinMaxScaler



# Selecting the columns to scale
columns_to_scale = ['AmountTransferred', 'TransactionFee', 'BlockHeight']

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the selected columns
dataframes['BlockchainTransactions'][columns_to_scale] = scaler.fit_transform(dataframes['BlockchainTransactions'][columns_to_scale])




def OnehotcodeEncoding(df,categorical_columns):

    df = pd.get_dummies(df, columns=categorical_columns)
    return df





dataframes['BlockchainTransactions'] = OnehotcodeEncoding(dataframes['BlockchainTransactions'], ['ReceiverAddress'])





# Define the columns to drop
columns_to_drop = ['TransactionID', 'SenderAddress']

# Specify the name of the DataFrame in the 'dataframes' dictionary
table_update = 'BlockchainTransactions'

# Call the function to drop columns
droper(dataframes, table_update, columns_to_drop)



dataframes['BehavioralPatterns'] = apply_nlp_label_encoding(dataframes['BehavioralPatterns'], ['TransactionSizeDistribution', 'GeographicInconsistencies', 'TimeOfDayPatterns', 'RegularVsIrregularBehavior', 'ChangesInBehaviorOverTime'])

from sklearn.preprocessing import StandardScaler

def numerical_preprocessing(data_frames, table_name, columns):
    if table_name in data_frames:
        scaler = StandardScaler()
        data_frames[table_name][columns] = scaler.fit_transform(data_frames[table_name][columns])
    else:
        print(f"Table '{table_name}' not found in the data_frames dictionary.")

# Assuming 'dataframes' is your dictionary of DataFrames
numerical_preprocessing(dataframes, "BehavioralPatterns", ['TransactionFrequency'])


# Define the columns to drop
columns_to_drop = ['UserID']

# Specify the name of the DataFrame in the 'dataframes' dictionary
table_update = 'BehavioralPatterns'

# Call the function to drop columns
droper(dataframes, table_update, columns_to_drop)

columns_to_drop = ['UserID']

# Specify the name of the DataFrame in the 'dataframes' dictionary
table_update = 'CreditAndFinancialHistory'

# Call the function to drop columns
droper(dataframes, table_update, columns_to_drop)
dataframes['CreditAndFinancialHistory']
best_scaling = apply_feature_scaling(dataframes['CreditAndFinancialHistory'], ['CreditScore', 'CreditCardUtilization', 'IncomeLevel','DebtToIncomeRatio'])

print("Best scaling method:", best_scaling)


from sklearn.preprocessing import RobustScaler


# Selecting the columns to scale
columns_to_scale = ['CreditScore', 'CreditCardUtilization', 'IncomeLevel', 'DebtToIncomeRatio']

# Initialize RobustScaler
scaler = RobustScaler()

# Fit and transform the selected columns
dataframes['CreditAndFinancialHistory'][columns_to_scale] = scaler.fit_transform(dataframes['CreditAndFinancialHistory'][columns_to_scale])
dataframes['CreditAndFinancialHistory'] = apply_nlp_label_encoding(dataframes['CreditAndFinancialHistory'], ['LoanRepaymentHistory', 'PastFraudulentActivity', ])
best_scaling = apply_feature_scaling(dataframes['NetworkAnalysis'], ['DegreeCentrality', 'ClusteringCoefficients'])

print("Best scaling method:", best_scaling)


from sklearn.preprocessing import StandardScaler

# Extract the DataFrame
df = dataframes['NetworkAnalysis']

# Columns to be scaled
columns_to_scale = ['DegreeCentrality', 'ClusteringCoefficients']

# Initialize the StandardScaler
scaler = StandardScaler()

# Apply StandardScaler to the specified columns
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# Update the DataFrame in the dataframes dictionary
dataframes['NetworkAnalysis'] = df


def lableEncoding(df,categorical_columns):
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()


    df[categorical_columns] = df[categorical_columns].apply(label_encoder.fit_transform)
    return df

dataframes['NetworkAnalysis'] = lableEncoding(dataframes['NetworkAnalysis'],['AnomaliesInNetwork'])

columns_to_drop = ['SenderUserID','ReceiverUserID']

# Specify the name of the DataFrame in the 'dataframes' dictionary
table_update = 'NetworkAnalysis'

# Call the function to drop columns
droper(dataframes, table_update, columns_to_drop)
dataframes['SentimentAnalysis'] = apply_nlp_label_encoding(dataframes['SentimentAnalysis'],['SocialMediaSentiment','TransactionSentiment', 'FinancialTransactionSentiment'])
# Define the columns to drop
columns_to_drop = ['UserID']

# Specify the name of the DataFrame in the 'dataframes' dictionary
table_update = 'SentimentAnalysis'

# Call the function to drop columns
droper(dataframes, table_update, columns_to_drop)
from sklearn.preprocessing import StandardScaler

# Extract the DataFrame
df = dataframes['CommunityBehavior']

# Columns to be scaled
columns_to_scale = ['ReputationScore']

# Initialize the StandardScaler
scaler = StandardScaler()

# Apply StandardScaler to the specified columns
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# Update the DataFrame in the dataframes dictionary
dataframes['CommunityBehavior'] = df


# In[44]:


dataframes['CommunityBehavior'] = apply_nlp_label_encoding(dataframes['CommunityBehavior'],['ParticipationInForums','FeedbackFromPeers','CommunityEndorsementsWarnings'])
# Define the columns to drop
columns_to_drop = ['UserID']

# Specify the name of the DataFrame in the 'dataframes' dictionary
table_update = 'CommunityBehavior'

# Call the function to drop columns
droper(dataframes, table_update, columns_to_drop)
# Assuming 'dataframes' is your dictionary of DataFrames
best_scaling = apply_feature_scaling(dataframes['SystemAndPlatformScores'], ['SystemTrustScore', 'PlatformReliabilityScore'])

print("Best scaling method:", best_scaling)


from sklearn.preprocessing import MinMaxScaler
df_system_platform_scores = dataframes['SystemAndPlatformScores'][['SystemTrustScore', 'PlatformReliabilityScore']]

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the data
scaled_data = scaler.fit_transform(df_system_platform_scores)

# Create a new DataFrame with scaled data
scaled_df_system_platform_scores = pd.DataFrame(scaled_data, columns=['SystemTrustScore_scaled', 'PlatformReliabilityScore_scaled'])

# Update the original DataFrame with the scaled scores
dataframes['SystemAndPlatformScores'] = pd.concat([dataframes['SystemAndPlatformScores'].drop(['SystemTrustScore', 'PlatformReliabilityScore'], axis=1), scaled_df_system_platform_scores], axis=1)
# Define the columns to drop
columns_to_drop = ['UserID']

# Specify the name of the DataFrame in the 'dataframes' dictionary
table_update = 'SystemAndPlatformScores'

# Call the function to drop columns
droper(dataframes, table_update, columns_to_drop)

def lableEncoding(df, categorical_columns):
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()

    for column in categorical_columns:
        df[column] = label_encoder.fit_transform(df[column])

    return df
dataframes['SystemAndPlatformScores'] = lableEncoding(dataframes['SystemAndPlatformScores'],['PastSecurityIncidents'])
# Define the columns to drop
columns_to_drop = ['UserID','IPAddress','DeviceLocation','DeviceID','DeviceTypeAndVersion']

# Specify the name of the DataFrame in the 'dataframes' dictionary
table_update = 'DeviceAndIPInformation'

# Call the function to drop columns
droper(dataframes, table_update, columns_to_drop)
dataframes['DeviceAndIPInformation'] = lableEncoding(dataframes['DeviceAndIPInformation'],['ProxyOrVPNUsage'])
dataframes['MachineLearningFeatures'] = lableEncoding(dataframes['MachineLearningFeatures'],['HistoricalFraudData','AnomalyDetectionFeatures','ClusteringFeatures','DeepLearningModelFeatures'])
# Define the columns to drop
columns_to_drop = ['UserID']

# Specify the name of the DataFrame in the 'dataframes' dictionary
table_update = 'MachineLearningFeatures'

# Call the function to drop columns
droper(dataframes, table_update, columns_to_drop)
# Define the columns to drop
columns_to_drop = ['TransactionID','BlacklistedEntities']

# Specify the name of the DataFrame in the 'dataframes' dictionary
table_update = 'HistoricalFraudData'

# Call the function to drop columns
droper(dataframes, table_update, columns_to_drop)
dataframes['HistoricalFraudData'] = lableEncoding(dataframes['HistoricalFraudData'],['FraudulentTransactions','PreviousFraudulentActivity'])

import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def apply_mice(data, max_iter=10, random_state=None):


    # Convert to numpy array if DataFrame is provided
    if hasattr(data, 'to_numpy'):
        data = data.to_numpy()

    # Initialize MICE imputer
    mice_imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)

    # Fill missing values using MICE
    filled_data = mice_imputer.fit_transform(data)

    return filled_data

import pandas as pd

def combiner(dataframes, table_names):
    combined_dataframes = {}
    for table_name in table_names:
        combined_dataframes[table_name] = dataframes[table_name]
        
    combined_df = pd.concat(combined_dataframes.values(), axis=1)
    return combined_df

# Assuming dataframes is a dictionary containing DataFrames for each table
combine_df = combiner(dataframes, table_names)

table_names = [
    "BlockchainTransactions",
    "UserProfile",
    "BehavioralPatterns",
    "CreditAndFinancialHistory",
    "NetworkAnalysis",
    "SentimentAnalysis",
    "CommunityBehavior",
    "SystemAndPlatformScores",
    "DeviceAndIPInformation",
    "MachineLearningFeatures",
    "HistoricalFraudData",
    
]

combine_df=combiner(dataframes,table_names)

combine_df.drop
combine_df.drop(columns=['Timestamp', 'AccountCreationDate'], inplace=True)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
from tensorflow.keras import layers

# Load the dataset
data = combine_df

# Separate features and target variable
X = data.drop(columns=['FraudulentTransactions'])
y = data['FraudulentTransactions']

X_encoded = X
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Assuming X contains your features and y contains your target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the logistic regression model
logistic_regression = LogisticRegression()

# Fit the model on the training data
logistic_regression.fit(X_train, y_train)

# Predict the target variable on the testing data
y_pred = logistic_regression.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print("Accuracy:", accuracy)

# Convert data types if necessary
X_train = X_train.astype('float32')
y_train = y_train.astype('float32')
X_test = X_test.astype('float32')
y_test = y_test.astype('float32')

# Build the neural network model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[X_test.shape[1]]),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=15, batch_size=20, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report



rf = RandomForestClassifier()
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=2, n_jobs=-1, verbose=2)

grid_search.fit(X_train, y_train)
print("Best parameters found: ", grid_search.best_params_)
best_rf = grid_search.best_estimator_

y_pred = best_rf.predict(X_test)
print(classification_report(y_test, y_pred))
