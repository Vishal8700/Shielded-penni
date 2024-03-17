import pandas as pd
import numpy as np
import mysql.connector
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Connect to MySQL database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Vishal123",
    database="users_data_new"
)
cursor = conn.cursor()

# Function to fetch data from MySQL and create DataFrame
def fetch_data_to_dataframe(table_name):
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, conn)
    return df

# Function to drop columns from DataFrame
def droper(data_frames, table_name, col_arr):
    data_frames[table_name] = data_frames[table_name].drop(columns=col_arr)

# Function to handle date columns
def handle_dates(data, column_names):
    df = data.copy()
    for column_name in column_names:
        df[column_name] = pd.to_datetime(df[column_name])
    return df

# Function to apply NLP label encoding
def apply_nlp_label_encoding(df, column_names):
    stemmer = PorterStemmer()
    label_encoder = LabelEncoder()
    for column in column_names:
        if column in df.columns and df[column].dtype == 'object':
            df[column] = df[column].apply(lambda x: ' '.join([stemmer.stem(word) for word in word_tokenize(x.lower())]))
            df[column] = label_encoder.fit_transform(df[column])
    return df

# Function to apply feature scaling
def apply_feature_scaling(dataframe, column_names):
    X = dataframe[column_names].values
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test)
    return scaled_X_train, scaled_X_test

# Function to perform one-hot encoding
def OnehotcodeEncoding(df, categorical_columns):
    df = pd.get_dummies(df, columns=categorical_columns)
    return df

# Function to combine multiple DataFrames
def combiner(dataframes, table_names):
    combined_dataframes = {}
    for table_name in table_names:
        combined_dataframes[table_name] = dataframes[table_name]
    combined_df = pd.concat(combined_dataframes.values(), axis=1)
    return combined_df

# Function to apply MICE (Multiple Imputation by Chained Equations)
def apply_mice(data, max_iter=10, random_state=None):
    if hasattr(data, 'to_numpy'):
        data = data.to_numpy()
    mice_imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
    filled_data = mice_imputer.fit_transform(data)
    return filled_data

# Fetching data and creating DataFrames for each table
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
    "HistoricalFraudData"
]
dataframes = {}
for table_name in table_names:
    dataframes[table_name] = fetch_data_to_dataframe(table_name)

# Close the database connection
conn.close()

# Dropping specified columns from UserProfile table
columns_to_drop = ['Name', 'Address', 'Email', 'UserID', 'PhoneNumber']
table_update = 'UserProfile'
droper(dataframes, table_update, columns_to_drop)

# Handling dates in UserProfile and BlockchainTransactions tables
dataframes["UserProfile"] = handle_dates(dataframes["UserProfile"], ['AccountCreationDate'])
dataframes['BlockchainTransactions'] = handle_dates(dataframes['BlockchainTransactions'], ['Timestamp'])

# Applying NLP label encoding
dataframes["UserProfile"] = apply_nlp_label_encoding(dataframes["UserProfile"], ['KYCStatus'])
dataframes['BehavioralPatterns'] = apply_nlp_label_encoding(dataframes['BehavioralPatterns'], ['TransactionSizeDistribution', 'GeographicInconsistencies', 'TimeOfDayPatterns', 'RegularVsIrregularBehavior', 'ChangesInBehaviorOverTime'])
dataframes['SentimentAnalysis'] = apply_nlp_label_encoding(dataframes['SentimentAnalysis'],['SocialMediaSentiment','TransactionSentiment', 'FinancialTransactionSentiment'])
dataframes['CommunityBehavior'] = apply_nlp_label_encoding(dataframes['CommunityBehavior'],['ParticipationInForums','FeedbackFromPeers','CommunityEndorsementsWarnings'])

# Applying feature scaling
scaled_X_train, scaled_X_test = apply_feature_scaling(dataframes['BlockchainTransactions'], ['AmountTransferred', 'TransactionFee', 'BlockHeight'])

# Applying one-hot encoding
dataframes['BlockchainTransactions'] = OnehotcodeEncoding(dataframes['BlockchainTransactions'], ['ReceiverAddress'])

# Dropping specified columns from BlockchainTransactions and NetworkAnalysis tables
columns_to_drop = ['TransactionID', 'SenderAddress']
table_update = 'BlockchainTransactions'
droper(dataframes, table_update, columns_to_drop)

columns_to_drop = ['SenderUserID', 'ReceiverUserID']
table_update = 'NetworkAnalysis'
droper(dataframes, table_update, columns_to_drop)

# Applying MICE for missing value imputation
def lableEncoding(df,categorical_columns):
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()


    df[categorical_columns] = df[categorical_columns].apply(lableEncoding)
    return df

dataframes['HistoricalFraudData'] = lableEncoding(dataframes['HistoricalFraudData'],['FraudulentTransactions','PreviousFraudulentTransaction'])


# Combining DataFrames
combine_df = combiner(dataframes, table_names)

# Dropping specified columns from the combined DataFrame
combine_df.drop(columns=['Timestamp', 'AccountCreationDate'], inplace=True)

# Splitting the combined DataFrame into features (X) and target variable (y)
X = combine_df.drop(columns=['FraudulentTransactions'])
y = combine_df['FraudulentTransactions']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training a logistic regression model
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
y_pred = logistic_regression.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy:", accuracy)

# Building and training a neural network model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[X_test.shape[1]]),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=15, batch_size=20, validation_data=(X_test, y_test))
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Neural Network Test Accuracy:", test_accuracy)

