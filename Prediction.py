import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import the CSV file
df = pd.read_csv('27feb-31maart.csv')


# Convert date column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Extract year, month, day, and hour from date column
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['hour'] = df['date'].dt.hour
# Voeg een nieuwe kolom toe met de dagnamen
df['dag'] = df['date'].dt.day_name()
df['boven_500'] = df['CO2'].apply(lambda x: 1 if x > 500 else 0)

# Calculate the average CO2 value for each date
df_grouped = df.groupby(['year', 'month', 'day', 'hour']).agg({'CO2': 'mean'}).reset_index()

# Create a new column indicating whether someone was using the room or not
df_grouped['usage'] = np.where(df_grouped['CO2'] > 500, 1, 0)

# Set features as year, month, day, and hour
feature_cols = ['year', 'month', 'day', 'hour']

# Set the target variable as 'usage'
target_col = 'usage'

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_grouped[feature_cols], df_grouped[target_col], test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model using the training sets
model.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Confusion matrix:\n{confusion}')

# Predict the usage for a given date
date = datetime(2022, 7, 5, 22)  # Example date: April 5th, 2022 at 2pm
date_values = [date.year, date.month, date.day, date.hour]
usage = model.predict([date_values])[0]

# Calculate the date in hours
if usage == 1:
    print(f'The room is predicted to be in use on {date} ({date.strftime("%Y-%m-%d %H:%M:%S")} in hours).')
else:
    print(f'The room is predicted to be empty on {date} ({date.strftime("%Y-%m-%d %H:%M:%S")} in hours).')

X= df.iloc[:,1:14]   #all features
Y= df.iloc[:,-1]   #target output (floods)
# Bekijk de resulterende DataFrame
print(df.head())



