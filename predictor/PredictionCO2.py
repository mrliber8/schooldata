import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class CO2PredictionModel:
    def __init__(self, train_file, test_file, sequence_length=10, window_size=1):
        self.train_file = train_file
        self.test_file = test_file
        self.sequence_length = sequence_length
        self.window_size = window_size

    def load_data(self):
        train_data = pd.read_csv(self.train_file, parse_dates=['timestamp'])
        test_data = pd.read_csv(self.test_file, parse_dates=['timestamp'])
        return train_data, test_data

    def preprocess_data(self, train_data, test_data):
        scaler = MinMaxScaler()
        scaled_train_data = scaler.fit_transform(train_data['value'].values.reshape(-1, 1))
        scaled_test_data = scaler.transform(test_data['value'].values.reshape(-1, 1))
        return scaler, scaled_train_data, scaled_test_data

    def generate_sequences(self, data):
        X = []
        y = []
        for i in range(len(data) - self.sequence_length - self.window_size + 1):
            X.append(data[i:i+self.sequence_length])
            y.append(data[i+self.sequence_length:i+self.sequence_length+self.window_size])
        return np.array(X), np.array(y)

    def build_model(self):
        model = Sequential()
        model.add(LSTM(64, activation='relu', input_shape=(self.sequence_length, 1)))
        model.add(Dense(self.window_size))
        model.compile(optimizer='adam', loss='mse')
        return model

    def train_model(self, model, X_train, y_train, epochs=100, batch_size=32):
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def evaluate_model(self, model, X_train, y_train, X_test, y_test):
        train_loss = model.evaluate(X_train, y_train)
        test_loss = model.evaluate(X_test, y_test)
        print(f"Train Loss: {train_loss}")
        print(f"Test Loss: {test_loss}")

    def predict_values(self, model, X_test, scaler):
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        return predictions

    def plot_predictions(self, test_data, sequence_length, predictions):
        actual_values = scaler.inverse_transform(y_test.reshape(-1, 1))
        predictions_df = pd.DataFrame({
            'Timestamp': test_data['timestamp'].values[sequence_length:],
            'CO2_Prediction': predictions.flatten(),
            'CO2_Actual': actual_values.flatten()
        })
        print(predictions_df)
        plt.plot(predictions_df['Timestamp'], predictions_df['CO2_Prediction'], label='Prediction')
        plt.plot(predictions_df['Timestamp'], predictions_df['CO2_Actual'], label='Actual')
        plt.xlabel('Timestamp')
        plt.ylabel('CO2 Value')
        plt.title('CO2 Predictions')
        plt.legend()
        plt.show()

# Create an instance of the CO2PredictionModel class
model = CO2PredictionModel('train.csv', 'test.csv')

# Load the data
train_data, test_data = model.load_data()

# Preprocess the data
scaler, scaled_train_data, scaled_test_data = model.preprocess_data(train_data, test_data)

# Generate sequences for training and test data
X_train, y_train = model.generate_sequences(scaled_train_data)
X_test, y_test = model.generate_sequences(scaled_test_data)

# Build the LSTM model
lstm_model = model.build_model()

# Train the model
model.train_model(lstm_model, X_train, y_train, epochs=1, batch_size=32)

# Evaluate the model
model.evaluate_model(lstm_model, X_train, y_train, X_test, y_test)

# Predict CO2 values for the test set
predictions = model.predict_values(lstm_model, X_test, scaler)

# Plot the predicted and actual values
model.plot_predictions(test_data, model.sequence_length, predictions)
