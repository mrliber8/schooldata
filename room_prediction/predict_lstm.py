import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

class predict_lstm:

    def create_target_variable(self, df, window_size=30, co2_threshold=600):
        """
        Cretae a variable (in_room) based on the CO2 values and their rolling window statistics.

        :param df: (pd.DataFrame) A pandas dataframe with two columns: 'timestamp' and 'co2'.
        :param window_size: (int) The number of periods to include in the rolling window.
        :param co2_threshold: CO2 value above which it is considered that someone is in the room.
        :return df_copy: (pd.DataFrame) A new dataframe with an additional 'in_room' column indicating if someone is in the room.
        """
        df_copy = df.copy()
        df_copy.index = pd.to_datetime(df_copy['timestamp'])

        # Calculate the rolling mean, median, and standard deviation
        df_copy['rolling_mean'] = df_copy['co2'].rolling(window=window_size, min_periods=1).mean()
        df_copy['rolling_median'] = df_copy['co2'].rolling(window=window_size, min_periods=1).median()
        df_copy['rolling_std'] = df_copy['co2'].rolling(window=window_size, min_periods=1).std().fillna(0)

        # Create a custom condition to determine if someone is in the room
        condition = (df_copy['co2'] > co2_threshold) & (df_copy['rolling_mean'] > co2_threshold) & (
                df_copy['rolling_median'] > co2_threshold)

        df_copy['in_room'] = condition.astype(int)

        return df_copy


    def preprocess_lstm_data(self, df, sequence_length=24 * 30):
        """
        Prepare the data so the LSTm model can make predictions

        :param df: (pd.DataFrame) A pandas dataframe
        :param sequence_length: (int) A value for the sequence length, standard is 24*30(24 hours with 30 measurements per hour)
        :return: X, y: (np.array) Two numpy arrays that the LSTM model uses for the predictions
        """
        df_copy = self.create_target_variable(df)
        data = df_copy['in_room'].values

        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])

        if len(X) == 0:  # If no sequence was created, create one without a target
            X.append(data[:sequence_length])
            y.append(np.nan)

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        return X, y

    def predict_with_saved_model(self, new_df, sequence_length=24 * 30, model_path='lstm_model_daily_predict.h5'):
        """
        Make predictions using the saved model

        :param new_df: (pd.dataframe) A pandas dataframe with two columns: 'timestamp' and 'co2'
        :param sequence_length: (int) A value for the sequence length, standard is 24*30(24 hours with 30 measurements per hour)
        :param model_path: (string) The path to the LSTM model
        :return: prediction: (pd.dataframe) A pandas dataframe containing timestamps and the prediction in binary
        """
        X_new, y_new = self.preprocess_lstm_data(new_df, sequence_length)

        # Load the saved model
        loaded_model = load_model(model_path)

        # Make predictions on new data
        y_pred_new = loaded_model.predict(X_new)
        y_pred_new = np.round(y_pred_new).astype(int)

        if np.isnan(y_new).all():  # If target is NaN, just use the end of the dataframe for timestamp
            timestamp_index_new = new_df['timestamp'].iloc[-1:]
        else:
            timestamp_index_new = pd.to_datetime(new_df['timestamp']).iloc[sequence_length:].reset_index(drop=True)

        # Add timestamps to the predictions
        predictions_new = pd.DataFrame({'timestamp': timestamp_index_new, 'in_room': y_pred_new.flatten()})

        return predictions_new

    def predict_occupancy_lstm(self, df):
        """
        Kicks off the data preparation and the predictions

        :param df: (pd.dataframe) A pandas dataframe with two columns: 'timestamp' and 'co2'
        :return: prediction: (pd.dataframe) A pandas dataframe containing timestamps and the prediction in binary
        """
        if 'value' in df.columns:
            df.rename(columns={'value': 'co2'}, inplace=True)

        prediction = self.predict_with_saved_model(df)

        return prediction