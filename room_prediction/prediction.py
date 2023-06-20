import pandas as pd
from room_prediction.predict_lstm import predict_lstm

class Prediction:
    def predict_occupancy_logic(self, df, window_size=30, co2_threshold=600):
        """
        Cretae a variable (in_room) based on the CO2 values and their rolling window statistics.

        :param df: (pd.DataFrame) A pandas dataframe with two columns: 'timestamp' and 'co2'.
        :param window_size: (int) The number of periods to include in the rolling window.
        :param co2_threshold: CO2 value above which it is considered that someone is in the room.
        :return df_copy: (pd.DataFrame) A new dataframe with an additional 'in_room' column indicating if someone is in the room.
        """

        #Copy the dataframe and use the new dataframe for the calcualtions
        df_copy = df.copy()
        #Set the timestamp as index
        df_copy.index = pd.to_datetime(df_copy.index)

        # Calculate the rolling mean, median, and standard deviation
        df_copy['rolling_mean'] = df_copy['co2'].rolling(window=window_size, min_periods=1).mean()
        df_copy['rolling_median'] = df_copy['co2'].rolling(window=window_size, min_periods=1).median()
        #df_copy['rolling_std'] = df_copy['co2'].rolling(window=window_size, min_periods=1).std().fillna(0)

        # Condition to determine if someone is in the room
        condition = (df_copy['co2'] > co2_threshold) & (df_copy['rolling_mean'] > co2_threshold) & (
                    df_copy['rolling_median'] > co2_threshold)

        # Set the value as an int
        df_copy['in_room'] = condition.astype(int)

        return df_copy


    def get_first_last(self, df):
        """
        Get a dataframe displaying the first and last time of a day that someone is in the room

        :param df: (pd.DataFrame) A pandas dataframe with the columns: 'timestamp' and 'in_room'.
        :return: result_df: (pd.dataframe) A pandas dataframe grouped by date displaying the times
        """

        # Ensure it is datetime
        df.index = pd.to_datetime(df.index)

        # Create a new column for the date (without time)
        # print(type(df.iloc[0]))
        df['date'] = df.index.datetime.date

        # Filter out rows where value is not 1
        df = df[df['in_room'] == 1]

        # Group by the date and get the min and max datetime for each date
        result_df = df.groupby('date').agg(first=('timestamp', 'min'), last=('timestamp', 'max'))

        return result_df


    def main(self, df):
        """
        Main function for the prediction file
        """
        # Read in the data
        # df = pd.read_csv('1683496800-1683755999-datapoint_3291.csv')
        # Makes it clearer to understand
        df.rename(columns = {'value':'co2'}, inplace = True)
        # Convert to datetime
        df.index = pd.to_datetime(df.index, unit='s')

        # Predict the occupancy based on logic
        df_with_occupancy_logic = self.predict_occupancy_logic(df)
        print(df_with_occupancy_logic)
        df_first_last = self.get_first_last(df_with_occupancy_logic)
        print(df_first_last)

        # Predict the occupancy based on the LSTM model
        df_with_occupancy_lstm = predict_lstm.predict_occupancy_lstm(df)
        print(df_with_occupancy_lstm)
        #df_first_last = get_first_last(df_with_occupancy_lstm)
        #print(df_first_last)

