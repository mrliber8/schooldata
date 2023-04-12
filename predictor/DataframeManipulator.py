import pandas as pd
import datetime

from sklearn.preprocessing import OneHotEncoder

class DataframeManipulator:
    def __init__(self) -> None:
        pass

    def prepare_dataframe(self, csv):
        """
        Prepares the dataframe by:
            Reading the csv
            Splitting the datetime into months, days and hours
            One hot encoding the months, days and hours

        Returns the dataframe
        """
        df = pd.read_csv(csv)
        self.split_datetime(df)
        df = self.one_hot_encode(df, "month")
        df = self.one_hot_encode(df, "day")
        df = self.one_hot_encode(df, "hour")

        return df

    def split_datetime(self, df):
        """
        Splits the datetime into months, days and hours
        """
        df['month'] = df.apply(lambda row: (int(row.timestamp[5:7])), axis = 1)
        df['day'] = df.apply(lambda row: (datetime.datetime(int(row.timestamp[:4]), int(row.timestamp[5:7]), int(row.timestamp[8:10])).weekday() + 1), axis = 1) 
        df['hour'] = df.apply(lambda row: (int(row.timestamp[11:13])), axis = 1)

    def one_hot_encode(self, df, column):
        """
        One hot encodes a column of a dataframe and returns the dataframe with the one hot encode columns added
        """
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoder_df = pd.DataFrame(encoder.fit_transform(df[[column]]).toarray())
        encoder_df.columns = encoder.get_feature_names_out()
        df = df.join(encoder_df)
        return df
