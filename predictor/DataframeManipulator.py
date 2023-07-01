import pandas as pd


class DataframeManipulator:
    def __init__(self) -> None:
        pass

    def prepare_dataframe(self, df):
        """
        Prepares the dataframe by:
            Reading the csv
            Splitting the datetime into months, days and hours
            One hot encoding the months, days and hours

        Returns the dataframe
        """
        self.split_datetime(df)
        df = self.one_hot_encode(df, "month")
        df = self.one_hot_encode(df, "day")
        df = self.one_hot_encode(df, "hour")

        return df

    def split_datetime(self, df):
        """
        Splits the datetime into months, days and hours
        """
        df['month'] = df.index.month
        df['day'] = [i.weekday() + 1 for i in df.index]
        df['hour'] = df.index.hour

    def one_hot_encode(self, df, column):
        """
        One hot encodes a column of a dataframe and returns the dataframe with the one hot encode columns added
        """
        # Get one hot encoding of columns B
        one_hot = pd.get_dummies(df[column], column)
        # Drop column B as it is now encoded
        df = df.drop(column, axis = 1)
        # Join the encoded df
        df = df.join(one_hot)
        return df
