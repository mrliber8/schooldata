<<<<<<< HEAD
# Random forest co2 predictor

import pandas as pd
=======
>>>>>>> a3657e73ccc4957d5fa5f68ebbb1e45c65b4e6ff
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score


class Co2Predictor:
    def __init__(self) -> None: 

        # The features used to predict the Co2 values
        # It's one hot encoded as it's categorial data   
        self.FEATURES = [
            'hour_0', 'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6', 'hour_7', 'hour_8', 'hour_9',
            'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18', 'hour_19',
            'hour_20', 'hour_21', 'hour_22', 'hour_23',
            
            'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6',
            'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12',

            'day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6', 'day_7'
                ]

    def predict(self, train_df, validation_df):
        """
        Predicts the Co2 values of the validation data based on the train data given
        Returns a list of predicted Co2 values
        """
        
        # Filter features that are not in the validation dataframe
        # Filtering is needed when a prediction is only need for a specific couple of months
        filtered_features = self.filter_unused_features(validation_df)

        # Create the X's and y's for the train and validation for the predictive model
        train_y = train_df.co2
        train_X = train_df[filtered_features]
        # If validation dataframe has a Co2 value, then compared the prediction to the actual value
        if "co2" in validation_df:
            val_y = validation_df['co2'].to_numpy()
        val_X = validation_df[filtered_features]


        # Setup the predictive model
        rf = RandomForestRegressor(random_state=10, max_depth=10)
        rf.fit(train_X, train_y)
        predicted_val_y = rf.predict(val_X)
                
        # If validation dataframe has a Co2 value, then compared the prediction to the actual value
        if "co2" in validation_df:
            rf_val_mae = mean_absolute_error(val_y, predicted_val_y)
            print("Gemmidelde absolute afwijking: ", rf_val_mae)

            self.chart_comparison(val_y, predicted_val_y, validation_df)
        return predicted_val_y

    def filter_unused_features(self, df):
        """
        Returns a list of features which are used in the validation data
        """
        filtered_features = []
        for feature in self.FEATURES:
            if feature in df:
                filtered_features.append(feature)
        return filtered_features

    def chart_comparison(self, val_y, predicted_val_y, df):
        """
        Creates and shows a chart displaying the predicted Co2 vs the actual values
        """

        plt.plot(val_y)
        plt.plot(predicted_val_y)
        plt.ylabel('CO2 Measurement in PPM')
        plt.xlabel('Timestamp')
        listlen = len(df) - 1

        # Creates the ticks on the X-axis
        points = []
        ticks = []
        tick_count = 13
        for i in range(tick_count):
            points.append(str(df.iloc[round(listlen / (tick_count - 1) * i)].name)[:10])
            ticks.append(round(listlen / (tick_count - 1) * i))
        plt.xticks(ticks, points)

        plt.show()
