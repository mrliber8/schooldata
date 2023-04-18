import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class predict_and_show:



    def get_the_data_ready(self):
        df = pd.read_csv('csv/validation.csv', sep=",")
        return df


    def check_occupancy(self, df):
        testlist = []
        x = 0
        windowsize = 10


        while x in range(len(df.index)):
            #Get the average of the CO2 Values
            average_co2 = df['value'][x:x + windowsize].mean()
            # Set the CO2 Values in a Numpy Array
            co2_value_check_array = df.iloc[x:x + windowsize, 1].values.reshape(-1, 1)  # values converts it into a numpy array


            df['timestamp'] = pd.to_datetime(df['timestamp'])


            # Set the timestamps in a Numpy Array
            timestamp_value_check_array = df.iloc[x:x + windowsize, 0].values.reshape(-1, 1)  # values converts it into a numpy array

            # Make an instance of LinearRegression and fit the Values in it
            model = LinearRegression().fit(timestamp_value_check_array, co2_value_check_array)
            # Score the Model
            #r_sq = model.score(timestamp_value_check_array, co2_value_check_array)
            # Get the Coefficient
            line_slope = model.coef_
            # Print the values
            print(df['timestamp'][x], line_slope, 1/line_slope)


            if average_co2 > 800 or (line_slope >= 0 and average_co2 > 500):
                # If line is positive fill with 1's
                l = [1] * windowsize
                testlist = testlist + l
            elif average_co2 <= 500 or line_slope < 0:
                # If average CO2 values are below 500 or the slope is negative fill the list with zeros
                l = [0] * windowsize
                testlist = testlist + l

            x += windowsize  # Slide the window

        while len(testlist) > len(df.index):
            testlist.pop()
        return testlist



    def plot_graph(self, df, testlist, predicted_value_y):



        #Turn it into a numpy array
        timestamp_array = np.array(df['timestamp'])
        testlist_array = np.array(testlist)

        # Turn it into a pandas dataframe
        dataset = pd.DataFrame(np.hstack((timestamp_array.reshape(-1, 1), testlist_array.reshape(-1, 1))))

        # Turn it into human readable dates
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        #Create the figure and make axis 1
        fig, ax1 = plt.subplots()

        # plot line graph on axis #1
        ax1 = sns.lineplot(
            x=df['timestamp'],
            y=df['value'],
            data=df,
            sort=False,
            color='red'
        )
        ax1.set_ylabel('CO2 Measurement in PPM', color='red')
        ax1.legend(['CO2 Measurement'], loc="upper left")

        # set up the 2nd axis
        ax2 = ax1.twinx()

        ax2 = sns.lineplot(
            x=df['timestamp'],
            y=testlist_array,
            data=dataset,
            sort=False,
            color='blue'
        )

        ax2.set_ylim()
        ax2.set_ylabel('Prediction: Is the room occupied? Window = 20 Min', color='blue')
        ax2.legend(['predicted occupancy'], loc="upper right")

        plt.show()

    def setup_co2(self, predicted_val_y):

        # Read in the CSV
        df = self.get_the_data_ready()

        # Check the occupancy
        testlist = self.check_occupancy(df)

        # Show us the magic
        self.plot_graph(df, testlist, predicted_val_y)
