from predictor.Co2Predictor import Co2Predictor
from predictor.DataframeManipulator import DataframeManipulator
import csv
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression




def main():
    # # Read in the CSV and get the data ready
    # timestamplist, valuelist = get_the_data_ready()

    # # Check the occupancy
    # testlist = check_occupancy(timestamplist, valuelist, 10)

    # # Show us the magic
    # plot_double_graph(timestamplist, valuelist, testlist)

    df_man = DataframeManipulator()
    pred = Co2Predictor()

    train_df = df_man.prepare_dataframe("csv/train.csv")
    validation_df = df_man.prepare_dataframe("csv/validation.csv")

    print(train_df.head())

    pred.predict(train_df, validation_df)



# def get_the_data_ready():
#     #Read in the CSV File
#     with open('1679526000-1679612399-datapoint_3291.csv') as csvfile:
#         data = list(csv.reader(csvfile, delimiter=","))


#     #Convert from string to int, float and datetime
#     for row in data[1:]:
#         #Convert from string to int
#         row[0] = int(row[0])
#         #Convert from int to timestamp
#         #row[0] = datetime.utcfromtimestamp(row[0]).strftime('%Y-%m-%d %H:%M:%S ')
#         #Convert from string to float
#         row[1] = float(row[1])


#     #Seperate the csv data into two different lists
#     timestamplist = []
#     valuelist = []
#     for datas in data:
#         timestamplist.append(datas[0])
#         valuelist.append(datas[1])


#     #Remove the headers to only get workable data
#     timestamplist.pop(0)
#     valuelist.pop(0)
#     return timestamplist, valuelist


# def plot_double_graph(timestamplist, valuelist, testlist):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Time in Unix-timestamps')
    ax1.set_ylabel('CO2 Measurement in PPM', color=color)
    ax1.plot(timestamplist, valuelist, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    """
    It automatically places every timestamp on the x-axis, so we have to calculate ourself what we put on it. 
    We do this every 20% meaning we get 6 points on the x-axis.
    """
    listlen = len(timestamplist)
    point0 = timestamplist[1]
    point1 = timestamplist[round(listlen / 100 * 20)]
    point2 = timestamplist[round(listlen / 100 * 40)]
    point3 = timestamplist[round(listlen / 100 * 60)]
    point4 = timestamplist[round(listlen / 100 * 80)]
    point5 = timestamplist[-1]

    # Set values for the x and y-axis
    fig.gca().set_xticks([point0, point1, point2, point3, point4, point5])

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis


    color = 'tab:blue'
    ax2.set_ylabel('Is the average CO2 below or above 500? Window = 20 Min', color=color)  # we already handled the x-label with ax1
    ax2.plot(timestamplist, testlist, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show() # Show us the Magic


# def check_occupancy(timestamplist, valuelist, windowsize):
#     testlist = []
#     x = 0
#     while x in range(len(timestamplist)):
#         # Get the average of the CO2 Values
#         sum_check_list = sum(valuelist[x:x + windowsize]) / windowsize
#         # Set the CO2 Values in a Numpy Array
#         check_list = np.array(valuelist[x:x + windowsize])
#         # Set the timestamps in a Numpy Array
#         check_time_list = np.array(timestamplist[x:x + windowsize]).reshape((-1, 1))
#         # Make an instance of LinearRegression and fit the Values in it
#         model = LinearRegression().fit(check_time_list, check_list)
#         # Score the Model
#         r_sq = model.score(check_time_list, check_list)
#         # Get the Coefficient
#         line_slope = model.coef_
#         print(timestamplist[x], line_slope, 1/line_slope)


#         if sum_check_list > 800 or (line_slope >= 0 and sum_check_list > 500):
#             # If line is positive fill with 1's
#             l = [1] * windowsize
#             testlist = testlist + l
#         elif sum_check_list <= 500 or line_slope < 0:
#             # If average CO2 values are below 500 or the slope is negative fill the list with zeros
#             l = [0] * windowsize
#             testlist = testlist + l

#         """
#         # If average CO2 values are below 500 or the slope is negative fill the list with zeros
#         if sum_check_list < 500 or line_slope < 0:
#             l = [0] * windowsize
#             #counter += 1
#             testlist = testlist + l
#         elif line_slope >= 0: # If line is positive fill with 1's
#             l = [1] * windowsize
#             #counter += 4
#             testlist = testlist + l
#         """
#         x += windowsize # Slide the window

#     # We now fill the window EVERY time with 10 values, so pop the last values until the lists match in size
#     while len(testlist) > len(timestamplist):
#         testlist.pop()
#     return testlist


if __name__ == "__main__":
    main()

