import csv
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np


#Read in the CSV File
with open('1679526000-1679612399-datapoint_3291.csv') as csvfile:
    data = list(csv.reader(csvfile, delimiter=","))


#Convert from string to float
for row in data[1:]:
    #Convert from string to int
    row[0] = int(row[0])
    #Convert from int to timestamp
    row[0] = datetime.utcfromtimestamp(row[0]).strftime('%Y-%m-%d %H:%M:%S ')
    #Convert from string to float
    row[1] = float(row[1])


#Seperate into two different lists
timestamplist = []
valuelist = []
for datas in data:
    timestamplist.append(datas[0])
    valuelist.append(datas[1])
    #print(datas[0], datas[1])


#Pop the headers
timestamplist.pop(0)
valuelist.pop(0)
print(timestamplist)


def plot_double_graph(timestamplist, valuelist, testlist):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Time in Unix-timestamps')
    ax1.set_ylabel('CO2 Measurement in PPM', color=color)
    ax1.plot(timestamplist, valuelist, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Get values for the x-axis
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
    plt.show()


testlist = []
for item in valuelist:
    if item < 500:
        testlist.append(0)
    else:
        testlist.append(1)


#plot_double_graph(timestamplist, valuelist, testlist)

testlist = []
x = 0
while x in range(len(timestamplist)):
    check_list = valuelist[x:x+10]
    sum_check_list = sum(check_list)
    if (sum_check_list / 10) < 500:
        l = [0] * 10
        testlist = testlist + l
    elif (sum(valuelist[x:x+10]) / 10) >= 500:
        l = [1] * 10
        testlist = testlist + l
    x += 10


while len(testlist) > len(timestamplist):
    testlist.pop()


plot_double_graph(timestamplist, valuelist, testlist)