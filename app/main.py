import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from var_dump import var_dump
import datetime

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score


df = pd.read_csv("train.csv")
print(df.head())

df['in_use'] = df.apply(lambda row: (int(row.value > 500)), axis = 1)
df['minutes'] = df.apply(lambda row: (int(row.timestamp[11:13]) * 60 + int(row.timestamp[14:16])), axis = 1)
df['month'] = df.apply(lambda row: (int(row.timestamp[5:7])), axis = 1)
df['day'] = df.apply(lambda row: (datetime.datetime(int(row.timestamp[:4]), int(row.timestamp[5:7]), int(row.timestamp[8:10])).weekday()), axis = 1) 

print(df)
encoder = OneHotEncoder(handle_unknown='ignore')
encoder_df = pd.DataFrame(encoder.fit_transform(df[['day']]).toarray())
encoder_df.columns = encoder.get_feature_names_out()
df = df.join(encoder_df)
print(df)

# Create target object and call it y
train_y = df.value
# Create X
features = ['minutes', 'month', 'day_0','day_1','day_2','day_3','day_4','day_5','day_6']
train_X = df[features]

##########################################################################################################################
df_val = pd.read_csv("validation.csv")
print(df_val.head())

df_val['in_use'] = df.apply(lambda row: (int(row.value > 500)), axis = 1)
df_val['minutes'] = df.apply(lambda row: (int(row.timestamp[11:13]) * 60 + int(row.timestamp[14:16])), axis = 1)
df_val['month'] = df.apply(lambda row: (int(row.timestamp[5:7])), axis = 1)
df_val['day'] = df.apply(lambda row: (datetime.datetime(int(row.timestamp[:4]), int(row.timestamp[5:7]), int(row.timestamp[8:10])).weekday()), axis = 1) 

print(df_val)
encoder = OneHotEncoder(handle_unknown='ignore')
encoder_df = pd.DataFrame(encoder.fit_transform(df[['day']]).toarray())
encoder_df.columns = encoder.get_feature_names_out()
df_val = df_val.join(encoder_df)
print(df_val)

# Create target object and call it y
val_y = df_val.value
# Create X
features = ['minutes', 'month', 'day_0','day_1','day_2','day_3','day_4','day_5','day_6']
val_X = df_val[features]




# Split into validation and training data
# train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

rf = RandomForestRegressor(random_state=10, max_depth=10)
rf.fit(train_X, train_y)
rf_val_mae = mean_absolute_error(val_y, rf.predict(val_X))
print(rf_val_mae)





plt.plot(train_y)
plt.plot(rf.predict(train_X))
plt.show()


# l = df.iloc[:,-1:].values.tolist()
# print(l)
# for x in l:
    
#     xpoints = np.array(l)
#     ypoints = np.array([400, 1300])

# plt.plot(xpoints, ypoints)
# plt.show()




# #Convert from string to float
# for row in data[1:]:
#     #Convert from string to int
#     row[0] = int(row[0])
#     #Convert from int to timestamp
#     row[0] = datetime.utcfromtimestamp(row[0]).strftime('%Y-%m-%d %H:%M:%S ')
#     #Convert from string to float
#     row[1] = float(row[1])


# #Seperate into two different lists
# timestamplist = []
# valuelist = []
# for datas in data:
#     timestamplist.append(datas[0])
#     valuelist.append(datas[1])
#     #print(datas[0], datas[1])


# #Pop the headers
# timestamplist.pop(0)
# valuelist.pop(0)
# print(timestamplist)


# def plot_double_graph(timestamplist, valuelist, testlist):
#     fig, ax1 = plt.subplots()

#     color = 'tab:red'
#     ax1.set_xlabel('Time in Unix-timestamps')
#     ax1.set_ylabel('CO2 Measurement in PPM', color=color)
#     ax1.plot(timestamplist, valuelist, color=color)
#     ax1.tick_params(axis='y', labelcolor=color)

#     # Get values for the x-axis
#     listlen = len(timestamplist)
#     point0 = timestamplist[1]
#     point1 = timestamplist[round(listlen / 100 * 20)]
#     point2 = timestamplist[round(listlen / 100 * 40)]
#     point3 = timestamplist[round(listlen / 100 * 60)]
#     point4 = timestamplist[round(listlen / 100 * 80)]
#     point5 = timestamplist[-1]

#     # Set values for the x and y-axis
#     fig.gca().set_xticks([point0, point1, point2, point3, point4, point5])

#     ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis


#     color = 'tab:blue'
#     ax2.set_ylabel('Is the average CO2 below or above 500? Window = 20 Min', color=color)  # we already handled the x-label with ax1
#     ax2.plot(timestamplist, testlist, color=color)
#     ax2.tick_params(axis='y', labelcolor=color)

#     fig.tight_layout()  # otherwise the right y-label is slightly clipped
#     plt.show()


# testlist = []
# for item in valuelist:
#     if item < 500:
#         testlist.append(0)
#     else:
#         testlist.append(1)


# #plot_double_graph(timestamplist, valuelist, testlist)

# testlist = []
# x = 0
# while x in range(len(timestamplist)):
#     check_list = valuelist[x:x+10]
#     sum_check_list = sum(check_list)
#     if (sum_check_list / 10) < 500:
#         l = [0] * 10
#         testlist = testlist + l
#     elif (sum(valuelist[x:x+10]) / 10) >= 500:
#         l = [1] * 10
#         testlist = testlist + l
#     x += 10


# while len(testlist) > len(timestamplist):
#     testlist.pop()


# plot_double_graph(timestamplist, valuelist, testlist)